"""VPGARCH-SV model: dynamics, positivity, and variance-moment machinery.

Implements Eqs. (9)-(12) and Proposition 3 (Sec. 2.2.1), the one-step
transition matrix Q_m on Pol_m(R_+) (Sec. 2.3.1), the moment-stability
condition of Proposition 5, and the NGARCH-SV/NGARCH nesting of Remark 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src import innovations as innov
from src.poly_basis import build_Q_variance, conditional_moment_1d


@dataclass
class VPGARCHParams:
    tau: float
    rho: float
    r: float
    lam: float
    sigma2: float
    phi: float
    eta: float
    a0: float
    a1: float
    b0: float
    b1: float

    @property
    def omega(self) -> float:
        """omega(tau) = [1 - phi(tau)] * sigma^2(tau), Eq. (11)."""
        return (1.0 - self.phi) * self.sigma2


def g_vp(x: np.ndarray | float, p: VPGARCHParams) -> np.ndarray | float:
    """G(x) = a0 + a1*x + x^2, the argument of the sqrt in psi_2, Eq. (12)."""
    return p.a0 + p.a1 * x + x**2


def deltas(p: VPGARCHParams) -> tuple[float, float, float]:
    """delta_2, delta_1, delta_0 of Proposition 3."""
    delta2 = p.eta**2 - 4.0 * p.b1 * (p.phi - p.b1)
    delta1 = p.eta**2 * p.a1 - 4.0 * (p.b0 * p.phi + p.b1 * p.omega - 2.0 * p.b0 * p.b1)
    delta0 = p.eta**2 * p.a0 - 4.0 * p.b0 * (p.omega - p.b0)
    return delta2, delta1, delta0


def check_positivity(p: VPGARCHParams) -> dict:
    """Proposition 3: necessary and sufficient conditions for h_next >= 0 for all x>=0, eps2 in R."""
    delta2, delta1, delta0 = deltas(p)
    conds = {
        "a0_nonneg": p.a0 >= 0,
        "a1_bound": p.a1 >= -2.0 * np.sqrt(max(p.a0, 0.0)),
        "b0_nonneg": p.b0 >= 0,
        "b1_pos": p.b1 > 0,
        "delta2_nonpos": delta2 <= 1e-12,
        "delta0_nonpos": delta0 <= 1e-12,
        "delta1_bound": delta1 <= 2.0 * np.sqrt(max(delta2, 0.0) * max(delta0, 0.0)) + 1e-10,
    }
    conds["all"] = all(conds.values())
    return conds


def h_next(x: np.ndarray, eps2: np.ndarray, p: VPGARCHParams) -> np.ndarray:
    """One-step variance update, Eq. (10) with psi loadings (11)-(12)."""
    G = g_vp(x, p)
    return (
        p.omega
        + p.phi * x
        + p.eta * np.sqrt(np.clip(G, 0.0, None)) * eps2
        + (p.b0 + p.b1 * x) * (eps2**2 - 1.0)
    )


def simulate_paths(
    p: VPGARCHParams, n_steps: int, n_paths: int, dist: str, rng: np.random.Generator, nu: float | None = None
) -> dict:
    """Simulate N paths of (y, h) for n_steps periods under P^(n)."""
    y = np.zeros((n_paths, n_steps + 1))
    h = np.zeros((n_paths, n_steps + 1))
    h[:, 0] = p.phi * 0 + p.sigma2  # deterministic initial state h_0 = sigma^2
    for k in range(n_steps):
        eps1, eps2 = innov.sample_pair(dist, p.rho, n_paths, rng, nu)
        y[:, k + 1] = p.r * p.tau + p.lam * h[:, k] + np.sqrt(p.tau * h[:, k]) * eps1
        h[:, k + 1] = h_next(h[:, k], eps2, p)
        if np.any(h[:, k + 1] < 0):
            raise ValueError("negative variance encountered: positivity restriction violated")
    return {"y": y, "h": h}


def transition_matrix(p: VPGARCHParams, m: int, dist: str, nu: float | None = None) -> np.ndarray:
    """Q_m on Pol_m(R_+) built from the exact moment recursion (Sec. 2.3.1)."""
    even_m = innov.even_moments_eps2(2 * m, dist, p.rho, nu)
    return build_Q_variance(m, p.phi, p.omega, p.eta, [p.a0, p.a1, 1.0], p.b0, p.b1, even_m)


def multi_step_moment(Q: np.ndarray, x0: float, s: int, degree: int) -> float:
    """E[h_{t+s}^degree | h_t = x0] via H_m(x0)^T Q^s p_{h^degree}, Proposition 1."""
    return conditional_moment_1d(Q, x0, s, degree)


def a_vp_moment(p: VPGARCHParams, j: int, dist: str, nu: float | None = None) -> float:
    """E[A_VP^j], A_VP = phi + eta*eps2 + b1*(eps2^2 - 1), the degree-j diagonal entry q_jj (Prop. 5)."""
    # A_VP expanded via binomial theorem in eps2 using available even moments.
    rng = np.random.default_rng(0)
    n = 500_000
    eps2 = innov.sample_pair(dist, 0.0, n, rng, nu)[1]
    a = p.phi + p.eta * eps2 + p.b1 * (eps2**2 - 1.0)
    return float(np.mean(a**j))


def diagonal_vp(p: VPGARCHParams, degree: int, dist: str, nu: float | None = None) -> float:
    """Exact E[A_VP^degree] via quadrature/moment expansion (diagonal entry q_jj of Prop. 5).

    A_VP = (phi - b1) + eta*eps2 + b1*eps2^2, expanded by the multinomial theorem.
    """
    c0 = p.phi - p.b1
    c1 = p.eta
    c2 = p.b1
    total = 0.0
    for i0 in range(degree + 1):
        for i1 in range(degree - i0 + 1):
            i2 = degree - i0 - i1
            coeff = (
                math.factorial(degree)
                / (math.factorial(i0) * math.factorial(i1) * math.factorial(i2))
            )
            order = i1 + 2 * i2
            if order % 2 != 0:
                continue
            if dist == "gaussian":
                mom = innov.moment_gaussian(0, order, 0.0)
            else:
                mom = innov.moment_nig(0, order, 0.0, nu)
            total += coeff * (c0**i0) * (c1**i1) * (c2**i2) * mom
    return float(total)


def stability_condition(p: VPGARCHParams, max_degree: int, dist: str, nu: float | None = None) -> dict:
    """Proposition 5: rho(Q_rr) < 1 for r=1..max_degree ensures finite stationary moments."""
    out = {}
    for r in range(1, max_degree + 1):
        out[r] = diagonal_vp(p, r, dist, nu)
    out["stable"] = all(v < 1.0 for v in out.values())
    return out


def stationary_moments(p: VPGARCHParams, m: int, dist: str, nu: float | None = None) -> np.ndarray:
    """Stationary moments (1, E[h], E[h^2], ..., E[h^m]) solving mu = Q^T mu with mu[0]=1 (Prop. SM.2)."""
    Q = transition_matrix(p, m, dist, nu)
    # Solve (I - Q^T)[1:,1:] mu[1:] = Q^T[1:,0] for the non-trivial moments.
    QT = Q.T
    A = np.eye(m) - QT[1:, 1:]
    b = QT[1:, 0]
    mu_rest = np.linalg.solve(A, b)
    return np.concatenate(([1.0], mu_rest))


# --------------------------------------------------------------------------
# Remark 1: NGARCH-SV and NGARCH nested special cases
# --------------------------------------------------------------------------
def ngarch_sv_params(tau, rho, r, lam, sigma2, phi, alpha, gamma) -> VPGARCHParams:
    """NGARCH-SV: a0=a1=b0=0, b1=alpha, eta=-2*alpha*gamma (Remark 1)."""
    return VPGARCHParams(
        tau=tau, rho=rho, r=r, lam=lam, sigma2=sigma2, phi=phi,
        eta=-2.0 * alpha * gamma, a0=0.0, a1=0.0, b0=0.0, b1=alpha,
    )


def ngarch_params(tau, r, lam, sigma2, phi, alpha, gamma) -> VPGARCHParams:
    """NGARCH: the rho=1 special case of NGARCH-SV (Remark 1, Engle and Ng 1993)."""
    return ngarch_sv_params(tau, 1.0, r, lam, sigma2, phi, alpha, gamma)
