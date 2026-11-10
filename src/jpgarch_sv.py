"""JPGARCH-SV model: dynamics, positivity, and joint (return, variance) moments.

Implements Eqs. (9)-(11),(14) and Proposition 4 (Sec. 2.2.2), the joint
transition matrix on Pol_m(R x R_+) (Sec. 2.2.2/2.3), cumulative-return
moments via the augmented state Z=(R,h) (Corollary 2(ii)), and the
HNGARCH-SV/HNGARCH nesting of Remark 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import innovations as innov
from src.poly_basis import (
    build_Q_joint_jpgarch,
    build_Q_variance,
    basis_index_2d,
    basis_vector_2d,
)


@dataclass
class JPGARCHParams:
    tau: float
    rho: float
    r: float
    lam: float
    sigma2: float
    phi: float
    eta: float
    b0: float
    b1: float

    @property
    def omega(self) -> float:
        return (1.0 - self.phi) * self.sigma2


def check_positivity(p: JPGARCHParams) -> dict:
    """Proposition 4: necessary and sufficient conditions for h_next >= 0 for all x>=0, eps2 in R."""
    eta_bound = 2.0 * (np.sqrt(max(p.b0 * (p.phi - p.b1), 0.0)) + np.sqrt(max(p.b1 * (p.omega - p.b0), 0.0)))
    conds = {
        "b0_nonneg": p.b0 >= 0,
        "b1_nonneg": p.b1 >= 0,
        "b0_le_omega": p.b0 <= p.omega + 1e-12,
        "b1_le_phi": p.b1 <= p.phi + 1e-12,
        "b0_plus_b1_pos": (p.b0 + p.b1) > 0,
        "eta_bound": p.eta**2 <= eta_bound**2 + 1e-12,
    }
    conds["all"] = all(conds.values())
    return conds


def h_next(x: np.ndarray, eps2: np.ndarray, p: JPGARCHParams) -> np.ndarray:
    """One-step variance update, Eq. (14): psi_2(x)=eta*sqrt(x), psi_3(x)=b0+b1*x."""
    return (
        p.omega
        + p.phi * x
        + p.eta * np.sqrt(np.clip(x, 0.0, None)) * eps2
        + (p.b0 + p.b1 * x) * (eps2**2 - 1.0)
    )


def simulate_paths(
    p: JPGARCHParams, n_steps: int, n_paths: int, dist: str, rng: np.random.Generator, nu: float | None = None
) -> dict:
    y = np.zeros((n_paths, n_steps + 1))
    h = np.zeros((n_paths, n_steps + 1))
    h[:, 0] = p.sigma2
    for k in range(n_steps):
        eps1, eps2 = innov.sample_pair(dist, p.rho, n_paths, rng, nu)
        y[:, k + 1] = p.r * p.tau + p.lam * h[:, k] + np.sqrt(p.tau * h[:, k]) * eps1
        h[:, k + 1] = h_next(h[:, k], eps2, p)
        if np.any(h[:, k + 1] < 0):
            raise ValueError("negative variance encountered: positivity restriction violated")
    return {"y": y, "h": h}


def transition_matrix_variance(p: JPGARCHParams, m: int, dist: str, nu: float | None = None) -> np.ndarray:
    """Variance-only Q_m (special case of build_Q_variance with G(x)=x, i.e. a0=0, a1=1)."""
    even_m = innov.even_moments_eps2(2 * m, dist, p.rho, nu)
    return build_Q_variance(m, p.phi, p.omega, p.eta, [0.0, 1.0], p.b0, p.b1, even_m)


def transition_matrix_joint(p: JPGARCHParams, m: int, dist: str, nu: float | None = None) -> np.ndarray:
    """Joint transition matrix Q_m on Pol_m(R x R_+) for X=(Y,h), Sec. 2.2.2."""
    moment_fn = innov.moment_nig if dist == "nig" else innov.moment_gaussian
    return build_Q_joint_jpgarch(
        m, p.tau, p.r, p.lam, p.phi, p.omega, p.eta, p.b0, p.b1, p.rho, dist, nu, moment_fn
    )


def joint_moment(Q: np.ndarray, m: int, x1_0: float, x2_0: float, s: int, p_order: int, q_order: int) -> float:
    """E[Y_{t+s}^p * h_{t+s}^q | Y_t=x1_0, h_t=x2_0] via H_m(X_t)^T Q^s p_{p,q} (Proposition 1)."""
    basis = basis_index_2d(m)
    idx = {b: k for k, b in enumerate(basis)}
    target = np.zeros(len(basis))
    target[idx[(p_order, q_order)]] = 1.0
    Qs = np.linalg.matrix_power(Q, s)
    return float(basis_vector_2d(m, x1_0, x2_0) @ Qs @ target)


def diagonal_jp(p: JPGARCHParams, degree: int, dist: str, nu: float | None = None) -> float:
    """E[A_JP^degree], A_JP = phi + eta*eps2 + b1*(eps2^2-1), for moment stability (Prop. 5 analogue)."""
    c0 = p.phi - p.b1
    c1 = p.eta
    c2 = p.b1
    import math

    total = 0.0
    for i0 in range(degree + 1):
        for i1 in range(degree - i0 + 1):
            i2 = degree - i0 - i1
            order = i1 + 2 * i2
            if order % 2 != 0:
                continue
            coeff = math.factorial(degree) / (math.factorial(i0) * math.factorial(i1) * math.factorial(i2))
            mom = innov.moment_gaussian(0, order, 0.0) if dist == "gaussian" else innov.moment_nig(0, order, 0.0, nu)
            total += coeff * (c0**i0) * (c1**i1) * (c2**i2) * mom
    return float(total)


def stationary_moments_variance(p: JPGARCHParams, m: int, dist: str, nu: float | None = None) -> np.ndarray:
    Q = transition_matrix_variance(p, m, dist, nu)
    QT = Q.T
    A = np.eye(m) - QT[1:, 1:]
    b = QT[1:, 0]
    mu_rest = np.linalg.solve(A, b)
    return np.concatenate(([1.0], mu_rest))


def conditional_corr(x: np.ndarray, p: JPGARCHParams, m4: float) -> np.ndarray:
    """State-dependent conditional return-variance correlation (Sec. 2.2.2 discussion).

    Cov[y_next, h_next | h_t=x] = rho*eta*sqrt(tau)*x (linear in x).
    Var[h_next|x] = w0 + w1*x + w2*x^2, w0=(m4-1)*b0^2, w1=eta^2+2*(m4-1)*b0*b1, w2=(m4-1)*b1^2.
    Var[y_next|x] = tau*x.
    """
    w0 = (m4 - 1.0) * p.b0**2
    w1 = p.eta**2 + 2.0 * (m4 - 1.0) * p.b0 * p.b1
    w2 = (m4 - 1.0) * p.b1**2
    var_h = w0 + w1 * x + w2 * x**2
    var_y = p.tau * x
    cov = p.rho * p.eta * np.sqrt(p.tau) * x
    return cov / np.sqrt(var_y * var_h)


# --------------------------------------------------------------------------
# Remark 2: HNGARCH-SV and HNGARCH nested special cases
# --------------------------------------------------------------------------
def hngarch_sv_params(tau, rho, r, lam, sigma2, phi, alpha, gamma) -> JPGARCHParams:
    """HNGARCH-SV: b0=alpha, b1=0, eta=-2*alpha*gamma (Remark 2)."""
    return JPGARCHParams(
        tau=tau, rho=rho, r=r, lam=lam, sigma2=sigma2, phi=phi,
        eta=-2.0 * alpha * gamma, b0=alpha, b1=0.0,
    )


def hngarch_params(tau, r, lam, sigma2, phi, alpha, gamma) -> JPGARCHParams:
    """HNGARCH: the rho=1 special case of HNGARCH-SV (Remark 2, Heston and Nandi 2000)."""
    return hngarch_sv_params(tau, 1.0, r, lam, sigma2, phi, alpha, gamma)


# --------------------------------------------------------------------------
# Cumulative returns and cross-date moments (Corollary 1, Corollary 2(ii))
# --------------------------------------------------------------------------
def simulate_cumulative_return(
    p: JPGARCHParams, j_periods: int, n_paths: int, dist: str, rng: np.random.Generator, nu: float | None = None
) -> dict:
    """Augmented state Z=(R,h): R_{k,j} accumulates returns while h evolves (Corollary 2(ii))."""
    R = np.zeros(n_paths)
    h = np.full(n_paths, p.sigma2)
    for _ in range(j_periods):
        eps1, eps2 = innov.sample_pair(dist, p.rho, n_paths, rng, nu)
        y_incr = p.r * p.tau + p.lam * h + np.sqrt(p.tau * h) * eps1
        R = R + y_incr
        h = h_next(h, eps2, p)
    return {"R": R, "h": h}


def cumulative_return_moment(p: JPGARCHParams, m: int, j_periods: int, dist: str, nu: float | None = None) -> float:
    """E[R_{0,j}] and E[R_{0,j}^2] via the joint transition matrix of Z=(R,h), degree m.

    Because Z=(R,h) obeys the same joint transition structure as (Y,h) (the
    increment added to R is the same y_next as for Y), the joint matrix
    Q_m built by transition_matrix_joint applies unchanged; only the number
    of periods (j_periods) differs from the single-step case.
    """
    Q = transition_matrix_joint(p, m, dist, nu)
    return joint_moment(Q, m, 0.0, p.sigma2, j_periods, 1, 0)
