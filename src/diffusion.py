"""Weak convergence to polynomial diffusions under P (Sec. 3.1-3.2).

Implements the parameter scaling of Assumption 1, the càdlàg extension of
Eq. (19), the limiting diffusion of Proposition 7, the polynomial generator
matrix of Corollary 3 (via scipy.linalg.expm), and Euler-Maruyama simulation
of the limiting SDE for weak-convergence verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from src import innovations as innov


@dataclass
class DiffusionLimitParams:
    """Limiting-diffusion coefficients under Assumption 1 (Sec. 3.1)."""

    model: str  # "VP" or "JP"
    kappa: float  # lim (1-phi(tau))/tau
    sigma2: float  # lim sigma^2(tau)
    r: float
    lam: float  # lim lambda(tau)/tau
    eta: float  # lim eta(tau)/sqrt(tau)
    b0: float  # lim b0(tau)/sqrt(tau)
    b1: float  # lim b1(tau)/sqrt(tau)
    a0: float = 0.0  # VP only: lim a0(tau) (eta^2*a0=0 boundary condition)
    a1: float = 0.0  # VP only: lim a1(tau)
    rho: float = 0.0


def v_m(x: np.ndarray | float, p: DiffusionLimitParams, m4: float) -> np.ndarray | float:
    """v_M(x): the diffusion coefficient of the variance process, Proposition 7.

    v_VP(x) = eta^2*(a0+a1*x+x^2) + (m4-1)*(b0+b1*x)^2
    v_JP(x) = eta^2*x + (m4-1)*(b0+b1*x)^2
    """
    quad = p.b0 + p.b1 * x
    if p.model == "VP":
        return p.eta**2 * (p.a0 + p.a1 * x + x**2) + (m4 - 1.0) * quad**2
    return p.eta**2 * x + (m4 - 1.0) * quad**2


def g_m(x: np.ndarray | float, p: DiffusionLimitParams) -> np.ndarray | float:
    """g_M(x): the argument under the square root multiplying rho in the return-variance covariance."""
    if p.model == "VP":
        return p.a0 + p.a1 * x + x**2
    return x


def rho_m(x: np.ndarray, p: DiffusionLimitParams, m4: float) -> np.ndarray:
    """rho_M(x) = rho*eta*sqrt(g_M(x)) / sqrt(v_M(x)), the state-dependent return-variance correlation."""
    g = np.clip(g_m(x, p), 0.0, None)
    v = np.clip(v_m(x, p, m4), 1e-300, None)
    return p.rho * p.eta * np.sqrt(g) / np.sqrt(v)


def check_assumption1_boundary(p: DiffusionLimitParams) -> dict:
    """Boundary/non-attainment conditions stated before Proposition 7 (Sec. 3.2)."""
    conds = {"b0_zero": abs(p.b0) < 1e-12}
    if p.model == "JP":
        conds["non_attainment"] = 2.0 * p.kappa * p.sigma2 >= p.eta**2 - 1e-10
    else:
        conds["eta2_a0_zero"] = abs(p.eta**2 * p.a0) < 1e-10
        conds["non_attainment"] = 2.0 * p.kappa * p.sigma2 >= p.eta**2 * p.a1 - 1e-10
    conds["all"] = all(conds.values())
    return conds


def simulate_diffusion_euler(
    p: DiffusionLimitParams, y0: float, h0: float, horizon: float, n_grid: int, n_paths: int,
    dist: str, rng: np.random.Generator, nu: float | None = None,
) -> dict:
    """Euler-Maruyama simulation of the limiting diffusion, Proposition 7:

    dY_t = (r + lambda*h_t)dt + sqrt(h_t) dW_{1,t}
    dh_t = kappa*(sigma2 - h_t)dt + sqrt(v_M(h_t)) dW_{2,t}
    d<W_1,W_2>_t = rho_M(h_t) dt
    """
    dt = horizon / n_grid
    sqrt_dt = np.sqrt(dt)
    m4 = innov.fourth_moment(dist, nu)
    y = np.full(n_paths, y0)
    h = np.full(n_paths, h0)
    for _ in range(n_grid):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        rho_state = rho_m(h, p, m4)
        w1 = z1
        w2 = rho_state * z1 + np.sqrt(np.clip(1.0 - rho_state**2, 0.0, 1.0)) * z2
        h_safe = np.clip(h, 0.0, None)
        y = y + (p.r + p.lam * h_safe) * dt + np.sqrt(h_safe) * sqrt_dt * w1
        h = h + p.kappa * (p.sigma2 - h) * dt + np.sqrt(np.clip(v_m(h_safe, p, m4), 0.0, None)) * sqrt_dt * w2
        h = np.clip(h, 0.0, None)  # Feller-type reflection at the simulation boundary (Prop. SM.4)
    return {"y": y, "h": h}


def generator_matrix_variance(p: DiffusionLimitParams, m: int, m4: float) -> np.ndarray:
    """Polynomial generator G_m on Pol_m(R_+) for the variance diffusion (Proposition 6).

    For p(x)=x^k: A f(x) = kappa*(sigma2-x)*k*x^{k-1} + 0.5*v_M(x)*k*(k-1)*x^{k-2}.
    G_m[l,k] = coefficient of x^l in A(x^k).
    """
    import sympy as sp

    x = sp.symbols("x")
    quad = p.b0 + p.b1 * x
    if p.model == "VP":
        vM = p.eta**2 * (p.a0 + p.a1 * x + x**2) + (m4 - 1.0) * quad**2
    else:
        vM = p.eta**2 * x + (m4 - 1.0) * quad**2
    G = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        xk = x**k
        drift_term = p.kappa * (p.sigma2 - x) * sp.diff(xk, x)
        diff_term = sp.Rational(1, 2) * vM * sp.diff(xk, x, 2)
        expr = sp.expand(drift_term + diff_term)
        poly = sp.Poly(expr, x)
        for l in range(m + 1):
            c = poly.coeff_monomial(x**l)
            G[l, k] = float(c) if c is not None else 0.0
    return G


def discrete_to_generator_matrix(Q: np.ndarray, tau: float) -> np.ndarray:
    """Corollary 3: G_m = lim (Q_m^{(n)} - I)/tau as tau -> 0; finite-tau approximation."""
    return (Q - np.eye(Q.shape[0])) / tau


def moment_via_generator(G: np.ndarray, x0: float, t: float, degree: int) -> float:
    """E[h_t^degree | h_0=x0] = H_m(x0)^T exp(t*G) p_{h^degree} (Corollary 3)."""
    m = G.shape[0] - 1
    H = np.array([x0**i for i in range(m + 1)])
    p = np.zeros(m + 1)
    p[degree] = 1.0
    return float(H @ expm(t * G) @ p)
