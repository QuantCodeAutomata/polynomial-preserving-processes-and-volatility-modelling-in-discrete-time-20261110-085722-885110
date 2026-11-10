"""Weak convergence to polynomial jump-diffusions under Levy increment scaling (Sec. 3.3).

Implements the Levy-increment innovation construction of Eq. (24), the
parameter scaling of Assumption 2, the limiting jump-diffusion SDE of
Eq. (27)/Proposition 8, and the extended generator of Eq. (28) built from a
compound-Poisson Levy driver with symmetric jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LevyDriverParams:
    """Bivariate compound-Poisson Levy driver J=(J1,J2) with symmetric jumps.

    J_t = sqrt(C_J) W_t + sum_{s<=t} Delta_J_s, jump times a Poisson process
    with rate `jump_rate`; jump sizes (z1, z2) drawn i.i.d. symmetric with
    correlation `rho` and scaled so that C_J + integral z z^T Pi_J(dz) = [[1,rho],[rho,1]]
    (data_requirements, exp_4: centred, jointly symmetric Levy process with unit covariance).
    """

    jump_rate: float
    jump_scale: float  # scale of the symmetric jump law (here: Gaussian jump sizes)
    rho: float
    brownian_share: float = 0.5  # fraction of unit covariance assigned to the Brownian part C_J

    def c_j(self) -> np.ndarray:
        return self.brownian_share * np.array([[1.0, self.rho], [self.rho, 1.0]])

    def jump_cov_target(self) -> np.ndarray:
        """Target covariance of the jump part: (1-brownian_share)*[[1,rho],[rho,1]]."""
        return (1.0 - self.brownian_share) * np.array([[1.0, self.rho], [self.rho, 1.0]])

    def jump_moment_scale(self) -> float:
        """Scale s such that jump_rate * E[(s*Z)^2] = target variance, Z~N(0,1) per unit correlation block.

        With jump sizes (z1,z2) jointly Gaussian with correlation rho and common
        scale s, E[z1 z2]=s^2*rho and Var(z_i)=s^2, so the jump part covariance
        is s^2*[[1,rho],[rho,1]]; matching the target gives s^2 = (1-brownian_share)/jump_rate.
        """
        return np.sqrt((1.0 - self.brownian_share) / self.jump_rate)

    def mu(self, p_order: int, q_order: int) -> float:
        """Levy-measure moment mu_{p,q} = integral z1^p z2^q Pi_J(dz) for the Gaussian jump law.

        Pi_J has total mass `jump_rate` concentrated on jointly Gaussian jump
        sizes with scale s = jump_moment_scale() and correlation rho, so
        mu_{p,q} = jump_rate * s^(p+q) * E[Z1^p Z2^q] for the standardized pair.
        """
        from src import innovations as innov

        s = self.jump_moment_scale()
        return self.jump_rate * (s ** (p_order + q_order)) * innov.moment_gaussian(p_order, q_order, self.rho)


def simulate_levy_increments(
    driver: LevyDriverParams, n_steps: int, tau: float, n_paths: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized innovations eps_{i,(k+1)tau} = Delta J_{i,(k+1)tau} / sqrt(tau), Eq. (24).

    Delta J over one interval of length tau = Brownian increment (cov C_J*tau)
    plus a compound-Poisson jump sum with Poisson(jump_rate*tau) jump count.
    """
    c_j = driver.c_j()
    L = np.linalg.cholesky(c_j + 1e-14 * np.eye(2))
    z = rng.standard_normal((n_steps, n_paths, 2))
    brownian = (z @ L.T) * np.sqrt(tau)

    s = driver.jump_moment_scale()
    jump_counts = rng.poisson(driver.jump_rate * tau, size=(n_steps, n_paths))
    jump_sum = np.zeros((n_steps, n_paths, 2))
    max_jumps = int(jump_counts.max())
    if max_jumps > 0:
        jump_corr = np.array([[1.0, driver.rho], [driver.rho, 1.0]])
        L_j = np.linalg.cholesky(jump_corr)
        for step in range(n_steps):
            for path in range(n_paths):
                nj = jump_counts[step, path]
                if nj > 0:
                    zz = rng.standard_normal((nj, 2)) @ L_j.T * s
                    jump_sum[step, path, :] = zz.sum(axis=0)
    delta_j = brownian + jump_sum
    eps = delta_j / np.sqrt(tau)
    return eps[:, :, 0], eps[:, :, 1]


@dataclass
class JumpDiffusionParams:
    model: str  # "VP" or "JP"
    kappa: float
    sigma2: float
    r: float
    lam: float
    eta: float
    b0: float
    b1: float
    a0: float = 0.0
    a1: float = 0.0
    driver: LevyDriverParams | None = None


def g_m(x, p: JumpDiffusionParams):
    if p.model == "VP":
        return p.a0 + p.a1 * x + x**2
    return x


def check_proposition8(p: JumpDiffusionParams) -> dict:
    """Proposition 8 admissibility conditions (Sec. 3.3), checked on a positive x-grid."""
    xs = np.linspace(1e-6, 4.0 * p.sigma2, 200)
    g = g_m(xs, p)
    lhs = p.eta**2 * g
    rhs = 2.0 * xs * (p.b0 + p.b1 * xs)
    cond_main = np.all(lhs <= rhs + 1e-10)
    conds = {
        "b0_le_kappa_sigma2": p.b0 <= p.kappa * p.sigma2 + 1e-10,
        "eta2_g_le_2x_bx": bool(cond_main),
    }
    if p.model == "VP":
        conds["vp_a0_cond"] = p.eta**2 * p.a0 <= 4.0 * p.b0 * (p.kappa * p.sigma2 - p.b0) + 1e-10
    conds["all"] = all(v for k, v in conds.items() if isinstance(v, (bool, np.bool_)))
    return conds


def simulate_discrete_scaled(
    p: JumpDiffusionParams, n: int, horizon: float, n_paths: int, rng: np.random.Generator, y0: float, h0: float
) -> dict:
    """Discrete-time VPGARCH-SV/JPGARCH-SV path with Assumption-2 parameter scaling tau=1/n.

    phi(tau)=1-kappa*tau, sigma2(tau)=sigma2, lambda(tau)=lambda*tau, b_i(tau)=b_i*tau,
    eta(tau)=eta*sqrt(tau); for VP, a1(tau)=a1, a0(tau)=a0*tau (Assumption 2, Sec. 3.3).
    """
    tau = 1.0 / n
    n_steps = int(round(horizon * n))
    phi_tau = 1.0 - p.kappa * tau
    omega_tau = (1.0 - phi_tau) * p.sigma2
    eta_tau = p.eta * np.sqrt(tau)
    b0_tau = p.b0 * tau
    b1_tau = p.b1 * tau
    a0_tau = p.a0 * tau
    a1_tau = p.a1

    eps1_all, eps2_all = simulate_levy_increments(p.driver, n_steps, tau, n_paths, rng)

    y = np.full(n_paths, y0)
    h = np.full(n_paths, h0)
    for k in range(n_steps):
        eps1, eps2 = eps1_all[k], eps2_all[k]
        G = (a0_tau + a1_tau * h + h**2) if p.model == "VP" else h
        sqrt_G = np.sqrt(np.clip(G, 0.0, None))
        y = y + p.r * tau + p.lam * tau * h + np.sqrt(tau * np.clip(h, 0.0, None)) * eps1
        h = omega_tau + phi_tau * h + eta_tau * sqrt_G * eps2 + (b0_tau + b1_tau * h) * (eps2**2 - 1.0)
        h = np.clip(h, 0.0, None)
    return {"y": y, "h": h}


def simulate_limit_jump_diffusion(
    p: JumpDiffusionParams, horizon: float, n_paths: int, rng: np.random.Generator, y0: float, h0: float,
    n_grid: int = 2000,
) -> dict:
    """Euler simulation of the limiting jump-diffusion, Eq. (27):

    dY_t = (r+lambda*h_)dt + sqrt(h_) dJ_{1,t}
    dh_t = kappa*(sigma2-h_)dt + eta*sqrt(g_M(h_)) dJ_{2,t} + (b0+b1*h_)(d[J2]_t - dt)
    """
    dt = horizon / n_grid
    driver = p.driver
    eps1_all, eps2_all = simulate_levy_increments(driver, n_grid, dt, n_paths, rng)
    # quadratic variation increment of J2 over [t,t+dt]: Brownian part dt*C_J[1,1] + jump part sum(z2^2)
    c_j = driver.c_j()
    s = driver.jump_moment_scale()
    y = np.full(n_paths, y0)
    h = np.full(n_paths, h0)
    for k in range(n_grid):
        eps1, eps2 = eps1_all[k], eps2_all[k]
        dJ1 = eps1 * np.sqrt(dt)
        dJ2 = eps2 * np.sqrt(dt)
        # quadratic variation increment of J2: Brownian contributes c_j[1,1]*dt deterministically in expectation;
        # approximate d[J2]_t - dt using the realized squared increment minus its Brownian part's mean.
        d_qv_minus_dt = (dJ2**2 - c_j[1, 1] * dt)
        h_safe = np.clip(h, 0.0, None)
        G = g_m(h_safe, p)
        y = y + (p.r + p.lam * h_safe) * dt + np.sqrt(h_safe) * dJ1
        h = h + p.kappa * (p.sigma2 - h) * dt + p.eta * np.sqrt(np.clip(G, 0.0, None)) * dJ2 + (p.b0 + p.b1 * h_safe) * d_qv_minus_dt
        h = np.clip(h, 0.0, None)
    return {"y": y, "h": h}


def generator_matrix_variance(p: JumpDiffusionParams, m: int) -> np.ndarray:
    """Extended generator G_m on Pol_m(R_+), Eq. (28): drift + diffusion + jump-integral terms.

    For p(x)=x^k, the jump term integral[(x+D(x,z))^k - x^k - k x^{k-1} D(x,z)] Pi_J(dz)
    is evaluated by expanding the binomial and replacing integrals of z-powers
    by the Levy-measure moments mu_{p,q} from LevyDriverParams.mu, with
    D(x,z) = (b0+b1*x)*z2^2 for the variance coordinate (Eq. 28 with D_M).
    """
    import math

    import sympy as sp

    x = sp.symbols("x")
    G = np.zeros((m + 1, m + 1))
    driver = p.driver
    for k in range(m + 1):
        xk = x**k
        drift = p.kappa * (p.sigma2 - x) * sp.diff(xk, x)
        b_of_x = p.b0 + p.b1 * x
        # jump term: mu_{0,2j} moments of z2 give E[(b_of_x*z2^2)^j] type contributions
        jump_expr = sp.Integer(0)
        for j in range(1, k + 1):
            binom = math.comb(k, j)
            coeff = binom * (b_of_x**j) * (x ** (k - j))
            mu = driver.mu(0, 2 * j)
            jump_expr += coeff * mu
        expr = sp.expand(drift + jump_expr)
        poly = sp.Poly(expr, x)
        for l in range(m + 1):
            c = poly.coeff_monomial(x**l)
            G[l, k] = float(c) if c is not None else 0.0
    return G
