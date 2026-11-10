"""Risk-neutral dynamics under the variance-dependent pricing kernel (Appendix B).

Implements the exponential-affine kernel of Eq. (B.1), the no-arbitrage
restrictions of Eq. (B.2), the Gaussian joint cgf C^P, the risk-neutral
tilting formulas of Proposition 9, and the polynomial-preservation
conditions under Q of Proposition 10 for jointly Gaussian innovations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def cgf_gaussian(z1: float, z2: float, z3: float, rho: float) -> float:
    """Joint cgf of (eps1, eps2, eps2^2) for jointly Gaussian (eps1,eps2) with correlation rho.

    C^P(z1,z2,z3) = (1-rho^2)*z1^2/2 - log(1-2*z3)/2 + (z2+rho*z1)^2/(2*(1-2*z3))  (Appendix B, Gaussian case).
    Domain: z3 < 1/2.
    """
    if z3 >= 0.5:
        raise ValueError("cgf domain violated: z3 must be < 1/2")
    return 0.5 * (1 - rho**2) * z1**2 - 0.5 * np.log(1 - 2 * z3) + (z2 + rho * z1) ** 2 / (2 * (1 - 2 * z3))


@dataclass
class PricingKernelGaussian:
    """Exponential-affine kernel N = exp(xi0 + xi1*y_next + xi2*h_next), Eq. (B.1), xi2=0 (monotonic kernel)."""

    tau: float
    r: float
    rho: float
    xi2: float = 0.0


def solve_xi1(kernel: PricingKernelGaussian, h: float, lam: float) -> float:
    """Solve for xi1 from the stock-pricing martingale restriction (Eq. B.2), xi2=0.

    With xi2=0, A1 = xi1*sqrt(tau*h), A2 = 0, B = 0 and the restriction becomes
    C^P(A1+sqrt(tau*h), 0, 0) - C^P(A1, 0, 0) = -lambda(tau)*h.
    """
    tau, rho = kernel.tau, kernel.rho
    sqrt_th = np.sqrt(tau * h)

    def f(xi1: float) -> float:
        a1 = xi1 * sqrt_th
        lhs = cgf_gaussian(a1 + sqrt_th, 0.0, 0.0, rho) - cgf_gaussian(a1, 0.0, 0.0, rho)
        return lhs + lam * h

    return brentq(f, -50.0, 50.0)


def xi0_from_bond_restriction(kernel: PricingKernelGaussian, h: float, xi1: float) -> float:
    """xi0 from E^P[N|F_t] = exp(-r*tau): xi0 = -r*tau - C^P(xi1*sqrt(tau*h), 0, 0)."""
    tau, rho, r = kernel.tau, kernel.rho, kernel.r
    a1 = xi1 * np.sqrt(tau * h)
    return -r * tau - cgf_gaussian(a1, 0.0, 0.0, rho)


def risk_neutral_tilts(kernel: PricingKernelGaussian, h: float, xi1: float) -> dict:
    """Tilted innovation mean/variance (Proposition 9), xi2=0 Gaussian case.

    P1 = (1-rho^2)*A1 + rho*P2, P2 = A2 (=0 here since xi2=0), R1 = 1-rho^2 (xi2=0 => R2=1).
    """
    tau, rho = kernel.tau, kernel.rho
    a1 = xi1 * np.sqrt(tau * h)
    p2 = 0.0
    p1 = (1 - rho**2) * a1 + rho * p2
    r2 = 1.0  # 1/(1-2*B) with B=xi2*psi3=0
    r1 = 1 - rho**2 + rho**2 * r2
    return {"P1": p1, "P2": p2, "R1": r1, "R2": r2, "h_star": r1 * h, "rho_star": rho * np.sqrt(r2 / r1) if r1 > 0 else np.nan}


def verify_martingale_restrictions(
    kernel: PricingKernelGaussian, h: float, xi0: float, xi1: float, lam: float, n_paths: int, rng: np.random.Generator
) -> dict:
    """Monte Carlo check of E^P[N]=exp(-r*tau) and E^P[N*exp(y_next)]=1."""
    from src import innovations as innov

    eps1, _ = innov.sample_gaussian_pair(kernel.rho, n_paths, rng)
    y_next = kernel.r * kernel.tau + lam * h + np.sqrt(kernel.tau * h) * eps1
    N = np.exp(xi0 + xi1 * y_next)
    return {
        "E_N": float(np.mean(N)),
        "target_bond": float(np.exp(-kernel.r * kernel.tau)),
        "E_N_exp_y": float(np.mean(N * np.exp(y_next))),
        "target_stock": 1.0,
    }


# --------------------------------------------------------------------------
# Proposition 10: polynomial preservation under Q for jointly Gaussian innovations
# --------------------------------------------------------------------------
def check_preservation_vpgarch_q(rho: float, eta: float, b0: float, b1: float, a0: float, a1: float, lam_tau_r1: float) -> dict:
    """VPGARCH-SV, rho>0 (Proposition 10(i)): preservation under Q additionally requires

    lambda(tau) = -tau*R1(tau)/2  and  eta*(1-rho^2)*[b1^2*a0 - b0*b1*a1 + b0^2] = 0.
    `lam_tau_r1` is lambda(tau) + tau*R1(tau)/2 (should be ~0 for the first condition).
    """
    divisibility = eta * (1 - rho**2) * (b1**2 * a0 - b0 * b1 * a1 + b0**2)
    return {
        "lambda_condition_residual": lam_tau_r1,
        "divisibility_residual": divisibility,
        "preserved": bool(abs(lam_tau_r1) < 1e-8 and abs(divisibility) < 1e-8),
    }


def check_preservation_jpgarch_q(rho: float, eta: float, b0: float, b1: float, xi2: float, lam_tau_r1: float) -> dict:
    """JPGARCH-SV, rho>0 (Proposition 10(ii)): preservation under Q requires

    b1*[lambda(tau)+tau*R1(tau)/2] = 0  and  b0*b1*(1-rho^2)*eta*xi2 = 0.
    """
    cond1 = b1 * lam_tau_r1
    cond2 = b0 * b1 * (1 - rho**2) * eta * xi2
    return {
        "cond1_residual": cond1,
        "cond2_residual": cond2,
        "preserved": bool(abs(cond1) < 1e-8 and abs(cond2) < 1e-8),
    }
