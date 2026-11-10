"""Joint innovation laws (epsilon_1, epsilon_2) used throughout Section 2.

Both innovation laws are i.i.d. jointly symmetric with standardized marginals
D_i(0,1), finite moments of all orders, and correlation rho in [0,1]
(paper_pipeline data_requirements for exp_1/exp_2). Two laws are implemented:

* Gaussian: (eps_1, eps_2) jointly Gaussian with correlation rho.
* Symmetric NIG: the variance-mixture representation of Appendix C,
  eps_i = sqrt(V) * Z_i, with V ~ IG(1, alpha) independent of the jointly
  Gaussian pair Z = (Z_1, Z_2) with correlation rho, alpha = 1/nu.  Since
  E[V] = 1, Corr(eps_1, eps_2) = Corr(Z_1, Z_2) = rho, and the fourth moment
  of each marginal is m_4 = 3 + 3/alpha = 3 + 3*nu (paper, Sec. 4.2 / App. C).

Joint moments E[eps_1^p eps_2^q] are computed by 2-D Gauss-Hermite quadrature
(numpy.polynomial.hermite_e), which integrates polynomials in jointly Gaussian
variables exactly; this is the standard quadrature rule for Gaussian moments
and is used here instead of a hand-written Isserlis/Wick recursion.
scipy.stats.invgauss supplies the inverse-Gaussian moments E[V^k] needed for
the NIG mixing variable.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial import hermite_e
from scipy import stats


def nig_alpha(nu: float) -> float:
    """NIG tail/shape parameter alpha = 1/nu (nu = inverse shape, Table 1)."""
    if nu <= 0:
        raise ValueError("nu must be positive")
    return 1.0 / nu


def nig_fourth_moment(nu: float) -> float:
    """m_4 = 3 + 3*nu for the standardized symmetric NIG marginal (Sec. 4.2)."""
    return 3.0 + 3.0 * nu


def gaussian_fourth_moment() -> float:
    """m_4 = 3 for the standard Gaussian marginal."""
    return 3.0


@lru_cache(maxsize=4)
def _hermgauss_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes/weights integrating against exp(-x^2/2)/sqrt(2*pi)."""
    nodes, weights = hermite_e.hermegauss(n)
    weights = weights / np.sqrt(2.0 * np.pi)
    return nodes, weights


def moment_gaussian(p: int, q: int, rho: float, n_nodes: int = 24) -> float:
    """E[Z1^p Z2^q] for jointly standard Gaussian (Z1, Z2) with correlation rho.

    Uses 2-D tensor Gauss-Hermite quadrature, exact for polynomial integrands
    of total degree < 2*n_nodes.
    """
    if p == 0 and q == 0:
        return 1.0
    nodes, weights = _hermgauss_nodes(n_nodes)
    # Z1 = X, Z2 = rho*X + sqrt(1-rho^2)*Y, X,Y iid N(0,1)
    x, wx = nodes, weights
    y, wy = nodes, weights
    X, Y = np.meshgrid(x, y, indexing="ij")
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    z1 = X
    z2 = rho * X + np.sqrt(max(0.0, 1.0 - rho**2)) * Y
    integrand = (z1**p) * (z2**q) * WX * WY
    return float(np.sum(integrand))


def moment_nig(p: int, q: int, rho: float, nu: float, n_nodes: int = 24) -> float:
    """E[eps1^p eps2^q] for the symmetric NIG variance-mixture law.

    eps = sqrt(V) * Z with V ~ IG(1, alpha) independent of Z (correlation rho).
    Odd total order vanishes by symmetry of Z; for even total order 2k,
    E[eps1^p eps2^q] = E[V^k] * E[Z1^p Z2^q].
    """
    total = p + q
    if total % 2 == 1:
        return 0.0
    k = total // 2
    alpha = nig_alpha(nu)
    # V ~ IG(mean=1, shape=alpha): scipy.stats.invgauss(mu_scipy, scale=s) is the law of
    # s*X for X standard invgauss(mu_scipy) (mean mu_scipy, shape 1); matching mean 1 and
    # Var(V) = 1/alpha gives mu_scipy = 1/alpha, s = alpha.
    mu_scipy, scale = 1.0 / alpha, alpha
    e_vk = stats.invgauss.moment(k, mu_scipy, scale=scale) if k > 0 else 1.0
    return float(e_vk) * moment_gaussian(p, q, rho)


def even_moments_eps2(max_order: int, dist: str, rho: float = 0.0, nu: float | None = None) -> np.ndarray:
    """Even raw moments E[eps_2^(2k)] for k=0..max_order//2, marginal law."""
    orders = np.arange(0, max_order + 1, 2)
    out = np.zeros(len(orders))
    for i, o in enumerate(orders):
        if dist == "gaussian":
            out[i] = moment_gaussian(0, o, rho)
        elif dist == "nig":
            out[i] = moment_nig(0, o, rho, nu)
        else:
            raise ValueError(f"unknown distribution {dist}")
    return out


def sample_gaussian_pair(rho: float, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw i.i.d. jointly Gaussian (eps1, eps2) pairs with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    draws = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=size)
    return draws[:, 0], draws[:, 1]


def sample_nig_pair(
    rho: float, nu: float, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw i.i.d. symmetric NIG (eps1, eps2) pairs via the variance mixture (App. C)."""
    alpha = nig_alpha(nu)
    # Same IG(mean=1, shape=alpha) parametrization as moment_nig.
    mu_scipy, scale = 1.0 / alpha, alpha
    v = stats.invgauss.rvs(mu_scipy, scale=scale, size=size, random_state=rng)
    z1, z2 = sample_gaussian_pair(rho, size, rng)
    sqrt_v = np.sqrt(v)
    return sqrt_v * z1, sqrt_v * z2


def sample_pair(dist: str, rho: float, size: int, rng: np.random.Generator, nu: float | None = None):
    """Dispatch to the Gaussian or NIG innovation sampler."""
    if dist == "gaussian":
        return sample_gaussian_pair(rho, size, rng)
    if dist == "nig":
        if nu is None:
            raise ValueError("nu must be given for the NIG law")
        return sample_nig_pair(rho, nu, size, rng)
    raise ValueError(f"unknown distribution {dist}")


def fourth_moment(dist: str, nu: float | None = None) -> float:
    """Dispatch to the marginal fourth moment m_4 of the chosen innovation law."""
    if dist == "gaussian":
        return gaussian_fourth_moment()
    if dist == "nig":
        if nu is None:
            raise ValueError("nu must be given for the NIG law")
        return nig_fourth_moment(nu)
    raise ValueError(f"unknown distribution {dist}")
