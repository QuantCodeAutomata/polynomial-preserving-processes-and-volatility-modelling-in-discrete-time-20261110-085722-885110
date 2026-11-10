"""Transition-matrix construction on the monomial bases H_m of Section 2.1.

Custom implementation (paper Sec. 2.1, Eq. (4)-(5); Proposition 1). No
general-purpose library builds these finite-dimensional polynomial-moment
transition operators, so the construction is implemented directly from the
one-step conditional-moment recursions of Sections 2.2.1/2.2.2, using sympy
only for exact polynomial bookkeeping in the state variable(s).

Both variance recursions take the form

    h_next = A(x) + eta * sqrt(G(x)) * eps2 + C(x) * (eps2^2 - 1),

with A, C affine in x and G a low-degree polynomial (G(x) = a0 + a1*x + x^2
for VPGARCH-SV, G(x) = x for JPGARCH-SV). Expanding h_next^k via the
multinomial theorem, every surviving term (after taking expectations over the
symmetric eps2) has an even power of sqrt(G(x)), which collapses back to the
polynomial G(x)^{power/2}; this is what makes the one-step operator map
Pol_m(R_+) into itself (Proposition 2(i)).
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import sympy as sp

x = sp.symbols("x")
x1, x2 = sp.symbols("x1 x2")


def _poly_from_coeffs(coeffs: list[float], var: sp.Symbol) -> sp.Expr:
    """Polynomial a0 + a1*var + a2*var**2 + ... from ascending coefficients."""
    return sum(c * var**i for i, c in enumerate(coeffs))


def build_Q_variance(
    m: int,
    phi: float,
    omega: float,
    eta: float,
    g_coeffs: list[float],
    b0: float,
    b1: float,
    even_moments_eps2: np.ndarray,
) -> np.ndarray:
    """One-step transition matrix Q_m on Pol_m(R_+) for the variance process h.

    h_next = (omega + phi*x) + eta*sqrt(G(x))*eps2 + (b0 + b1*x)*(eps2^2 - 1)

    Parameters
    ----------
    m : maximum polynomial degree.
    even_moments_eps2 : array with even_moments_eps2[k] = E[eps2^(2k)], k=0..m.

    Returns
    -------
    Q : (m+1, m+1) ndarray, Q[l, k] = coefficient of x^l in E[h_next^k | x].
    """
    A = omega + phi * x
    G = _poly_from_coeffs(g_coeffs, x)
    C = b0 + b1 * x
    Q = np.zeros((m + 1, m + 1))
    Q[0, 0] = 1.0
    for k in range(1, m + 1):
        expr = sp.Integer(0)
        for i1 in range(0, k + 1):
            for i2 in range(0, k - i1 + 1, 2):  # only even i2 survives
                i3 = k - i1 - i2
                multinom = math.factorial(k) // (
                    math.factorial(i1) * math.factorial(i2) * math.factorial(i3)
                )
                g_power = sp.expand(G**(i2 // 2))
                term_poly = (A**i1) * (eta**i2) * g_power * (C**i3)
                scalar = 0.0
                for j in range(0, i3 + 1):
                    order = i2 + 2 * j
                    scalar += (
                        math.comb(i3, j)
                        * (-1) ** (i3 - j)
                        * even_moments_eps2[order // 2]
                    )
                expr += multinom * scalar * term_poly
        expr = sp.expand(expr)
        poly = sp.Poly(expr, x)
        for l in range(0, k + 1):
            Q[l, k] = float(poly.coeff_monomial(x**l)) if poly.coeff_monomial(x**l) is not None else 0.0
    return Q


def basis_vector_1d(m: int, x_value: float) -> np.ndarray:
    """H_m(x) = (1, x, x^2, ..., x^m)."""
    return np.array([x_value**i for i in range(m + 1)])


def conditional_moment_1d(Q: np.ndarray, x_value: float, s: int, degree: int) -> float:
    """E[(h_{t+s})^degree | h_t = x_value] = H_m(x)^T (Q^s) p_{h^degree}."""
    m = Q.shape[0] - 1
    p = np.zeros(m + 1)
    p[degree] = 1.0
    Qs = np.linalg.matrix_power(Q, s)
    return float(basis_vector_1d(m, x_value) @ Qs @ p)


def basis_index_2d(m: int) -> list[tuple[int, int]]:
    """Degree-ordered basis exponents for Pol_m(R x R_+): (i, j) with i+j<=m.

    Within each total degree d, ordered (x1^d, x1^(d-1) x2, ..., x2^d), matching
    Example 1 / Eq. (5) in the paper: H_2 = (1, x1, x2, x1^2, x1 x2, x2^2).
    """
    out = []
    for d in range(m + 1):
        for i in range(d, -1, -1):
            j = d - i
            out.append((i, j))
    return out


def build_Q_joint_jpgarch(
    m: int,
    tau: float,
    r: float,
    lam: float,
    phi: float,
    omega: float,
    eta: float,
    b0: float,
    b1: float,
    rho: float,
    dist: str,
    nu: float | None,
    moment_fn,
) -> np.ndarray:
    """Joint one-step transition matrix Q_m for X=(Y,h) under JPGARCH-SV.

    Y_next = x1 + (r*tau + lam*x2) + sqrt(tau*x2)*eps1
    h_next = (omega + phi*x2) + eta*sqrt(x2)*eps2 + (b0 + b1*x2)*(eps2^2 - 1)

    For JPGARCH-SV, sqrt(tau*x2) and eta*sqrt(x2) share the same square-root
    base sqrt(x2), so sqrt(x2)^(i3+i5) is polynomial whenever i3+i5 is even;
    joint symmetry of (eps1, eps2) forces every other combination to vanish
    (Proposition 2(ii), psi_2 in span{sqrt(x)}).
    """
    basis = basis_index_2d(m)
    idx = {b: k for k, b in enumerate(basis)}
    N = len(basis)
    Q = np.zeros((N, N))
    Q[0, 0] = 1.0
    A0 = r * tau + lam * x2
    A1 = omega + phi * x2
    C = b0 + b1 * x2

    def eps_moment(p_order: int, q_order: int) -> float:
        return moment_fn(p_order, q_order, rho, nu) if dist == "nig" else moment_fn(p_order, q_order, rho)

    for (p, q) in basis[1:]:
        expr = sp.Integer(0)
        # Y_next^p expansion: (x1 + A0 + sqrt(tau*x2)*eps1)^p
        for i1 in range(0, p + 1):
            for i2 in range(0, p - i1 + 1):
                i3 = p - i1 - i2
                mult_y = math.factorial(p) // (
                    math.factorial(i1) * math.factorial(i2) * math.factorial(i3)
                )
                # h_next^q expansion: (A1 + eta*sqrt(x2)*eps2 + C*(eps2^2-1))^q
                for i4 in range(0, q + 1):
                    for i5 in range(0, q - i4 + 1):
                        i6 = q - i4 - i5
                        if (i3 + i5) % 2 != 0:
                            continue  # vanishes by joint symmetry
                        mult_h = math.factorial(q) // (
                            math.factorial(i4) * math.factorial(i5) * math.factorial(i6)
                        )
                        half = (i3 + i5) // 2
                        sqrt_part = (tau ** (i3 / 2.0)) * (eta**i5) * sp.expand(x2**half)
                        poly_part = (x1**i1) * (A0**i2) * (A1**i4) * (C**i6) * sqrt_part
                        scalar = 0.0
                        for j in range(0, i6 + 1):
                            e2_order = i5 + 2 * j
                            scalar += (
                                math.comb(i6, j)
                                * (-1) ** (i6 - j)
                                * eps_moment(i3, e2_order)
                            )
                        expr += mult_y * mult_h * scalar * poly_part
        expr = sp.expand(expr)
        poly = sp.Poly(expr, x1, x2)
        col = idx[(p, q)]
        for (bi, bj) in basis:
            row = idx[(bi, bj)]
            coeff = poly.coeff_monomial(x1**bi * x2**bj)
            Q[row, col] = float(coeff) if coeff is not None else 0.0
    return Q


def basis_vector_2d(m: int, x1_value: float, x2_value: float) -> np.ndarray:
    basis = basis_index_2d(m)
    return np.array([x1_value**i * x2_value**j for (i, j) in basis])
