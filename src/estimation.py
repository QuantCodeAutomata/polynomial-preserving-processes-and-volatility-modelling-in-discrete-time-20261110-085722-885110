"""Maximum-likelihood estimation of HNGARCH, JPGARCH, NGARCH and VPGARCH on
daily excess log returns (Sec. 4.1-4.2), under Gaussian and symmetric NIG
innovations, plus information criteria, variance term structures and the
variance-of-variance news-impact curve (VoVNIC) of Eq. (31).

All four models share the return/variance recursion of Eqs. (9)-(11):

    y_{t+1} = r*tau + lambda*h_t + sqrt(tau*h_t)*eps_t
    h_{t+1} = omega + phi*h_t + psi_2(h_t)*eps_t + psi_3(h_t)*(eps_t^2-1)

with tau=1, rho=1 (the GARCH special case, Sec. 4.1) and model-specific
loadings psi_2, psi_3 (Sec. 4.2):

    HNGARCH : psi_2 = eta*sqrt(h),                 psi_3 = b0            (b1=0)
    JPGARCH : psi_2 = eta*sqrt(h),                 psi_3 = b0 + b1*h
    NGARCH  : psi_2 = eta*h,                        psi_3 = b1*h          (a0=a1=b0=0)
    VPGARCH : psi_2 = eta*sqrt(a0 + a1*h + h^2),    psi_3 = b0 + b1*h     (a0=0 fixed)

Excess-return convention: r is fixed at 0 because the dependent variable is
already the excess log return (data_requirements, exp_6), so the whole
conditional mean lambda*h_t is the equity risk premium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import special
from scipy.optimize import minimize

from src.jpgarch_sv import JPGARCHParams
from src.jpgarch_sv import check_positivity as check_positivity_jp
from src.vpgarch_sv import VPGARCHParams
from src.vpgarch_sv import check_positivity as check_positivity_vp

# Integer codes for the psi_2 functional form, dispatched in the numba core
# below; psi_3 = b0 + b1*h covers all four models (Sec. 4.2) once b0, b1 are
# set to zero where the model omits that term (HNGARCH: b1=0, NGARCH: b0=0).
_PSI2_SQRT_LINEAR = 0  # HNGARCH, JPGARCH: eta*sqrt(h)
_PSI2_LINEAR = 1  # NGARCH: eta*h
_PSI2_SQRT_QUADRATIC = 2  # VPGARCH: eta*sqrt(a0+a1*h+h^2)
_PSI2_CODE = {"HNGARCH": _PSI2_SQRT_LINEAR, "JPGARCH": _PSI2_SQRT_LINEAR,
              "NGARCH": _PSI2_LINEAR, "VPGARCH": _PSI2_SQRT_QUADRATIC}


@njit(cache=True)
def _filter_variance_core(y, psi2_code, phi, sigma2, lam, eta, a0, a1, b0, b1):
    """Numba-compiled Eq. (9)-(10) forward recursion (h_0=sigma2, Sec. 4.2)."""
    n = y.shape[0]
    h = np.empty(n)
    eps = np.empty(n)
    h[0] = sigma2
    omega = (1.0 - phi) * sigma2
    ok = True
    for t in range(n - 1):
        ht = h[t]
        sqrt_h = np.sqrt(ht) if ht > 1e-12 else 1e-6
        eps[t] = (y[t] - lam * ht) / sqrt_h
        if psi2_code == _PSI2_SQRT_LINEAR:
            psi2_val = eta * np.sqrt(ht) if ht > 0.0 else 0.0
        elif psi2_code == _PSI2_LINEAR:
            psi2_val = eta * ht
        else:
            inner = a0 + a1 * ht + ht * ht
            psi2_val = eta * np.sqrt(inner) if inner > 0.0 else 0.0
        psi3_val = b0 + b1 * ht
        h[t + 1] = omega + phi * ht + psi2_val * eps[t] + psi3_val * (eps[t] ** 2 - 1.0)
        if h[t + 1] <= 0.0:
            ok = False
            break
    if ok:
        sqrt_h = np.sqrt(h[-1]) if h[-1] > 1e-12 else 1e-6
        eps[-1] = (y[-1] - lam * h[-1]) / sqrt_h
    return h, eps, ok

MODELS = ("HNGARCH", "JPGARCH", "NGARCH", "VPGARCH")

# Free-parameter names per model (phi, sigma2 always first two; lam, eta next).
PARAM_NAMES = {
    "HNGARCH": ("phi", "sigma2", "lam", "eta", "b0"),
    "JPGARCH": ("phi", "sigma2", "lam", "eta", "b0", "b1"),
    "NGARCH": ("phi", "sigma2", "lam", "eta", "b1"),
    "VPGARCH": ("phi", "sigma2", "lam", "eta", "a1", "b0", "b1"),
}


def _unpack(model: str, theta: np.ndarray) -> dict:
    """Raw-to-constrained parameter map. phi in (0,1), sigma2>0, b1>0 via exp/logit;
    other parameters unconstrained ex ante and screened by the positivity check."""
    names = PARAM_NAMES[model]
    raw = dict(zip(names, theta))
    phi = 1.0 / (1.0 + math.exp(-raw["phi"])) * 0.999
    sigma2 = math.exp(raw["sigma2"])
    out = {"phi": phi, "sigma2": sigma2, "lam": raw["lam"], "eta": raw["eta"], "r": 0.0}
    if model == "HNGARCH":
        out.update(a0=0.0, a1=0.0, b0=math.exp(raw["b0"]), b1=0.0)
    elif model == "JPGARCH":
        out.update(a0=0.0, a1=0.0, b0=math.exp(raw["b0"]), b1=math.exp(raw["b1"]))
    elif model == "NGARCH":
        out.update(a0=0.0, a1=0.0, b0=0.0, b1=math.exp(raw["b1"]))
    elif model == "VPGARCH":
        out.update(a0=0.0, a1=math.exp(raw["a1"]), b0=math.exp(raw["b0"]), b1=math.exp(raw["b1"]))
    else:
        raise ValueError(f"unknown model {model}")
    return out


def _positivity_penalty(model: str, p: dict) -> float:
    """Large penalty when Propositions 3 (VP/NGARCH) or 4 (JP/HNGARCH) are violated."""
    if model in ("VPGARCH", "NGARCH"):
        vp = VPGARCHParams(tau=1.0, rho=1.0, r=0.0, lam=p["lam"], sigma2=p["sigma2"], phi=p["phi"],
                            eta=p["eta"], a0=p["a0"], a1=p["a1"], b0=p["b0"], b1=p["b1"])
        conds = check_positivity_vp(vp)
    else:
        jp = JPGARCHParams(tau=1.0, rho=1.0, r=0.0, lam=p["lam"], sigma2=p["sigma2"], phi=p["phi"],
                            eta=p["eta"], b0=p["b0"], b1=p["b1"])
        conds = check_positivity_jp(jp)
    if conds["all"]:
        return 0.0
    violation = sum(max(0.0, -v) if isinstance(v, (int, float)) else (0.0 if v else 1.0)
                     for k, v in conds.items() if k != "all")
    return 1e6 * (1.0 + violation)


def psi2(h: np.ndarray, model: str, p: dict) -> np.ndarray:
    if model in ("HNGARCH", "JPGARCH"):
        return p["eta"] * np.sqrt(np.clip(h, 0.0, None))
    if model == "NGARCH":
        return p["eta"] * h
    return p["eta"] * np.sqrt(np.clip(p["a0"] + p["a1"] * h + h**2, 0.0, None))  # VPGARCH


def psi3(h: np.ndarray, model: str, p: dict) -> np.ndarray:
    if model in ("HNGARCH",):
        return np.full_like(h, p["b0"])
    if model == "NGARCH":
        return p["b1"] * h
    return p["b0"] + p["b1"] * h  # JPGARCH, VPGARCH


def filter_variance(y: np.ndarray, model: str, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover (h_t, eps_t) by iterating Eq. (9)-(10) forward from h_0=sigma2 (Sec. 4.1).

    Delegates to a numba-compiled core (`_filter_variance_core`) since maximum
    likelihood needs thousands of likelihood evaluations on n~6,965 daily
    observations (Sec. 4.1); the recursion itself is exactly Eq. (9)-(10),
    r=0 (excess-return convention).
    """
    h, eps, ok = _filter_variance_core(
        np.asarray(y, dtype=np.float64), _PSI2_CODE[model],
        p["phi"], p["sigma2"], p["lam"], p["eta"], p["a0"], p["a1"], p["b0"], p["b1"],
    )
    if not ok:
        h = np.where(h <= 0, np.nan, h)
    return h, eps


def neg_loglik_gaussian(theta: np.ndarray, model: str, y: np.ndarray) -> float:
    p = _unpack(model, theta)
    penalty = _positivity_penalty(model, p)
    if penalty > 0:
        return penalty
    h, eps = filter_variance(y, model, p)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e8
    ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + eps**2)
    return float(-np.sum(ll))


def nig_log_density(eps: np.ndarray, nu: float) -> np.ndarray:
    """Standardized symmetric-NIG log-density, Appendix C: alpha=1/nu,
    f(x) = (alpha/pi)*exp(alpha)*K_1(alpha*sqrt(1+x^2))/sqrt(1+x^2) for the unit-scale
    symmetric NIG with mean 0 and variance 1 (beta=0 case of the general NIG law)."""
    alpha = 1.0 / nu
    arg = alpha * np.sqrt(1.0 + eps**2)
    k1 = special.kv(1.0, arg)
    k1 = np.clip(k1, 1e-300, None)
    return np.log(alpha) - np.log(np.pi) + alpha + np.log(k1) - 0.5 * np.log(1.0 + eps**2)


def neg_loglik_nig(theta: np.ndarray, model: str, y: np.ndarray) -> float:
    p = _unpack(model, theta[:-1])
    nu = math.exp(theta[-1])
    penalty = _positivity_penalty(model, p)
    if penalty > 0:
        return penalty
    h, eps = filter_variance(y, model, p)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e8
    ll = nig_log_density(eps, nu) - 0.5 * np.log(h)
    return float(-np.sum(ll))


def _per_obs_loglik(theta: np.ndarray, model: str, y: np.ndarray, dist: str) -> np.ndarray:
    """Per-observation log-likelihood contributions (for the OPG standard errors)."""
    if dist == "gaussian":
        p = _unpack(model, theta)
        h, eps = filter_variance(y, model, p)
        return -0.5 * (np.log(2 * np.pi) + np.log(np.clip(h, 1e-300, None)) + eps**2)
    p = _unpack(model, theta[:-1])
    nu = math.exp(theta[-1])
    h, eps = filter_variance(y, model, p)
    return nig_log_density(eps, nu) - 0.5 * np.log(np.clip(h, 1e-300, None))


def opg_standard_errors(theta: np.ndarray, model: str, y: np.ndarray, dist: str, eps_fd: float = 1e-5) -> np.ndarray:
    """SE = sqrt(diag(inverse(sum_i grad_i grad_i^T))), the outer-product-of-gradients estimator."""
    n = len(y)
    k = len(theta)
    base = _per_obs_loglik(theta, model, y, dist)
    grads = np.zeros((n, k))
    for j in range(k):
        bump = np.zeros(k)
        bump[j] = eps_fd * max(1.0, abs(theta[j]))
        up = _per_obs_loglik(theta + bump, model, y, dist)
        grads[:, j] = (up - base) / bump[j]
    opg = grads.T @ grads
    try:
        cov = np.linalg.inv(opg)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(opg)
    diag = np.clip(np.diag(cov), 0.0, None)
    return np.sqrt(diag)


@dataclass
class FitResult:
    model: str
    dist: str
    params: dict
    nu: float | None
    loglik: float
    n_params: int
    n_obs: int
    aic: float
    bic: float
    hqic: float
    se: dict


def fit_model(y: np.ndarray, model: str, dist: str, x0: np.ndarray | None = None, n_starts: int = 3, seed: int = 0) -> FitResult:
    """MLE of `model` under `dist` innovations by scipy.optimize.minimize (Sec. 4.2)."""
    names = PARAM_NAMES[model]
    k = len(names) + (1 if dist == "nig" else 0)
    rng = np.random.default_rng(seed)
    if x0 is None:
        v = float(np.var(y))
        x0 = np.zeros(k)
        all_names = list(names) + (["nu"] if dist == "nig" else [])
        for i, name in enumerate(all_names):
            if name == "phi":
                x0[i] = 2.0  # phi near 0.87 after the logit map
            elif name == "sigma2":
                x0[i] = math.log(v)
            elif name == "lam":
                x0[i] = 0.0
            elif name == "eta":
                # HNGARCH/JPGARCH bound eta^2 <= 4*b0*(phi-b1) with b0 ~ variance units
                # (Prop. 4), far tighter than the VPGARCH/NGARCH bound eta^2 <= 4*b1*(phi-b1)
                # with dimensionless b1 (Prop. 3); scale the start accordingly.
                x0[i] = -0.2 * math.sqrt(v) if model in ("HNGARCH", "JPGARCH") else -0.05
            elif name == "b0":
                x0[i] = math.log(0.1 * v)  # b0 has variance units (<=omega, Prop. 3/4)
            elif name == "a1":
                x0[i] = math.log(0.1 * v)  # a1 has variance units (added inside sqrt(.), Eq. 12)
            elif name == "b1":
                x0[i] = math.log(0.05)  # b1 is a dimensionless multiplier of h (<phi, Prop. 4)
            elif name == "nu":
                x0[i] = math.log(0.4)  # nu~0.4 near the Sec. 4.2 NIG estimates (kurtosis~4.2)
    objective = neg_loglik_gaussian if dist == "gaussian" else neg_loglik_nig
    best = None
    starts = [x0] + [x0 + rng.normal(scale=0.3, size=k) for _ in range(n_starts - 1)]
    for start in starts:
        res = minimize(objective, start, args=(model, y), method="Nelder-Mead",
                        options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x
    if dist == "gaussian":
        params = _unpack(model, theta)
        nu = None
    else:
        params = _unpack(model, theta[:-1])
        nu = math.exp(theta[-1])
    loglik = -best.fun
    se_vec = opg_standard_errors(theta, model, y, dist)
    se = dict(zip(list(names) + (["nu"] if dist == "nig" else []), se_vec))
    n = len(y)
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * math.log(n)
    hqic = -2 * loglik + 2 * k * math.log(math.log(n))
    return FitResult(model=model, dist=dist, params=params, nu=nu, loglik=loglik, n_params=k,
                      n_obs=n, aic=aic, bic=bic, hqic=hqic, se=se)


# --------------------------------------------------------------------------
# Variance term structure, half-life, and VoVNIC (Eq. 31)
# --------------------------------------------------------------------------
def variance_term_structure(phi: float, sigma2: float, h0: float, s_max: int) -> np.ndarray:
    """E_t[h_{t+s}] = sigma2 + phi^s*(h0 - sigma2), s=0..s_max (Sec. 4.2)."""
    s = np.arange(s_max + 1)
    return sigma2 + phi**s * (h0 - sigma2)


def half_life(phi: float) -> float:
    """Mean-reversion half-life in trading days: ln(0.5)/ln(phi)."""
    return math.log(0.5) / math.log(phi)


def d_coeffs(model: str, p: dict, m4: float) -> tuple[float, float, float]:
    """d0, d1, d2 with d(x) = d0 + d1*x + d2*x^2 = Var[h_{t+1} | h_t=x] (Sec. 2.3.1/4.2).

    VPGARCH: d0=eta^2*a0+(m4-1)*b0^2, d1=eta^2*a1+2*(m4-1)*b0*b1, d2=eta^2+(m4-1)*b1^2.
    JPGARCH/HNGARCH/NGARCH: d0=(m4-1)*b0^2, d1=eta^2+2*(m4-1)*b0*b1, d2=(m4-1)*b1^2.
    """
    eta, b0, b1 = p["eta"], p["b0"], p["b1"]
    if model == "VPGARCH":
        a0, a1 = p["a0"], p["a1"]
        return eta**2 * a0 + (m4 - 1) * b0**2, eta**2 * a1 + 2 * (m4 - 1) * b0 * b1, eta**2 + (m4 - 1) * b1**2
    return (m4 - 1) * b0**2, eta**2 + 2 * (m4 - 1) * b0 * b1, (m4 - 1) * b1**2


def eov_nic(model: str, p: dict, sigma2: float, eps: np.ndarray, j: int) -> np.ndarray:
    """EoVNIC_{j+1}(eps) = sigma2 + phi^j*(psi_2(sigma2)*eps + psi_3(sigma2)*(eps^2-1)) (Eq. 31)."""
    phi = p["phi"]
    psi2_s = float(psi2(np.array([sigma2]), model, p)[0])
    psi3_s = float(psi3(np.array([sigma2]), model, p)[0])
    return sigma2 + phi**j * (psi2_s * eps + psi3_s * (eps**2 - 1.0))


def vovnic(model: str, p: dict, m4: float, eps: np.ndarray, s_max: int) -> np.ndarray:
    """VoVNIC_s(eps) = Var_{t+1}[h_{t+s}], s=1..s_max, via the recursion of Eq. (31):

    VoVNIC_s(eps) = sum_{j=0}^{s-2} (phi^2+d2)^{s-2-j} * d(EoVNIC_{j+1}(eps)).
    """
    phi = p["phi"]
    d0, d1, d2 = d_coeffs(model, p, m4)
    sigma2 = p["sigma2"]

    def d_of(x):
        return d0 + d1 * x + d2 * x**2

    out = np.zeros((s_max + 1, len(eps)))
    persistence = phi**2 + d2
    for s in range(2, s_max + 1):
        acc = np.zeros(len(eps))
        for j in range(0, s - 1):
            e = eov_nic(model, p, sigma2, eps, j + 1)
            acc += persistence ** (s - 2 - j) * d_of(e)
        out[s] = acc
    return out
