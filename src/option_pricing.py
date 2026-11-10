"""Monte Carlo option pricing under the physical measure with a variance-
dependent pricing kernel, and weekly IVRMSE calibration (Sec. 4.3).

Prices are computed as kernel-weighted averages of terminal payoffs under P
(Eq. B.1, xi2=0): price = E^P[ (prod_k N_k) * payoff ], with N_k the one-period
pricing kernel exp(xi0+xi1*y_k) and xi1 solved from the martingale restriction
of Eq. (B.2) at every simulated state (Sec. 4.3, "equity risk-preference
parameter ... determined from the martingale restriction").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src import innovations as innov
from src.risk_neutral import cgf_gaussian


def bs_price(spot: float, strike: float, rate: float, ttm: float, vol: float, is_call: bool) -> float:
    """Black-Scholes price for validation and implied-volatility inversion."""
    if vol <= 0 or ttm <= 0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)
    if is_call:
        return spot * norm.cdf(d1) - strike * math.exp(-rate * ttm) * norm.cdf(d2)
    return strike * math.exp(-rate * ttm) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_vol(price: float, spot: float, strike: float, rate: float, ttm: float, is_call: bool) -> float:
    """Invert the Black-Scholes formula by bisection (scipy.optimize.brentq)."""
    intrinsic = max(spot - strike * math.exp(-rate * ttm), 0.0) if is_call else max(strike * math.exp(-rate * ttm) - spot, 0.0)
    if price <= intrinsic + 1e-12:
        return 1e-6

    def f(vol):
        return bs_price(spot, strike, rate, ttm, vol, is_call) - price

    try:
        return brentq(f, 1e-6, 5.0)
    except ValueError:
        return float("nan")


def implied_vol_vec(
    prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    rate: float,
    ttms: np.ndarray,
    is_call: np.ndarray,
    init_vol: np.ndarray | float = 0.2,
    n_iter: int = 80,
) -> np.ndarray:
    """Vectorized Black-Scholes implied-volatility inversion by damped Newton-
    Raphson with vega (fully array-valued so that one weekly cross-section of
    several thousand quotes is inverted in a handful of numpy passes instead
    of one scipy.optimize.brentq call per option, as used by `implied_vol`).
    """
    strikes = np.asarray(strikes, dtype=float)
    ttms = np.asarray(ttms, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    prices = np.asarray(prices, dtype=float)
    disc_strike = strikes * np.exp(-rate * ttms)
    intrinsic = np.where(is_call, np.maximum(spot - disc_strike, 0.0), np.maximum(disc_strike - spot, 0.0))
    vol = np.full(strikes.shape, init_vol, dtype=float) if np.isscalar(init_vol) else np.array(init_vol, dtype=float)
    vol = np.clip(vol, 1e-3, 3.0)
    valid = (prices > intrinsic + 1e-12) & (ttms > 0)
    for _ in range(n_iter):
        sqrt_t = np.sqrt(np.maximum(ttms, 1e-12))
        d1 = (np.log(spot / np.maximum(strikes, 1e-12)) + (rate + 0.5 * vol**2) * ttms) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        call_price = spot * norm.cdf(d1) - strikes * np.exp(-rate * ttms) * norm.cdf(d2)
        model_price = np.where(is_call, call_price, call_price - spot + strikes * np.exp(-rate * ttms))
        vega = spot * norm.pdf(d1) * sqrt_t
        vega = np.where(vega < 1e-8, 1e-8, vega)
        step = (model_price - prices) / vega
        vol = np.clip(vol - step, 1e-4, 5.0)
    result = np.where(valid, vol, 1e-6)
    result = np.where(np.isfinite(result), result, np.nan)
    return result


def _xi1_gaussian(h: float, tau: float, r: float, lam: float, rho: float) -> float:
    """xi1 solving the stock martingale restriction with xi2=0, Gaussian eps1 (Eq. B.2)."""
    sqrt_th = math.sqrt(max(tau * h, 0.0))

    def f(xi1):
        a1 = xi1 * sqrt_th
        return cgf_gaussian(a1 + sqrt_th, 0.0, 0.0, rho) - cgf_gaussian(a1, 0.0, 0.0, rho) + lam * h

    try:
        return brentq(f, -100.0, 100.0)
    except ValueError:
        return 0.0


def _xi1_nig(h: float, tau: float, r: float, lam: float, alpha: float) -> float:
    """xi1 for symmetric-NIG eps1 with xi2=0, admissible (negative) branch (Badescu et al. 2011, Cor. A.3).

    The univariate NIG cgf is C_eps(s) = alpha - sqrt(alpha^2 - alpha*s^2), domain |s|<=sqrt(alpha).
    The restriction C_eps(xi1*sqrt(tau*h) + sqrt(tau*h)) - C_eps(xi1*sqrt(tau*h)) = -lambda*tau*h
    is solved on the branch keeping both arguments inside the cgf domain.
    """
    sqrt_th = math.sqrt(max(tau * h, 0.0))
    if sqrt_th <= 0:
        return 0.0
    bound = math.sqrt(alpha) / sqrt_th - 1.0  # keeps xi1*sqrt_th+sqrt_th within domain

    def c_eps(s):
        inside = alpha**2 - alpha * s**2
        if inside < 0:
            return np.inf
        return alpha - math.sqrt(inside)

    def f(xi1):
        a1 = xi1 * sqrt_th
        return c_eps(a1 + sqrt_th) - c_eps(a1) + lam * tau * h

    try:
        return brentq(f, -bound + 1e-9, bound - 1e-9)
    except ValueError:
        return 0.0


@dataclass
class OptionMCConfig:
    n_paths: int = 50_000
    seed: int = 0


def mc_option_prices(
    model_params: dict,
    model: str,
    dist: str,
    nu: float | None,
    strikes: np.ndarray,
    ttms_days: np.ndarray,
    is_call: np.ndarray,
    spot: float,
    rate_daily: float,
    cfg: OptionMCConfig,
) -> np.ndarray:
    """Price a cross-section of options by simulating the physical-measure paths
    of Eqs. (9)-(10) and weighting payoffs by the cumulative pricing kernel.

    `model_params` carries phi, sigma2, lam, eta, a0, a1, b0, b1 (as in
    src/estimation.py); `ttms_days` is the per-option maturity in trading
    days; one simulation is run to the maximum maturity and every shorter
    maturity reuses the common-random-number path up to its own horizon.
    """
    rng = np.random.default_rng(cfg.seed)
    max_t = int(np.max(ttms_days))
    n = cfg.n_paths
    p = model_params
    h = np.full(n, p["sigma2"])
    log_s = np.zeros(n)
    log_kernel = np.zeros(n)
    rho = 1.0  # GARCH special case (Sec. 4.1: rho=1, tau=1)
    cum_by_maturity: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    target_mats = set(int(t) for t in ttms_days)
    alpha = innov.nig_alpha(nu) if dist == "nig" else None

    for t in range(1, max_t + 1):
        if dist == "gaussian":
            eps1, eps2 = innov.sample_gaussian_pair(rho, n, rng)
            xi1 = _xi1_gaussian(float(np.mean(h)), 1.0, rate_daily, p["lam"], rho)
        else:
            eps1, eps2 = innov.sample_pair("nig", rho, n, rng, nu)
            xi1 = _xi1_nig(float(np.mean(h)), 1.0, rate_daily, p["lam"], alpha)
        y_next = rate_daily + p["lam"] * h + np.sqrt(np.clip(h, 0.0, None)) * eps1
        log_s = log_s + y_next
        log_kernel = log_kernel + xi1 * y_next  # xi0 cancels: price uses self-normalized kernel weights

        if model in ("VPGARCH", "NGARCH"):
            G = p["a0"] + p["a1"] * h + h**2
        else:
            G = h
        sqrt_G = np.sqrt(np.clip(G, 0.0, None))
        if model == "NGARCH":
            psi2_val = p["eta"] * h
        else:
            psi2_val = p["eta"] * sqrt_G
        if model == "HNGARCH":
            psi3_val = np.full(n, p["b0"])
        elif model == "NGARCH":
            psi3_val = p["b1"] * h
        else:
            psi3_val = p["b0"] + p["b1"] * h
        omega = (1.0 - p["phi"]) * p["sigma2"]
        h = omega + p["phi"] * h + psi2_val * eps2 + psi3_val * (eps2**2 - 1.0)
        h = np.clip(h, 0.0, None)

        if t in target_mats:
            cum_by_maturity[t] = (log_s.copy(), log_kernel.copy())

    weight = np.exp(log_kernel)
    weight = weight / np.mean(weight)  # self-normalized importance weights (Eq. B.1 kernel average)
    prices = np.empty(len(strikes))
    for i, (k, t, call) in enumerate(zip(strikes, ttms_days, is_call)):
        log_s_t, log_kernel_t = cum_by_maturity[int(t)]
        w_t = np.exp(log_kernel_t)
        w_t = w_t / np.mean(w_t)
        s_t = spot * np.exp(log_s_t)
        payoff = np.maximum(s_t - k, 0.0) if call else np.maximum(k - s_t, 0.0)
        prices[i] = math.exp(-rate_daily * t) * float(np.mean(w_t * payoff))
    return prices


def ivrmse(model_prices: np.ndarray, market_ivs: np.ndarray, spot: float, strikes: np.ndarray,
           rate_daily: float, ttms_days: np.ndarray, is_call: np.ndarray) -> float:
    """IVRMSE = sqrt(mean((IV^model - IV^market)^2)) (Sec. 4.3)."""
    model_ivs = implied_vol_vec(
        model_prices, spot, strikes, rate_daily, np.asarray(ttms_days, dtype=float) / 252.0, is_call,
        init_vol=np.clip(market_ivs, 0.05, 2.0),
    )
    valid = np.isfinite(model_ivs)
    return float(np.sqrt(np.mean((model_ivs[valid] - market_ivs[valid]) ** 2)))
