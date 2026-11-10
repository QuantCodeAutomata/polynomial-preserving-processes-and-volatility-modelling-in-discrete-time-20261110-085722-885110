"""Experiment 7: Empirical option-based estimation (Sec. 4.3).

Calibrates HNGARCH, JPGARCH, NGARCH, and VPGARCH under Gaussian and symmetric
NIG innovations to an S&P 500 index-option cross-section by minimizing the
implied-volatility root mean squared error (IVRMSE), with option prices from
Monte Carlo simulation of the physical-measure dynamics (Eqs. 9-10) weighted
by the variance-dependent pricing kernel with xi2=0 (Eq. B.1, Sec. 4.3).

The paper calibrates 1,443 weekly OptionMetrics/WRDS cross-sections
(Sec. 4.1); OptionMetrics is not reachable through the QuantX Dataloader or
the Massive API used here, so this script calibrates to a single live SPY
option-chain cross-section obtained from Massive's option-chain snapshot
endpoint (see "Not reproduced" in results/RESULTS.md). Monte Carlo paths and
optimizer iterations are also reduced from the paper's configuration; both
reductions are recorded in results/RESULTS.md.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data.data_loader import load_option_chain_snapshot, load_risk_free_rate
from src import config
from src.estimation import MODELS, PARAM_NAMES, _positivity_penalty, _unpack
from src.option_pricing import OptionMCConfig, ivrmse, mc_option_prices

RESULTS_DIR = Path("results")


def load_filtered_chain() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fetch and filter the live SPY option-chain snapshot per Sec. 4.1's rules.

    Returns (strikes, ttm_days, is_call, market_iv, spot, rate_daily).
    """
    chain = load_option_chain_snapshot(config.EXP7_UNDERLYING, use_cache=True)
    spot = float(chain["underlying_price"].dropna().iloc[0])
    # The snapshot is a live cross-section (not point-in-time historical data),
    # so maturities are measured from the download time.
    ref_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    ttm_cal = (chain["expiration_date"] - ref_date).dt.days.astype(float)

    # The live snapshot carries no bid/ask quotes (unlike OptionMetrics), so the
    # $0.375 minimum-price filter of Sec. 4.1 cannot be applied; liquidity is
    # screened through open interest and volume instead (see RESULTS.md).
    otm_call = (chain["contract_type"] == "call") & (chain["strike_price"] >= spot)
    otm_put = (chain["contract_type"] == "put") & (chain["strike_price"] <= spot)
    mask = (
        (otm_call | otm_put)
        & (ttm_cal >= config.EXP7_MIN_MATURITY_DAYS)
        & (ttm_cal <= config.EXP7_MAX_MATURITY_DAYS)
        & chain["implied_volatility"].notna()
        & (chain["implied_volatility"] > 0.01)
        & (chain["implied_volatility"] < 3.0)
        & ((chain["open_interest"].fillna(0) > 0) | (chain["volume"].fillna(0) > 0))
    )
    filtered = chain.loc[mask].copy()
    filtered["ttm_cal"] = ttm_cal.loc[mask]
    if len(filtered) == 0:
        raise RuntimeError("no option observations survived the Sec. 4.1 filters")

    rf = load_risk_free_rate(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    )
    yield_3m_pct = float(rf["yield_3_month"].iloc[-1])
    rate_daily = math.log(1.0 + yield_3m_pct / 100.0) / 252.0

    ttm_trading_days = np.maximum(1, np.round(filtered["ttm_cal"].values * (252.0 / 365.0))).astype(int)
    strikes = filtered["strike_price"].values.astype(float)
    is_call = (filtered["contract_type"].values == "call")
    market_iv = filtered["implied_volatility"].values.astype(float)
    return strikes, ttm_trading_days, is_call, market_iv, spot, rate_daily


def _init_theta(model: str, dist: str) -> np.ndarray:
    """Starting values: moderate persistence and a small negative leverage
    loading, typical of daily equity-index GARCH fits (Sec. 4.2 discusses
    negative eta estimates across all four models)."""
    names = PARAM_NAMES[model]
    daily_var = (0.16 ** 2) / 252.0  # ~16% annualized volatility, long-run variance
    raw = {
        "phi": math.log(0.95 / (1.0 - 0.95 / 0.999)),
        "sigma2": math.log(daily_var),
        "lam": 2.0,
        "eta": -0.5 if model in ("HNGARCH", "JPGARCH", "VPGARCH") else -60.0,
        "a1": math.log(0.5),
        "b0": math.log(daily_var * 0.05),
        "b1": math.log(0.05),
    }
    theta = np.array([raw[n] for n in names])
    if dist == "nig":
        theta = np.append(theta, math.log(0.43))  # nu starting value, Sec. 4.2's estimated range
    return theta


def calibrate_one(model: str, dist: str, strikes, ttms, is_call, market_iv, spot, rate_daily,
                   n_paths: int, maxiter: int, seed: int) -> dict:
    n_free = len(PARAM_NAMES[model])

    def objective(theta: np.ndarray) -> float:
        p = _unpack(model, theta[:n_free])
        penalty = _positivity_penalty(model, p)
        if penalty > 0:
            return penalty
        nu = float(np.exp(theta[n_free])) if dist == "nig" else None
        if dist == "nig" and (nu <= 0.05 or nu > 5.0):
            return 1e6
        cfg = OptionMCConfig(n_paths=n_paths, seed=seed)
        prices = mc_option_prices(p, model, dist, nu, strikes, ttms, is_call, spot, rate_daily, cfg)
        val = ivrmse(prices, market_iv, spot, strikes, rate_daily, ttms, is_call)
        return val if np.isfinite(val) else 1e3

    theta0 = _init_theta(model, dist)
    best = None
    for restart in range(config.EXP7_OPTIM_RESTARTS):
        theta_start = theta0 if restart == 0 else theta0 + config.RNG.normal(scale=0.2, size=theta0.shape)
        res = minimize(objective, theta_start, method="Nelder-Mead",
                        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    p_hat = _unpack(model, best.x[:n_free])
    nu_hat = float(np.exp(best.x[n_free])) if dist == "nig" else None
    return {"model": model, "dist": dist, "ivrmse": float(best.fun), "params": p_hat, "nu": nu_hat,
            "n_options": int(len(strikes)), "n_paths": n_paths, "converged": bool(res.success)}


def main(n_paths: int | None = None, maxiter: int | None = None) -> dict:
    n_paths = n_paths or config.EXP7_MC_PATHS_RUN
    maxiter = maxiter or config.EXP7_OPTIM_MAXITER
    t0 = time.time()
    strikes, ttms, is_call, market_iv, spot, rate_daily = load_filtered_chain()
    print(f"Filtered option cross-section: {len(strikes)} contracts, spot={spot:.2f}, "
          f"maturities {ttms.min()}-{ttms.max()} trading days, rate_daily={rate_daily:.6g}")

    results = []
    for dist in ("gaussian", "nig"):
        for model in MODELS:
            r = calibrate_one(model, dist, strikes, ttms, is_call, market_iv, spot, rate_daily,
                               n_paths, maxiter, seed=0)
            print(f"{model:9s} {dist:9s} IVRMSE={r['ivrmse']:.4f}  nu={r['nu']}")
            results.append(r)

    summary = {
        "n_options": int(len(strikes)),
        "spot": spot,
        "rate_daily": rate_daily,
        "n_paths_paper": config.EXP7_MC_PATHS_PAPER,
        "n_paths_run": n_paths,
        "n_weeks_paper": config.EXP7_N_WEEKS_PAPER,
        "n_snapshots_run": config.EXP7_N_SNAPSHOTS_RUN,
        "runtime_sec": time.time() - t0,
        "results": results,
    }
    with open(RESULTS_DIR / "exp7_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(strikes / spot)
    for dist in ("gaussian", "nig"):
        for r in results:
            if r["dist"] != dist:
                continue
            p = r["params"]
            nu = r["nu"]
            cfg = OptionMCConfig(n_paths=n_paths, seed=1)
            prices = mc_option_prices(p, r["model"], dist, nu, strikes, ttms, is_call, spot, rate_daily, cfg)
            model_iv = np.array([
                np.nan if not np.isfinite(prices[i]) else prices[i] for i in range(len(prices))
            ])
            from src.option_pricing import implied_vol
            model_iv = np.array([implied_vol(prices[i], spot, strikes[i], rate_daily, ttms[i] / 252.0, bool(is_call[i]))
                                  for i in range(len(prices))])
            ax.plot((strikes / spot)[order], model_iv[order], alpha=0.5,
                    label=f"{r['model']}-{dist}" if dist == "gaussian" else None, lw=1)
    ax.scatter(strikes / spot, market_iv, s=8, color="black", label="market", zorder=5)
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Model vs. market implied volatility (Sec. 4.3 IVRMSE calibration)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_iv_smile.png", dpi=150)
    plt.close(fig)

    print(f"exp_7 finished in {time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
