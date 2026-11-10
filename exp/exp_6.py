"""Experiment 6: maximum-likelihood estimation of HNGARCH, JPGARCH, NGARCH and
VPGARCH on daily S&P 500 excess log returns (Sec. 4.1-4.2).

Run: python -m exp.exp_6
"""

from __future__ import annotations

import json
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import load_ohlcv, load_risk_free_rate
from src import config as cfg
from src import estimation as est
from src import innovations as innov

RESULTS_DIR = "results"


def load_excess_returns() -> tuple[np.ndarray, pd.DataFrame]:
    """Daily S&P 500 (SPY proxy) excess log returns, Sec. 4.1.

    Excess log return = log(close_t/close_{t-1}) minus the daily 3-month
    Treasury yield (annualized percent / 100 / 252), the risk-free rate used
    in Section 4.1 to compute daily excess log returns.
    """
    bars, meta = load_ohlcv([cfg.EXP6_TICKER], cfg.EXP6_START_DATE, cfg.EXP6_END_DATE, timeframe="1d")
    bars = bars.sort_values("date").reset_index(drop=True)
    bars["log_close"] = np.log(bars["close"])
    bars["log_return"] = bars["log_close"].diff()
    bars = bars.dropna(subset=["log_return"]).reset_index(drop=True)

    rf = load_risk_free_rate(cfg.EXP6_START_DATE, cfg.EXP6_END_DATE)
    rf = rf.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(bars, rf, on="date", direction="backward")
    merged["yield_3_month"] = merged["yield_3_month"].ffill().bfill()
    merged["r_daily"] = merged["yield_3_month"] / 100.0 / 252.0
    merged["excess_log_return"] = merged["log_return"] - merged["r_daily"]
    merged.attrs["ticker_details"] = meta.get("ticker_details")
    merged.attrs["non_massive_source"] = meta.get("non_massive_source")
    return merged["excess_log_return"].to_numpy(dtype=np.float64), merged


def fit_all_models(y: np.ndarray, n_starts: int) -> dict[tuple[str, str], est.FitResult]:
    results = {}
    for model in est.MODELS:
        for dist in ("gaussian", "nig"):
            t0 = time.time()
            fit = est.fit_model(y, model, dist, n_starts=n_starts)
            results[(model, dist)] = fit
            print(f"  {model:9s} {dist:8s} loglik={fit.loglik:12.2f} "
                  f"aic={fit.aic:10.2f} time={time.time() - t0:6.2f}s")
    return results


def lr_tests(results: dict[tuple[str, str], est.FitResult]) -> list[dict]:
    """Likelihood-ratio tests of the two nested pairs for each innovation law
    (Sec. 4.2): JPGARCH nests HNGARCH at b1=0; VPGARCH nests NGARCH at
    a1=0, b0=0 (a0 is fixed at zero in both, Sec. 4.2). Conventional chi^2
    reference distributions are used, noting the paper's caveat (Sec. 4.2)
    that boundary restrictions mean these need not be the exact asymptotic null.
    """
    out = []
    for dist in ("gaussian", "nig"):
        for full, restricted, df in (("JPGARCH", "HNGARCH", 1), ("VPGARCH", "NGARCH", 2)):
            lr_full = results[(full, dist)]
            lr_restr = results[(restricted, dist)]
            lr_stat = 2.0 * (lr_full.loglik - lr_restr.loglik)
            p_value = float(chi2.sf(max(lr_stat, 0.0), df))
            out.append({"dist": dist, "full_model": full, "restricted_model": restricted,
                        "lr_stat": lr_stat, "df": df, "p_value": p_value})
    return out


def term_structures_and_half_lives(results: dict[tuple[str, str], est.FitResult]) -> dict:
    out = {}
    for (model, dist), fit in results.items():
        phi = fit.params["phi"]
        sigma2 = fit.params["sigma2"]
        ts = est.variance_term_structure(phi, sigma2, sigma2, 252)
        out[f"{model}_{dist}"] = {
            "half_life_days": est.half_life(phi) if 0 < phi < 1 else float("nan"),
            "term_structure_day252": float(ts[-1]),
            "sigma2": sigma2,
            "phi": phi,
        }
    return out


def vovnic_curves(results: dict[tuple[str, str], est.FitResult]) -> dict:
    """VoVNIC curves (Eq. 31) comparing VPGARCH and NGARCH under Gaussian
    innovations, Figure 1 (Sec. 4.2): weaker variance-of-variance propagation
    for VPGARCH than NGARCH at a given return-innovation shock."""
    out = {}
    eps_grid = np.array(cfg.EXP6_VOVNIC_EPSILONS)
    m4 = innov.gaussian_fourth_moment()
    for model in ("VPGARCH", "NGARCH"):
        fit = results[(model, "gaussian")]
        curve = est.vovnic(model, fit.params, m4, eps_grid, max(cfg.EXP6_VOVNIC_HORIZONS))
        out[model] = {
            str(h): curve[h].tolist() for h in cfg.EXP6_VOVNIC_HORIZONS
        }
    out["epsilons"] = eps_grid.tolist()
    return out


def main(n_starts: int | None = None) -> dict:
    n_starts = n_starts or cfg.EXP6_OPTIM_RESTARTS
    y, merged = load_excess_returns()
    print(f"Loaded {len(y)} daily excess log returns from {merged['date'].min()} to {merged['date'].max()}")
    print(f"Paper reports n={cfg.EXP6_N_OBS_PAPER} observations (Table 1 notes)")

    results = fit_all_models(y, n_starts)
    lr = lr_tests(results)
    term = term_structures_and_half_lives(results)
    vov = vovnic_curves(results)

    summary_rows = []
    for (model, dist), fit in results.items():
        row = {"model": model, "dist": dist, "loglik": fit.loglik, "n_params": fit.n_params,
               "aic": fit.aic, "bic": fit.bic, "hqic": fit.hqic}
        row.update({f"param_{k}": v for k, v in fit.params.items()})
        if fit.nu is not None:
            row["nu"] = fit.nu
            row["kurtosis"] = 3.0 + 3.0 * fit.nu
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{RESULTS_DIR}/exp6_mle_summary.csv", index=False)

    gain_nig = {}
    for model in est.MODELS:
        gain_nig[model] = results[(model, "nig")].loglik - results[(model, "gaussian")].loglik

    output = {
        "n_obs": len(y),
        "n_obs_paper": cfg.EXP6_N_OBS_PAPER,
        "sample_start": str(merged["date"].min()),
        "sample_end": str(merged["date"].max()),
        "optim_restarts": n_starts,
        "data_sources": {
            "ticker_details": merged.attrs.get("ticker_details"),
            "non_massive_source": merged.attrs.get("non_massive_source"),
        },
        "fits": {f"{m}_{d}": {
            "loglik": r.loglik, "aic": r.aic, "bic": r.bic, "hqic": r.hqic,
            "n_params": r.n_params, "params": r.params, "nu": r.nu, "se": r.se,
        } for (m, d), r in results.items()},
        "loglik_gain_nig_over_gaussian": gain_nig,
        "lr_tests": lr,
        "term_structures": term,
        "vovnic": vov,
    }
    with open(f"{RESULTS_DIR}/exp6_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s = np.arange(253)
    for model in est.MODELS:
        fit = results[(model, "gaussian")]
        ts = est.variance_term_structure(fit.params["phi"], fit.params["sigma2"], fit.params["sigma2"] * 1.5, 252)
        axes[0].plot(s, ts, label=model)
    axes[0].set_xlabel("horizon s (trading days)")
    axes[0].set_ylabel("E_t[h_{t+s}]")
    axes[0].set_title("Expected-variance term structure (Gaussian fits)")
    axes[0].legend()

    for model in ("VPGARCH", "NGARCH"):
        curve = vov[model][str(252)]
        axes[1].plot(vov["epsilons"], curve, marker="o", label=model)
    axes[1].set_xlabel("return innovation epsilon")
    axes[1].set_ylabel("VoVNIC_252(epsilon)")
    axes[1].set_title("Variance-of-variance news impact at s=252 (Eq. 31)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp6_term_structure_vovnic.png", dpi=120)
    plt.close(fig)

    return output


if __name__ == "__main__":
    main()
