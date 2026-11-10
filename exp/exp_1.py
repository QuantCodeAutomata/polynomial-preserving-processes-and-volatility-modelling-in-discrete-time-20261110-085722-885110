"""exp_1 — VPGARCH-SV: specification, positivity, and variance-moment analysis.

Reproduces Section 2.2.1 (Eqs. (9)-(12)), Proposition 3 (nonnegativity),
Section 2.3.1 (the one-step transition matrix Q_m on Pol_m(R_+) and the
s-step moment formula of Proposition 1), Proposition 5 (moment stability)
and Remark 1 (the nested NGARCH-SV / NGARCH special cases).

Run: python -m exp.exp_1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import vpgarch_sv as vp

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_params() -> vp.VPGARCHParams:
    return vp.VPGARCHParams(
        tau=cfg.EXP1_TAU, rho=cfg.EXP1_RHO, r=cfg.EXP1_R, lam=cfg.EXP1_LAMBDA,
        sigma2=cfg.EXP1_SIGMA2, phi=cfg.EXP1_PHI, eta=cfg.EXP1_ETA,
        a0=cfg.EXP1_A0, a1=cfg.EXP1_A1, b0=cfg.EXP1_B0, b1=cfg.EXP1_B1,
    )


def verify_positivity_grid(p: vp.VPGARCHParams, n_grid: int = 2000) -> dict:
    """Proposition 3: verify h_next >= 0 on a grid of states x>=0 and innovations eps2 in R."""
    conds = vp.check_positivity(p)
    rng = np.random.default_rng(cfg.RNG_SEED)
    x_grid = rng.uniform(0.0, 10.0 * p.sigma2, n_grid)
    eps2_grid = rng.uniform(-8.0, 8.0, n_grid)
    h_next = vp.h_next(x_grid, eps2_grid, p)
    conds["min_h_next_on_grid"] = float(np.min(h_next))
    conds["grid_all_nonnegative"] = bool(np.all(h_next >= -1e-10))
    return conds


def monte_carlo_moments(p: vp.VPGARCHParams, horizons: tuple, dist: str, nu: float | None, n_paths: int) -> dict:
    """Simulate N paths and estimate E[h_s^i | h_0] for i=1,2 at each horizon (Sec. 2.3.1)."""
    rng = np.random.default_rng(cfg.RNG_SEED + 1)
    max_s = max(horizons)
    paths = vp.simulate_paths(p, max_s, n_paths, dist, rng, nu)
    h = paths["h"]
    out = {}
    for s in horizons:
        out[s] = {"mean_h": float(np.mean(h[:, s])), "mean_h2": float(np.mean(h[:, s] ** 2)),
                   "se_mean_h": float(np.std(h[:, s]) / np.sqrt(n_paths)),
                   "se_mean_h2": float(np.std(h[:, s] ** 2) / np.sqrt(n_paths))}
    return out


def matrix_moments(p: vp.VPGARCHParams, horizons: tuple, dist: str, nu: float | None, m: int) -> dict:
    """Exact moments via Q_m^s, Proposition 1."""
    Q = vp.transition_matrix(p, m, dist, nu)
    out = {}
    for s in horizons:
        out[s] = {"mean_h": vp.multi_step_moment(Q, p.sigma2, s, 1),
                   "mean_h2": vp.multi_step_moment(Q, p.sigma2, s, 2) if m >= 2 else None}
    return out, Q


def remark1_check(dist: str, nu: float | None, n_paths: int = 20_000) -> dict:
    """Remark 1: NGARCH-SV special case (a0=a1=b0=0, b1=alpha, eta=-2*alpha*gamma) and
    its rho=1 NGARCH restriction; verify the variance recursion matches the explicit
    NGARCH form h_{t+1} = omega + phi*h_t + alpha*h_t*[eps2^2 - 2*gamma*eps2 - 1]."""
    alpha, gamma = 0.06, 0.4
    p_sv = vp.ngarch_sv_params(tau=1.0, rho=0.7, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=alpha, gamma=gamma)
    p_ngarch = vp.ngarch_params(tau=1.0, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=alpha, gamma=gamma)
    rng = np.random.default_rng(cfg.RNG_SEED + 2)
    x = 1.2e-4
    eps2 = rng.standard_normal(5000)
    h1 = vp.h_next(np.full(5000, x), eps2, p_ngarch)
    h2 = p_ngarch.omega + p_ngarch.phi * x + alpha * x * (eps2**2 - 2 * gamma * eps2 - 1.0)
    max_abs_diff = float(np.max(np.abs(h1 - h2)))
    pos_sv = vp.check_positivity(p_sv)
    pos_ngarch = vp.check_positivity(p_ngarch)
    return {
        "alpha": alpha, "gamma": gamma,
        "max_abs_diff_recursion_vs_explicit_ngarch": max_abs_diff,
        "matches": bool(max_abs_diff < 1e-9),
        "ngarch_sv_positivity_all": pos_sv["all"],
        "ngarch_positivity_all": pos_ngarch["all"],
    }


def run(dist: str, nu: float | None, n_paths: int) -> dict:
    p = build_params()
    positivity = verify_positivity_grid(p)
    assert positivity["all"], "Proposition 3 restrictions must hold for the chosen parameters"

    mc = monte_carlo_moments(p, cfg.EXP1_HORIZONS, dist, nu, n_paths)
    mat, Q = matrix_moments(p, cfg.EXP1_HORIZONS, dist, nu, cfg.EXP1_MAX_DEGREE)

    comparison = {}
    for s in cfg.EXP1_HORIZONS:
        se = mc[s]["se_mean_h"]
        diff = mat[s]["mean_h"] - mc[s]["mean_h"]
        comparison[s] = {
            "matrix_mean_h": mat[s]["mean_h"], "mc_mean_h": mc[s]["mean_h"],
            "mc_se_mean_h": se, "diff_in_se_units": diff / se if se > 0 else None,
            "matrix_mean_h2": mat[s]["mean_h2"], "mc_mean_h2": mc[s]["mean_h2"],
        }

    stability = vp.stability_condition(p, cfg.EXP1_MAX_DEGREE, dist, nu)
    stationary = vp.stationary_moments(p, cfg.EXP1_MAX_DEGREE, dist, nu).tolist() if stability["stable"] else None

    return {
        "dist": dist, "nu": nu, "n_paths": n_paths,
        "params": p.__dict__,
        "positivity": positivity,
        "moment_comparison": comparison,
        "stability": stability,
        "stationary_moments": stationary,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for dist, nu in (("gaussian", None), ("nig", cfg.EXP1_NIG_NU)):
        results[dist] = run(dist, nu, cfg.MC_PATHS_VERIFICATION_RUN)

    results["remark1_ngarch_nesting"] = {
        dist_name: remark1_check(dist_name, nu_val)
        for dist_name, nu_val in (("gaussian", None), ("nig", cfg.EXP1_NIG_NU))
    }

    out_path = RESULTS_DIR / "exp1_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_path}")

    # Plot: matrix-predicted vs Monte Carlo mean variance at each horizon (Gaussian case).
    horizons = sorted(cfg.EXP1_HORIZONS)
    gauss = results["gaussian"]["moment_comparison"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, [gauss[s]["matrix_mean_h"] for s in horizons], "o-", label="Q_m matrix formula")
    ax.errorbar(horizons, [gauss[s]["mc_mean_h"] for s in horizons],
                yerr=[1.96 * gauss[s]["mc_se_mean_h"] for s in horizons], fmt="x", label="Monte Carlo (95% CI)")
    ax.set_xlabel("horizon s (periods)")
    ax.set_ylabel("E[h_{t+s} | h_t]")
    ax.set_title("VPGARCH-SV: exact vs Monte Carlo variance moments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_moment_convergence.png", dpi=120)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp1_moment_convergence.png'}")


if __name__ == "__main__":
    main()
