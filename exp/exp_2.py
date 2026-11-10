"""exp_2 — JPGARCH-SV: specification, positivity, and joint return-variance moments.

Reproduces Section 2.2.2 (Eqs. (9)-(11),(14)), Proposition 4 (nonnegativity),
the joint transition matrix on Pol_m(R x R_+) and Proposition 1 moment
formula, cumulative-return moments via the augmented state Z=(R,h)
(Corollary 2(ii)), and Remark 2 (the nested HNGARCH-SV / HNGARCH cases).

Run: python -m exp.exp_2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import jpgarch_sv as jp
from src import innovations as innov

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_params() -> jp.JPGARCHParams:
    return jp.JPGARCHParams(
        tau=cfg.EXP2_TAU, rho=cfg.EXP2_RHO, r=cfg.EXP2_R, lam=cfg.EXP2_LAMBDA,
        sigma2=cfg.EXP2_SIGMA2, phi=cfg.EXP2_PHI, eta=cfg.EXP2_ETA,
        b0=cfg.EXP2_B0, b1=cfg.EXP2_B1,
    )


def verify_positivity_grid(p: jp.JPGARCHParams, n_grid: int = 2000) -> dict:
    conds = jp.check_positivity(p)
    rng = np.random.default_rng(cfg.RNG_SEED)
    x_grid = rng.uniform(0.0, 10.0 * p.sigma2, n_grid)
    eps2_grid = rng.uniform(-8.0, 8.0, n_grid)
    h_n = jp.h_next(x_grid, eps2_grid, p)
    conds["min_h_next_on_grid"] = float(np.min(h_n))
    conds["grid_all_nonnegative"] = bool(np.all(h_n >= -1e-10))
    return conds


def joint_moment_check(p: jp.JPGARCHParams, dist: str, nu, m: int, horizons, n_paths: int) -> dict:
    """Compare the exact joint-matrix moments of (Y,h) with Monte Carlo at several horizons.

    Since Q_m is block upper triangular in degree (Sec. 2.1, Example 1), a
    degree-2 query (E[Y], E[h], E[Y*h]) never receives a contribution from
    the degree-3 block analytically. Building Q at the minimal degree m=2
    needed for these targets avoids amplifying floating-point residuals in
    that unused higher-degree block over long horizons (up to s=252 steps).
    """
    m = min(m, 2)
    rng = np.random.default_rng(cfg.RNG_SEED + 1)
    paths = jp.simulate_paths(p, max(horizons), n_paths, dist, rng, nu)
    Q = jp.transition_matrix_joint(p, m, dist, nu)
    out = {}
    for s in horizons:
        y_s, h_s = paths["y"][:, s], paths["h"][:, s]
        exact_Eh = jp.joint_moment(Q, m, 0.0, p.sigma2, s, 0, 1)
        exact_EY = jp.joint_moment(Q, m, 0.0, p.sigma2, s, 1, 0)
        exact_EYh = jp.joint_moment(Q, m, 0.0, p.sigma2, s, 1, 1)
        out[s] = {
            "exact_E[h]": exact_Eh, "mc_E[h]": float(np.mean(h_s)), "mc_se_E[h]": float(np.std(h_s) / np.sqrt(n_paths)),
            "exact_E[Y]": exact_EY, "mc_E[Y]": float(np.mean(y_s)), "mc_se_E[Y]": float(np.std(y_s) / np.sqrt(n_paths)),
            "exact_E[Y*h]": exact_EYh, "mc_E[Y*h]": float(np.mean(y_s * h_s)),
        }
    return out


def cumulative_return_check(p: jp.JPGARCHParams, dist: str, nu, m: int, j_periods_list, n_paths: int) -> dict:
    """Corollary 2(ii): cumulative-return moments via the augmented state Z=(R,h)."""
    rng = np.random.default_rng(cfg.RNG_SEED + 2)
    out = {}
    for j in j_periods_list:
        exact = jp.cumulative_return_moment(p, m, j, dist, nu)
        mc = jp.simulate_cumulative_return(p, j, n_paths, dist, rng, nu)
        out[j] = {"exact_E[R]": exact, "mc_E[R]": float(np.mean(mc["R"])),
                   "mc_se_E[R]": float(np.std(mc["R"]) / np.sqrt(n_paths))}
    return out


def remark2_check() -> dict:
    """Remark 2: HNGARCH-SV (b1=0) nests HNGARCH at rho=1 (Heston and Nandi 2000)."""
    alpha, gamma = 1.2e-6, 120.0
    p_sv = jp.hngarch_sv_params(tau=1.0, rho=0.6, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=alpha, gamma=gamma)
    p_h = jp.hngarch_params(tau=1.0, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=alpha, gamma=gamma)
    rng = np.random.default_rng(cfg.RNG_SEED + 3)
    x = 1.1e-4
    eps2 = rng.standard_normal(5000)
    h1 = jp.h_next(np.full(5000, x), eps2, p_h)
    # Paper Remark 2 (p. 13): h_{t+1} = omega + phi*h_t + alpha*[eps2^2 - 2*gamma*sqrt(h_t)*eps2 - 1].
    h2 = p_h.omega + p_h.phi * x + alpha * (eps2 ** 2 - 2 * gamma * np.sqrt(x) * eps2 - 1)
    max_abs_diff = float(np.max(np.abs(h1 - h2)))
    return {
        "alpha": alpha, "gamma": gamma, "rho_sv": 0.6,
        "max_abs_diff_recursion_vs_hn2000_form": max_abs_diff,
        "matches": bool(max_abs_diff < 1e-9),
        "hngarch_sv_positivity_all": jp.check_positivity(p_sv)["all"],
        "hngarch_positivity_all": jp.check_positivity(p_h)["all"],
    }


def conditional_corr_shape(p: jp.JPGARCHParams, m4: float) -> dict:
    """Expected outcome: |corr| increases up to x=b0/b1 then decreases (Sec. 2.2.2)."""
    x = np.linspace(1e-8, 6.0 * p.sigma2, 400)
    corr = jp.conditional_corr(x, p, m4)
    peak_x = float(x[np.argmax(np.abs(corr))])
    theoretical_peak = p.b0 / p.b1 if p.b1 > 0 else None
    return {"peak_x_numeric": peak_x, "theoretical_b0_over_b1": theoretical_peak, "corr_curve": {"x": x.tolist(), "abs_corr": np.abs(corr).tolist()}}


def run(dist: str, nu) -> dict:
    p = build_params()
    positivity = verify_positivity_grid(p)
    assert positivity["all"], "Proposition 4 restrictions must hold for the chosen parameters"

    joint = joint_moment_check(p, dist, nu, cfg.EXP2_MAX_DEGREE, cfg.EXP2_HORIZONS, cfg.MC_PATHS_VERIFICATION_RUN)
    cum_periods = (5, 10, cfg.EXP2_CUM_RETURN_J)
    cumret = cumulative_return_check(p, dist, nu, cfg.EXP2_MAX_DEGREE, cum_periods, cfg.MC_PATHS_VERIFICATION_RUN)

    m4 = 3.0 if dist == "gaussian" else 3.0 + 3.0 / nu
    corr_shape = conditional_corr_shape(p, m4)
    return {"dist": dist, "nu": nu, "params": p.__dict__, "positivity": positivity,
            "joint_moments": joint, "cumulative_return_moments": cumret, "conditional_corr_shape": corr_shape}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for dist, nu in (("gaussian", None), ("nig", cfg.EXP2_NIG_NU)):
        results[dist] = run(dist, nu)
    results["remark2_hngarch_nesting"] = remark2_check()

    out_path = RESULTS_DIR / "exp2_results.json"
    serializable = json.loads(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved {out_path}")

    curve = results["gaussian"]["conditional_corr_shape"]["corr_curve"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["x"], curve["abs_corr"])
    ax.axvline(results["gaussian"]["conditional_corr_shape"]["theoretical_b0_over_b1"], ls="--", color="gray", label="b0/b1")
    ax.set_xlabel("variance state x")
    ax.set_ylabel("|conditional return-variance correlation|")
    ax.set_title("JPGARCH-SV: state-dependent conditional correlation (Sec. 2.2.2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_conditional_corr.png", dpi=120)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'exp2_conditional_corr.png'}")


if __name__ == "__main__":
    main()
