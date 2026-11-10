"""exp_3 — Weak convergence to polynomial diffusions under P (Sec. 3.1-3.2).

Verifies, for the VPGARCH-SV and JPGARCH-SV discrete-time processes under
the Assumption-1 parameter scaling (Sec. 3.1), that:

  (i) the terminal distribution of the discrete-time process converges
      weakly, as tau=1/n -> 0, to the terminal distribution of the limiting
      polynomial diffusion of Proposition 7 (Sec. 3.2), measured via the
      two-sample Kolmogorov-Smirnov statistic between discrete and
      continuous terminal samples;
  (ii) the discrete one-step moment-matrix power Q_m^{n*L} converges to the
       continuous matrix exponential exp(L*G_m) of Corollary 3, where G_m is
       the polynomial generator of Proposition 6.

Run: python -m exp.exp_3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import ks_2samp

from src import config as cfg
from src import diffusion
from src import innovations as innov
from src import jpgarch_sv as jp
from src import vpgarch_sv as vp

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_limit_params(model: str) -> diffusion.DiffusionLimitParams:
    """Build the Assumption-1 limiting-diffusion coefficients (Sec. 3.1) for one model.

    b0=EXP3_B0=0.0 per the boundary condition "b0=0 forced by Proposition 7".
    """
    if model == "VP":
        return diffusion.DiffusionLimitParams(
            model="VP", kappa=cfg.EXP3_KAPPA, sigma2=cfg.EXP3_SIGMA2, r=cfg.EXP3_R,
            lam=cfg.EXP3_LAMBDA, eta=cfg.EXP3_ETA, b0=cfg.EXP3_B0, b1=cfg.EXP3_B1,
            a0=cfg.EXP3_A0, a1=cfg.EXP3_A1, rho=cfg.EXP3_RHO,
        )
    if model == "JP":
        return diffusion.DiffusionLimitParams(
            model="JP", kappa=cfg.EXP3_KAPPA, sigma2=cfg.EXP3_SIGMA2, r=cfg.EXP3_R,
            lam=cfg.EXP3_LAMBDA, eta=cfg.EXP3_ETA, b0=cfg.EXP3_B0, b1=cfg.EXP3_B1,
            a0=0.0, a1=0.0, rho=cfg.EXP3_RHO,
        )
    raise ValueError(f"unknown model {model!r}")


def simulate_diffusion_reference(p: diffusion.DiffusionLimitParams, seed: int) -> dict:
    """Terminal sample of the limiting diffusion (Proposition 7), the convergence target."""
    rng = np.random.default_rng(seed)
    return diffusion.simulate_diffusion_euler(
        p, y0=cfg.EXP3_Y0, h0=cfg.EXP3_H0, horizon=cfg.EXP3_HORIZON_L,
        n_grid=cfg.EXP3_N_GRID_FINE, n_paths=cfg.EXP3_N_PATHS_RUN, dist="gaussian", rng=rng,
    )


def run_model_convergence(model: str, model_index: int) -> dict:
    """Weak-convergence check (Sec. 3.2) for one model: discrete terminal sample vs diffusion reference."""
    p = build_limit_params(model)
    boundary = diffusion.check_assumption1_boundary(p)

    ref = simulate_diffusion_reference(p, seed=cfg.RNG_SEED + model_index)
    y_ref, h_ref = ref["y"], ref["h"]

    sim_fn = vp.simulate_paths if model == "VP" else jp.simulate_paths

    skipped: dict[int, str] = {}
    per_n: dict[int, dict] = {}
    discrete_params_by_n: dict[int, Any] = {}
    for n in cfg.EXP3_N_LIST_RUN:
        try:
            discrete_p = diffusion.discrete_params_from_assumption1(p, n)
        except ValueError as exc:
            skipped[n] = str(exc)
            continue

        n_steps = round(cfg.EXP3_HORIZON_L * n)
        rng = np.random.default_rng(cfg.RNG_SEED + 1000 * (model_index + 1) + n)
        try:
            sim = sim_fn(discrete_p, n_steps, cfg.EXP3_N_PATHS_RUN, "gaussian", rng)
        except ValueError as exc:
            skipped[n] = f"simulation failure: {exc}"
            continue

        y_T, h_T = sim["y"][:, -1], sim["h"][:, -1]
        ks_h = ks_2samp(h_T, h_ref)
        ks_y = ks_2samp(y_T, y_ref)
        per_n[n] = {
            "tau": 1.0 / n,
            "ks_stat_h": float(ks_h.statistic), "ks_pvalue_h": float(ks_h.pvalue),
            "ks_stat_y": float(ks_y.statistic), "ks_pvalue_y": float(ks_y.pvalue),
            "mean_h_discrete": float(np.mean(h_T)), "std_h_discrete": float(np.std(h_T)),
            "mean_y_discrete": float(np.mean(y_T)), "std_y_discrete": float(np.std(y_T)),
        }
        discrete_params_by_n[n] = discrete_p

    ns_ok = sorted(per_n.keys())
    monotonic_improvement = None
    if len(ns_ok) >= 2:
        ks_seq = [per_n[n]["ks_stat_h"] for n in ns_ok]
        # Allow one non-monotonic step given Monte Carlo sampling noise.
        violations = sum(1 for i in range(1, len(ks_seq)) if ks_seq[i] > ks_seq[i - 1] + 1e-9)
        monotonic_improvement = bool(violations <= 1)

    return {
        "boundary_conditions": boundary,
        "reference_diffusion_terminal": {
            "mean_h": float(np.mean(h_ref)), "std_h": float(np.std(h_ref)),
            "mean_y": float(np.mean(y_ref)), "std_y": float(np.std(y_ref)),
        },
        "skipped_n": skipped,
        "per_n": per_n,
        "monotonic_improvement": monotonic_improvement,
        "discrete_params_by_n": discrete_params_by_n,
    }


def matrix_convergence_corollary3(p_vp: diffusion.DiffusionLimitParams, discrete_params_by_n: dict) -> dict:
    """Corollary 3: Q_m^{n*L} -> exp(L*G_m) as n -> infinity, for the VP variance coordinate."""
    m = cfg.EXP3_MAX_DEGREE
    m4 = innov.fourth_moment("gaussian")
    G = diffusion.generator_matrix_variance(p_vp, m, m4)
    L = cfg.EXP3_HORIZON_L
    cont_exp = expm(L * G)

    x0 = cfg.EXP3_H0
    H0_vec = np.array([x0**i for i in range(m + 1)])
    e1 = np.zeros(m + 1)
    e1[1] = 1.0
    moment_continuous = float(H0_vec @ cont_exp @ e1)
    moment_generator = diffusion.moment_via_generator(G, x0, L, 1)

    per_n: dict[int, dict] = {}
    for n, discrete_p in sorted(discrete_params_by_n.items()):
        Q = vp.transition_matrix(discrete_p, m, "gaussian")
        s = round(n * L)
        Qs = np.linalg.matrix_power(Q, s)
        frob_norm_diff = float(np.linalg.norm(Qs - cont_exp, ord="fro"))
        moment_discrete = float(H0_vec @ Qs @ e1)
        per_n[n] = {
            "s_steps": s,
            "frob_norm_diff": frob_norm_diff,
            "moment_discrete_Qs": moment_discrete,
            "moment_continuous_expm": moment_continuous,
            "moment_via_generator": moment_generator,
        }

    return {
        "degree_m": m,
        "moment_continuous_expm_degree1": moment_continuous,
        "moment_via_generator_degree1": moment_generator,
        "per_n": per_n,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    results: dict[str, Any] = {"models": {}}
    discrete_params_vp: dict[int, Any] = {}
    for model_index, model in enumerate(("VP", "JP")):
        res = run_model_convergence(model, model_index)
        discrete_params = res.pop("discrete_params_by_n")
        if model == "VP":
            discrete_params_vp = discrete_params
        results["models"][model] = res

        n_ok = sorted(res["per_n"].keys())
        n_skipped = sorted(res["skipped_n"].keys())
        print(f"[{model}] successful n: {n_ok}, skipped n: {n_skipped}")
        if res["monotonic_improvement"] is False:
            print(f"[{model}] WARNING: KS statistic for h is not weakly decreasing in n")
        elif res["monotonic_improvement"] is None:
            print(f"[{model}] WARNING: fewer than 2 successful n values, cannot verify monotonic improvement")

    p_vp = build_limit_params("VP")
    matrix_section = matrix_convergence_corollary3(p_vp, discrete_params_vp)
    results["matrix_convergence_corollary3"] = matrix_section
    n_ok_mat = sorted(matrix_section["per_n"].keys())
    if len(n_ok_mat) >= 2:
        diffs = [matrix_section["per_n"][n]["frob_norm_diff"] for n in n_ok_mat]
        mat_violations = sum(1 for i in range(1, len(diffs)) if diffs[i] > diffs[i - 1] + 1e-9)
        matrix_section["monotonic_improvement"] = bool(mat_violations <= 1)
        if not matrix_section["monotonic_improvement"]:
            print("WARNING: ||Q_m^{nL} - expm(L*G_m)|| is not weakly decreasing in n")
    else:
        matrix_section["monotonic_improvement"] = None
        print("WARNING: fewer than 2 successful n values, cannot verify matrix-norm convergence")

    out_path = RESULTS_DIR / "exp3_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_path}")

    fig, (ax_ks, ax_mat) = plt.subplots(1, 2, figsize=(11, 4.5))

    for model, marker in (("VP", "o-"), ("JP", "s-")):
        per_n = results["models"][model]["per_n"]
        if per_n:
            ns = sorted(per_n.keys())
            ks_vals = [per_n[n]["ks_stat_h"] for n in ns]
            ax_ks.plot(ns, ks_vals, marker, label=f"{model}GARCH-SV")
    ax_ks.set_xscale("log")
    ax_ks.set_xlabel("n = 1/tau (periods per year)")
    ax_ks.set_ylabel("KS statistic, h_T discrete vs diffusion")
    ax_ks.set_title("Weak convergence of h_T (Prop. 7)")
    ax_ks.legend()

    if matrix_section["per_n"]:
        ns = sorted(matrix_section["per_n"].keys())
        diffs = [matrix_section["per_n"][n]["frob_norm_diff"] for n in ns]
        ax_mat.plot(ns, diffs, "^-", color="tab:red", label=r"$\|Q_m^{nL} - \exp(L G_m)\|_F$ (VP)")
    ax_mat.set_xscale("log")
    ax_mat.set_yscale("log")
    ax_mat.set_xlabel("n = 1/tau")
    ax_mat.set_ylabel("Frobenius norm difference")
    ax_mat.set_title("Moment-matrix convergence (Corollary 3)")
    ax_mat.legend()

    fig.tight_layout()
    fig_path = RESULTS_DIR / "exp3_convergence.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
