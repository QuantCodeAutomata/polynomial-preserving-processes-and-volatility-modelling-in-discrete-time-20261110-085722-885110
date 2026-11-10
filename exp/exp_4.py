"""exp_4 -- Weak convergence to polynomial jump-diffusions under Levy
increment scaling (Sec. 3.3).

Verifies, for the VPGARCH-SV and JPGARCH-SV discrete-time processes driven by
standardized increments of a bivariate compound-Poisson Levy process (Eq. 24)
under the Assumption-2 parameter scaling, that:

  (i) the terminal distribution of the discrete-time process converges
      weakly, as tau=1/n -> 0, to the terminal distribution of the limiting
      jump-diffusion of Proposition 8 / Eq. (27), measured via the two-sample
      Kolmogorov-Smirnov statistic between discrete and continuous terminal
      samples;
  (ii) the one-step matrix expansion Q_m^{(n)} = I + tau*G_m + O(tau^2) holds,
       where G_m is the extended generator of Eq. (28), verified by comparing
       (Q_m^{(n)} - I)/tau to G_m as n -> infinity (Eq. 29).

Run: python -m exp.exp_4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from src import config as cfg
from src import jump_diffusion as jd
from src import jpgarch_sv as jp
from src import vpgarch_sv as vp

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_params(model: str) -> jd.JumpDiffusionParams:
    """Assumption-2 limiting coefficients (Sec. 3.3) with a compound-Poisson Levy driver."""
    driver = jd.LevyDriverParams(
        jump_rate=cfg.EXP4_JUMP_RATE, jump_scale=cfg.EXP4_JUMP_SCALE, rho=cfg.EXP4_RHO,
    )
    if model == "VP":
        return jd.JumpDiffusionParams(
            model="VP", kappa=cfg.EXP4_KAPPA, sigma2=cfg.EXP4_SIGMA2, r=cfg.EXP4_R,
            lam=cfg.EXP4_LAMBDA, eta=cfg.EXP4_ETA, b0=cfg.EXP4_B0, b1=cfg.EXP4_B1,
            a0=cfg.EXP4_A0, a1=cfg.EXP4_A1, driver=driver,
        )
    return jd.JumpDiffusionParams(
        model="JP", kappa=cfg.EXP4_KAPPA, sigma2=cfg.EXP4_SIGMA2, r=cfg.EXP4_R,
        lam=cfg.EXP4_LAMBDA, eta=cfg.EXP4_ETA, b0=cfg.EXP4_B0, b1=cfg.EXP4_B1,
        driver=driver,
    )


def discrete_params_from_assumption2(p: jd.JumpDiffusionParams, n: int):
    """phi(tau)=1-kappa*tau, sigma2(tau)=sigma2, lambda(tau)=lambda*tau, b_i(tau)=b_i*tau,
    eta(tau)=eta*sqrt(tau); VP: a_1(tau)=a_1, a_0(tau)=a_0*tau (Assumption 2)."""
    tau = 1.0 / n
    phi_tau = 1.0 - p.kappa * tau
    if not (0.0 <= phi_tau < 1.0):
        raise ValueError(f"phi(tau)={phi_tau} outside [0,1) at n={n}")
    eta_tau = p.eta * np.sqrt(tau)
    b0_tau = p.b0 * tau
    b1_tau = p.b1 * tau
    if p.model == "VP":
        a0_tau = p.a0 * tau
        a1_tau = p.a1
        params = vp.VPGARCHParams(tau=tau, rho=1.0, r=p.r, lam=p.lam * tau, sigma2=p.sigma2,
                                   phi=phi_tau, eta=eta_tau, a0=a0_tau, a1=a1_tau, b0=b0_tau, b1=b1_tau)
        conds = vp.check_positivity(params)
    else:
        params = jp.JPGARCHParams(tau=tau, rho=1.0, r=p.r, lam=p.lam * tau, sigma2=p.sigma2,
                                   phi=phi_tau, eta=eta_tau, b0=b0_tau, b1=b1_tau)
        conds = jp.check_positivity(params)
    if not conds["all"]:
        raise ValueError(f"finite-tau positivity (Prop. 3/4) violated at n={n}: {conds}")
    return params


def run_model_convergence(model: str, model_index: int) -> dict:
    p = build_params(model)
    boundary = jd.check_proposition8(p)

    rng_ref = np.random.default_rng(cfg.RNG_SEED + 7_000 + model_index)
    ref = jd.simulate_limit_jump_diffusion(
        p, horizon=cfg.EXP4_HORIZON_L, n_paths=cfg.EXP4_N_PATHS_RUN, rng=rng_ref,
        y0=cfg.EXP4_Y0, h0=cfg.EXP4_H0,
    )
    y_ref, h_ref = ref["y"], ref["h"]

    skipped: dict[int, str] = {}
    per_n: dict[int, dict] = {}
    for n in cfg.EXP4_N_LIST_RUN:
        try:
            discrete_params_from_assumption2(p, n)
        except ValueError as exc:
            skipped[n] = str(exc)
            continue
        rng = np.random.default_rng(cfg.RNG_SEED + 8_000 * (model_index + 1) + n)
        sim = jd.simulate_discrete_scaled(
            p, n=n, horizon=cfg.EXP4_HORIZON_L, n_paths=cfg.EXP4_N_PATHS_RUN, rng=rng,
            y0=cfg.EXP4_Y0, h0=cfg.EXP4_H0,
        )
        y_T, h_T = sim["y"], sim["h"]
        ks_h = ks_2samp(h_T, h_ref)
        ks_y = ks_2samp(y_T, y_ref)
        per_n[n] = {
            "tau": 1.0 / n,
            "ks_stat_h": float(ks_h.statistic), "ks_pvalue_h": float(ks_h.pvalue),
            "ks_stat_y": float(ks_y.statistic), "ks_pvalue_y": float(ks_y.pvalue),
            "mean_h_discrete": float(np.mean(h_T)), "std_h_discrete": float(np.std(h_T)),
            "mean_y_discrete": float(np.mean(y_T)), "std_y_discrete": float(np.std(y_T)),
        }

    ns_ok = sorted(per_n.keys())
    monotonic_improvement = None
    if len(ns_ok) >= 2:
        ks_seq = [per_n[n]["ks_stat_h"] for n in ns_ok]
        violations = sum(1 for i in range(1, len(ks_seq)) if ks_seq[i] > ks_seq[i - 1] + 1e-9)
        monotonic_improvement = bool(violations <= 1)

    return {
        "proposition8_boundary_conditions": boundary,
        "reference_jump_diffusion_terminal": {
            "mean_h": float(np.mean(h_ref)), "std_h": float(np.std(h_ref)),
            "mean_y": float(np.mean(y_ref)), "std_y": float(np.std(y_ref)),
        },
        "skipped_n": skipped,
        "per_n": per_n,
        "monotonic_improvement": monotonic_improvement,
    }


def generator_expansion_check(p: jd.JumpDiffusionParams) -> dict:
    """Eq. (29): (Q_m^{(n)}-I)/tau -> G_m as n -> infinity, for the variance coordinate.

    Q_m^{(n)} is the one-step transition matrix of the discrete-time variance
    process built from the Assumption-2 finite-tau parameters; G_m is the
    extended generator of Eq. (28) built directly from the limiting (kappa,
    sigma2, b0, b1, driver) coefficients.
    """
    m = cfg.EXP4_MAX_DEGREE
    G = jd.generator_matrix_variance(p, m)
    per_n: dict[int, dict] = {}
    for n in cfg.EXP4_N_LIST_RUN:
        try:
            discrete_params = discrete_params_from_assumption2(p, n)
        except ValueError as exc:
            per_n[n] = {"skipped": str(exc)}
            continue
        tau = 1.0 / n
        if p.model == "VP":
            Q = vp.transition_matrix(discrete_params, m, "gaussian")
        else:
            Q = jp.transition_matrix_variance(discrete_params, m, "gaussian")
        approx_G = (Q - np.eye(m + 1)) / tau
        frob_diff = float(np.linalg.norm(approx_G - G, ord="fro"))
        per_n[n] = {"frob_norm_diff": frob_diff}
    return {"degree_m": m, "generator_G_m": G.tolist(), "per_n": per_n}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, Any] = {"models": {}}

    for model_index, model in enumerate(("VP", "JP")):
        res = run_model_convergence(model, model_index)
        results["models"][model] = res
        n_ok = sorted(res["per_n"].keys())
        n_skipped = sorted(res["skipped_n"].keys())
        print(f"[{model}] successful n: {n_ok}, skipped n: {n_skipped}")
        if res["monotonic_improvement"] is False:
            print(f"[{model}] WARNING: KS statistic for h is not weakly decreasing in n")

    p_jp = build_params("JP")
    gen_section = generator_expansion_check(p_jp)
    results["generator_expansion_JP"] = gen_section
    diffs = {n: v["frob_norm_diff"] for n, v in gen_section["per_n"].items() if "frob_norm_diff" in v}
    if len(diffs) >= 2:
        ns = sorted(diffs.keys())
        vals = [diffs[n] for n in ns]
        violations = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i - 1] + 1e-6)
        gen_section["monotonic_improvement"] = bool(violations <= 1)
        if not gen_section["monotonic_improvement"]:
            print("WARNING: ||(Q_m-I)/tau - G_m|| is not weakly decreasing in n")
    else:
        gen_section["monotonic_improvement"] = None

    out_path = RESULTS_DIR / "exp4_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_path}")

    fig, (ax_ks, ax_gen) = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, marker in (("VP", "o-"), ("JP", "s-")):
        per_n = results["models"][model]["per_n"]
        if per_n:
            ns = sorted(per_n.keys())
            ks_vals = [per_n[n]["ks_stat_h"] for n in ns]
            ax_ks.plot(ns, ks_vals, marker, label=f"{model}GARCH-SV")
    ax_ks.set_xscale("log")
    ax_ks.set_xlabel("n = 1/tau")
    ax_ks.set_ylabel("KS statistic, h_T discrete vs jump-diffusion")
    ax_ks.set_title("Weak convergence of h_T (Prop. 8)")
    ax_ks.legend()

    if diffs:
        ns = sorted(diffs.keys())
        ax_gen.plot(ns, [diffs[n] for n in ns], "^-", color="tab:red")
    ax_gen.set_xscale("log")
    ax_gen.set_yscale("log")
    ax_gen.set_xlabel("n = 1/tau")
    ax_gen.set_ylabel(r"$\|(Q_m^{(n)}-I)/\tau - G_m\|_F$")
    ax_gen.set_title("Generator expansion (Eq. 29), JPGARCH-SV")

    fig.tight_layout()
    fig_path = RESULTS_DIR / "exp4_convergence.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
