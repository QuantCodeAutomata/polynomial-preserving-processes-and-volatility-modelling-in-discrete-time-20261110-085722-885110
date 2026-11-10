"""exp_5 -- Risk-neutral dynamics and polynomial preservation under Q (Appendix B).

Builds the variance-dependent exponential-affine pricing kernel of Eq. (B.1),
solves the no-arbitrage restrictions of Eq. (B.2) for the jointly Gaussian
innovation law (Appendix B.1), derives the risk-neutral tilting formulas of
Proposition 9, and verifies under which parameter restrictions polynomial
preservation of the VPGARCH-SV / JPGARCH-SV dynamics survives the change of
measure from P to Q (Proposition 10).

Run: python -m exp.exp_5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import innovations as innov
from src import jpgarch_sv as jp
from src import risk_neutral as rn
from src import vpgarch_sv as vp

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_vp_params() -> vp.VPGARCHParams:
    """VPGARCH-SV parameters for the preservation study (Sec. 4.3 calibration style)."""
    return vp.VPGARCHParams(
        tau=cfg.EXP5_TAU, rho=cfg.EXP5_RHO, r=cfg.EXP5_R, lam=cfg.EXP5_LAMBDA,
        sigma2=cfg.EXP5_SIGMA2, phi=cfg.EXP5_PHI, eta=cfg.EXP5_ETA,
        a0=cfg.EXP5_A0, a1=cfg.EXP5_A1, b0=cfg.EXP5_B0, b1=cfg.EXP5_B1,
    )


def build_jp_params() -> jp.JPGARCHParams:
    """JPGARCH-SV parameters for the preservation study (Sec. 4.3 calibration style)."""
    return jp.JPGARCHParams(
        tau=cfg.EXP5_TAU, rho=cfg.EXP5_RHO, r=cfg.EXP5_R, lam=cfg.EXP5_LAMBDA,
        sigma2=cfg.EXP5_SIGMA2, phi=cfg.EXP5_PHI, eta=cfg.EXP5_ETA,
        b0=cfg.EXP5_B0, b1=cfg.EXP5_B1,
    )


def base_case_kernel_and_tilts(rng: np.random.Generator) -> dict:
    """Base case xi2=0 (Sec. 4.3 empirical kernel): solve xi1, xi0, verify Eq. (B.2), report Prop. 9 tilts."""
    kernel = rn.PricingKernelGaussian(tau=cfg.EXP5_TAU, r=cfg.EXP5_R, rho=cfg.EXP5_RHO, xi2=0.0)
    h0 = cfg.EXP5_H0

    xi1 = rn.solve_xi1(kernel, h=h0, lam=cfg.EXP5_LAMBDA)
    xi0 = rn.xi0_from_bond_restriction(kernel, h=h0, xi1=xi1)

    mc = rn.verify_martingale_restrictions(
        kernel, h=h0, xi0=xi0, xi1=xi1, lam=cfg.EXP5_LAMBDA,
        n_paths=cfg.MC_PATHS_VERIFICATION_RUN, rng=rng,
    )

    # Recompute the raw simulated arrays (same construction as verify_martingale_restrictions)
    # to obtain the Monte Carlo standard errors of the two sample means.
    eps1, _ = innov.sample_gaussian_pair(kernel.rho, cfg.MC_PATHS_VERIFICATION_RUN, rng)
    y_next = kernel.r * kernel.tau + cfg.EXP5_LAMBDA * h0 + np.sqrt(kernel.tau * h0) * eps1
    N = np.exp(xi0 + xi1 * y_next)
    N_exp_y = N * np.exp(y_next)

    n = cfg.MC_PATHS_VERIFICATION_RUN
    se_N = float(np.std(N, ddof=1) / np.sqrt(n))
    se_N_exp_y = float(np.std(N_exp_y, ddof=1) / np.sqrt(n))

    bond_ok = bool(abs(mc["E_N"] - mc["target_bond"]) < 3.0 * se_N)
    stock_ok = bool(abs(mc["E_N_exp_y"] - mc["target_stock"]) < 3.0 * se_N_exp_y)
    assert bond_ok, (
        f"bond-pricing martingale restriction failed: E_N={mc['E_N']} target={mc['target_bond']} "
        f"3*SE={3 * se_N}"
    )
    assert stock_ok, (
        f"stock-pricing martingale restriction failed: E_N_exp_y={mc['E_N_exp_y']} "
        f"target={mc['target_stock']} 3*SE={3 * se_N_exp_y}"
    )

    tilts = rn.risk_neutral_tilts(kernel, h=h0, xi1=xi1)

    return {
        "kernel": {"tau": kernel.tau, "r": kernel.r, "rho": kernel.rho, "xi2": kernel.xi2},
        "h0": h0,
        "xi1": xi1,
        "xi0": xi0,
        "martingale_check": {
            **mc,
            "se_E_N": se_N,
            "se_E_N_exp_y": se_N_exp_y,
            "bond_restriction_ok": bond_ok,
            "stock_restriction_ok": stock_ok,
        },
        "tilts": tilts,
        "h_star_from_tilts": tilts["R1"] * h0,
    }


def proposition10_preservation(rho: float) -> dict:
    """Proposition 10 preservation study with a state-independent B = xi2_const.

    Choosing xi2(x;tau) = xi2_const / psi_3(x;tau) makes B = xi2(x)*psi_3(x) = xi2_const
    identically a constant, so R1(tau), R2(tau) are constant even though xi2 itself is
    state-dependent. We then solve analytically for the lambda(tau) value that makes
    Proposition 10's "lambda(tau) = -tau*R1(tau)/2" condition hold exactly, and feed the
    (by-construction ~0) residual to the VP/JP preservation checks.
    """
    B = cfg.EXP5_XI2
    # Domain restriction of the Gaussian cgf (Eq. B.1): B must be < 1/2. (The config value
    # EXP5_XI2 = -5.0 easily satisfies this; it is not required to be small in magnitude.)
    assert B < 0.5, "B must be < 1/2 for the Gaussian cgf domain (Eq. B.1)"

    gen = rn.risk_neutral_tilts_general(rho=rho, A1=0.0, A2=0.0, B=B)
    r1_general, r2_general = gen["R1"], gen["R2"]

    lam_preserving = -cfg.EXP5_TAU * r1_general / 2.0
    lam_tau_r1 = lam_preserving + cfg.EXP5_TAU * r1_general / 2.0
    assert abs(lam_tau_r1) < 1e-10, "lambda(tau)=-tau*R1/2 residual must vanish by construction"

    return {
        "B": B,
        "R1_general": r1_general,
        "R2_general": r2_general,
        "lam_preserving": lam_preserving,
        "lam_tau_r1_residual": lam_tau_r1,
    }


def vp_preservation_checks(prop10: dict) -> dict:
    """Proposition 10(i): config (b0=0, preserved) vs perturbed (b0>0, not preserved) VPGARCH-SV."""
    lam_tau_r1 = prop10["lam_tau_r1_residual"]
    config_case = rn.check_preservation_vpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=cfg.EXP5_B0, b1=cfg.EXP5_B1,
        a0=cfg.EXP5_A0, a1=cfg.EXP5_A1, lam_tau_r1=lam_tau_r1,
    )
    assert config_case["preserved"], "config case (b0=0) must preserve polynomiality under Q"

    b0_perturbed = cfg.EXP5_B0 + 0.05 * cfg.EXP5_B1
    perturbed_case = rn.check_preservation_vpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=b0_perturbed, b1=cfg.EXP5_B1,
        a0=cfg.EXP5_A0, a1=cfg.EXP5_A1, lam_tau_r1=lam_tau_r1,
    )
    assert not perturbed_case["preserved"], "perturbed case (b0>0) must break polynomial preservation"

    return {"config_case": config_case, "perturbed_case": perturbed_case, "b0_perturbed": b0_perturbed}


def jp_preservation_checks(prop10: dict) -> dict:
    """Proposition 10(ii): config (b0=0, preserved) vs perturbed (b0>0, not preserved) JPGARCH-SV."""
    lam_tau_r1 = prop10["lam_tau_r1_residual"]
    B = prop10["B"]

    config_case = rn.check_preservation_jpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=cfg.EXP5_B0, b1=cfg.EXP5_B1,
        xi2=B, lam_tau_r1=lam_tau_r1,
    )
    assert config_case["preserved"], "config case (b0=0) must preserve polynomiality under Q (cond2 has factor b0)"

    b0_perturbed = cfg.EXP5_B0 + 0.05 * cfg.EXP5_B1
    perturbed_case = rn.check_preservation_jpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=b0_perturbed, b1=cfg.EXP5_B1,
        xi2=B, lam_tau_r1=lam_tau_r1,
    )
    assert not perturbed_case["preserved"], "perturbed case (b0>0, b1!=0) must break polynomial preservation"

    return {"config_case": config_case, "perturbed_case": perturbed_case, "b0_perturbed": b0_perturbed}


def hngarch_sv_special_case(prop10: dict) -> dict:
    """Remark 2 (b1=0): HNGARCH-SV retains preservation without the extra restriction on lambda.

    With b1=0, cond1 = b1*lam_tau_r1 = 0 identically, regardless of lam_tau_r1, so preservation
    under Q depends only on cond2 (which vanishes whenever b0=0 or eta=0 or rho^2=1).
    """
    arbitrary_lam_tau_r1 = 1.23  # deliberately nonzero, to show cond1 vanishes regardless
    case_b0_zero = rn.check_preservation_jpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=cfg.EXP5_B0, b1=0.0,
        xi2=prop10["B"], lam_tau_r1=arbitrary_lam_tau_r1,
    )
    assert case_b0_zero["cond1_residual"] == 0.0, "cond1 = b1*lam_tau_r1 must vanish identically when b1=0"

    b0_nonzero = cfg.EXP5_B0 + 0.05 * cfg.EXP5_B1 if cfg.EXP5_B1 != 0 else 0.01
    case_b0_nonzero = rn.check_preservation_jpgarch_q(
        rho=cfg.EXP5_RHO, eta=cfg.EXP5_ETA, b0=b0_nonzero, b1=0.0,
        xi2=prop10["B"], lam_tau_r1=arbitrary_lam_tau_r1,
    )
    assert case_b0_nonzero["cond1_residual"] == 0.0, "cond1 = b1*lam_tau_r1 must vanish identically when b1=0"

    return {
        "arbitrary_lam_tau_r1": arbitrary_lam_tau_r1,
        "b0_zero_case": case_b0_zero,
        "b0_nonzero_case": case_b0_nonzero,
        "b0_nonzero_value": b0_nonzero,
    }


def plot_preservation(vp_res: dict, jp_res: dict) -> Path:
    labels = ["VP: config\n(b0=0)", "VP: perturbed\n(b0>0)", "JP: config\n(b0=0)", "JP: perturbed\n(b0>0)"]
    values = [
        vp_res["config_case"]["divisibility_residual"],
        vp_res["perturbed_case"]["divisibility_residual"],
        jp_res["config_case"]["cond2_residual"],
        jp_res["perturbed_case"]["cond2_residual"],
    ]
    colors = ["tab:green", "tab:red", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="residual = 0 (preserved)")
    ax.set_ylabel("Preservation residual (divisibility / cond2)")
    ax.set_title("Proposition 10: polynomial preservation under Q\n(config b0=0 preserves, perturbed b0>0 breaks it)")
    ax.legend()
    fig.tight_layout()

    fig_path = RESULTS_DIR / "exp5_preservation.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    return fig_path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.RNG_SEED)

    vp_params = build_vp_params()
    jp_params = build_jp_params()
    vp_positivity = vp.check_positivity(vp_params)
    jp_positivity = jp.check_positivity(jp_params)
    # The EXP5_* calibration (eta, a0, a1, b0, b1) is chosen (grid search, not specified in
    # the paper) so that Proposition 3 (VPGARCH-SV) and Proposition 4 (JPGARCH-SV) positivity
    # restrictions both hold at this (sigma2, phi) pair; verify this explicitly.
    assert vp_positivity["all"], f"VPGARCH-SV positivity restriction violated: {vp_positivity}"
    assert jp_positivity["all"], f"JPGARCH-SV positivity restriction violated: {jp_positivity}"

    base_case = base_case_kernel_and_tilts(rng)
    prop10 = proposition10_preservation(rho=cfg.EXP5_RHO)
    vp_checks = vp_preservation_checks(prop10)
    jp_checks = jp_preservation_checks(prop10)
    hngarch_sv_check = hngarch_sv_special_case(prop10)

    results: dict[str, Any] = {
        "positivity": {"vp": vp_positivity, "jp": jp_positivity},
        "base_case": base_case,
        "proposition10_general_tilts": prop10,
        "vp_preservation": vp_checks,
        "jp_preservation": jp_checks,
        "hngarch_sv_special_case": hngarch_sv_check,
    }

    out_path = RESULTS_DIR / "exp5_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_path}")

    print(f"Base-case martingale check: bond_ok={base_case['martingale_check']['bond_restriction_ok']}, "
          f"stock_ok={base_case['martingale_check']['stock_restriction_ok']}")
    print(f"h_star = R1*h0 = {base_case['h_star_from_tilts']:.6e} (h0={base_case['h0']:.6e})")
    print(f"VP preserved: config={vp_checks['config_case']['preserved']}, "
          f"perturbed={vp_checks['perturbed_case']['preserved']}")
    print(f"JP preserved: config={jp_checks['config_case']['preserved']}, "
          f"perturbed={jp_checks['perturbed_case']['preserved']}")

    fig_path = plot_preservation(vp_checks, jp_checks)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
