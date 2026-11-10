"""Unit tests for src/ model components, run on synthetic/simulated inputs
(no market data). Checks signs, positivity, moment-formula correctness
against Monte Carlo, and absence of look-ahead in the variance filter."""

import numpy as np
import pytest

from src import config as cfg
from src import innovations as innov
from src.vpgarch_sv import (
    VPGARCHParams, check_positivity, transition_matrix, multi_step_moment,
    simulate_paths, stability_condition, stationary_moments, ngarch_sv_params,
)
from src.jpgarch_sv import (
    JPGARCHParams, check_positivity as check_positivity_jp,
    transition_matrix_joint, joint_moment, simulate_paths as simulate_paths_jp,
    hngarch_sv_params, cumulative_return_moment,
)
from src.diffusion import DiffusionLimitParams, generator_matrix_variance, discrete_to_generator_matrix
from src.risk_neutral import cgf_gaussian, solve_xi1, PricingKernelGaussian, risk_neutral_tilts, verify_martingale_restrictions
from src.estimation import variance_term_structure, half_life, d_coeffs, vovnic


def _vp_params():
    return VPGARCHParams(tau=cfg.EXP1_TAU, rho=cfg.EXP1_RHO, r=cfg.EXP1_R, lam=cfg.EXP1_LAMBDA,
                          sigma2=cfg.EXP1_SIGMA2, phi=cfg.EXP1_PHI, eta=cfg.EXP1_ETA,
                          a0=cfg.EXP1_A0, a1=cfg.EXP1_A1, b0=cfg.EXP1_B0, b1=cfg.EXP1_B1)


def _jp_params():
    return JPGARCHParams(tau=cfg.EXP2_TAU, rho=cfg.EXP2_RHO, r=cfg.EXP2_R, lam=cfg.EXP2_LAMBDA,
                          sigma2=cfg.EXP2_SIGMA2, phi=cfg.EXP2_PHI, eta=cfg.EXP2_ETA,
                          b0=cfg.EXP2_B0, b1=cfg.EXP2_B1)


def test_vpgarch_positivity_holds_for_config_params():
    p = _vp_params()
    conds = check_positivity(p)
    assert conds["all"], conds


def test_jpgarch_positivity_holds_for_config_params():
    p = _jp_params()
    conds = check_positivity_jp(p)
    assert conds["all"], conds


def test_vpgarch_variance_never_negative_on_simulated_paths():
    p = _vp_params()
    rng = np.random.default_rng(0)
    out = simulate_paths(p, n_steps=200, n_paths=2000, dist="gaussian", rng=rng)
    assert np.all(out["h"] >= 0.0)


def test_jpgarch_variance_never_negative_on_simulated_paths():
    p = _jp_params()
    rng = np.random.default_rng(0)
    out = simulate_paths_jp(p, n_steps=200, n_paths=2000, dist="gaussian", rng=rng)
    assert np.all(out["h"] >= 0.0)


def test_vpgarch_one_step_mean_formula_matches_monte_carlo():
    p = _vp_params()
    Q = transition_matrix(p, m=2, dist="gaussian")
    x0 = p.sigma2
    mean_formula = multi_step_moment(Q, x0, s=1, degree=1)
    rng = np.random.default_rng(1)
    eps2 = innov.sample_gaussian_pair(p.rho, 200_000, rng)[1]
    from src.vpgarch_sv import h_next
    mean_mc = np.mean(h_next(x0, eps2, p))
    assert mean_formula == pytest.approx(mean_mc, rel=0.02)


def test_vpgarch_multistep_moment_matches_monte_carlo():
    p = _vp_params()
    Q = transition_matrix(p, m=2, dist="gaussian")
    x0 = p.sigma2
    s = 10
    formula = multi_step_moment(Q, x0, s=s, degree=1)
    rng = np.random.default_rng(2)
    n = 40_000
    h = np.full(n, x0)
    for _ in range(s):
        eps2 = innov.sample_gaussian_pair(p.rho, n, rng)[1]
        from src.vpgarch_sv import h_next
        h = h_next(h, eps2, p)
    mc = np.mean(h)
    assert formula == pytest.approx(mc, rel=0.05)


def test_vpgarch_stability_condition_degree_one_equals_phi():
    """E[A_VP^1] = phi + eta*E[eps2] + b1*(E[eps2^2]-1) = phi for symmetric mean-zero eps2."""
    p = _vp_params()
    out = stability_condition(p, max_degree=1, dist="gaussian")
    assert out[1] == pytest.approx(p.phi, abs=1e-6)


def test_vpgarch_stationary_moments_solve_fixed_point():
    p = _vp_params()
    m = 2
    mu = stationary_moments(p, m, dist="gaussian")
    Q = transition_matrix(p, m, dist="gaussian")
    # stationary moments are a fixed point of E[h^i] = H_m(mu)^T Q p_{h^i}... check via Q^T mu = mu
    assert mu[0] == 1.0
    lhs = Q.T @ mu
    assert lhs[1:] == pytest.approx(mu[1:], rel=1e-6)


def test_ngarch_sv_nesting_remark1():
    p = ngarch_sv_params(tau=1.0, rho=1.0, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=0.05, gamma=0.5)
    assert p.a0 == 0.0 and p.a1 == 0.0 and p.b0 == 0.0
    assert p.eta == pytest.approx(-2.0 * 0.05 * 0.5)


def test_jpgarch_joint_moment_mean_return_matches_monte_carlo():
    p = _jp_params()
    Q = transition_matrix_joint(p, m=2, dist="gaussian")
    mean_formula = joint_moment(Q, 2, 0.0, p.sigma2, s=1, p_order=1, q_order=0)
    rng = np.random.default_rng(3)
    n = 200_000
    eps1, _ = innov.sample_gaussian_pair(p.rho, n, rng)
    y_next = p.r * p.tau + p.lam * p.sigma2 + np.sqrt(p.tau * p.sigma2) * eps1
    assert mean_formula == pytest.approx(np.mean(y_next), abs=5e-4)


def test_jpgarch_cross_date_cumulative_mean_matches_monte_carlo():
    p = _jp_params()
    j = 5
    formula = cumulative_return_moment(p, m=2, j_periods=j, dist="gaussian")
    rng = np.random.default_rng(4)
    from src.jpgarch_sv import simulate_cumulative_return
    out = simulate_cumulative_return(p, j_periods=j, n_paths=100_000, dist="gaussian", rng=rng)
    assert formula == pytest.approx(np.mean(out["R"]), rel=0.1, abs=5e-4)


def test_hngarch_sv_nesting_remark2():
    p = hngarch_sv_params(tau=1.0, rho=1.0, r=0.0, lam=0.02, sigma2=1e-4, phi=0.9, alpha=0.05, gamma=0.5)
    assert p.b1 == 0.0
    assert p.eta == pytest.approx(-2.0 * 0.05 * 0.5)


def test_generator_matrix_matches_discrete_difference_quotient_small_tau():
    """Corollary 3: (Q_m - I)/tau -> G_m as tau -> 0."""
    dp = DiffusionLimitParams(model="JP", kappa=5.0, sigma2=0.04, r=0.02, lam=2.0, eta=0.5, b0=0.0, b1=0.0)
    m4 = innov.gaussian_fourth_moment()
    G_exact = generator_matrix_variance(dp, m=2, m4=m4)

    tau = 1e-4
    n = round(1.0 / tau)
    phi_tau = 1.0 - dp.kappa * tau
    eta_tau = dp.eta * np.sqrt(tau)
    even_m = innov.even_moments_eps2(4, "gaussian", dp.rho)
    from src.poly_basis import build_Q_variance
    omega_tau = (1 - phi_tau) * dp.sigma2
    Q = build_Q_variance(2, phi_tau, omega_tau, eta_tau, [0.0, 1.0], 0.0, 0.0, even_m)
    G_approx = discrete_to_generator_matrix(Q, tau)
    assert G_approx == pytest.approx(G_exact, abs=0.05)


def test_risk_neutral_cgf_gaussian_matches_definition():
    """C^P(0,0,0) = 0 (cgf at the origin)."""
    assert cgf_gaussian(0.0, 0.0, 0.0, rho=0.5) == pytest.approx(0.0, abs=1e-10)


def test_risk_neutral_martingale_restrictions_hold_at_solution():
    kernel = PricingKernelGaussian(tau=1.0, r=0.0, rho=0.6)
    h = 1e-4
    lam = 0.02
    xi1 = solve_xi1(kernel, h, lam)
    from src.risk_neutral import xi0_from_bond_restriction
    xi0 = xi0_from_bond_restriction(kernel, h, xi1)
    rng = np.random.default_rng(5)
    out = verify_martingale_restrictions(kernel, h, xi0, xi1, lam, n_paths=500_000, rng=rng)
    assert out["E_N"] == pytest.approx(out["target_bond"], rel=0.02)
    assert out["E_N_exp_y"] == pytest.approx(out["target_stock"], rel=0.02)


def test_variance_term_structure_converges_to_long_run_level():
    ts = variance_term_structure(phi=0.95, sigma2=1e-4, h0=2e-4, s_max=500)
    assert ts[-1] == pytest.approx(1e-4, abs=1e-8)
    assert ts[0] == 2e-4


def test_half_life_positive_and_consistent_with_term_structure():
    phi = 0.95
    hl = half_life(phi)
    assert hl > 0
    ts = variance_term_structure(phi, sigma2=0.0, h0=1.0, s_max=int(round(hl)))
    assert ts[-1] == pytest.approx(0.5, rel=0.05)


def test_d_coeffs_nonnegative_for_jpgarch():
    p = {"eta": -0.01, "b0": 1e-6, "b1": 0.8}
    d0, d1, d2 = d_coeffs("JPGARCH", p, m4=3.0)
    assert d0 >= 0 and d2 >= 0


def test_vovnic_nonnegative():
    p = {"phi": 0.95, "eta": -0.01, "b0": 1e-6, "b1": 0.8, "sigma2": 1e-4, "a0": 0.0, "a1": 0.0}
    out = vovnic("JPGARCH", p, m4=3.0, eps=np.array([-4.0, 0.0, 4.0]), s_max=10)
    assert np.all(out[2:] >= -1e-12)


def test_innovations_gaussian_moments_known_values():
    assert innov.moment_gaussian(0, 0, 0.3) == pytest.approx(1.0, abs=1e-6)
    assert innov.moment_gaussian(2, 0, 0.3) == pytest.approx(1.0, abs=1e-4)
    assert innov.moment_gaussian(1, 1, 0.3) == pytest.approx(0.3, abs=1e-4)
    assert innov.moment_gaussian(1, 0, 0.3) == pytest.approx(0.0, abs=1e-8)


def test_nig_fourth_moment_matches_analytic_quadrature():
    """m_4 = 3 + 3*nu (closed form) must match the quadrature-based moment used elsewhere in src."""
    nu = 0.45
    m4_formula = innov.nig_fourth_moment(nu)
    m4_quad = innov.moment_nig(0, 4, 0.0, nu)
    assert m4_formula == pytest.approx(m4_quad, rel=1e-6)
    assert m4_formula > 3.0  # heavier-tailed than Gaussian (Table 1 kurtosis 4.17-4.47)


def test_nig_simulation_mean_and_variance_match_standardization():
    """Sample mean/variance are low-order moments with bounded sampling error, unlike the
    heavy-tailed 4th moment; they verify the mixture sampler is correctly standardized."""
    nu = 0.45
    rng = np.random.default_rng(6)
    eps1, _ = innov.sample_nig_pair(0.0, nu, 500_000, rng)
    assert np.mean(eps1) == pytest.approx(0.0, abs=0.02)
    assert np.var(eps1) == pytest.approx(1.0, rel=0.05)
