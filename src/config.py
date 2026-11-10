"""Module-level experiment parameters.

Every value carries a one-line comment stating its meaning, its unit, and its
source (a paper section/equation, or "not specified in the paper" when the
paper leaves the value open and a value had to be chosen to run the code).
"""

import numpy as np

# ---------------------------------------------------------------------------
# General / shared
# ---------------------------------------------------------------------------
RNG_SEED = 20240101  # base random seed, not specified in the paper
MC_PATHS_VERIFICATION = 100_000  # Monte Carlo paths for moment-formula checks, paper Sec. 2.3 (>=100,000 required)
MC_PATHS_VERIFICATION_RUN = 20_000  # reduced path count used in the executed run to fit the time budget (see RESULTS.md)

# ---------------------------------------------------------------------------
# exp_1: VPGARCH-SV specification, positivity, variance moments
# ---------------------------------------------------------------------------
# Illustrative VPGARCH-SV parameters satisfying Proposition 3 (Sec. 2.2.1); values
# not given numerically in the paper (which reports empirical estimates in Table 1
# for the rho=1 GARCH case only), chosen to respect the positivity restrictions.
EXP1_TAU = 1.0  # time step tau (one "period"), normalization, not specified in the paper
EXP1_RHO = 0.6  # correlation between epsilon_1 and epsilon_2, Eq. (9)-(10), not specified in the paper
EXP1_R = 0.0  # risk-free rate r, Eq. (11), not specified in the paper
EXP1_LAMBDA = 0.02  # conditional expected excess log return loading lambda(tau), Eq. (11)
EXP1_SIGMA2 = 1.0e-4  # long-run variance sigma^2(tau), Eq. (11)
EXP1_PHI = 0.9599227223123792  # variance persistence phi(tau), Eq. (11); found by a positivity-feasible random search (Prop. 3)
EXP1_ETA = -0.012637177804458707  # loading eta(tau) in psi_2, Eq. (12); satisfies Prop. 3 jointly with the other parameters below
EXP1_A0 = 0.0  # VPGARCH-SV quadratic-loading intercept a_0(tau)=0, Eq. (12); paper fixes a_0=0 empirically (Sec. 4.2)
EXP1_A1 = 0.04465605606610989  # VPGARCH-SV quadratic-loading linear term a_1(tau), Eq. (12)
EXP1_B0 = 2.8297558906644125e-06  # psi_3 intercept b_0(tau), Eq. (11)
EXP1_B1 = 0.059633635538181566  # psi_3 slope b_1(tau), Eq. (11)
EXP1_H0 = EXP1_SIGMA2  # initial variance h_0, set to the long-run level sigma^2 (Sec. 4.2 convention)
EXP1_MAX_DEGREE = 3  # maximum polynomial degree m used to build Q_m, Sec. 2.1 (paper works out m=2 explicitly)
EXP1_HORIZONS = (1, 5, 20, 60, 252)  # forecast horizons s (trading periods) for moment-formula verification, not specified in the paper
EXP1_NIG_NU = 0.45  # NIG inverse shape parameter nu=1/alpha, within the 0.39-0.49 range reported in Table 1

# ---------------------------------------------------------------------------
# exp_2: JPGARCH-SV specification, positivity, joint moments
# ---------------------------------------------------------------------------
EXP2_TAU = 1.0  # time step, not specified in the paper
EXP2_RHO = 0.6  # correlation rho in [0,1], Eq. (9)-(10)
EXP2_R = 0.0  # risk-free rate r
EXP2_LAMBDA = 0.02  # lambda(tau), Eq. (11)
EXP2_SIGMA2 = 1.0e-4  # long-run variance sigma^2(tau)
EXP2_PHI = 0.95  # persistence phi(tau); chosen within Prop. 4 positivity and degree-2 moment stability
EXP2_ETA = -0.0025  # loading eta(tau) in psi_2(x)=eta*sqrt(x), Eq. (14); satisfies Prop. 4
EXP2_B0 = 2.0e-06  # psi_3 intercept b_0(tau), within Proposition 4 bounds (b_0 <= omega)
EXP2_B1 = 0.08  # psi_3 slope b_1(tau); kept small so phi^2+eta^2+(m4-1)*b1^2 < 1 even for NIG (Prop. 5-type stability)
EXP2_H0 = EXP2_SIGMA2  # initial variance h_0
EXP2_MAX_DEGREE = 3  # maximum polynomial degree m for the joint (y,h) basis, Sec. 2.2.2
EXP2_HORIZONS = (1, 5, 20, 60, 252)  # horizons s for joint-moment verification, not specified in the paper
EXP2_NIG_NU = 0.45  # NIG inverse shape parameter, same range as exp_1 (Table 1)
EXP2_CUM_RETURN_J = 20  # number of periods j for cumulative-return moments, Corollary 2(ii), not specified in the paper

# ---------------------------------------------------------------------------
# exp_3: weak convergence to polynomial diffusions (Assumption 1 / Proposition 7)
# ---------------------------------------------------------------------------
EXP3_KAPPA = 0.5  # mean-reversion speed kappa (annualized units), Assumption 1, not specified in the paper
EXP3_SIGMA2 = 0.04  # long-run variance level sigma^2, Assumption 1
EXP3_R = 0.02  # risk-free rate r
EXP3_LAMBDA = 2.0  # limiting return-variance loading lambda, Assumption 1
EXP3_ETA = 0.1  # limiting variance-diffusion loading eta, Assumption 1 (JPGARCH-SV: 2*kappa*sigma^2>=eta^2 must hold)
EXP3_B0 = 0.0  # boundary condition b_0=0 forced by Proposition 7
# b_1 chosen (search over a small grid, not specified in the paper) so that the finite-tau
# positivity restrictions of Propositions 3/4 hold simultaneously for most n in EXP3_N_LIST_RUN
# under the exact Assumption-1 scaling eta(tau)=eta*sqrt(tau), b_1(tau)=b_1*sqrt(tau); the paper
# notes (Sec. 3.2) that the finite-step sequences satisfying both Assumption 1 and finite-tau
# positivity are not uniquely specified, so a few boundary n values are skipped (see RESULTS.md).
EXP3_B1 = 0.5
EXP3_A0 = 0.0  # VPGARCH-SV: eta^2*a_0=0 boundary condition, Proposition 7
EXP3_A1 = 0.0  # VPGARCH-SV: a_1=0 chosen jointly with b_1 above to satisfy finite-tau positivity (see note on EXP3_B1)
EXP3_RHO = 0.5  # correlation rho, Sec. 3.1
EXP3_Y0 = 0.0  # initial log-price state
EXP3_H0 = EXP3_SIGMA2  # initial variance state, set at the long-run level
EXP3_HORIZON_L = 1.0  # time horizon L (years), Sec. 3.2, not specified in the paper
EXP3_N_GRID_FINE = 5000  # Euler grid steps for the limiting diffusion, not specified in the paper ("fine grid")
EXP3_N_LIST = (1, 2, 4, 8, 16, 32, 64, 128)  # number of periods per year n=1/tau, Sec. 3.2
EXP3_N_LIST_RUN = (1, 2, 4, 8, 16, 32)  # reduced n grid actually executed to fit the time budget (see RESULTS.md)
EXP3_N_PATHS = 10_000  # number of simulated paths, Sec. 3.2 ("N>=10,000")
EXP3_N_PATHS_RUN = 4_000  # reduced path count used in the executed run (see RESULTS.md)
EXP3_MAX_DEGREE = 2  # polynomial degree m for the generator-matrix comparison, Corollary 3

# ---------------------------------------------------------------------------
# exp_4: weak convergence to polynomial jump-diffusions (Assumption 2 / Proposition 8)
# ---------------------------------------------------------------------------
EXP4_KAPPA = 5.0  # mean-reversion speed kappa, Assumption 2
EXP4_SIGMA2 = 0.04  # long-run variance sigma^2
EXP4_R = 0.02  # risk-free rate
EXP4_LAMBDA = 2.0  # loading lambda
EXP4_ETA = 0.3  # loading eta, Assumption 2
EXP4_B0 = 0.1  # b_0 (JPGARCH-SV: 0<=b_0<=kappa*sigma^2), Proposition 8
EXP4_B1 = 0.3  # b_1 (JPGARCH-SV: eta^2<=2*b_1)
EXP4_A1 = 1.0  # VPGARCH-SV a_1 (eta^2*a_1<=2*b_0)
EXP4_A0 = 0.0  # VPGARCH-SV a_0 (eta^2*a_0<=4*b_0*(kappa*sigma^2-b_0))
EXP4_RHO = 0.5  # Levy driver correlation rho
EXP4_JUMP_RATE = 10.0  # Poisson jump intensity of the compound-Poisson Levy driver (per unit time), not specified in the paper
EXP4_JUMP_SCALE = 0.3  # scale of the symmetric jump-size law, not specified in the paper
EXP4_Y0 = 0.0  # initial log-price state
EXP4_H0 = EXP4_SIGMA2  # initial variance state
EXP4_HORIZON_L = 1.0  # time horizon L (years), Sec. 3.3
EXP4_N_LIST = (1, 2, 4, 8, 16, 32, 64, 128)  # number of periods per year n, Sec. 3.3
EXP4_N_LIST_RUN = (1, 2, 4, 8, 16, 32)  # reduced n grid actually executed (see RESULTS.md)
EXP4_N_PATHS = 10_000  # number of simulated paths, Sec. 3.3
EXP4_N_PATHS_RUN = 4_000  # reduced path count used in the executed run (see RESULTS.md)
EXP4_MAX_DEGREE = 2  # polynomial degree m for the generator-matrix comparison, Eq. (29)

# ---------------------------------------------------------------------------
# exp_5: risk-neutral dynamics and polynomial preservation under Q
# ---------------------------------------------------------------------------
EXP5_LAMBDA = 0.02  # variance-risk price lambda(tau), base case before the Proposition 10 analytical overwrite
EXP5_TAU = 1.0  # time step, not specified in the paper
EXP5_RHO = 0.6  # correlation rho, Appendix B
EXP5_R = 0.0  # risk-free rate r
EXP5_SIGMA2 = 1.0e-4  # long-run variance
EXP5_PHI = 0.96  # persistence phi(tau)
# eta, b1, a1 chosen jointly (grid search, not specified in the paper) so that both
# Proposition 3 (VPGARCH-SV) and Proposition 4 (JPGARCH-SV) positivity restrictions hold
# at this (sigma2, phi) pair.
EXP5_ETA = -0.002  # loading eta(tau)
EXP5_B0 = 0.0  # b_0(tau)=0, required by Proposition 10(i)/(ii) for joint preservation under Q
EXP5_B1 = 0.3  # b_1(tau)
EXP5_A0 = 0.0  # VPGARCH-SV a_0(tau), Proposition 10(i)
EXP5_A1 = 0.005  # VPGARCH-SV a_1(tau)
EXP5_H0 = EXP5_SIGMA2  # initial variance
EXP5_XI2 = -5.0  # scalar risk-preference coefficient xi_2(tau) < 1/2, Eq. (B.6), not specified in the paper

# ---------------------------------------------------------------------------
# exp_6: maximum-likelihood estimation on S&P 500 returns (Sec. 4.1-4.2)
# ---------------------------------------------------------------------------
EXP6_START_DATE = "1996-01-01"  # sample start, Sec. 4.1 ("January 1996")
EXP6_END_DATE = "2023-08-31"  # sample end, Sec. 4.1 ("August 2023")
EXP6_TICKER = "SPY"  # S&P 500 proxy; the paper uses CRSP/WRDS index returns directly (see RESULTS.md)
EXP6_RISK_FREE_SOURCE = "massive_treasury_yields_3m"  # Federal Reserve H.15 3-month yield via Massive's Treasury Yields endpoint, Sec. 4.1
EXP6_N_OBS_PAPER = 6965  # number of daily return observations reported in the paper, Table 1 notes
EXP6_H0_RULE = "long_run_variance"  # h_0 set to the long-run variance sigma^2, Sec. 4.2
EXP6_A0_FIXED_ZERO = True  # VPGARCH a_0 fixed at zero in estimation, Sec. 4.2 ("based on preliminary estimates")
EXP6_OPTIM_RESTARTS = 8  # number of random restarts for the optimizer, not specified in the paper ("multiple starting values")
EXP6_VOVNIC_EPSILONS = (-4.0, -2.0, 0.0, 2.0, 4.0)  # return-innovation grid for VoVNIC, Figure 1
EXP6_VOVNIC_HORIZONS = (2, 21, 63, 126, 252)  # horizons s (trading days) for VoVNIC/EoVNIC, Figure 1 panels

# ---------------------------------------------------------------------------
# exp_7: weekly option calibration (Sec. 4.3)
# ---------------------------------------------------------------------------
# The paper calibrates to 1,443 historical weekly OptionMetrics/WRDS cross-sections
# spanning 1996-2023 (Sec. 4.1). No historical options database of that depth is
# reachable through the Dataloader or Massive APIs available here; Massive serves
# only live/recent SPY option-chain snapshots. The calibration below is therefore
# executed on the live SPY option chain snapshot(s) obtainable at run time, as a
# methodological demonstration of the same IVRMSE calibration procedure; see
# "Deviations from the paper" and "Not reproduced" in results/RESULTS.md.
EXP7_UNDERLYING = "SPY"  # S&P 500 ETF proxy for the option-chain snapshot
EXP7_N_WEEKS_PAPER = 1443  # number of weekly calibration dates reported in the paper, Sec. 4.1
EXP7_N_SNAPSHOTS_RUN = 1  # number of live option-chain snapshots actually calibrated (data availability, not a time reduction)
EXP7_MC_PATHS_PAPER = 50_000  # Monte Carlo paths per calibration, Sec. 4.3
EXP7_MC_PATHS_RUN = 4_000  # reduced path count used in the executed run (see RESULTS.md)
EXP7_XI2 = 0.0  # variance risk-preference parameter fixed at zero, Sec. 4.3 (monotonic pricing kernel)
EXP7_MIN_MATURITY_DAYS = 7  # minimum option maturity filter (calendar days), Sec. 4.1
EXP7_MAX_MATURITY_DAYS = 365  # maximum option maturity filter (calendar days), Sec. 4.1
EXP7_MIN_PRICE = 0.375  # minimum option price filter (USD), Sec. 4.1

RNG = np.random.default_rng(RNG_SEED)
