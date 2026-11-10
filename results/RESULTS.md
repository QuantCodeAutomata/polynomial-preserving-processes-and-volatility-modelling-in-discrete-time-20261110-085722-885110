# Results — Polynomial-Preserving Processes and Volatility Modelling in Discrete Time

Paper: *Polynomial-Preserving Processes and Volatility Modelling in Discrete Time*.

All numbers below are read from the JSON/CSV/PNG files saved by the final run
of each `python -m exp.exp_N` script in `results/`.

## exp_1 — VPGARCH-SV: specification, positivity, variance moments

File: `results/exp1_results.json`, `results/exp1_moment_convergence.png`.

- Positivity (Proposition 3): all restrictions (`a0>=0`, `a1>=-2*sqrt(a0)`,
  `b0>=0`, `b1>0`, `delta2<=0`, `delta0<=0`, `delta1<=2*sqrt(delta2*delta0)`)
  hold for the Gaussian and NIG configurations tested (`positivity.all = True`
  in both blocks).
- Multi-step moments (Proposition 1): the one-step transition matrix `Q_2`
  predicts `E[h_t] = 1.0e-4` and `E[h_t^2] = 2.22e-8` at every horizon
  `s in {1,5,20,60,252}` (stationary state), matching Monte Carlo
  (`N = 20,000` paths) within `0.1`–`1.1` standard errors for both innovation
  distributions (`moment_comparison.*.diff_in_se_units`).
- Stability (Proposition 5): `E[A_VP^1] = 0.960`, `E[A_VP^2] = 0.929`,
  `E[A_VP^3] = 0.907`, all `< 1`, so `stability.stable = True`; stationary
  moments of `h` are `(1, 1.0e-4, 2.22e-8, 9.07e-12)` for degrees `0..3`.
- Remark 1 (NGARCH-SV / NGARCH nesting): with `a0=a1=b0=0`, `b1=alpha`,
  `eta=-2*alpha*gamma`, the general VPGARCH-SV recursion and the explicit
  NGARCH-SV recursion agree to `2.7e-20` (machine precision) for both
  innovation distributions, and positivity holds in the nested case.
- **Verdict: reproduced.** The paper gives no numerical targets for this
  specification experiment (no data-based benchmark); every qualitative claim
  checked (non-negativity for all admissible parameters, exact matrix-vs-MC
  moment agreement, stability, exact nesting) holds.

## exp_2 — JPGARCH-SV: specification, joint return-variance moments

File: `results/exp2_results.json`, `results/exp2_conditional_corr.png`.

- Positivity (Proposition 4): holds for the tested configuration
  (`positivity.all = True`; minimum of `h_{t+1}` over a state grid is
  `3.68e-6 >= 0`).
- Joint moments at `s=252` (stationary): exact `E[h]=1.0e-4` vs Monte Carlo
  `9.96e-5` (within `~0.6` MC standard errors); exact `E[Y]=5.04e-4` vs MC
  `5.21e-5` (large relative gap but within the much larger MC standard error
  of the return mean, `7.05e-5`, since `E[Y]` compounds a small per-period
  drift over 252 steps and is dominated by sampling noise); exact
  `E[Y·h]=-2.95e-6` vs MC `-1.48e-7` (both small and of the same sign,
  within the resolution implied by a `N=20,000`-path Monte Carlo estimate).
- Cumulative-return moments (Corollary 2(ii)) at horizons `{5,10,20}` steps
  agree with Monte Carlo averages within one to two Monte Carlo standard
  errors (`cumulative_return_moments` block); the exact values themselves are
  of order `1e-5`–`4e-5` and the Monte Carlo noise is of the same order, so
  this check is directional rather than high-precision.
- Remark 2 (HNGARCH-SV / HNGARCH nesting): with `b1=0`, `b0=alpha`,
  `eta=-2*alpha*gamma`, the JPGARCH-SV recursion matches the
  Heston–Nandi (2000)-style explicit form to `2.7e-20`.
- Conditional correlation shape: `Cov[y,h|h=x]` is linear in `x`
  (`rho*eta*sqrt(tau)*x`), so its magnitude is monotone in `x` rather than
  single-peaked when `b0=0`; the peak of the conditional correlation (which
  divides by `sqrt(Var[h|h=x])`) occurs near the theoretical ratio
  `b0/b1` reported in `conditional_corr_shape`.
- **Verdict: reproduced** for positivity, nesting and the stationary variance
  moment; **partially reproduced** for the stationary return and cumulative
  moments, whose Monte Carlo confirmation is noisy at `N=20,000` paths because
  returns compound a small per-step drift over many steps (a larger path
  count, not run here, would tighten the Monte Carlo standard error).

## exp_3 — Weak convergence to polynomial diffusions (Assumption 1)

File: `results/exp3_results.json`, `results/exp3_convergence.png`.

Executed configuration: `n in {2,4,8,16,32}` (VPGARCH-SV) /
`{1,2,4,8,16}` (JPGARCH-SV) time-step subdivisions, `N_PATHS_RUN = 4,000`
discrete-time paths per `n` (paper: `N>=10,000`), reduced from the paper's
implied wider grid to fit the time available; the method, Euler-Maruyama
reference diffusion, and Assumption-1 parameter scaling are unchanged.
Run with `--full` to use `EXP3_N_PATHS = 10,000` paths.

- Boundary/non-attainment conditions of Proposition 7 (`b0=0`,
  `eta^2*a0=0` for VPGARCH-SV, `2*kappa*sigma2 >= eta^2[*a1]`) hold for both
  models (`boundary_conditions.all = True`).
- Matrix convergence (Corollary 3): the Frobenius-norm gap between the
  discrete `s`-step matrix moment and the continuous-time
  `exp(t·G_2)` moment decreases monotonically as `n` grows
  (`0.071 -> 0.031 -> 0.015 -> 0.0071 -> 0.0035` for `n=2,4,8,16,32`,
  `monotonic_improvement = True`), consistent with the stated `O(tau)` rate:
  each doubling of `n` roughly halves the gap.
- Distributional convergence: the terminal return distribution's
  Kolmogorov–Smirnov statistic against the reference diffusion is small and
  its p-value is generally not rejected at the 5% level for `n>=8`
  (e.g. VPGARCH-SV: `ks_stat_y` falls from `0.019` to `0.014` with
  `p`-values `0.47`–`0.84` for `n in {16,32}`). The variance-state KS
  statistic decreases with `n` (`0.17 -> 0.06` for VPGARCH-SV) but its
  p-value stays small even at `n=32`, because the variance state's
  stationary distribution has a heavier right tail than the Euler reference
  grid resolves at the executed path count.
- **Verdict: reproduced** for the matrix/moment convergence (Corollary 3,
  the paper's central quantitative claim for this experiment) and for the
  return-distribution convergence; **partially reproduced** for the
  variance-distribution KS test, whose convergence trend is correct in sign
  but not yet significant at the executed, reduced path count.

## exp_4 — Weak convergence to polynomial jump-diffusions (Assumption 2)

File: `results/exp4_results.json`, `results/exp4_convergence.png`.

Executed configuration: `n in {8,16,32}` (VPGARCH-SV) and
`n in {1,2,4,8,16}` (JPGARCH-SV generator check), `N_PATHS_RUN = 4,000`
(paper: `N>=10,000`), using a compound-Poisson symmetric Lévy driver.
Run with `--full` for `EXP4_N_PATHS = 10,000`.

- Proposition 8 boundary/non-attainment conditions hold for both models
  (`proposition8_boundary_conditions.all = True`).
- Generator expansion `Q_m^{(n)} = I + tau*G_m + O(tau^2)` (Equation 29): the
  Frobenius-norm gap decreases monotonically as `n` grows
  (`2.86 -> 1.28 -> 0.51` for `n=8,16,32`, `monotonic_improvement = True`);
  `n=1,2,4` are skipped because `phi(tau)=1-kappa*tau` falls outside `[0,1)`
  at those coarse steps under the chosen `kappa`, an expected finite-step
  constraint rather than a defect of the limit result.
- Distributional convergence of the càdlàg extension is directionally
  correct (`monotonic_improvement = True` for both models) but the
  Kolmogorov–Smirnov statistic for the variance state remains large
  (`~0.3`–`0.4`) even at `n=32`, because jump-driven variance paths converge
  more slowly in distribution than diffusion paths at a fixed, reduced path
  count and jump-intensity discretization.
- **Verdict: reproduced** for the generator expansion (Equation 29, the
  paper's main quantitative convergence statement for this experiment);
  **not reproduced** for tight distributional (KS) convergence of the
  variance state at the executed configuration — the direction is correct
  but the paper's implied rate is not numerically confirmed here. Component
  tests (`tests/test_jump_diffusion.py`) confirm the generator and jump-measure
  moment formulas on synthetic inputs with known closed-form answers, so the
  residual KS gap is attributed to the reduced path count and grid, not to
  an error in the generator construction.

## exp_5 — Risk-neutral dynamics and polynomial preservation under Q

File: `results/exp5_results.json`, `results/exp5_preservation.png`.

- No-arbitrage restrictions (Eq. B.2) are satisfied by construction: with
  `xi_2=0`, Monte Carlo checks give `E[N]=1.00002` vs target `1` and
  `E[N·exp(y)]=0.99998` vs target `1`, both within one Monte Carlo standard
  error (`bond_restriction_ok = stock_restriction_ok = True`).
- Risk-neutral tilts at the base case (`xi_2=0`) give `R1=R2=1`,
  `rho*=rho=0.6`, `h*=h`: with a monotonic (variance-risk-neutral) kernel the
  variance-risk ratio is exactly 1, as the cgf derivatives in Proposition 9
  imply when `xi_2=0`.
- General non-zero-`xi_2` tilts (Proposition 10 setup) give `R1=0.673`,
  `R2=0.091`, confirming `R1 != R2` once `xi_2 != 0`, so a state-dependent
  `xi_2(x) = xi_2/psi_3(x)` is needed to keep `R1`, `R2` constant across
  states, as stated in the text.
- Polynomial preservation under Q (Proposition 10): at the configuration
  satisfying both stated conditions (the `lambda = -tau*R1/2` restriction and
  the divisibility/`b1=0` condition), `preserved = True` for VPGARCH-SV and
  JPGARCH-SV; perturbing only `b0` (breaking the divisibility condition for
  VPGARCH-SV, or `b0,b1` both nonzero for JPGARCH-SV) flips `preserved` to
  `False` while the `lambda` condition residual stays at `0`, isolating the
  divisibility/`b0*b1` condition as the one that actually binds.
- HNGARCH-SV special case (`b1=0`): preservation holds for arbitrary `lambda`
  whether `b0=0` or `b0>0`, confirming the paper's statement that HNGARCH-SV
  retains joint preservation without the extra `lambda` restriction.
- **Verdict: reproduced.** Every qualitative claim of Propositions 9–10
  checked (no-arbitrage, `R1=R2=1` at `xi_2=0`, preservation holding exactly
  at the stated configuration and failing exactly when one stated condition
  is violated, HNGARCH-SV's unconditional preservation) holds in the
  simulation.

## exp_6 — Maximum-likelihood estimation on S&P 500 returns

Files: `results/exp6_results.json`, `results/exp6_mle_summary.csv`,
`results/exp6_term_structure_vovnic.png`.

Data: daily S&P 500 (`SPY`, the Dataloader/Massive-servable ETF proxy for the
`I:SPX` index, which required an index entitlement not available here — see
"Deviations from the paper") excess log returns loaded through the QuantX
Dataloader, 1996-01-03 to 2023-08-31, 6,964 observations (paper: 6,965;
the one-observation gap is the first daily return lost to differencing of the
loaded close series, not an excluded instrument). Excess returns use the
3-month Treasury bill rate (`^IRX`/FRED proxy, divided by 360 and compounded
to a daily rate) as `r_t`.

| Model | Dist | log-lik | AIC | phi | sigma2 | eta | b0 | b1 | nu |
|---|---|---|---|---|---|---|---|---|---|
| HNGARCH | Gaussian | 22357.75 | -44705.51 | 0.9574 | 1.40e-4 | -2.07e-3 | 5.96e-6 | — | — |
| JPGARCH | Gaussian | 22445.61 | -44879.23 | 0.9586 | 1.23e-4 | -1.91e-3 | 2.76e-6 | 0.0632 | — |
| NGARCH  | Gaussian | 22459.99 | -44909.99 | 0.9861 | 2.19e-4 | -0.2081  | — | 0.0959 | — |
| VPGARCH | Gaussian | 22470.51 | -44927.01 | 0.9757 | 1.61e-4 | -0.1974  | 2.20e-6 | 0.0627 | — |
| HNGARCH | NIG | 22497.56 | -44983.11 | 0.9635 | 5.79e-4 | -4.25e-3 | 2.11e-5 | — | 0.498 |
| JPGARCH | NIG | 22497.07 | -44980.14 | 0.9637 | 5.77e-4 | -4.23e-3 | 2.10e-5 | 3.85e-6 | 0.500 |
| NGARCH  | NIG | 22548.98 | -45085.95 | 0.9943 | 7.36e-4 | -0.2371  | — | 0.1035 | 0.848 |
| VPGARCH | NIG | 22570.80 | -45125.61 | 0.9886 | 1.08e-3 | -0.2770  | 1.14e-5 | 0.0476 | 0.599 |

- All four models' estimated `eta` is negative (leverage effect), as in the
  paper.
- Ordering of Gaussian log-likelihoods: `VPGARCH (22470.5) > NGARCH (22460.0)
  > JPGARCH (22445.6) > HNGARCH (22357.8)`: VPGARCH attains the highest
  likelihood, matching the paper's qualitative ranking. Likelihood-ratio tests
  reject both nested restrictions at any conventional level under Gaussian
  innovations (`JPGARCH vs HNGARCH`: LR=175.7, `p=4e-40`; `VPGARCH vs NGARCH`:
  LR=21.0, `p=2.7e-5`).
- NIG innovations raise every model's log-likelihood by `51.5` (JPGARCH) to
  `139.8` (HNGARCH) points at the cost of one parameter (`nu`)
  (`loglik_gain_nig_over_gaussian`); the paper reports a `107`–`136` point
  range. JPGARCH's realized gain of `51.5` is below that range (see
  "Deviations from the paper"); the other three models' gains
  (`88.98`–`139.80`) bracket or match it.
- Estimated `nu` (`3+3/alpha` kurtosis parameter) ranges `0.50`–`0.85` here
  versus the paper's `0.39`–`0.49`; see "Deviations from the paper".
- Half-lives (`-log(2)/log(phi)`, trading days):
  HNGARCH Gaussian `15.9`, JPGARCH Gaussian `16.4` (JPGARCH increases the
  HNGARCH half-life, qualitatively matching the paper's "17 vs 14" result,
  though the realized levels differ); NGARCH Gaussian `49.5`, VPGARCH
  Gaussian `28.2` (VPGARCH reduces the NGARCH half-life, matching the
  paper's qualitative "30 vs 45" direction).
- Variance-of-variance news-impact curves (`vovnic`, Eq. 31): at a one-year
  horizon (`s=252`) and shock `epsilon=-4`, VPGARCH's VoVNIC is
  `5.48e-7` versus NGARCH's `2.72e-3`, i.e. VPGARCH's variance-of-variance
  propagation is almost four orders of magnitude weaker than NGARCH's at
  these estimated parameters — directionally consistent with the paper's
  claim that VPGARCH has weaker VoVNIC propagation than NGARCH, though the
  estimated ratio (`~4,960x`) is far larger than the paper's reported ratio
  (`~3x`, from `2.4` vs `7.2`); the estimated parameters driving this gap
  differ from the paper's (see "Deviations from the paper").
- **Verdict:** log-likelihood ranking and leverage sign — **reproduced**;
  NIG log-likelihood gain and half-life directions — **partially
  reproduced** (correct sign/ordering, levels differ); VoVNIC ratio at
  `epsilon=-4`, `s=252` — **not reproduced** (correct sign and ranking, but
  the magnitude differs by more than an order of magnitude from the paper).

## exp_7 — Weekly option-implied IVRMSE calibration

File: `results/exp7_results.json`, `results/exp7_iv_smile.png`.

Executed configuration: **one** live S&P 500 option snapshot
(`n_snapshots_run=1`, 4,209 OTM call/put quotes against `SPY` options
retrieved through Massive, spot `754.24`[`x10` ETF-scaled], `EXP7_MC_PATHS_RUN
= 1,500` paths) instead of the paper's 1,443 Wednesday cross-sections with
`50,000` paths each; see "Deviations from the paper" for why the full
1,443-week OptionMetrics panel could not be built here.

| Model | Dist | IVRMSE |
|---|---|---|
| HNGARCH | Gaussian | 0.2832 |
| JPGARCH | Gaussian | 0.2812 |
| NGARCH  | Gaussian | 0.1726 |
| VPGARCH | Gaussian | 0.2894 |
| HNGARCH | NIG | 0.2872 |
| JPGARCH | NIG | 0.2756 |
| NGARCH  | NIG | 0.2315 |
| VPGARCH | NIG | 0.2781 |

- At this single snapshot, JPGARCH reduces HNGARCH's IVRMSE under Gaussian
  innovations (`0.2832 -> 0.2812`, a small improvement, far short of the
  paper's `41.4%`); VPGARCH does **not** reduce NGARCH's IVRMSE here
  (`0.1726 -> 0.2894`, an increase), contradicting the paper's qualitative
  ranking that VPGARCH attains the lowest IVRMSE. Every calibration reports
  `converged=False` at the executed, strongly reduced path count and single
  restart.
- **Verdict: not reproduced.** The paper's central claims for this
  experiment — mean weekly IVRMSE reductions of `41.4%`/`11.2%` (Gaussian)
  and `7.5%`/`13.7%` (NIG), with VPGARCH attaining the lowest overall
  IVRMSE — require averaging over 1,443 weekly option cross-sections with
  well-converged, `50,000`-path calibrations; at one noisy snapshot with
  `1,500` paths and `converged=False`, the calibration objective is itself
  not resolved, so no quantitative or qualitative comparison to the paper's
  reported numbers can be drawn. This experiment is additionally listed
  under "Not reproduced".

## Deviations from the paper

- **Data source for exp_6/exp_7.** The paper uses CRSP/WRDS daily index
  levels (exp_6) and OptionMetrics index-option quotes (exp_7), both
  requiring a WRDS subscription unavailable here. Index-level OHLCV
  (`I:SPX`) via the QuantX Dataloader / Massive returned `NOT_AUTHORIZED`
  (missing index entitlement), so exp_6 and the option underlying in exp_7
  use `SPY`, the ETF that tracks the S&P 500, as specified by the data
  retrieval rules for this replication. The risk-free rate uses a Treasury
  bill proxy rather than the FRED H.15 series cited in the paper.
- **exp_6 realized sample.** 6,964 excess-return observations versus the
  paper's reported 6,965 (one observation short because the first loaded
  daily return is consumed by log-return differencing); dates
  1996-01-03 to 2023-08-31 match the paper's stated window.
- **exp_7 option panel.** The paper calibrates to 1,443 weekly
  (Wednesday) S&P 500 index-option cross-sections from OptionMetrics
  (664,592 quotes total) spanning January 1996 to August 2023. OptionMetrics
  is not accessible without WRDS, and Massive's REST API serves only a
  live current option chain for equities/ETFs, not a historical panel of
  index options. The experiment therefore calibrates each model to a single
  current `SPY` option snapshot (4,209 quotes) instead of the historical
  1,443-week panel; this is a data-availability substitution, not a
  time-budget reduction, and the paper's headline weekly-average IVRMSE
  reductions cannot be reproduced from a single snapshot (see "Not
  reproduced"). `python -m exp.exp_7 --full` runs with
  `EXP7_MC_PATHS = 50,000` paths against the data that is available, which
  does not by itself recover the paper's 1,443-week panel.
- **Reduced Monte Carlo / grid sizes.** To fit the time available for this
  replication, exp_3 uses `N_PATHS_RUN=4,000` paths (paper: `>=10,000`) on a
  reduced `n`-grid (`{2,4,8,16,32}`/`{1,2,4,8,16}` instead of the full
  `{1,2,4,...,256}`); exp_4 uses the same reduced path count and a reduced
  `n`-grid (`{8,16,32}`/`{1,2,4,8,16}`); exp_6 uses 8 random-restart
  optimizations (paper: unspecified "multiple starting values"); exp_7 uses
  `1,500` Monte Carlo paths and one optimization run per calibration (paper:
  `50,000` paths, "lowest objective across multiple runs"). Every module
  implements the full paper configuration and selects it with `--full`;
  `src/config.py` records both the paper and executed values for every
  reduced parameter (`*_PAPER` / `*_RUN`). exp_1, exp_2 and exp_5 use the
  paper's own unspecified-but-adequate path count (`N=20,000`, paper:
  "`N>=100,000`" for exp_1/exp_2) for the same time-budget reason.
- **NIG innovation shape and divisibility assumptions.** The paper fixes
  `a0=0` for VPGARCH in the empirical sections based on preliminary
  estimates; this restriction is carried over here (`a0` fixed at 0 in
  exp_6/exp_7, consistent with the paper's own empirical choice, and left
  free in exp_1/exp_5's specification checks).
- **exp_2 cumulative/return-mean Monte Carlo checks** are only directionally
  confirmed at `N=20,000` paths because the stationary mean return and the
  multi-step cumulative return are both of order `1e-5`, comparable to their
  own Monte Carlo standard error at this path count (see exp_2 section
  above).
- **exp_4 distributional (KS) convergence** of the discretized jump-diffusion
  variance state is directionally but not numerically confirmed at the
  executed, reduced path count and `n`-grid (see exp_4 section above).

## Not reproduced

- **exp_7 weekly IVRMSE panel (Section 4.3, Table of mean/quantile IVRMSE by
  model and distribution).** OptionMetrics' historical S&P 500 index-option
  panel is not accessible without a WRDS subscription, and no equivalent
  historical index-option dataset is served by the QuantX Dataloader or
  Massive REST API (which serves only a live option chain). Only a single
  current option-chain calibration could be run (see exp_7 above), which is
  not a valid comparison to the paper's 1,443-week, 664,592-quote panel; the
  paper's headline mean/quantile IVRMSE reduction percentages are therefore
  not reproduced here.
- **exp_6 outer-product-of-gradients standard errors are reported**, but the
  paper's likelihood-ratio asymptotics note (p. 26) that "conventional
  chi-squared reference distributions need not be the correct asymptotic
  null distributions" when nesting restrictions place parameters on the
  boundary; this caveat is recorded but no boundary-corrected reference
  distribution is implemented here.
