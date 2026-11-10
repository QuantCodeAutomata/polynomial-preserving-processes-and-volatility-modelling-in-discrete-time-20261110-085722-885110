# Polynomial-Preserving Processes and Volatility Modelling in Discrete Time

Replication of selected results from *"Polynomial-Preserving Processes and
Volatility Modelling in Discrete Time"*.

This repository implements and verifies:

1. **exp_1** — the variance-polynomial hybrid GARCH–SV model (VPGARCH-SV):
   dynamics, positivity (Proposition 3), one-step transition matrix `Q_m` on
   `Pol_m(R_+)`, multi-step moments (Proposition 1), stability (Proposition 5),
   and the nested NGARCH-SV / NGARCH special cases (Remark 1).
2. **exp_2** — the jointly polynomial hybrid GARCH–SV model (JPGARCH-SV):
   dynamics, positivity (Proposition 4), the joint `(y, h)` transition matrix,
   cumulative-return moments (Corollary 2), cross-date moments (Corollary 1),
   and the nested HNGARCH-SV / HNGARCH special cases (Remark 2).
3. **exp_3** — weak convergence of both discrete-time models to polynomial
   diffusions under Assumption 1 and Proposition 7, with moment-matrix
   convergence to matrix exponentials (Corollary 3).
4. **exp_4** — weak convergence to polynomial jump-diffusions under Lévy
   increment scaling (Assumption 2, Proposition 8), including the extended
   generator and the COGARCH special case.
5. **exp_5** — risk-neutral dynamics induced by the variance-dependent
   pricing kernel (Appendix B.1) and the conditions under which polynomial
   preservation survives the change of measure (Propositions 9–10).
6. **exp_6** — maximum-likelihood estimation of HNGARCH, JPGARCH, NGARCH and
   VPGARCH on daily S&P 500 excess log returns, under Gaussian and symmetric
   NIG innovations, with information criteria, term structures and
   variance-of-variance news-impact curves (Section 4.1–4.2).
7. **exp_7** — weekly option-implied calibration of the four models by
   minimizing implied-volatility RMSE via Monte Carlo pricing under the
   physical measure with a variance-dependent pricing kernel (Section 4.3).

## Layout

```
src/        model, moment, diffusion, jump-diffusion, risk-neutral,
            estimation and option-pricing modules (flat, no subpackages)
exp/        one script per experiment, run as `python -m exp.exp_N`
data/       data_loader.py (QuantX Dataloader / Massive REST market data)
tests/      pytest unit and market-data tests
results/    tables, figures, logs, and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for exp_6/exp_7
```

## Tests

```bash
python -m pytest -m "not market_data"   # fast unit tests, no network
python -m pytest                        # includes market-data tests (needs API keys)
```

## Running the experiments

```bash
python -m exp.exp_1   # VPGARCH-SV specification, positivity, variance moments
python -m exp.exp_2   # JPGARCH-SV specification, positivity, joint moments
python -m exp.exp_3   # weak convergence to polynomial diffusions
python -m exp.exp_4   # weak convergence to polynomial jump-diffusions
python -m exp.exp_5   # risk-neutral dynamics and polynomial preservation under Q
python -m exp.exp_6   # MLE on S&P 500 returns (needs DATALOADER_API_KEY)
python -m exp.exp_7   # weekly option IVRMSE calibration (needs DATALOADER_API_KEY, MASSIVE_API_KEY)
```

Each experiment accepts `--full` to run the paper's full configuration
(Monte Carlo path counts, grid sizes, number of weekly calibration dates);
by default a reduced configuration is executed so the full suite completes
within the time available for this replication (see `results/RESULTS.md`
for the exact counts used and the estimated time of the full run).

`reproduce.sh` runs the test suite and every experiment with the commands
above.

Results (tables, figures, logs) are written directly under `results/`, next
to `results/RESULTS.md`, which reports every experiment's outcome against the
paper together with the "Deviations from the paper" and "Not reproduced"
sections.
