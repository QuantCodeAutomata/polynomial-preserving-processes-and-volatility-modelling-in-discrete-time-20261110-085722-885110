# Repository guide for coding agents and contributors

This repository implements and verifies selected results from *Polynomial-Preserving
Processes and Volatility Modelling in Discrete Time*: the VPGARCH-SV and
JPGARCH-SV discrete-time stochastic-volatility models, their polynomial
moment structure, their continuous-time diffusion and jump-diffusion limits,
their risk-neutral dynamics, and their maximum-likelihood / option-implied
calibration to S&P 500 returns and options.

## Layout

```
src/        flat library modules: model dynamics, polynomial transition
            matrices, moment formulas, diffusion/jump-diffusion limits,
            risk-neutral dynamics, MLE estimation, Monte Carlo option
            pricing, and config.py (all experiment parameters)
exp/        one script per experiment (exp_1 .. exp_7), run as
            `python -m exp.exp_N`
data/       data_loader.py (QuantX Dataloader / Massive REST market data);
            downloaded data is never committed, only cached under
            data/cache/ (gitignored)
tests/      pytest unit tests (synthetic/small inputs) and market-data tests
            marked @pytest.mark.market_data
results/    tables, figures and logs written directly by the exp/ scripts,
            plus results/RESULTS.md with the paper comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for exp_6/exp_7
```

## Tests

```bash
python -m pytest -m "not market_data"   # fast unit tests, no network, default
python -m pytest                        # also runs market-data tests (needs API keys)
```

## Experiments

```bash
python -m exp.exp_1   # VPGARCH-SV specification, positivity, variance moments
python -m exp.exp_2   # JPGARCH-SV specification, joint moments
python -m exp.exp_3   # weak convergence to polynomial diffusions
python -m exp.exp_4   # weak convergence to polynomial jump-diffusions
python -m exp.exp_5   # risk-neutral dynamics and polynomial preservation under Q
python -m exp.exp_6   # MLE on S&P 500 returns (needs DATALOADER_API_KEY, MASSIVE_API_KEY)
python -m exp.exp_7   # option-implied IVRMSE calibration (needs DATALOADER_API_KEY, MASSIVE_API_KEY)
```

Every script accepts `--full` to run the paper's full configuration (Monte
Carlo path counts, `n`-grids, number of calibration dates); by default a
reduced configuration runs so the full set of experiments completes in the
time available (see `results/RESULTS.md` for the exact counts and the
reasons). Re-running an experiment command overwrites its files under
`results/`; the numbers and verdicts in `results/RESULTS.md` must then be
checked against the newly written files and updated by hand — they are not
regenerated automatically.

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables (no classes, no dataclasses), each with a comment
  naming its meaning, unit, and the paper section/equation it comes from, or
  stating that the paper does not specify it. Parameters reduced to fit the
  time available keep both a `*_PAPER` and a `*_RUN` variable.
- Market data is loaded only through `data/data_loader.py`, which wraps the
  QuantX Dataloader REST API (`POST /fetch-ohlcvs`) for OHLCV bars and the
  Massive REST API for data the Dataloader does not serve (here, the live
  option chain used by exp_7). No other module fetches market data.
  `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment
  via `.env` (see `.env.example`) and never written to code, logs or results.
- `tests/` contains all pytest files. Unit tests of `src/` run on synthetic
  inputs or closed-form reference values and carry no marker. Tests that call
  `data/data_loader.py` are marked `@pytest.mark.market_data`, use a small,
  session-cached slice of data, and are skipped automatically when
  `DATALOADER_API_KEY` is not set.
- Deviations from the paper (data source, sample realized vs. paper-stated,
  reduced Monte Carlo/grid configurations, and any material gap to the
  paper's reported numbers) are recorded only in `results/RESULTS.md`, never
  in code comments or config values beyond a one-line source note.
