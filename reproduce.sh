#!/usr/bin/env bash
# Runs the test suite and every experiment exactly as described in README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
python -m exp.exp_6
python -m exp.exp_7
