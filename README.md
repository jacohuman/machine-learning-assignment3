# Recurrent neural networks for time series forecasting

A Python implementation comparing Elman, Jordan and multi-recurrent
networks on five time series datasets.

## Directory structure

```text
src/
  datasets.py        Load, transform and standardise data; construct windows
  rnn.py             Three recurrence equations and the shared training loop
  experiment.py      Chronological validation, final training and evaluation
data/                Five local datasets, metadata and source information
tests/
  test_core.py       Basic recurrence, training and preprocessing tests
requirements.txt     Pinned Python dependencies
```

Running the experiment creates `results/`, which is excluded from Git.

## Setup

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tests and reproduction

First run the quick implementation tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These three tests check the equations, learning on a simple signal and causal
preprocessing. They do not reproduce the  experiments.

To train all models again and check the reported results:

```bash
python src/experiment.py
python tests/check_results.py
```

The experiment runs 135 validation evaluations and 45 final neural runs, plus
naive, drift and monthly seasonal-naive baselines. It tests hidden sizes 4, 8
and 16 using three expanding validation folds. The final 20% is reserved for
testing, with seeds 1, 2 and 3. Full settings are in `src/experiment.py` and
`src/rnn.py`.

The report's best neural models by mean test root mean squared error (RMSE) are:

| Dataset | Model | RMSE |
| --- | --- | ---: |
| Nile | Jordan | 127.672 |
| Sunspots | Elman | 30.717 |
| Carbon dioxide | Elman | 0.430 |
| El Niño | Multi | 0.498 |
| GDP | Multi | 89.010 |

Drift beats all three neural models on GDP, with RMSE 85.958. Mean absolute error
(MAE) and variation across seeds are included in `summary.csv`.

The other generated files are `runs.csv` for individual test runs,
`predictions.csv` for neural predictions, and `histories.csv` for learning curves
from the selected models' last validation folds. No network access  is needed to run the experiments.
