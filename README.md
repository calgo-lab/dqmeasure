# dqmeasure: Data-Quality Measures from ISO/IEC 25024 and ISO/IEC 5259

We catalogue the overlapping quality measures between ISO/IEC 25024 and ISO/IEC 5259-2
and implement them as code. Each measure can learn a reference on a clean instance of a
table and measures a *dirty* instance, or take expert-user inputs directly.

Read our documentation at [https://calgo-lab.de/dqmeasure/](https://calgo-lab.de/dqmeasure/).

## Repository structure

This is a [uv](https://docs.astral.sh/uv/) **workspace** with two members:

- `packages/dqmeasure` is the library. It takes prepared train/test frames and computes
  measures with minimal dependencies.
- `experiments` consumes the library. Sample-data generation, error injection
  with [`tab_err`](https://github.com/calgo-lab/tab_err), and analysis notebooks live here.

## Setup

Create the venv and install both members:
```bash
uv sync --all-packages
```

## Common tasks

Run the library test suite:
```
uv run pytest packages/dqmeasure/tests -q
```

Format and lint the whole workspace (Ruff):

```
uv run ruff format .
uv run ruff check .
```

Type-check the library (mypy, strict)

```
uv run mypy packages/dqmeasure
```

Run / open the end-to-end example notebook

```
uv run --package experiments jupyter notebook experiments/notebooks/data_accuracy_range.ipynb
```

Execute the notebook headlessly, e.g. in CI:
```
uv run jupyter nbconvert --to notebook --execute --inplace \
    experiments/notebooks/data_accuracy_range.ipynb
```

## Quickstart

```python
import polars as pl
from dqmeasure import DataAccuracyRange

train = pl.DataFrame({"temperature": [0.0, 20.0, 100.0]})
measure = DataAccuracyRange().fit(train)

test = pl.DataFrame({"temperature": [25.0, 150.0, -5.0]})
measure.predict(test)
measure.score(test)
```

For the full workflow, see
[`experiments/notebooks/data_accuracy_range.ipynb`](experiments/notebooks/data_accuracy_range.ipynb).

