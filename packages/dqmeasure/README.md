# dqmeasure

Data-quality measures from **ISO/IEC 25024** and **ISO/IEC 5259-2**, implemented as
scikit-learn-style estimators.

A measure learns a **reference** from a *clean* (train) instance of a table and then
measures a *dirty* (test) instance:

```python
from dqmeasure import DataAccuracyRange

measure = DataAccuracyRange().fit(train)
units   = measure.predict(test)   # per-unit condition results (positional measures)
scores  = measure.score(test)     # {subject: X}, the ISO quality measure value
```

`X` is reported exactly as the standard defines it; each measure carries an
`orientation` (higher- or lower-is-better). The conceptual model is described in
`docs/dqmeasure-model.md`.

The core is written against [Narwhals](https://narwhals-dev.github.io/narwhals/), so inputs
may be Polars or pandas frames; results come back in the caller's backend.

## Documentation

API docs are built with [MkDocs](https://www.mkdocs.org/) + Material +
[mkdocstrings](https://mkdocstrings.github.io/) from the in-source docstrings. The tooling
lives in the `docs` dependency group declared in this package's `pyproject.toml`; the site
sources are `mkdocs.yml` and `docs/` here.

Run from the **workspace root**:

```bash
# live preview with reload at http://127.0.0.1:8000
uv run --package dqmeasure --group docs \
  mkdocs serve -f packages/dqmeasure/mkdocs.yml

# one-shot strict build (fails on broken cross-references)
uv run --package dqmeasure --group docs \
  mkdocs build -f packages/dqmeasure/mkdocs.yml --strict
```
