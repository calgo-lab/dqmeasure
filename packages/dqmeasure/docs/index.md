# dqmeasure

Data-quality measures from **ISO/IEC 25024** and **ISO/IEC 5259-2**, implemented as
scikit-learn-style estimators.

A measure is scoped to **a single column** or to **the whole table** and learns a **reference** from a
*clean* (train) instance of a table, then measures a *dirty* (test) instance:

```python
from dqmeasure import DataAccuracyRange, RecordCompleteness

measure = DataAccuracyRange("temperature").fit(train)
units   = measure.predict(test)  # per-cell condition results, a series
x       = measure.score(test)    # the ISO quality measure value, one float

x_table = RecordCompleteness().score(test)  # a table-scoped measure takes no column
```

Every measure scores **higher is better**: where a standard defines `X` in the opposite direction, we report
`1 − X`. Otherwise `X` follows the standard, with opinionated defaults where it leaves room for
interpretation. The conceptual model is described in [the measure model](dqmeasure-model.md).

The core is written against [Narwhals](https://narwhals-dev.github.io/narwhals/), so inputs
may be **Polars** or **pandas** frames; results come back in the caller's backend.

See the [API reference](reference.md) for the full estimator surface.
