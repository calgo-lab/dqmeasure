# dqmeasure

Data-quality measures from **ISO/IEC 25024** and **ISO/IEC 5259-2**, implemented as
scikit-learn-style estimators.

A measure learns a **reference** from a *clean* (train) instance of a table and then
measures a *dirty* (test) instance:

```python
from dqmeasure import DataAccuracyRange

measure = DataAccuracyRange().fit(train)
units   = measure.predict(test)
scores  = measure.score(test)
```

`X` is reported exactly as the standard defines it; each measure carries an
`orientation` (higher- or lower-is-better) so downstream code never has to guess. The
conceptual model — units, conditions, subjects, and the two measure tiers — is described
in [the measure model](dqmeasure-model.md).

The core is written against [Narwhals](https://narwhals-dev.github.io/narwhals/), so inputs
may be **Polars** or **pandas** frames; results come back in the caller's backend.

See the [API reference](reference.md) for the full estimator surface.
