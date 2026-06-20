# The data quality measure model

Every data quality measure in `dqmeasure` follows the model described in this document.
The model follows the measurement framework of **ISO/IEC 25021** and the measure
definitions of **ISO/IEC 25024** and **ISO/IEC 5259-2**, with a small number of
simplifying assumptions listed [at the end](#8-simplifying-assumptions).

## 1. The ISO measurement chain

ISO/IEC 25021 defines how a quality measure is constructed:
```text
Target entity
The artifact being measured
        │
        ▼
Property to quantify
A measurable property of the target entity
        │
        ▼
Measurement method
The operations used to quantify the property
        │
        ▼
Quality Measure Element, QME
The resulting base value
        │
        ▼
Measurement function
An algorithm that combines two or more QMEs
        │
        ▼
Quality Measure, QM(X)
A derived measure that indicates a quality characteristic,
such as accuracy, completeness, consistency, or timeliness
```

In `dqmeasure` these concepts map onto tabular data:

| ISO concept | In `dqmeasure` |
|---|---|
| Target entity | Dataframe (ISO/IEC 25024 *data file*, ISO/IEC 5259-2 *data frame*) |
| Property to quantify | *data values* (cells), *data records* (rows), *data items* (columns) |
| Quality characteristic | accuracy, completeness, consistency, … (ISO/IEC 25012, ISO/IEC 5259-1) |

ISO/IEC 25024 documents each QM with an ID (e.g. `Acc-I-7`), a measurement function (
almost always a ratio `X = A/B` of two QMEs) and the target entities and properties it
applies to. ISO/IEC 5259-2 reuses most of these for ML datasets and adds its own
IDs.

## 2. Scope: parameterized measures

The standards define each measure relative to a requirements specification: For example, the
required interval of `Acc-I-7`, the expected value occurrences of `Com-ML-2`, and the
outlier criterion of `Acc-I-4` are all assumed to be given. In contrast, `dqmeasure` implements the
subset of measures for which the requirement can be estimated from a clean instance of the data.

We express this with an scikit-learn-style API:

- `fit(clean)` estimates the **reference** (an interval, a domain, expected counts, a
  distribution) and thereby turns the measurement method into a concrete
  **measurement procedure**, as described in ISO/IEC 25021, §4.9.
- `predict(dirty)` and `score(dirty)` execute that procedure on another instance.

In ML terms, the clean instance is the *train set* and the dirty instance the *test
set*. The `y` argument of the scikit-learn API is unused, because estimating the reference does
not require labels.

The reference is a set of named parameters (for `Acc-I-7` the interval bounds, for
`Com-ML-2` the expected occurrences). There are two ways to set them, and the measurement
works the same either way: an expert can specify them in the constructor, or `fit(clean)`
estimates the ones left open. A fully specified measure is ready to use without `fit`;
otherwise `fit` fills in the rest.

## 3. Two tiers of measures

The measures we consider fall into two different tiers and are distinguished by
their QMEs.

### Tier 1 — condition-count measures

The QMEs are **counts over a population of units**. With `U_T` the set of units in
scope within the target entity `T`:

```text
A = Σ condition(u)   for u ∈ U_T,   condition(u) ∈ [0, 1]
B = |U_T|
X = A/B   (or 1 − A/B)
```

For example, take `Acc-I-7` *data accuracy range* on an `age` column whose reference
interval, learned from the clean data, is `[18, 65]`, and a dirty instance holding the
values `[25, 17, 44, null, 103]`:

```text
units U_T    the non-null cells of the column:  25, 17, 44, 103
condition    18 ≤ value ≤ 65:                    1,  0,  1,   0
A            Σ condition(u)        = 2           (in-range values)
B            |U_T|                 = 4           (values an interval applies to)
X            A/B                   = 0.5
```

A boolean condition like "in range?" or "non-null?" is the `{0,1}` special case. Fractional
conditions are allowed: with `Com-I-1` record completeness the units `U_T` are the rows, and a
record that fills 8 of 10 data items contributes `condition(u) = 0.8` to `A`.

Most measures in ISO/IEC 25024 and ISO/IEC 5259-2 have this shape.

### Tier 2 — statistic measures

The QMEs are dataset-level statistics, not counts: For example, a fitted distribution, a mean, the
eigenvalues of the data's Gram matrix. The measurement function compares or combines
these statistics. Examples from ISO/IEC 5259-2:

- `Con-ML-2` *distribution of data values* — the standard explicitly delegates the
  function: "an appropriate distribution measure and measurement function should be
  determined according to the ML task".
- `Sim-ML-2` *samples tightness* — the spread `A − B` of the extreme eigenvalues.

No meaningful per-unit value exists for these measures, so they are `score()`-only
(see [§6](#6-mapping-to-the-fitpredictscore-api)).

## 4. Tier-1 measure: unit, condition, subject

We define three independent facets that describe every condition-count measure.

### Unit: what `A` and `B` count

The **unit** is either a position in the dataframe or not: A cell or a row we attach a score to
are positional units. But non-positional units live outside the frame, `Com-ML-2` for example
counts expected occurrences of each value in a column. If a value is absent, we cannot
position it to a cell or row, making the QME non-positional.

| Unit | Positional? | Example |
|---|---|---|
| cell (data value) | yes | value in range (`Acc-I-7`) |
| row (data record) | yes | record without empty items (`Com-ML-4`) |
| expected occurrence (domain member) | no | expected domain value frequency (`Com-ML-2`) |

### Condition: Applying the QM to the unit

The **condition** is determined by the QM. It evalutes a unit and may freely read more than the unit
itself and may span the whole row (a cross-column rule), the learned reference (the interval of
`Acc-I-7`), or side inputs.

### Subject: what `X` belongs to

The **subject** is what the resulting quality measure value attaches to — a column, a
cross-column rule, or the whole table. `score()` returns one `X` per subject.

Unit and subject are independent axes: for a per-column range check, `subject = column`
and `unit = cell` (`A` sums the cells *within* each column). But for a cross-column rule
evaluation ("recruitment date after birth date + 16 years"), `subject = the rule` and
`unit = row`.

## 5. Measurement function and orientation

The measurement function `X(A,B)` combines the QMEs into `X`. We consider two forms
that occur in the standards:

- `X = A/B`, by far the most common;
- `X = 1 − A/B`, occasionally, when `A` counts violations but the measure should still
  report conformance (e.g. `Com-ML-5` label completeness).

Independently of the function's form, each measure has an **orientation**:

- **higher-is-better**, which is the default. ISO/IEC 25024 normalizes most measures so that
  values toward 1 mean requirements are increasingly met.
- **lower-is-better**, the *risk* measures keep `X = A/B` with `A` counting
  violations: `Acc-I-4` *risk of data set inaccuracy* counts outliers ("for X, lower is
  better"), `Con-ML-1` *data record consistency* counts duplicate records.

`dqmeasure` reports `X` as the standard defines it and exposes the orientation as machine-readable
metadata on each measure. Code that compares or thresholds scores must consult the orientation rather
than assume "bigger is better".

Edge convention: when a subject has no units in scope (`B = 0`), `X` is undefined and
reported as `NaN`.

## 6. Mapping to the fit/predict/score API

| API | ISO reading | Returns |
|---|---|---|
| `fit(clean)` | estimate the reference; instantiate the measurement method as a measurement procedure | the fitted measure (`self`) |
| `predict(dirty)` | per-unit condition results (tier 1 with positional units only) | one `condition(u)` score per unit, aligned with the frame |
| `score(dirty)` | compute the QMEs and apply the measurement function | `{subject: X}` |
| orientation | direction in which `X` improves | metadata on the measure |

`fit()` is optional: a measure whose reference is fully specified in the constructor can
`score()` and `predict()` without it.

`predict()` and `score()` are two views of the same measurement: for positional units,
`score()` is the aggregation of `predict()` per subject. For non-positional units,
there is no per-position array whose aggregate is `X`, and `predict()` collapses into
`score()`. The same holds for tier-2 measures, whose QMEs are not per-unit at all.
Both are `score()`-only.

## 7. Examples

We discuss the implementation of three measures from the standards in detail.

### `Acc-I-7` Data accuracy range: tier 1, positional

> "Are data values included in the required interval?" (ISO/IEC 25024, Table 1)

```text
QM          Acc-I-7 Data accuracy range
Subject     numeric columns
Unit        cell (data value); positional
Reference   per-column interval [min, max]; specified in the constructor or estimated by fit()
Condition   value lies inside the reference interval
A           number of values in the interval
B           number of values for which an interval is defined (non-null values)
X           A/B, higher is better
```

`predict()` returns the per-cell condition (`{0,1}` for out/in range), and
`score()` returns the per-column measure.

### `Com-ML-2` Value occurrence completeness — tier 1, non-positional

> "Ratio of the number of occurrences of a given data value to the expected number of
> value occurrences in data items with the same domain" (ISO/IEC 5259-2, Table 2)

```text
QM          Com-ML-2 Value occurrence completeness
Subject     categorical columns
Unit        expected occurrence of a domain value; non-positional
Reference   the domain and expected occurrences; specified in the constructor or estimated at fit()
A           observed occurrences
B           expected occurrences
X           A/B, higher is better
```

A domain value that never appears in the dirty data still contributes its expected
occurrences to `B`, meaning the measure is non-positional. Expected counts are occurrence
proportions learned at `fit()` and scaled to the measured instance's size, so clean and
dirty instances need not be the same size. There is no per-cell array that sums to `X`,
so the measure has no `predict()` and is `score()`-only.

### `Con-ML-2` Distribution of data values - tier 2

> "The statistical distribution of data values for a given feature in the dataset. An
> appropriate distribution measure and measurement function should be determined
> according to the ML task." (ISO/IEC 5259-2, Table 3)

```text
QM          Con-ML-2 Distribution of data values
Subject     each column
QMEs        the reference distribution (fitted on clean data)
            and the observed distribution (on dirty data)
Function    a distribution distance or test statistic,
            chosen per task as the standard prescribes
X           the distance, lower is better
```

`fit()` estimates the per-column reference distribution; `score()` compares the dirty
data's distribution against it. No per-unit value exists: `score()`-only.

## 8 Simplifying assumptions

Where `dqmeasure` deviates from a literal reading of the standards:

1. Single-table target entity: The target entity is one dataframe. Measures over
   other target entities (data models, data dictionaries, DBMS configuration,
   presentation devices) are out of scope.
2. References can be learned and specified. ISO assumes references come from a
   requirements specification; `dqmeasure` estimates them from a clean instance at
   `fit()` time. A learned reference is an estimate and inherits the clean data's
   blind spots.
3. Inherent point of view only. Only measures from ISO/IEC 25012's "inherent"
   point of view (properties of the data itself) are implemented. "system-dependent"
   measures (hardware, access infrastructure) are out of scope.
