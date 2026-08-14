# The data quality measure model

Every data quality measure in `dqmeasure` follows the model in this document.
The model builds on the measurement framework of **ISO/IEC 25021** and the measure
definitions of **ISO/IEC 25024** and **ISO/IEC 5259-2**. We make a small number of
simplifying assumptions, listed in [§7](#7-simplifying-assumptions) with their effects
noted in [§8](#8-effects-of-the-simplifying-assumptions).

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

In `dqmeasure`, we map these concepts onto tabular data:

| ISO concept | In `dqmeasure` |
|---|---|
| Target entity | Dataframe (ISO/IEC 25024 *data file*, *data set*; ISO/IEC 5259-2 *data frame*, *dataset*) |
| Property to quantify | *data values* (cells), *data records* (rows), *data items* (columns) |
| Quality characteristic | accuracy, completeness, consistency, etc. (ISO/IEC 25012, ISO/IEC 5259-1) |

ISO/IEC 25024 documents each QM with an ID (e.g. `Acc-I-7`), a measurement function (almost always a
ratio `X = A/B` of two QMEs) and the target entities and properties it applies to.  ISO/IEC 5259-2
reuses most of these for ML datasets and adds its own IDs.

## 2. Scope: parameterized measures

The standards define each measure with a requirements specification: For example, the
required interval of `Acc-I-7` or the expected value occurrences of `Com-ML-2` are assumed to be given.
In contrast, `dqmeasure` implements the subset of measures for which the requirement can be estimated
from a clean instance of the data.

We model an scikit-learn-style API:

- `dqmeasure.fit(clean)` estimates the **reference** (an interval, a domain, expected counts, a
  distribution), turning the `dqmeasure` into a concrete **measurement procedure**,
  as described in ISO/IEC 25021, §4.9.
- `dqmeasure.predict(dirty)` and `score(dirty)` execute that procedure on another instance.

In ML terms, the clean instance is the *train set* and the dirty instance the *test
set*. The `y` argument of the scikit-learn API is unused, because estimating the reference does
not require labels.

The reference is a set of named parameters (for `Acc-I-7` the interval bounds, for
`Com-ML-2` the expected occurrences) that we use to run the measure. Our approaches are opinionated
wherever the standards leave room for interpretation. We take this decision in order to make the library
easier to use.

The reference can be set in two ways: either an expert specifies them in the measure's constructor,
or `fit(clean)` estimates the ones left open. A fully specified measure is ready to use without `fit`.

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

A boolean condition like "in range?" or "non-null?" is the `{0,1}` special case; fractional
condition results are allowed but rare in the standards.

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
(see [§2](#2-scope-parameterized-measures)).

## 4. Tier-1 measure: unit, condition, subject

We define three independent facets that describe every condition-count measure.

### Unit: what `A` and `B` count

The **unit** is either a position in the dataframe or not: A cell we attach a score to is a
positional unit. But non-positional units live outside the frame, `Com-ML-2` for example
counts expected occurrences of each value in a column. If a value is absent, we cannot
position it to a cell, making the QME non-positional.

| Unit | Positional? | Example |
|---|---|---|
| cell (data value) | yes | value in range (`Acc-I-7`) |
| record (data record) | yes | no empty data item (`Com-ML-4`) |
| expected occurrence (domain member) | no | expected domain value frequency (`Com-ML-2`) |

### Condition: Applying the QM to the unit

The **condition** is determined by the QM. It evalutes a unit and may freely read more than the unit
itself and may span the whole row (a cross-column rule), other rows of the same column (`Cur-I-1`
reads the temporally preceding update event), the learned reference (the interval of
`Acc-I-7`), or side inputs.

### Subject: what `X` belongs to

The **subject** is what the one `X` belongs to, and it comes in exactly two scopes: a
**column** measure is constructed for one named column
(`DataAccuracyRange("temperature")`), a **table** measure for the whole dataframe
(`RecordCompleteness()`). The scope is taken from the measure's ISO definition.
A table-scoped constructor takes no column argument, and each class exposes it as
`scope` metadata. To measure several columns with a column measure, construct one
instance per column.

Scope and **context** are independent: `Acc-I-2` semantic data accuracy is scoped to one
column, but its condition reads the whole row as context.

## 5. Measurement function and orientation

The measurement function `X(A,B)` combines the QMEs into `X`. We consider two forms
that occur in the standards:

- `X = A/B`, by far the most common;
- `X = 1 − A/B`, occasionally, when `A` counts violations but the measure should still
  report conformance (e.g. `Com-ML-5` label completeness, `Com-I-5` empty records in a
  data file).

Every `dqmeasure` score is oriented the same way: **higher is better**, with values toward 1
meaning requirements are increasingly met. ISO/IEC 25024 normalizes most measures that way
already. Where a standard defines `X` in the opposite direction, the measure reports `1 - X` instead.
Scores can be compared and thrasholded more easily this way. This applies to the *risk* measures
`Acc-I-4` (outliers) and `Con-I-3` (duplicated values), the duplicate ratio `Con-ML-1`, and
the distance `Con-ML-2`.

Edge convention: when a subject has no units in scope (`B = 0`), `X` is undefined and
reported as `NaN`.

## 6. Examples

We discuss the implementation of three measures from the standards in detail.

### `Acc-I-7` Data accuracy range: tier 1, positional

> "Are data values included in the required interval?" (ISO/IEC 25024, Table 1)

```text
QM          Acc-I-7 Data accuracy range
Subject     one numeric column, named in the constructor
Unit        cell (data value); positional
Reference   the interval [min, max]; specified in the constructor or estimated by fit()
Condition   value lies inside the reference interval
A           number of values in the interval
B           number of values for which an interval is defined (non-null values)
X           A/B
```

`predict()` returns the per-cell condition (`{0,1}` for out/in range) as a series, and
`score()` returns the column's measure.

### `Com-ML-2` Value occurrence completeness — tier 1, non-positional

> "Ratio of the number of occurrences of a given data value to the expected number of
> value occurrences in data items with the same domain" (ISO/IEC 5259-2, Table 2)

```text
QM          Com-ML-2 Value occurrence completeness
Subject     one categorical column, named in the constructor
Unit        expected occurrence of a domain value; non-positional
Reference   the domain and expected occurrences; specified in the constructor or estimated at fit()
A           observed occurrences
B           expected occurrences
X           A/B
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
Subject     one column, named in the constructor
QMEs        the reference distribution (fitted on clean data)
            and the observed distribution (on dirty data)
Function    the worst-case probability disagreement sup |P(A) − Q(A)|
            over the column type's natural events: half-lines for ordered
            columns (the Kolmogorov–Smirnov statistic), all value subsets
            for unordered columns (the total variation distance)
X           1 − the distance in [0, 1]
```

`fit()` estimates the column's reference distribution; `score()` compares the dirty
data's distribution against it. No per-unit value exists: `score()`-only. The standard
delegates the choice of function to the ML task; we resolve it with the single
sup-over-events principle above, which the column's dtype instantiates — so the
measure needs no method parameter.

## 7. Simplifying assumptions

Where `dqmeasure` deviates from a literal reading of the standards:

1. Single-table target entity: The target entity is one dataframe. Measures over
   other target entities (data models, data dictionaries, DBMS configuration,
   presentation devices) are out of scope.
2. References can be learned and specified: ISO assumes references come from a
   requirements specification; `dqmeasure` estimates them from a clean instance at
   `fit()` time. A learned reference is an estimate and inherits the clean data's
   blind spots. Measures with references that cannot be learned are excluded.
3. Inherent point of view only: Only measures from ISO/IEC 25012's "inherent"
   point of view (properties of the data itself) are implemented. "system-dependent"
   measures (hardware, access infrastructure) are out of scope.
4. Two scopes only: Every measure is constructed either for a single named column or
   for all columns of the dataframe, and `score()` returns one `X` either way.
5. No measures of measures: DQMs that quantify other DQMs are excluded.
6. Missing data is null-encoded: We assume a value is missing if and only if it is null
   in the dataframe; null-indicator sentinels (`""`, `"?"`, `-999`) must be
   normalized to nulls before measurement.
7. Attribute = feature = column. The standards use *attribute*, *feature* and *data
   item* in their respective domain. We assume they're the same thing.

## 8. Effects of the simplifying assumptions

The assumptions in [§7](#7-simplifying-assumptions) make some ISO/IEC 25024 and
5259-2 measures numerically coincide, though they remain distinct classes,
distinguished by the role of the column and their measurement function:

- `Com-ML-3` feature completeness equals `Com-I-2` feature completeness, both are implemented
  in the same measure. `Com-ML-5` label completeness uses the same measurement function. However,
  we decided to implement `Com-ML-5` as an independent measure because it contains the semantic
  meaning of *label* column completeness compared to *feature* column completeness.

The following measures are identical but have different scopes:

- `Com-ML-1` value completeness (table) relates to `Com-ML-3` (column).
- `Com-ML-4` record completeness (table) relates to `Com-ML-3` (column).
- `Com-I-5` empty records (table) relates to `Com-ML-3` (column).
- `Cur-ML-2` record currentness (table) relates to `Cur-ML-1` feature currentness (column).

Notably, `Con-ML-1` data record consistency (table, k = all columns) does **not** relate to
`Con-I-3` risk of data inconsistency (column, k = 1), because duplication isn't cell-separable.