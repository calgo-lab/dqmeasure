from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import narwhals as nw

from dqmeasure.base import BaseMeasure


def _is_ordered(dtype: Any) -> bool:
    """Whether a column's values carry a meaningful order: numeric or timestamp-like."""
    return bool(dtype.is_numeric()) or isinstance(dtype, (nw.Datetime, nw.Date))


def _is_unordered(dtype: Any) -> bool:
    """Whether a column's values are compared by identity only: categorical-like or boolean."""
    return isinstance(dtype, (nw.String, nw.Categorical, nw.Enum, nw.Boolean))


def _drop_nan(values: list[Any]) -> list[Any]:
    # NaN is a float value, not a null, but it has no place on the real line's order (v != v filters it).
    return [v for v in values if v == v]


def _ks_statistic(reference: list[Any], observed: list[Any]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup |F - G|`` over two sorted, non-empty samples."""
    m, n = len(reference), len(observed)
    i = j = 0
    d = 0.0
    while i < m and j < n:
        if reference[i] < observed[j]:
            i += 1
        elif observed[j] < reference[i]:
            j += 1
        else:  # tie: the ECDFs may only be compared after both jump past the shared value
            value = reference[i]
            while i < m and reference[i] == value:
                i += 1
            while j < n and observed[j] == value:
                j += 1
        d = max(d, abs(i / m - j / n))
    return max(d, abs(i / m - j / n))


class DataValueDistribution(BaseMeasure):
    """ISO/IEC 5259-2 `Con-ML-2` "Distribution of data values".

    Column measure, tier 2 (statistic): the QMEs are the reference distribution learned from clean data and
    the observed distribution of the measured frame, and ``X`` is the distance between them — no per-unit
    value exists, so the measure is ``score()``-only. The standard's ``X`` is the distance itself, so we
    report ``1 - X`` to keep every measure higher-is-better: ``X = 1`` means the distributions agree,
    ``X = 0`` that they are disjoint.

    The standard delegates the choice of distribution measure ("determined according to the ML task"). We
    resolve it with a single principle instead of a catalogue of tests: ``X`` is the worst-case disagreement
    in probability over the column type's natural events, ``sup |P(A) - Q(A)|``.

    * Ordered columns (numeric, dates, datetimes): the natural events are the half-lines ``(-∞, x]``, and the
      sup over them is the two-sample **Kolmogorov-Smirnov statistic** ``sup |F - G|`` of the two empirical
      CDFs.
    * Unordered columns (string, categorical, enum, boolean): with no order to exploit, the natural events
      are all subsets of values, and the sup evaluates to the **total variation distance**
      ``½ Σ |p - q|``. Values unseen in the reference contribute their full observed mass.

    Both instantiations are parameter-free — no bins, no kernel, no significance level — which is why there
    is no ``method`` parameter. Nulls are dropped on both sides: missingness is completeness' business, not
    distribution drift.

    Parameters
    ----------
    column:
        The column the measure applies to (numeric, date, datetime, string, categorical, enum, or boolean).
    expected:
        The reference distribution: a ``{value: proportion}`` mapping for an unordered column, or a reference
        sample (sequence of values whose empirical CDF is the reference) for an ordered column. ``None``
        (default) learns it from the clean data at [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Con-ML-2"
    iso_25024_id = None
    reference_params = ("expected",)

    expected_: Mapping[Any, float] | Sequence[Any]

    def __init__(self, column: str, expected: Mapping[Any, float] | Sequence[Any] | None = None) -> None:
        super().__init__(column=column)
        self.expected = expected

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        super()._validate(frame)
        dtype = frame.schema[self.column]
        if not _is_ordered(dtype) and not _is_unordered(dtype):
            raise ValueError(
                f"Column {self.column!r} has dtype {dtype}; this measure needs a numeric, date, datetime, "
                "string, categorical, enum, or boolean column"
            )

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        col = frame[self.column].drop_nulls()
        if _is_ordered(frame.schema[self.column]):
            return {"expected": sorted(_drop_nan(col.to_list()))}
        n = len(col)
        return {"expected": {value: count / n for value, count in col.value_counts().iter_rows()} if n else {}}

    def _score(self, frame: nw.DataFrame[Any]) -> float:
        observed = _drop_nan(frame[self.column].drop_nulls().to_list())
        if _is_ordered(frame.schema[self.column]):
            if isinstance(self.expected_, Mapping):
                raise ValueError(
                    f"Column {self.column!r} is ordered, so `expected` must be a reference sample "
                    "(sequence of values), not a mapping of proportions"
                )
            reference = sorted(_drop_nan(list(self.expected_)))
            if not reference or not observed:
                return float("nan")
            return 1.0 - _ks_statistic(reference, observed)
        if not isinstance(self.expected_, Mapping):
            raise ValueError(
                f"Column {self.column!r} is unordered, so `expected` must be a {{value: proportion}} "
                "mapping, not a sequence"
            )
        total = float(sum(self.expected_.values()))
        if not self.expected_ or total <= 0 or not observed:
            return float("nan")
        proportions = {value: count / len(observed) for value, count in _value_counts(observed).items()}
        values = set(self.expected_) | set(proportions)
        return 1.0 - 0.5 * sum(abs(proportions.get(v, 0.0) - self.expected_.get(v, 0.0) / total) for v in values)


def _value_counts(values: list[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
