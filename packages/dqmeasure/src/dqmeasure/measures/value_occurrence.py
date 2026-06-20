"""Value occurrence completeness from ISO/IEC 5259-2

*"Ratio of the number of occurrences of a given data value to the expected number of value occurrences in
data items with the same domain"*."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import BaseMeasure


class ValueOccurrenceCompleteness(BaseMeasure):
    """ISO/IEC 5259-2 `Com-ML-2` "Value occurrence completeness".

    Tier 1, non-positional measure. Unit is row, subject the whole table. Our interpretation of choices the
    standard leaves open:

    * We store occurrence proportions rather than raw counts, and expect that the proportions are the same on
      the measured frame.
    * Counted occurrences are capped at the number of expectation per value. This way, over-represented values
      cannot compensate for missing ones or push ``X`` past 1.

    Values outside the observed domain contribute to neither ``A`` nor ``B``; null values are not part of the
    domain.

    Parameters
    ----------
    columns:
        Optional subset of columns to evaluate. ``None`` auto-detects all string, categorical, and enum columns
        at fit time.
    expected:
        Expected occurrence proportions as a ``{column: {value: proportion}}`` dict, or ``None`` to learn them
        from the clean data at [`fit`][dqmeasure.base.BaseMeasure.fit]. Columns left out are learned.
    """

    iso_id = "Com-ML-2"
    orientation = "higher-is-better"
    reference_params = ("expected",)

    expected_: dict[str, dict[Any, float]]

    def __init__(self, columns: list[str] | None = None, expected: dict[str, dict[Any, float]] | None = None) -> None:
        super().__init__(columns=columns)
        self.expected = expected

    def _select_columns(self, frame: nw.DataFrame[Any]) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        categorical = (nw.String, nw.Categorical, nw.Enum)
        return [name for name, dtype in frame.schema.items() if isinstance(dtype, categorical)]

    def _fit_reference(self, frame: nw.DataFrame[Any], columns: list[str]) -> dict[str, dict[str, Any]]:
        # Per column: the domain with each value's occurrence proportion, relative to the clean frame's total row
        # count. Nulls in the clean data lower the expected occupancy and aren't part of the domain.
        n = len(frame)
        expected: dict[str, Any] = {}
        for name in columns:
            counts = frame[name].drop_nulls().value_counts()
            expected[name] = {value: count / n for value, count in counts.iter_rows()} if n else {}
        return {"expected": expected}

    def _score(self, frame: nw.DataFrame[Any]) -> dict[str, float]:
        n = len(frame)
        result: dict[str, float] = {}
        for name in self.columns_:
            proportions = self.expected_[name]
            observed = dict(frame[name].drop_nulls().value_counts().iter_rows()) if n else {}
            a = 0.0
            b = 0.0
            for value, proportion in proportions.items():
                expected = proportion * n
                a += min(float(observed.get(value, 0)), expected)
                b += expected
            result[name] = a / b if b else float("nan")
        return result
