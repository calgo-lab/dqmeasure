from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import BaseMeasure


class ValueOccurrenceCompleteness(BaseMeasure):
    """ISO/IEC 5259-2 `Com-ML-2` "Value occurrence completeness".

    Column measure, tier 1, non-positional: the unit is the expected occurrence of a domain value, which cannot
    be attached to a position in the frame, so the measure is ``score()``-only. Our interpretation of choices
    the standard leaves open:

    * We store occurrence proportions rather than raw counts, and expect that the proportions are the same on
      the measured frame.
    * Counted occurrences are capped at the number of expectation per value. This way, over-represented values
      cannot compensate for missing ones or push ``X`` past 1.

    Values outside the observed domain contribute to neither ``A`` nor ``B``; null values are not part of the
    domain.

    Parameters
    ----------
    column:
        The column the measure applies to. Typically categorical-like, but any dtype works.
    expected:
        Expected occurrence proportions as a ``{value: proportion}`` dict, or ``None`` to learn them from the
        clean data at [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Com-ML-2"
    iso_25024_id = None
    reference_params = ("expected",)

    expected_: dict[Any, float]

    def __init__(self, column: str, expected: dict[Any, float] | None = None) -> None:
        super().__init__(column=column)
        self.expected = expected

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        # The domain with each value's occurrence proportion, relative to the clean frame's total row count.
        # Nulls in the clean data lower the expected occupancy and aren't part of the domain.
        n = len(frame)
        counts = frame[self.column].drop_nulls().value_counts()
        return {"expected": {value: count / n for value, count in counts.iter_rows()} if n else {}}

    def _score(self, frame: nw.DataFrame[Any]) -> float:
        n = len(frame)
        observed = dict(frame[self.column].drop_nulls().value_counts().iter_rows()) if n else {}
        a = 0.0
        b = 0.0
        for value, proportion in self.expected_.items():
            expected = proportion * n
            a += min(float(observed.get(value, 0)), expected)
            b += expected
        return a / b if b else float("nan")
