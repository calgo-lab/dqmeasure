from __future__ import annotations

from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class DataAccuracyRange(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-7` "Data accuracy range".

    Column measure, tier 1, positional: unit = cell, subject = the column.

    Null cells are out of scope and count in neither ``A`` nor ``B``. A column of nulls scores
    ``nan`` rather than 0.

    Parameters
    ----------
    column:
        The numeric column the measure applies to.
    low, high:
        The interval bounds, or ``None`` to learn them from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit]. Specify both to skip ``fit`` entirely.
    method:
        How the reference interval is derived from the clean data. Currently only ``"minmax"`` (the observed
        minimum and maximum) is implemented, which is the default.
    inclusive:
        Whether the interval bounds count as in-range (``low <= v <= high``). When ``False`` the bounds are
        treated as out-of-range (``low < v < high``).
    """

    iso_5259_id = "Acc-ML-6"
    iso_25024_id = "Acc-I-7"
    reference_params = ("low", "high")

    low_: float
    high_: float

    def __init__(
        self,
        column: str,
        low: float | None = None,
        high: float | None = None,
        method: Literal["minmax"] = "minmax",
        inclusive: bool = True,
    ) -> None:
        super().__init__(column=column)
        self.low = low
        self.high = high
        self.method = method
        self.inclusive = inclusive

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, numeric=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "minmax":
            raise ValueError(f"Unsupported method: {self.method}")
        col = frame[self.column]
        return {"low": float(col.min()), "high": float(col.max())}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell, return 1.0 if in range, 0.0 if out of range. If a value is missing, return null, which
        # excludes missing values as the measure defines.
        # Encoding it as a null-preserving float rather than a boolean keeps the result identical across
        # backends: pandas' numpy-backed boolean columns cannot hold null and silently turns missing values
        # into False, which would inflate the score() output. Missing values are excluded from the measure.
        closed: Literal["both", "none"] = "both" if self.inclusive else "none"
        expr = (
            nw.when(~nw.col(self.column).is_null())
            .then(nw.col(self.column).is_between(self.low_, self.high_, closed=closed).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return frame.select(expr)[self.column]
