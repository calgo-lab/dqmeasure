"""Data accuracy range from ISO/IEC 25024

*"Are data values included in the required interval?"*
"""

from __future__ import annotations

from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class DataAccuracyRange(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-7` "Data accuracy range".

    Tier 1, positional: unit = cell, subject = column, higher is better.

    Parameters
    ----------
    columns:
        Optional subset of numeric columns to evaluate. ``None`` auto-detects all numeric columns at fit time.
    low, high:
        The interval bounds. Each may be a scalar (same bound for every column), a ``{column: bound}`` dict, or
        ``None`` to learn it from the clean data at [`fit`][dqmeasure.base.BaseMeasure.fit]. Specify both to skip
        ``fit`` entirely.
    method:
        How the reference interval is derived from the clean data. Currently ``"minmax"`` (the observed minimum
        and maximum), which is the default.
    inclusive:
        Whether the interval bounds count as in-range (``low <= v <= high``). When ``False`` the bounds are
        treated as out-of-range (``low < v < high``).
    """

    iso_id = "Acc-I-7"
    orientation = "higher-is-better"
    reference_params = ("low", "high")

    low_: dict[str, float]
    high_: dict[str, float]

    def __init__(
        self,
        columns: list[str] | None = None,
        low: float | dict[str, float] | None = None,
        high: float | dict[str, float] | None = None,
        method: Literal["minmax"] = "minmax",
        inclusive: bool = True,
    ) -> None:
        super().__init__(columns=columns)
        self.low = low
        self.high = high
        self.method = method
        self.inclusive = inclusive

    def _select_columns(self, frame: nw.DataFrame[Any]) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        return [name for name, dtype in frame.schema.items() if dtype.is_numeric()]

    def _fit_reference(self, frame: nw.DataFrame[Any], columns: list[str]) -> dict[str, dict[str, Any]]:
        if self.method != "minmax":
            raise ValueError(f"Unsupported method: {self.method!r}")
        low: dict[str, Any] = {}
        high: dict[str, Any] = {}
        for name in columns:
            col = frame[name]
            low[name] = float(col.min())
            high[name] = float(col.max())
        return {"low": low, "high": high}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        # Per-cell condition: 1.0 if in range, 0.0 if out of range, null if the value is missing. Encoding it as a
        # null-preserving float (rather than a bare boolean) keeps the result identical across backends: pandas'
        # numpy-backed boolean columns cannot hold null and would silently turn missing values into False, which
        # would also inflate the score() denominator. Missing values are excluded from the measure.
        closed: Literal["both", "none"] = "both" if self.inclusive else "none"
        exprs = [
            nw.when(~nw.col(name).is_null())
            .then(nw.col(name).is_between(self.low_[name], self.high_[name], closed=closed).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(name)
            for name in self.columns_
        ]
        return frame.select(exprs)
