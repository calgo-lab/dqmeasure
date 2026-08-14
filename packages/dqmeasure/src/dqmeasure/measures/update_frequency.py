from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class UpdateFrequency(PositionalMeasure):
    """ISO/IEC 25024 `Cur-I-1` "Update frequency".

    Column measure, tier 1, positional: unit = update event, subject = the column. The
    frame is read as the event log of **one** update stream: rows are update events and the column holds
    their timestamps (100 stock prices that should update every minute are 100 rows of one stream). The
    condition reads the temporally preceding event as context and checks that the event arrived within
    ``max_interval`` of it, so ``A`` counts the events keeping up the required frequency and ``B`` the events
    with a predecessor. The earliest event (no predecessor) and null timestamps are out of scope.

    Row order does not matter — events are ordered by timestamp internally, and
    [`predict`][dqmeasure.base.PositionalMeasure.predict] returns the results in the input's row order.
    Duplicate timestamps have a gap of zero and conform; exactly one of the tied earliest events is the
    out-of-scope first event.

    Parameters
    ----------
    column:
        The datetime column holding the update events' timestamps.
    max_interval:
        The required maximum time between consecutive updates, or ``None`` to learn it from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit].
    method:
        How the interval is derived from the clean data. Currently ``"max"`` (the largest inter-event gap
        observed in the clean stream), which is the default.
    """

    iso_5259_id = None
    iso_25024_id = "Cur-I-1"
    reference_params = ("max_interval",)

    max_interval_: timedelta

    def __init__(
        self,
        column: str,
        max_interval: timedelta | None = None,
        method: Literal["max"] = "max",
    ) -> None:
        super().__init__(column=column)
        self.max_interval = max_interval
        self.method = method

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, temporal=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "max":
            raise ValueError(f"Unsupported method: {self.method!r}")
        gaps = (
            frame.select(nw.col(self.column))
            .drop_nulls()
            .sort(self.column)
            .select(nw.col(self.column).diff().alias(self.column))[self.column]
            .drop_nulls()
        )
        if len(gaps) == 0:
            raise ValueError(
                f"{type(self).__name__}: needs at least two non-null timestamps in the clean data to learn max_interval"
            )
        return {"max_interval": gaps.max()}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # The frame is reduced to the measured column, so the helper names can only clash with it.
        index, gap = "_dqm_index", "_dqm_gap"
        while self.column in (index, gap):
            index, gap = index + "_", gap + "_"
        # Order by timestamp to take gaps, then restore the input's row order via the row index. The index
        # tiebreak makes the choice of "first event" among duplicate timestamps deterministic (neither
        # backend guarantees a stable sort), and nulls_last keeps null timestamps from corrupting real gaps:
        # their diffs, and the diff of whatever follows them, stay null.
        gaps = (
            frame.select(nw.col(self.column))
            .with_row_index(index)
            .sort([self.column, index], nulls_last=True)
            .with_columns(nw.col(self.column).diff().alias(gap))
            .sort(index)
        )
        # Null-preserving float, as in DataAccuracyRange; the guard is mandatory because on the pandas
        # backend a duration comparison against a missing gap yields False rather than null.
        expr = (
            nw.when(~nw.col(gap).is_null())
            .then((nw.col(gap) <= self.max_interval_).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return gaps.select(expr)[self.column]
