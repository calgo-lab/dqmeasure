from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class TimelinessOfDataItems(PositionalMeasure):
    """ISO/IEC 5259-2 `Tml-ML-1` "Timeliness of data items".

    Column measure, tier 1, positional: unit = row (a data item), subject = the
    availability-timestamp column. The standard defines timeliness as the latency
    between the time a phenomenon occurs and the time the data recorded for it becomes available for
    use — as opposed to currentness (`Cur-ML-1`), the age of recorded data relative to its use.
    ``event_column`` names when each phenomenon occurred; it is context, not scope. A row is in
    scope iff its event time is set; the condition checks that the data became available within
    ``max_latency`` of the event. Data that never became available (null availability timestamp
    with an event time set) is a failure, not out of scope.

    Data available before its event has a negative latency and is always timely. Time-zone-naive
    and -aware columns both work as long as the two columns are consistent with each other.

    `Tml-ML-1` coincides numerically with `Cur-I-2` timeliness of update under a relabeling of the
    columns; the measures stay distinct in the role of the columns: `Cur-I-2` counts items *needing
    updating* against a due time, `Tml-ML-1` counts every data item against its event time.

    Parameters
    ----------
    column:
        The datetime column holding when each data item became available (was recorded).
    event_column:
        The datetime column holding when the phenomenon each data item records occurred.
    max_latency:
        The allowed latency between event and availability, or ``None`` to learn it from the clean
        data at [`fit`][dqmeasure.base.BaseMeasure.fit].
    method:
        How the latency requirement is derived from the clean data. Currently ``"max"`` (the worst
        latency observed in rows with both timestamps set), which is the default.
    """

    iso_5259_id = "Tml-ML-1"
    iso_25024_id = None
    reference_params = ("max_latency",)

    max_latency_: timedelta

    def __init__(
        self,
        column: str,
        event_column: str,
        max_latency: timedelta | None = None,
        method: Literal["max"] = "max",
    ) -> None:
        super().__init__(column=column)
        self.event_column = event_column
        self.max_latency = max_latency
        self.method = method

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, temporal=True)
        _require_column(frame, self.event_column, temporal=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "max":
            raise ValueError(f"Unsupported method: {self.method!r}")
        # Datetime subtraction null-propagates, so this restricts to rows with both timestamps set.
        latencies = frame.select((nw.col(self.column) - nw.col(self.event_column)).alias(self.column))[
            self.column
        ].drop_nulls()
        if len(latencies) == 0:
            raise ValueError(
                f"{type(self).__name__}: no rows with both {self.column!r} and {self.event_column!r} set in "
                "the clean data"
            )
        return {"max_latency": latencies.max()}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        available, event = nw.col(self.column), nw.col(self.event_column)
        # Null-preserving float, as in DataAccuracyRange; the explicit guards are mandatory because on the
        # pandas backend a duration comparison against a missing timestamp yields False rather than null.
        timely = (
            nw.when(available.is_null())
            .then(nw.lit(0.0))
            .otherwise(((available - event) <= self.max_latency_).cast(nw.Float64))
        )
        expr = nw.when(~event.is_null()).then(timely).otherwise(nw.lit(None)).alias(self.column)
        return frame.select(expr)[self.column]
