from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class TimelinessOfUpdate(PositionalMeasure):
    """ISO/IEC 25024 `Cur-I-2` "Timeliness of update".

    Column measure, tier 1, positional: unit = row (a data item needing updating), subject = the
    update-timestamp column. ``due_column`` names when each update was due or requested; it
    is context, not scope (scope and context are independent). A row is in scope iff its due time is set, so
    ``B`` counts the items needing updating; the condition checks that the update landed within ``sla`` of
    the due time. A needed update that never happened (null update timestamp with a due time set) is a
    failure, not out of scope.

    An update before its due time has a negative delay and is always timely. Time-zone-naive and -aware
    columns both work as long as the two columns are consistent with each other.

    Parameters
    ----------
    column:
        The datetime column holding when each item was actually updated.
    due_column:
        The datetime column holding when each item's update was due; null means no update was needed.
    sla:
        The allowed delay between due time and update, or ``None`` to learn it from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit].
    method:
        How the SLA is derived from the clean data. Currently ``"max"`` (the worst delay observed in rows
        with both timestamps set), which is the default.
    """

    iso_5259_id = None
    iso_25024_id = "Cur-I-2"
    reference_params = ("sla",)

    sla_: timedelta

    def __init__(
        self,
        column: str,
        due_column: str,
        sla: timedelta | None = None,
        method: Literal["max"] = "max",
    ) -> None:
        super().__init__(column=column)
        self.due_column = due_column
        self.sla = sla
        self.method = method

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, temporal=True)
        _require_column(frame, self.due_column, temporal=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "max":
            raise ValueError(f"Unsupported method: {self.method!r}")
        # Datetime subtraction null-propagates, so this restricts to rows with both timestamps set.
        delays = frame.select((nw.col(self.column) - nw.col(self.due_column)).alias(self.column))[
            self.column
        ].drop_nulls()
        if len(delays) == 0:
            raise ValueError(
                f"{type(self).__name__}: no rows with both {self.column!r} and {self.due_column!r} set in "
                "the clean data"
            )
        return {"sla": delays.max()}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        updated, due = nw.col(self.column), nw.col(self.due_column)
        # Null-preserving float, as in DataAccuracyRange; the explicit guards are mandatory because on the
        # pandas backend a duration comparison against a missing timestamp yields False rather than null.
        timely = nw.when(updated.is_null()).then(nw.lit(0.0)).otherwise(((updated - due) <= self.sla_).cast(nw.Float64))
        expr = nw.when(~due.is_null()).then(timely).otherwise(nw.lit(None)).alias(self.column)
        return frame.select(expr)[self.column]
