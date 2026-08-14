from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class FeatureCurrentness(PositionalMeasure):
    """ISO/IEC 5259-2 `Cur-ML-1` "Feature currentness".

    Column measure, tier 1, positional: unit = cell, subject = the column. The column holds
    each data item's timestamp indicating when it was last updated. A column's currentness is measured through
    the timestamp column that dates it. The condition checks that the cell's age, defined as reference time minus
    the timestamp, lies within the required age range. ``A`` counts the items of the right age and ``B`` the
    non-null timestamps.

    The reference time is a measurement-time input, not part of the reference. With
    ``reference_time=None`` it is the wall clock, read once per ``fit``/``predict``/``score`` call, so the
    acceptable timestamp window moves with time. Pin ``reference_time`` for reproducible results. Time-zone-naive
    and -aware columns both work as long as the column and the reference time are consistent.  Note that the
    ``datetime.now()`` default is naive.

    The ISO standard calls for a date range, which includes `min_age` and `max_age`. That's somewhat unintuitive,
    a learned ``min_age`` makes data *fresher* than any clean item fail the range. To measure only a freshness ceiling,
    specify ``min_age=timedelta(0)``.

    Parameters
    ----------
    column:
        The datetime column the measure applies to.
    min_age, max_age:
        The required age range, or ``None`` to learn the bounds from the ages observed in the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit]. Specify both to skip ``fit`` entirely. Specify ``min_age=timedelta(0)``
        to not punish data that's fresher than the reference data's freshest row.
    reference_time:
        The instant ages are computed against, or ``None`` for the current wall clock at each call.
    """

    iso_5259_id = "Cur-ML-1"
    iso_25024_id = None
    reference_params = ("min_age", "max_age")

    min_age_: timedelta
    max_age_: timedelta

    def __init__(
        self,
        column: str,
        min_age: timedelta | None = None,
        max_age: timedelta | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        super().__init__(column=column)
        self.min_age = min_age
        self.max_age = max_age
        self.reference_time = reference_time

    def _reference_time(self) -> datetime:
        return self.reference_time if self.reference_time is not None else datetime.now()

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, temporal=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        ref = self._reference_time()
        ages = frame.select((nw.lit(ref) - nw.col(self.column)).alias(self.column))[self.column].drop_nulls()
        if len(ages) == 0:
            raise ValueError(
                f"{type(self).__name__}: column {self.column!r} has no non-null timestamps in the clean data"
            )
        return {"min_age": ages.min(), "max_age": ages.max()}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Null-preserving float, as in DataAccuracyRange. The explicit null guard is required: In pandas,
        # a duration comparison against a missing timestamp returns False rather than null,
        # which messes up the measure.
        age = nw.lit(self._reference_time()) - nw.col(self.column)
        expr = (
            nw.when(~nw.col(self.column).is_null())
            .then(((age >= self.min_age_) & (age <= self.max_age_)).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return frame.select(expr)[self.column]
