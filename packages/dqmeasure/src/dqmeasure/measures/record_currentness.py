from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column


class RecordCurrentness(PositionalMeasure):
    """ISO/IEC 5259-2 `Cur-ML-2` "Record currentness".

    Table measure, tier 1, positional: unit = record (row), subject = the whole table.

    The ratio of records where all data items fall within the required age range. The measure considers
    columns of datatype date or datetime to derive one or more ages per row. We call these columns "temporal"
    columns. For example, a tables' temporal columns could be `inserted_at` and `last_updated_at`, and they
    may have different required age ranges.

    This measure is the per-column analog of `Cur-ML-1` feature currentness.

    A record conforms when every one of its non-null temporal cells is of the right age. Null cells are ignored.
    The frame must have at least one temporal column, and every temporal column of the measured frame
    must be covered by the reference.

    The reference time is a measurement-time input: with ``reference_time=None`` it is the wall clock, read
    once per ``fit``/``predict``/``score`` call. Pin ``reference_time`` for reproducible results.

    Parameters
    ----------
    age_ranges:
        The required age range per temporal column, ``{column: (min_age, max_age)}``, or ``None`` to learn
        the ranges from the ages observed in the clean data at [`fit`][dqmeasure.base.BaseMeasure.fit].
    reference_time:
        The instant ages are computed against, or ``None`` for the current wall clock at each call.
    """

    iso_5259_id = "Cur-ML-2"
    iso_25024_id = None
    scope = "table"
    reference_params = ("age_ranges",)

    age_ranges_: dict[str, tuple[timedelta, timedelta]]

    def __init__(
        self,
        *,
        age_ranges: dict[str, tuple[timedelta, timedelta]] | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        self.age_ranges = age_ranges
        self.reference_time = reference_time

    def _reference_time(self) -> datetime:
        return self.reference_time if self.reference_time is not None else datetime.now()

    @staticmethod
    def _temporal_columns(frame: nw.DataFrame[Any]) -> list[str]:
        # Not dtype.is_temporal(), which also admits Duration and Time — timestamps only.
        return [c for c, dtype in frame.schema.items() if isinstance(dtype, (nw.Datetime, nw.Date))]

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        if not self._temporal_columns(frame):
            raise ValueError(f"{type(self).__name__} needs a frame with at least one datetime or date column")

    def _check_coverage(self, frame: nw.DataFrame[Any]) -> None:
        # Ensures all date and datetype columns are age checked.
        uncovered = [c for c in self._temporal_columns(frame) if c not in self.age_ranges_]
        if uncovered:
            raise ValueError(
                f"{type(self).__name__}: temporal columns {uncovered} are not covered by the reference "
                f"(covered: {sorted(self.age_ranges_)})"
                "If you want to exclude date or datetime columns from the evaluation, consider subsetting "
                "the frame or using Cur-ML-1 instead."
            )
        for column in self.age_ranges_:
            _require_column(frame, column, temporal=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        ref = self._reference_time()
        age_ranges: dict[str, tuple[timedelta, timedelta]] = {}
        for column in self._temporal_columns(frame):
            ages = frame.select((nw.lit(ref) - nw.col(column)).alias(column))[column].drop_nulls()
            if len(ages) == 0:
                raise ValueError(
                    f"{type(self).__name__}: column {column!r} has no non-null timestamps in the clean data"
                )
            age_ranges[column] = (ages.min(), ages.max())
        return {"age_ranges": age_ranges}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        self._check_coverage(frame)
        ref = self._reference_time()
        # Per temporal cell: in-range as null-preserving float, exactly as in FeatureCurrentness (the null
        # guard is mandatory on the pandas backend, where a duration comparison against a missing timestamp
        # yields False rather than null). Per record: 0.0 if one or more temporal columns are outside the age
        # range, 1.0 otherwise. A record is judged on its non-null timestamps and is out of scope only when
        # all of them are null.
        indicators = []
        for column, (min_age, max_age) in self.age_ranges_.items():
            age = nw.lit(ref) - nw.col(column)
            indicators.append(
                nw.when(~nw.col(column).is_null())
                .then(((age >= min_age) & (age <= max_age)).cast(nw.Float64))
                .otherwise(nw.lit(None))
                .alias(column)
            )
        expr = nw.min_horizontal(*indicators)
        return frame.select(expr.alias("record"))["record"]
