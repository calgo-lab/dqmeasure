from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class DataRecordConsistency(PositionalMeasure):
    """ISO/IEC 5259-2 `Con-ML-1` "Data record consistency".

    Table measure, tier 1, positional: unit = record (row), subject = the whole table.

    The ratio of records that occur exactly once in the dataset. The standard defines `Con-ML-1` as the
    ratio of *duplicate* records. We report ``1 - X`` to keep every measure higher-is-better. A record
    counts as a duplicate when the full row occurs more than once. This measure is the row-level case of
    [`RiskOfDataInconsistency`][dqmeasure.measures.inconsistency_risk.RiskOfDataInconsistency]. Records
    containing one or more null values are out of scope, because two nulls are not duplicates of each other.

    There is no reference to learn, so the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Con-ML-1"
    iso_25024_id = None
    scope = "table"

    def __init__(self) -> None:
        pass

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per row, returns 1.0 if the full row is unique, 0.0 if it occurs more than once, and null if the
        # row contains a null (out of scope).
        unique = (~frame.is_duplicated()).cast(nw.Float64).alias("record")
        no_null = frame.select(
            nw.min_horizontal(*((~nw.col(c).is_null()).cast(nw.Float64) for c in frame.columns)).alias("record")
        )["record"]
        nulls = nw.new_series("record", [None] * len(frame), dtype=nw.Float64, backend=frame.implementation)
        return unique.zip_with(no_null == 1.0, nulls)
