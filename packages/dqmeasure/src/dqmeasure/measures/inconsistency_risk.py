from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class RiskOfDataInconsistency(PositionalMeasure):
    """ISO/IEC 25024 `Con-I-3` "Risk of data inconsistency".

    Column measure, tier 1, positional: unit = cell, subject = the column.

    A cell counts as a duplication when its value occurs more than once in the column. The standard defines
    `Con-I-3` as the risk of inconsistency, i.e. the ratio of duplicated cells. We report ``1 - X`` to keep
    every measure higher-is-better: ``X`` is the ratio of cells holding a value unique in the column.
    Nulls are out of scope, as two nulls are not duplicates of each other.

    This measure concerns duplicate values in one column. The table-scoped
    [`DataRecordConsistency`][dqmeasure.measures.record_consistency.DataRecordConsistency] addresses entire rows.

    There is no reference to learn, and the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].
    The user should apply this measure to columns where a repeated value signals redundant storage (identifiers,
    names of entities stored once), not where repetition is natural (categories).

    Parameters
    ----------
    column:
        The column the measure applies to; any dtype works.
    """

    iso_5259_id = None
    iso_25024_id = "Con-I-3"

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # We need two helper columns, this ensures that names don't collide with actual column names
        index, count = "_dqm_index", "_dqm_count"
        if self.column in (index, count):
            index, count = index + "_", count + "_"

        # Use a group_by to get value counts and restore order through the index column.
        counts = frame.select(nw.col(self.column)).group_by(self.column).agg(nw.len().alias(count))
        counted = (
            frame.select(nw.col(self.column)).with_row_index(index).join(counts, on=self.column, how="left").sort(index)
        )

        # Per-cell: 1.0 if the value is unique in the column, 0.0 if it occurs more than once, null
        # if the value is missing (out of scope).
        expr = (
            nw.when(~nw.col(self.column).is_null())
            .then((nw.col(count) == 1).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return counted.select(expr)[self.column]
