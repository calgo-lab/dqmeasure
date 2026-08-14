from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class ValueCompleteness(PositionalMeasure):
    """ISO/IEC 5259-2 `Com-ML-1` "Value completeness".

    Table measure, tier 1, positional: unit = record (row), subject = the whole table.

    The ratio of non-null cells over all cells of the table. We implement this by calculating the fraction of
    non-null cells per row, then averaging them over the entire table. That per-record fraction is itself another
    measure: [`predict`][dqmeasure.base.PositionalMeasure.predict] reports ISO/IEC 25024 `Com-I-1`
    "Record completeness" for each record.

    There is no reference to learn, so the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].

    The table-wide value is also the mean of the per-column `Com-ML-3` scores,
    ``mean(FeatureCompleteness(c).score(df) for c in df.columns)``, since every column contributes the same
    ``B``.
    """

    iso_5259_id = "Com-ML-1"
    iso_25024_id = None
    scope = "table"

    def __init__(self) -> None:
        pass

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-row: the fraction of non-null cells in the row.
        expr = nw.mean_horizontal(*((~nw.col(c).is_null()).cast(nw.Float64) for c in frame.columns))
        return frame.select(expr.alias("record"))["record"]

    def _score(self, frame: nw.DataFrame[Any]) -> float:
        # Don't reuse _measure_units()' outputs, because rouding errors may compound.
        # Instead, calculate a and b directly.
        b = len(frame) * len(frame.columns)
        if b == 0:
            return float("nan")
        a = b - sum(frame[c].null_count() for c in frame.columns)
        return a / b
