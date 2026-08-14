from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class RecordCompleteness(PositionalMeasure):
    """ISO/IEC 5259-2 `Com-ML-4` "Record completeness".

    Table measure, tier 1, positional: unit = record (row), subject = the whole table.

    The ratio of rows that have no empty cell over all rows. nulls are the thing
    being measured, and a row with null(s) scores ``0.0``. There is no reference to learn, the
    measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Com-ML-4"
    iso_25024_id = None
    scope = "table"

    def __init__(self) -> None:
        pass

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-record condition: 1.0 if no cell of the row is null, else 0.0. Every record is in scope, the
        # output carries no nulls and score() divides by the row count.
        expr = nw.min_horizontal(*((~nw.col(c).is_null()).cast(nw.Float64) for c in frame.columns))
        return frame.select(expr.alias("record"))["record"]
