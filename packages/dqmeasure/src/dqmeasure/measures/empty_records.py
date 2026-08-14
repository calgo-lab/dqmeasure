from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class EmptyRecords(PositionalMeasure):
    """ISO/IEC 25024 `Com-I-5` "Empty records in a data file".

    Table measure, tier 1, positional: unit = record (row), subject = the whole table.

    Counts records where all data items are empty ``A`` over all records ``B`` and reports
    ``X = 1 - A/B``, the fraction of records that carry any data ("records exist but are empty").
    [`predict`][dqmeasure.base.PositionalMeasure.predict] reports ``1.0`` for a record with at least one
    non-null cell, and its mean is exactly the standard's ``1 - A/B``. The output carries no nulls. There
    is no reference to learn and the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = None
    iso_25024_id = "Com-I-5"
    scope = "table"

    def __init__(self) -> None:
        pass

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per row, returns 1.0 if at least one cell of the row is non-null, 0.0 if the record is empty.
        expr = nw.max_horizontal(*((~nw.col(c).is_null()).cast(nw.Float64) for c in frame.columns))
        return frame.select(expr.alias("record"))["record"]
