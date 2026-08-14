from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class FeatureCompleteness(PositionalMeasure):
    """ISO/IEC 25024 `Com-I-2` "Attribute completeness" (`Com-ML-3` "Feature completeness" in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column.

    The ratio of non-null values in the column. Every cell is a unit, and nulls are what's being measured.
    There is no reference to learn, so the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].

    The table-wide `Com-ML-1` "Value completeness" is the table-scoped
    [`ValueCompleteness`][dqmeasure.measures.value_completeness.ValueCompleteness]; since every column
    contributes the same ``B``, its value equals
    ``mean(FeatureCompleteness(c).score(df) for c in df.columns)``.

    Parameters
    ----------
    column:
        The column the measure applies to; any dtype works.
    """

    iso_5259_id = "Com-ML-3"
    iso_25024_id = "Com-I-2"

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell condition: 1.0 if the value is present, 0.0 if null. Every cell is in scope, so the output
        # carries no nulls and score() divides by the row count.
        expr = (~nw.col(self.column).is_null()).cast(nw.Float64).alias(self.column)
        return frame.select(expr)[self.column]
