"""Data record consistency from ISO/IEC 5259-2

*"The ratio of duplicate records in the dataset"*
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import TABLE_SUBJECT, PositionalMeasure

# These build the composite record key. Fill nulls before concatenating so two null-keyed records match. concat_str
# would otherwise yield null. The separator keeps ("a", "bc") distinct from ("ab", "c").
_NULL_SENTINEL = "\x00<null>"
_SEPARATOR = "\x1f"


class DataRecordConsistency(PositionalMeasure):
    """ISO/IEC 5259-2 `Con-ML-1` "Data record consistency".

    Tier 1, positional: unit = row, subject = the whole table, lower is better.

    The standard leaves "number of duplicate records" ambiguous. This implementation counts the excess copies:
    a group of k identical records contributes ``k - 1`` to ``A``, so an all-unique table scores 0 and a
    table of two identical records scores 0.5 (not 1).

    Detecting duplicates needs no reference from the clean data. [`fit`][dqmeasure.base.BaseMeasure.fit] only
    fixes the columns that define record identity, which standardizes the measurement procedure.

    Parameters
    ----------
    columns:
        Optional subset of columns that define record identity. ``None`` (default) means records must agree on
        **all** columns to count as duplicates.
    """

    iso_id = "Con-ML-1"
    orientation = "lower-is-better"

    def _select_columns(self, frame: nw.DataFrame[Any]) -> list[str]:
        if self.columns is not None:
            return list(self.columns)
        return list(frame.columns)

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        # Per-row condition: 1.0 if the row is an excess copy of an earlier identical row (compared on the key
        # columns), 0.0 otherwise. Rows are compared via a composite string key so multi-column identity works
        # uniformly across backends.
        key = nw.concat_str(
            (nw.col(name).cast(nw.String).fill_null(_NULL_SENTINEL) for name in self.columns_),
            separator=_SEPARATOR,
        )
        return frame.select((~key.is_first_distinct()).cast(nw.Float64).alias(TABLE_SUBJECT))
