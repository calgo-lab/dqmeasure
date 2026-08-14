from __future__ import annotations

from collections.abc import Collection
from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


def _shape(value: str) -> str:
    """A value's format shape: every digit becomes ``d``, every letter ``a``, other characters stay literal."""
    return "".join("d" if ch.isdigit() else "a" if ch.isalpha() else ch for ch in value)


class DataFormatConsistency(PositionalMeasure):
    """ISO/IEC 25024 `Con-I-2` "Data format consistency" (`Con-ML-3` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column.

    A value is format-consistent when its shape is one of the column's admissible format shapes. Here,
    a shape is a string where ``d`` indicates a digit, ``a`` a letter, and every other character is kept
    literally. For example, ``"202401" -> "dddddd"``, ``"2024-01" -> "dddd-dd"``.

    The measure applies to string-encoded columns only.

    Parameters
    ----------
    column:
        The column the measure applies to (string, categorical, or enum).
    formats:
        The admissible format shapes, written in the shape alphabet (e.g. ``{"dddd-dd"}``), or ``None`` to
        learn the set of shapes observed in the clean data at [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Con-ML-3"
    iso_25024_id = "Con-I-2"
    reference_params = ("formats",)

    formats_: Collection[str]

    def __init__(self, column: str, formats: Collection[str] | None = None) -> None:
        super().__init__(column=column)
        self.formats = formats

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        super()._validate(frame)
        dtype = frame.schema[self.column]
        if not isinstance(dtype, (nw.String, nw.Categorical, nw.Enum)):
            raise ValueError(
                f"Column {self.column!r} has dtype {dtype}; format consistency applies to string-encoded "
                "columns only (string, categorical, or enum). Typed columns have their format enforced by the schema"
            )

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        values = frame[self.column].drop_nulls().to_list()
        return {"formats": {_shape(value) for value in values}}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell condition: 1.0 if the value's shape is admissible, 0.0 if not, null if the value is missing
        # (out of scope, a null has no format). We use the same to_list/new_series bridge as SemanticDataAccuracy
        # to enable same behavior in pandas and polars.
        formats = set(self.formats_)
        nulls = frame[self.column].is_null().to_list()
        values = frame[self.column].to_list()
        data = [None if nulls[i] else float(_shape(values[i]) in formats) for i in range(len(frame))]
        return nw.new_series(self.column, data, nw.Float64, backend=frame.implementation)
