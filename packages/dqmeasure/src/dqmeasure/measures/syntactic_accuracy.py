from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class SyntacticDataAccuracy(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-1` "Syntactic data accuracy" (`Acc-ML-1` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column.

    A value is syntactically accurate when it equals a member of the column's domain. "The same as one from an
    identified source of validated information" (ISO/IEC 25024, Table 1, note 1). The clean data acts as that
    source: [`fit`][dqmeasure.base.BaseMeasure.fit] learns the domain as the set of distinct non-null values it
    observes.

    Checking values against the column's *data type* (the ISO/IEC 5259-2 reading of syntactic correctness) is
    deliberately not part of this measure: a typed dataframe already enforces its schema on load, and format
    conformance is a measure of its own (`Con-I-2` data format consistency).

    Parameters
    ----------
    column:
        The column the measure applies to. Typically categorical-like, but any dtype works.
    domain:
        The admissible values, or ``None`` to learn the domain from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit].
    method:
        How the domain is derived from the clean data. Currently ``"observed"`` (the set of distinct non-null
        values), which is the default.
    """

    iso_5259_id = "Acc-ML-1"
    iso_25024_id = "Acc-I-1"
    reference_params = ("domain",)

    domain_: Collection[Any]

    def __init__(
        self,
        column: str,
        domain: Collection[Any] | None = None,
        method: Literal["observed"] = "observed",
    ) -> None:
        super().__init__(column=column)
        self.domain = domain
        self.method = method

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "observed":
            raise ValueError(f"Unsupported method: {self.method!r}")
        return {"domain": set(frame[self.column].drop_nulls().unique().to_list())}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell condition: 1.0 if the value is a domain member, 0.0 if not, null if the value is missing (the
        # same null-preserving float encoding as DataAccuracyRange, and for the same backend-parity reasons).
        # Missing values are excluded from the measure: a null is incomplete, not syntactically inaccurate.
        expr = (
            nw.when(~nw.col(self.column).is_null())
            .then(nw.col(self.column).is_in(list(self.domain_)).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return frame.select(expr)[self.column]
