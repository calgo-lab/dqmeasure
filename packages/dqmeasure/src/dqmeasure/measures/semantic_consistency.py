from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class SemanticConsistency(PositionalMeasure):
    """ISO/IEC 25024 `Con-I-6` "Semantic consistency" (`Con-ML-4` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column. The measure is
    scoped to the column whose values the rules constrain, and each rule reads the rest of the row as
    context.

    The semantic rules are narwhals boolean expressions. Per row, the condition is 1 if every evaluable rule holds,
    0 if any evaluable rule fails, and null when no rule is evaluable.

    The rules are either specified in the constructor, e.g.
    ``rules=[nw.col("recruited") > nw.col("born")]``, or learned from clean data at
    [`fit`][dqmeasure.base.BaseMeasure.fit]. The rules origin doesn't matter: the DQM evaluates the expressions
    and counts the rows that satisfy them.

    Parameters
    ----------
    column:
        The column the measure applies to. Rules constrain this column's values.
    rules:
        The semantic rules as narwhals boolean expressions, or ``None`` to mine them from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit].
    """

    iso_5259_id = "Con-ML-4"
    iso_25024_id = "Con-I-6"
    reference_params = ("rules",)

    rules_: Sequence[nw.Expr]
    rule_descriptions_: list[str]
    """Human-readable forms of the mined rules; set when ``fit`` mined the reference."""

    def __init__(self, column: str, rules: Sequence[nw.Expr] | None = None) -> None:
        super().__init__(column=column)
        self.rules = rules

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        mined = self._mine(frame)
        if not mined:
            warnings.warn(
                f"{type(self).__name__}: no rule survived mining on the clean data; score() will return NaN.",
                stacklevel=2,
            )
        self.rule_descriptions_ = [description for description, _ in mined]
        return {"rules": [expr for _, expr in mined]}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        data: list[float | None]
        if not self.rules_:
            data = [None] * len(frame)
        else:
            results = frame.select([rule.alias(f"rule_{i}") for i, rule in enumerate(self.rules_)])
            columns = [results[f"rule_{i}"].to_list() for i in range(len(self.rules_))]
            data = []
            for row in zip(*columns, strict=True):
                evaluable = [v for v in row if v is not None and v == v]
                data.append(None if not evaluable else float(all(bool(v) for v in evaluable)))
        return nw.new_series(self.column, data, nw.Float64, backend=frame.implementation)

    def _mine(self, frame: nw.DataFrame[Any]) -> list[tuple[str, nw.Expr]]:
        """One rule per column that determines this one on the clean data."""
        rules: list[tuple[str, nw.Expr]] = []
        for context in frame.columns:
            if context == self.column:
                continue
            pairs = frame.select(context, self.column).drop_nulls().unique()
            keys, values = pairs[context].to_list(), pairs[self.column].to_list()
            # A key with two different values determines nothing.
            if not keys or len(set(keys)) < len(keys):
                continue
            # when/then evaluates its branch over every row, so unseen keys reach replace_strict regardless.
            expected = nw.col(context).replace_strict(
                keys, values, default=None, return_dtype=frame.schema[self.column]
            )
            # Null cells and unseen keys are out of scope, spelled out because pandas reads a null
            # comparison as False where Polars keeps it null.
            scope = ~nw.col(self.column).is_null() & ~nw.col(context).is_null() & nw.col(context).is_in(keys)
            rule = nw.when(scope).then(nw.col(self.column) == expected).otherwise(nw.lit(None))
            rules.append((f"{context} -> {self.column}", rule))
        return rules
