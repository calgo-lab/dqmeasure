from __future__ import annotations

import json
import operator
import warnings
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure
from dqmeasure.measures._llm import openai_completion, render_record

_ORDER_OPS: tuple[tuple[str, Callable[[Any, Any], Any]], ...] = (
    ("<", operator.lt),
    ("<=", operator.le),
    (">", operator.gt),
    (">=", operator.ge),
    ("==", operator.eq),
    ("!=", operator.ne),
)
_EQUALITY_OPS: tuple[tuple[str, Callable[[Any, Any], Any]], ...] = (("==", operator.eq), ("!=", operator.ne))

# A predicate that holds implies these also hold; the implied ones are pruned from the mined set.
_IMPLIES = {"<": ("<=", "!="), ">": (">=", "!="), "==": ("<=", ">=")}


def _pair_kind(subject: Any, context: Any) -> Literal["numeric", "temporal", "equality"] | None:
    """Which predicate family a subject/context dtype pair admits: order predicates, equality only, or none."""
    if subject.is_numeric() and context.is_numeric():
        return "numeric"
    if isinstance(subject, nw.Datetime) and isinstance(context, nw.Datetime):
        return "temporal"
    if isinstance(subject, nw.Date) and isinstance(context, nw.Date):
        return "temporal"
    if type(subject) is type(context) and isinstance(subject, (nw.String, nw.Boolean, nw.Categorical, nw.Enum)):
        return "equality"
    return None


def _guarded(predicate: nw.Expr, columns: Sequence[str]) -> nw.Expr:
    """Scope a predicate to rows where every involved column is set (null condition = unit out of scope).

    Comparisons propagate nulls under Polars but collapse to False under pandas' numpy semantics; the
    explicit guard gives mined predicates the same scope on every backend.
    """
    scope = ~nw.col(columns[0]).is_null()
    for name in columns[1:]:
        scope = scope & ~nw.col(name).is_null()
    return nw.when(scope).then(predicate).otherwise(nw.lit(None))


def _difference(column: str, context: str, kind: str) -> nw.Expr:
    """The difference between column and context; timestamps are compared in microseconds."""
    if kind == "numeric":
        return nw.col(column) - nw.col(context)
    return nw.col(column).dt.timestamp("us") - nw.col(context).dt.timestamp("us")


def _describe_threshold(threshold: float, kind: str) -> str:
    return str(timedelta(microseconds=threshold)) if kind == "temporal" else str(threshold)


class SemanticConsistency(PositionalMeasure):
    """ISO/IEC 25024 `Con-I-6` "Semantic consistency" (`Con-ML-4` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column. The measure is
    scoped to the column whose values the rules constrain, and each rule reads the rest of the row as
    context. The standard's example rule "recruitment date must be after birth date + 16 years" is a
    rule for the recruitment-date column with the birth-date column as context. 95 of 100 records satisfying
    it gives ``X = 95/100``.

    The reference is the semantic rule set. It is either specified as a narwhals boolean
    expressions in the constructor, e.g. ``nw.col("recruited") > nw.col("born")``, or learned from clean
    data at [`fit`][dqmeasure.base.BaseMeasure.fit]:

    * ``method="dc"`` (default) mines approximate single-tuple denial constraints: one row decides each one,
      and ``confidence`` sets the share of evaluable clean rows it must hold on. Candidates come from the
      predicate space of denial-constraint discovery (Chu et al., PVLDB 2013), where the atom
      ``column op context`` is a *predicate*: an order predicate (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``)
      when the subject and context dtypes are numeric or temporal, an equality predicate (``==``, ``!=``)
      otherwise. Candidates are mined per context column with a comparable dtype, tested on the clean data,
      and kept in their satisfying form (``recruitment > birth``, not ``¬(recruitment <= birth)``).

      Order predicates on numeric or temporal columns are replaced by a difference bound: ``recruitment > birth``
      becomes ``recruitment - birth >= 16 years``, the threshold being the smallest clean difference observed on
      columns within ``confidence``. A threshold of zero or less is dropped.
    * ``method="llm"`` asks a language model at fit time to propose rules for the column from the
      schema and sampled clean records, as expression strings over ``col(...)``. Each proposal is compiled
      in a restricted namespace and validated on the clean data. Proposals that fail to evaluate, are not
      boolean, or hold on a smaller share of the evaluable rows than ``confidence`` are discarded.

    Mined rules are stored in ``rules_`` with human-readable forms in ``rule_descriptions_``, so the fitted
    reference stays inspectable. Per row, the condition is 1 if every evaluable rule holds, 0 if any
    evaluable rule fails, and null when no rule is evaluable (e.g. the context is null). Mined rules are
    explicitly null-guarded on every backend (pandas and polars).

    Parameters
    ----------
    column:
        The column the measure applies to. Rules constrain this column's values.
    rules:
        The semantic rules as narwhals boolean expressions, or ``None`` to mine them from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit].
    method:
        How rules are mined at fit: ``"dc"`` (approximate single-tuple denial constraints) or ``"llm"``
        (language-model proposals validated on the clean data).
    confidence:
        Fraction between 0 and 1 of the evaluable clean rows a mined rule must satisfy to not be discarded.
    llm_model, llm_url:
        LLM method only. Model name and base URL (up to and including ``/v1``) of an OpenAI-compatible
        chat-completions endpoint.
    llm_api_key:
        LLM method only. Bearer token for the endpoint. ``None`` (default) falls back to the
        ``OPENAI_API_KEY`` environment variable; local servers typically need none.
    n_examples:
        LLM method only. Number of clean records sampled at fit time as examples in the prompt.
    random_state:
        LLM method only. Seed for the example sampling, making the measurement procedure reproducible.
    """

    iso_5259_id = "Con-ML-4"
    iso_25024_id = "Con-I-6"
    reference_params = ("rules",)

    rules_: Sequence[nw.Expr]
    rule_descriptions_: list[str]
    """Human-readable forms of the mined rules; set when ``fit`` mined the reference."""

    def __init__(
        self,
        column: str,
        rules: Sequence[nw.Expr] | None = None,
        method: Literal["dc", "llm"] = "dc",
        confidence: float = 0.99,
        llm_model: str = "llama3.2:3b",
        llm_url: str = "http://localhost:11434/v1",
        llm_api_key: str | None = None,
        n_examples: int = 20,
        random_state: int = 0,
    ) -> None:
        super().__init__(column=column)
        self.rules = rules
        self.method = method
        self.confidence = confidence
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.n_examples = n_examples
        self.random_state = random_state

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method not in ("dc", "llm"):
            raise ValueError(f"Unsupported method: {self.method!r}")
        mined = self._mine_dc(frame) if self.method == "dc" else self._mine_llm(frame)
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

    # -- rule evaluation on clean data --------------------------------------------------

    def _confidence_on(self, frame: nw.DataFrame[Any], rule: nw.Expr) -> float | None:
        """Fraction of evaluable clean rows satisfying the rule; None if it is broken or never evaluable."""
        try:
            evaluated = frame.select(rule.alias("rule"))["rule"].to_list()
        except Exception:  # a candidate that does not evaluate is simply not a rule
            return None
        values = [v for v in evaluated if v is not None and v == v]
        if not values or any(bool(v) != v for v in values):  # non-boolean output: not a predicate
            return None
        return sum(1 for v in values if v) / len(values)

    # -- method="dc": approximate single-tuple denial-constraint mining ------------------

    def _mine_dc(self, frame: nw.DataFrame[Any]) -> list[tuple[str, nw.Expr]]:
        schema = frame.schema
        rules: list[tuple[str, nw.Expr]] = []
        for context in frame.columns:
            if context == self.column:
                continue
            kind = _pair_kind(schema[self.column], schema[context])
            if kind is None:
                continue
            operators = _ORDER_OPS if kind in ("numeric", "temporal") else _EQUALITY_OPS
            survivors: dict[str, nw.Expr] = {}
            for symbol, op in operators:
                predicate = _guarded(op(nw.col(self.column), nw.col(context)), [self.column, context])
                confidence = self._confidence_on(frame, predicate)
                if confidence is not None and confidence >= self.confidence:
                    survivors[symbol] = predicate
            for symbol in list(survivors):
                if symbol in survivors:
                    for implied in _IMPLIES.get(symbol, ()):
                        survivors.pop(implied, None)
            bounds = self._difference_bounds(frame, context, kind, survivors) if kind != "equality" else []
            rules.extend((f"{self.column} {symbol} {context}", expr) for symbol, expr in survivors.items())
            rules.extend(bounds)
        return rules

    def _difference_bounds(
        self, frame: nw.DataFrame[Any], context: str, kind: str, survivors: dict[str, nw.Expr]
    ) -> list[tuple[str, nw.Expr]]:
        """Tighten a surviving order predicate into a difference bound, when strictly stronger.

        ``column > context`` becomes ``column - context ≥ t`` with ``t`` the share of clean data within the
        ``confidence`` threshold.
        """
        if not any(symbol in survivors for symbol in ("<", "<=", ">", ">=")):
            return []
        difference = _difference(self.column, context, kind)
        diffs = sorted(v for v in frame.select(difference.alias("d"))["d"].drop_nulls().to_list() if v == v)
        if not diffs:
            return []
        k = int((1 - self.confidence) * len(diffs))
        bounds: list[tuple[str, nw.Expr]] = []
        if (">" in survivors or ">=" in survivors) and diffs[k] > 0:
            threshold = diffs[k]
            survivors.pop(">", None)
            survivors.pop(">=", None)
            rule = _guarded(difference >= nw.lit(threshold), [self.column, context])
            bounds.append((f"{self.column} - {context} >= {_describe_threshold(threshold, kind)}", rule))
        if ("<" in survivors or "<=" in survivors) and diffs[len(diffs) - 1 - k] < 0:
            threshold = diffs[len(diffs) - 1 - k]
            survivors.pop("<", None)
            survivors.pop("<=", None)
            rule = _guarded(difference <= nw.lit(threshold), [self.column, context])
            bounds.append((f"{self.column} - {context} <= {_describe_threshold(threshold, kind)}", rule))
        return bounds

    # -- method="llm": language-model rule proposals --------------------------------------

    def _mine_llm(self, frame: nw.DataFrame[Any]) -> list[tuple[str, nw.Expr]]:
        llm = openai_completion(self.llm_model, self.llm_url, self.llm_api_key)
        response = llm(self._prompt(frame))
        rules: list[tuple[str, nw.Expr]] = []
        seen: set[str] = set()
        for text in _parse_proposals(response):
            if text in seen:
                continue
            seen.add(text)
            expr = _compile_rule(text)
            if expr is None:
                continue
            confidence = self._confidence_on(frame, expr)
            if confidence is not None and confidence >= self.confidence:
                rules.append((text, expr))
        return rules

    def _prompt(self, frame: nw.DataFrame[Any]) -> str:
        k = min(self.n_examples, len(frame))
        examples = "\n".join(
            render_record(row) for row in frame.sample(n=k, seed=self.random_state).iter_rows(named=True)
        )
        schema = ", ".join(f"{name} ({dtype})" for name, dtype in frame.schema.items())
        return (
            "You propose semantic consistency rules for one column of a table: conditions every correct "
            "record must satisfy, relating that column to the other columns.\n"
            "Answer with a JSON array of rule strings and nothing else. Each rule is a boolean expression "
            "built from col('name'), the comparison operators <, <=, >, >=, ==, !=, the arithmetic "
            "operators +, -, *, /, and numeric or string constants.\n\n"
            f"Table schema: {schema}\n"
            f"Records from the table that are known to be correct:\n{examples}\n\n"
            f"Rules constraining the column '{self.column}':"
        )


def _parse_proposals(response: str) -> list[str]:
    """The rule strings in the LLM response's JSON array; unparseable responses propose nothing."""
    start, end = response.find("["), response.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        proposals = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [proposal for proposal in proposals if isinstance(proposal, str)]


def _compile_rule(text: str) -> nw.Expr | None:
    """Compile a proposed rule string in a namespace of col/lit only; broken proposals compile to None."""
    try:
        expr = eval(text, {"__builtins__": {}}, {"col": nw.col, "lit": nw.lit})
    except Exception:  # any failure disqualifies the proposal, whatever its kind
        return None
    return expr if isinstance(expr, nw.Expr) else None
