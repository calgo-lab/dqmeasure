from __future__ import annotations

import inspect
from typing import Any, ClassVar, Literal, Self, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries

# what the measure's subject is: one named column, or the whole table
Scope = Literal["column", "table"]


class NotResolvedError(RuntimeError):
    """Raised when a measure is used before its reference is resolved (specified in the constructor or learned via
    [`fit`][dqmeasure.base.BaseMeasure.fit])."""


def _require_column(frame: nw.DataFrame[Any], column: str, *, numeric: bool = False, temporal: bool = False) -> None:
    """Validate that ``column`` exists in ``frame`` (and, optionally, is numeric or temporal)."""
    if column not in frame.columns:
        raise ValueError(f"Column {column!r} not found in the frame (columns: {frame.columns})")
    if numeric and not frame.schema[column].is_numeric():
        raise ValueError(f"Column {column!r} has dtype {frame.schema[column]}; this measure needs a numeric column")
    # Not dtype.is_temporal(), which also admits Duration and Time — timestamps only.
    if temporal and not isinstance(frame.schema[column], (nw.Datetime, nw.Date)):
        raise ValueError(
            f"Column {column!r} has dtype {frame.schema[column]}; this measure needs a datetime or date column"
        )


class BaseMeasure:
    """Data-quality measure base class.

    Every measure has one of two scopes, exposed as the ``scope`` class attribute: a **column** measure
    is constructed for exactly one column, named by the ``column`` constructor parameter. And a **table**
    measure applies to all columns of the frame, not requiring a ``column`` parameter. Either way
    [`score`][dqmeasure.base.BaseMeasure.score] yields exactly one quality measure value ``X``.

    A table-scoped subclass sets ``scope = "table"`` and defines its own ``__init__`` without a ``column``
    parameter.

    Subclasses set ``iso_5259_id`` and ``iso_25024_id`` (the measure's IDs in the two standards),
    and implement two hooks:

    * [`_fit_reference`][dqmeasure.base.BaseMeasure._fit_reference]: learn the reference from clean data.
    * [`_score`][dqmeasure.base.BaseMeasure._score]: compute the quality measure value ``X`` on a (dirty)
      dataframe.

    Measures whose units are positions in the dataframe should inherit from
    [`PositionalMeasure`][dqmeasure.base.PositionalMeasure] instead, which adds ``predict()`` and derives
    ``_score`` from it.

    Parameters
    ----------
    column:
        The column the measure applies to.
    """

    iso_5259_id: ClassVar[str | None]
    """The measure's ID in ISO/IEC 5259-2, or ``None`` if that standard has no counterpart."""

    iso_25024_id: ClassVar[str | None]
    """The measure's ID in ISO/IEC 25024, or ``None`` if that standard has no counterpart."""

    scope: ClassVar[Scope] = "column"
    """The measure's subject: one named column, or the whole table. Fixed by the ISO definition."""

    reference_params: ClassVar[tuple[str, ...]] = ()
    """Names of the constructor parameters that hold the measure's reference.

    Each may be specified in the constructor or left as ``None`` to be learned at
    [`fit`][dqmeasure.base.BaseMeasure.fit]. After resolution each appears as a fitted ``<name>_`` attribute.
    """

    def __init__(self, column: str) -> None:
        if self.scope != "column":
            raise TypeError(
                f"{type(self).__name__} is table-scoped and takes no column; "
                "table-scoped measures define their own __init__"
            )
        self.column = column

    def fit(self, X: IntoDataFrame) -> Self:
        """Learn the reference from a clean (training) dataframe.

        Sets one fitted ``<name>_`` attribute per reference parameter and returns ``self``. Parameters specified
        in the constructor are kept; only the rest is estimated from ``X``.
        """
        frame = nw.from_native(X, eager_only=True)
        self._validate(frame)
        self._resolve(frame)
        return self

    def score(self, X: IntoDataFrame) -> float:
        """Compute the quality measure value ``X`` for a (dirty) dataframe.

        Returns one value for the measure's subject (its column, or the whole table). Every measure is
        oriented so that **higher is better**. Where the standard defines ``X`` in the opposite direction,
        the measure reports ``1 - X``. When the subject has no units in scope (``B = 0``), the value is ``NaN``.
        """
        self._check_is_resolved()
        frame = nw.from_native(X, eager_only=True)
        self._validate(frame)
        return self._score(frame)

    # hooks for subclasses

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        """Check that ``frame`` supports this measure. Default: the column exists (column scope) or the frame
        has at least one column (table scope); subclasses may add dtype checks."""
        if self.scope == "column":
            _require_column(frame, self.column)
        elif not frame.columns:
            raise ValueError(f"{type(self).__name__} is table-scoped and needs a frame with at least one column")

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        """Estimate every reference parameter from clean data.

        Returns ``{param_name: value}``. The default has nothing to learn; measures with a non-empty
        ``reference_params`` override it.
        """
        return {}

    def _score(self, frame: nw.DataFrame[Any]) -> float:
        """Compute ``X`` for a (dirty) dataframe. Must be overridden."""
        raise NotImplementedError

    # helpers

    def _resolve(self, frame: nw.DataFrame[Any] | None) -> None:
        """Resolve the ``<name>_`` attributes from spec plus estimate.

        ``frame`` is the clean data when learning, or ``None`` when the reference is fully specified and no
        estimation is needed.
        """
        missing = [name for name in self.reference_params if getattr(self, name) is None]
        estimated: dict[str, Any] = {}
        if missing:
            if frame is None:
                raise NotResolvedError(
                    f"{type(self).__name__}: the reference is not fully specified ({', '.join(missing)}); "
                    "either give every parameter in the constructor or call fit() on clean data."
                )
            estimated = self._fit_reference(frame)
        for name in self.reference_params:
            value = getattr(self, name)
            setattr(self, f"{name}_", estimated[name] if value is None else value)
        self._resolved = True

    def _check_is_resolved(self) -> None:
        if not getattr(self, "_resolved", False):
            self._resolve(None)

    # minimal sklearn-style param protocol

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters, introspected from ``__init__``."""
        params: dict[str, Any] = {}
        for name, param in inspect.signature(type(self).__init__).parameters.items():
            if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


class PositionalMeasure(BaseMeasure):
    """Base class for tier-1 measures with positional units (cells or rows).

    A positional unit is a position in the dataframe you can point at and attach a score to. Such measures gain
    [`predict`][dqmeasure.base.PositionalMeasure.predict], and ``score()`` is its aggregation. Subclasses
    implement [`_measure_units`][dqmeasure.base.PositionalMeasure._measure_units] instead of ``_score``.

    Measures with non-positional units and tier-2 statistic measures have no per-unit output. They derive from
    [`BaseMeasure`][dqmeasure.base.BaseMeasure] directly and are ``score()``-only.
    """

    def predict(self, X: IntoDataFrame) -> IntoSeries:
        """Evaluate the condition per unit on a (dirty) dataframe.

        Returns a series with one entry per input row, holding the per-unit condition result
        ``condition(u) ∈ [0, 1]`` as a null-preserving float (null = unit out of scope). The return type matches
        the backend of ``X``.
        """
        self._check_is_resolved()
        frame = nw.from_native(X, eager_only=True)
        self._validate(frame)
        units = self._measure_units(frame)
        # _measure_units builds a fresh series, so narwhals' input type parameter is erased; cast back to the
        # caller's backend type that predict promises to return.
        return cast(IntoSeries, units.to_native())

    # -- hooks for subclasses ---------------------------------------------------------

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        """Return the per-unit condition results as a series. Must be overridden."""
        raise NotImplementedError

    def _score(self, frame: nw.DataFrame[Any]) -> float:
        """Aggregate the per-unit condition results to ``X``.

        Default: the mean ignoring nulls, i.e. the ISO ratio ``A / B`` where ``A`` sums the condition results
        and ``B`` counts the units in scope (non-null entries).
        """
        units = self._measure_units(frame).cast(nw.Float64)
        total = units.count()  # non-null count
        conforming = units.sum()
        return float(conforming / total) if total else float("nan")
