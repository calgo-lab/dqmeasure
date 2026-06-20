"""Base classes for data-quality measures."""

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Final, Literal, Self, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame

# direction in which the measure's X improves
Orientation = Literal["higher-is-better", "lower-is-better"]

# Score key used by measures whose subject is the whole table.
TABLE_SUBJECT: Final = "table"

# Sentinel: a reference value left to be learned at fit(), rather than specified up front.
_LEARN: Final = object()


class NotResolvedError(RuntimeError):
    """Raised when a measure is used before its reference is resolved (specified in the constructor or learned via
    [`fit`][dqmeasure.base.BaseMeasure.fit])."""


class BaseMeasure:
    """Data-quality measure base class.

    Every quality measure inherits from this class. Subclasses set ``iso_id`` (the measure's ID in ISO/IEC 25024 or
    5259-2, e.g. ``"Acc-I-7"``) and ``orientation``, and implement two hooks:

    * [`_fit_reference`][dqmeasure.base.BaseMeasure._fit_reference]: learn the reference from clean data.
    * [`_score`][dqmeasure.base.BaseMeasure._score]: compute the quality measure value ``X`` per subject on a
      (dirty) dataframe.

    Measures whose units are positions in the dataframe should inherit from
    [`PositionalMeasure`][dqmeasure.base.PositionalMeasure] instead, which adds ``predict()`` and derives
    ``_score`` from it.

    Parameters
    ----------
    columns:
        Optional subset of columns the measure applies to. ``None`` (default) means the applicable columns are
        auto-detected at [`fit`][dqmeasure.base.BaseMeasure.fit] time (see
        [`_select_columns`][dqmeasure.base.BaseMeasure._select_columns]).
    """

    iso_id: ClassVar[str]
    """The measure's ID in ISO/IEC 25024 or ISO/IEC 5259-2 (e.g. ``"Acc-I-7"``)."""

    orientation: ClassVar[Orientation]
    """Direction in which the measure's ``X`` improves."""

    reference_params: ClassVar[tuple[str, ...]] = ()
    """Names of the constructor parameters that hold the measure's reference.

    Each may be given as a scalar (broadcast to every column), a ``{column: value}`` dict (per column; columns left
    out are learned), or ``None`` (learned for every column). After resolution each appears as a fitted ``<name>_``
    attribute holding a ``{column: value}`` dict.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    # scikit-learn-style API

    def fit(self, X: IntoDataFrame) -> Self:
        """Learn the reference from a clean (training) dataframe.

        Sets the fitted attributes ``columns_`` (resolved applicable columns) and one ``<name>_`` per reference
        parameter, and returns ``self``. Parameters specified in the constructor are kept; only the rest is
        estimated from ``X``.
        """
        frame = nw.from_native(X, eager_only=True)
        self._resolve(self._select_columns(frame), frame)
        return self

    def score(self, X: IntoDataFrame) -> dict[str, float]:
        """Compute the quality measure value ``X`` for a (dirty) dataframe.

        Returns a mapping ``{subject: X}``: one value per subject (a column, a rule, or the whole table; table-subject
        measures use the key ``"table"``, available as ``dqmeasure.base.TABLE_SUBJECT``). When a subject has no
        units in scope (``B = 0``), its value is ``NaN``.
        """
        self._check_is_resolved()
        frame = nw.from_native(X, eager_only=True)
        return self._score(frame)

    # hooks for subclasses

    def _select_columns(self, frame: nw.DataFrame[Any]) -> list[str]:
        """Resolve which columns the measure applies to.

        Defaults to ``self.columns``. Subclasses may override for other column types. Raises an error if
        ``self.columns`` isn't set and no override is applied.
        """
        if self.columns is not None:
            return list(self.columns)
        raise NotImplementedError

    def _fit_reference(self, frame: nw.DataFrame[Any], columns: list[str]) -> dict[str, dict[str, Any]]:
        """Estimate every reference parameter for ``columns`` from clean data.

        Returns ``{param_name: {column: value}}``. The default has nothing to learn; measures with a non-empty
        ``reference_params`` override it.
        """
        return {}

    def _score(self, frame: nw.DataFrame[Any]) -> dict[str, float]:
        """Compute ``{subject: X}`` for a (dirty) dataframe. Must be overridden."""
        raise NotImplementedError

    # helpers

    def _resolve(self, columns: list[str], frame: nw.DataFrame[Any] | None) -> None:
        """Resolve ``columns_`` and the ``<name>_`` attributes from spec plus estimate.

        ``frame`` is the clean data when learning, or ``None`` when the reference is fully specified and no
        estimation is needed.
        """
        needs_learning = any(self._specified(name, col) is _LEARN for name in self.reference_params for col in columns)
        estimated: dict[str, dict[str, Any]] = {}
        if needs_learning:
            if frame is None:
                raise NotResolvedError(
                    f"{type(self).__name__}: the reference is not fully specified; "
                    "either give every parameter in the constructor or call fit() on clean data."
                )
            estimated = self._fit_reference(frame, columns)
        for name in self.reference_params:
            resolved: dict[str, Any] = {}
            for col in columns:
                value = self._specified(name, col)
                resolved[col] = estimated[name][col] if value is _LEARN else value
            setattr(self, f"{name}_", resolved)
        self.columns_ = columns

    def _specified(self, name: str, column: str) -> Any:
        """The value specified for parameter ``name`` at ``column``, or ``_LEARN`` if none."""
        spec = getattr(self, name)
        if spec is None:
            return _LEARN
        if isinstance(spec, dict):
            return spec.get(column, _LEARN)
        return spec  # scalar: broadcast to every column

    def _check_is_resolved(self) -> None:
        if hasattr(self, "columns_"):
            return
        if self.columns is None:
            raise NotResolvedError(
                f"{type(self).__name__}: columns are unknown without data; "
                "set columns=... and the reference in the constructor, or call fit() on clean data."
            )
        self._resolve(list(self.columns), None)

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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


class PositionalMeasure(BaseMeasure):
    """Base class for tier-1 measures with positional units (cells or rows).

    A positional unit is a position in the dataframe you can point at and attach a score to. Such measures gain
    [`predict`][dqmeasure.base.PositionalMeasure.predict], and ``score()`` is its aggregation per subject. Subclasses
    implement [`_measure_units`][dqmeasure.base.PositionalMeasure._measure_units] instead of ``_score``.

    Measures with non-positional units and tier-2 statistic measures have no per-unit output. They derive from
    [`BaseMeasure`][dqmeasure.base.BaseMeasure] directly and are ``score()``-only.
    """

    def predict(self, X: IntoDataFrame) -> IntoDataFrame:
        """Evaluate the condition per unit on a (dirty) dataframe.

        Returns a dataframe with one column per subject and one row per input row, holding the per-unit condition
        result ``condition(u) ∈ [0, 1]`` as a null-preserving float (null = unit out of scope). The return type
        matches the backend of ``X``.
        """
        self._check_is_resolved()
        frame = nw.from_native(X, eager_only=True)
        units = self._measure_units(frame)
        # _measure_units builds a fresh dataframe, so narwhals' input type parameter is erased; cast back to the
        # caller's backend type that predict promises to return.
        return cast(IntoDataFrame, units.to_native())

    # -- hooks for subclasses ---------------------------------------------------------

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Return a per-unit condition dataframe, one column per subject. Must be overridden."""
        raise NotImplementedError

    def _score(self, frame: nw.DataFrame[Any]) -> dict[str, float]:
        """Aggregate the per-unit condition results to one ``X`` per subject.

        Default: the mean of each subject column ignoring nulls, i.e. the ISO ratio ``A / B`` where ``A`` sums the
        condition results and ``B`` counts the units in scope (non-null entries).
        """
        units = self._measure_units(frame)
        result: dict[str, float] = {}
        for name in units.columns:
            col = units[name].cast(nw.Float64)
            total = col.count()  # non-null count
            conforming = col.sum()
            result[name] = float(conforming / total) if total else float("nan")
        return result
