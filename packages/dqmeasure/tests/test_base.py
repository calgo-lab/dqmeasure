from __future__ import annotations

from datetime import date, datetime, timedelta

import narwhals as nw
import polars as pl
import pytest

import dqmeasure
from dqmeasure import BaseMeasure, DataAccuracyRange, PositionalMeasure
from dqmeasure.base import _require_column

# Every concrete measure the package exports. The two base classes are abstract and declare no metadata.
MEASURES = sorted(
    (
        cls
        for name in dqmeasure.__all__
        if isinstance(cls := getattr(dqmeasure, name), type)
        and issubclass(cls, BaseMeasure)
        and cls not in (BaseMeasure, PositionalMeasure)
    ),
    key=lambda cls: cls.__name__,
)


@pytest.mark.parametrize("measure", MEASURES, ids=lambda cls: cls.__name__)
def test_every_measure_declares_metadata(measure):
    # The ID attributes and scope are ClassVars without defaults that no library code reads, so a
    # measure that forgets to declare one would otherwise fail nowhere. A measure comes from at least
    # one of the two standards, so the two IDs are never both None.
    for iso_id in (measure.iso_5259_id, measure.iso_25024_id):
        assert iso_id is None or (isinstance(iso_id, str) and iso_id)
    assert measure.iso_5259_id or measure.iso_25024_id
    assert measure.scope in ("column", "table")


@pytest.mark.parametrize("attribute", ["iso_5259_id", "iso_25024_id"])
def test_iso_ids_are_unique(attribute):
    # Two measures claiming the same ID within one standard means one of them is mislabelled against it.
    seen: dict[str, str] = {}
    clashes = []
    for measure in MEASURES:
        iso_id = getattr(measure, attribute)
        if iso_id is None:
            continue
        if iso_id in seen:
            clashes.append(f"{iso_id}: {seen[iso_id]} and {measure.__name__}")
        seen[iso_id] = measure.__name__
    assert not clashes, f"duplicate {attribute}s: {'; '.join(clashes)}"


def test_get_params_introspects_init_signature():
    measure = DataAccuracyRange("temp", inclusive=False)
    assert measure.get_params() == {
        "column": "temp",
        "low": None,
        "high": None,
        "method": "minmax",
        "inclusive": False,
    }


def test_get_params_on_base_subclass_without_extra_params():
    class Minimal(BaseMeasure):
        iso_5259_id = "Tst-0"
        iso_25024_id = None

    assert Minimal("a").get_params() == {"column": "a"}


def test_set_params_roundtrip():
    measure = DataAccuracyRange("temp").set_params(inclusive=False)
    assert measure.get_params()["inclusive"] is False


class TableNoInit(BaseMeasure):
    """A table-scoped measure that forgot to define its own ``__init__``."""

    iso_5259_id = "Tst-1"
    iso_25024_id = None
    scope = "table"


class TableMinimal(BaseMeasure):
    """A well-formed table-scoped measure without parameters."""

    iso_5259_id = "Tst-2"
    iso_25024_id = None
    scope = "table"

    def __init__(self) -> None:
        pass

    def _score(self, frame) -> float:
        return 1.0


def test_table_scope_without_own_init_fails_loudly():
    with pytest.raises(TypeError, match="table-scoped"):
        TableNoInit("a")


def test_table_scope_get_params_is_empty():
    assert TableMinimal().get_params() == {}


def test_table_scope_rejects_zero_column_frame():
    with pytest.raises(ValueError, match="at least one column"):
        TableMinimal().score(pl.DataFrame())


def test_require_column_temporal():
    frame = nw.from_native(
        pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1)],
                "day": [date(2024, 1, 1)],
                "gap": [timedelta(days=1)],
                "x": [1.0],
            }
        ),
        eager_only=True,
    )
    _require_column(frame, "ts", temporal=True)
    _require_column(frame, "day", temporal=True)
    # Duration and non-temporal dtypes are rejected: temporal means a point in time, not a span.
    with pytest.raises(ValueError, match="datetime"):
        _require_column(frame, "gap", temporal=True)
    with pytest.raises(ValueError, match="datetime"):
        _require_column(frame, "x", temporal=True)
