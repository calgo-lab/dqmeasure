"""Tests for DataAccuracyRange, parametrised over Polars and pandas backends."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

from dqmeasure import DataAccuracyRange
from dqmeasure.base import NotResolvedError

# Backends to run every test against. pandas is optional; skip if not installed.
BACKENDS = ["polars"]
try:  # pragma: no cover - import guard
    import pandas as pd

    BACKENDS.append("pandas")
except ImportError:  # pragma: no cover
    pass


def make_frame(data: dict[str, Sequence[float | None]], backend: str):
    """Build a native frame in the requested backend."""
    if backend == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def test_measure_metadata():
    assert DataAccuracyRange.iso_id == "Acc-I-7"
    assert DataAccuracyRange.orientation == "higher-is-better"


def test_fit_learns_minmax(backend):
    train = make_frame({"temp": [0.0, 50.0, 100.0], "hum": [10.0, 20.0, 30.0]}, backend)
    measure = DataAccuracyRange().fit(train)

    assert measure.columns_ == ["temp", "hum"]
    assert measure.low_ == {"temp": 0.0, "hum": 10.0}
    assert measure.high_ == {"temp": 100.0, "hum": 30.0}


def test_score_is_iso_ratio(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange().fit(train)

    # 5 of these are out of [0, 100]: -10, 150, 200, 999, -1  -> 95/100 in range.
    values = [50.0] * 95 + [-10.0, 150.0, 200.0, 999.0, -1.0]
    test = make_frame({"temp": values}, backend)

    scores = measure.score(test)
    assert scores["temp"] == pytest.approx(0.95)


def test_predict_cell_level_and_nulls_preserved(backend):
    import math

    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange().fit(train)

    test = make_frame({"temp": [50.0, 150.0, None]}, backend)
    cells = measure.predict(test)

    # Returned frame is in the caller's backend.
    assert type(cells) is type(test)

    # Per-cell measure: 1.0 in range, 0.0 out of range, missing stays missing.
    # Read values back in a backend-agnostic way via Narwhals.
    import narwhals as nw

    col = nw.from_native(cells, eager_only=True)["temp"].to_list()
    assert col[0] == 1.0  # 50 is in range
    assert col[1] == 0.0  # 150 is out of range
    assert col[2] is None or math.isnan(col[2])  # null in -> missing out


def test_inclusive_flag(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)

    inclusive = DataAccuracyRange(inclusive=True).fit(train)
    exclusive = DataAccuracyRange(inclusive=False).fit(train)

    test = make_frame({"temp": [0.0, 100.0]}, backend)  # both values are the bounds
    assert inclusive.score(test)["temp"] == pytest.approx(1.0)
    assert exclusive.score(test)["temp"] == pytest.approx(0.0)


def test_explicit_columns_subset(backend):
    train = make_frame({"a": [0.0, 10.0], "b": [0.0, 10.0]}, backend)
    measure = DataAccuracyRange(columns=["a"]).fit(train)

    assert measure.columns_ == ["a"]
    assert set(measure.low_) == {"a"}


def test_specified_bounds_skip_fit(backend):
    # Bounds given by an expert (broadcast scalars): the measure is usable without fit.
    measure = DataAccuracyRange(columns=["temp"], low=0.0, high=100.0)
    values = [50.0] * 95 + [-10.0, 150.0, 200.0, 999.0, -1.0]
    test = make_frame({"temp": values}, backend)

    assert measure.score(test)["temp"] == pytest.approx(0.95)
    assert measure.low_ == {"temp": 0.0}
    assert measure.high_ == {"temp": 100.0}


def test_specified_bounds_per_column_dict(backend):
    measure = DataAccuracyRange(columns=["a", "b"], low={"a": 0.0, "b": 10.0}, high={"a": 5.0, "b": 20.0})
    test = make_frame({"a": [0.0, 6.0], "b": [15.0, 25.0]}, backend)

    scores = measure.score(test)
    assert scores["a"] == pytest.approx(0.5)  # 0.0 in [0, 5], 6.0 out
    assert scores["b"] == pytest.approx(0.5)  # 15.0 in [10, 20], 25.0 out


def test_partial_spec_learns_rest(backend):
    # low is specified (broadcast), high is left to be learned from the clean data at fit.
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange(low=-10.0).fit(train)

    assert measure.low_ == {"temp": -10.0}
    assert measure.high_ == {"temp": 100.0}


def test_not_fitted_raises(backend):
    test = make_frame({"temp": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        DataAccuracyRange().predict(test)
    with pytest.raises(NotResolvedError):
        DataAccuracyRange().score(test)
