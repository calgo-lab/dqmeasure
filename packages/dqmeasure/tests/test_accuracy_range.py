from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import DataAccuracyRange
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame


def test_fit_learns_minmax(backend):
    train = make_frame({"temp": [0.0, 50.0, 100.0], "hum": [10.0, 20.0, 30.0]}, backend)
    measure = DataAccuracyRange("temp").fit(train)

    assert measure.low_ == 0.0
    assert measure.high_ == 100.0


def test_score_is_iso_ratio(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange("temp").fit(train)

    # 5 of these are out of [0, 100]: -10, 150, 200, 999, -1  -> 95/100 in range.
    values = [50.0] * 95 + [-10.0, 150.0, 200.0, 999.0, -1.0]
    test = make_frame({"temp": values}, backend)

    assert measure.score(test) == pytest.approx(0.95)


def test_missing_values_are_out_of_scope(backend):
    # The null is not a unit, so B counts the 4 non-null cells rather than all 5 rows: A/B = 2/4. Letting
    # nulls into B would give 0.4.
    train = make_frame({"age": [18.0, 40.0, 65.0]}, backend)
    measure = DataAccuracyRange("age").fit(train)

    test = make_frame({"age": [25.0, 17.0, 44.0, None, 103.0]}, backend)

    assert measure.score(test) == pytest.approx(0.5)


def test_all_null_column_scores_nan(backend):
    # Every cell out of scope means the score is nan, not 0.0
    train = make_frame({"age": [18.0, 65.0]}, backend)
    measure = DataAccuracyRange("age").fit(train)

    frame = nw.from_native(make_frame({"age": [1.0, 2.0]}, backend), eager_only=True)
    test = frame.with_columns(age=nw.lit(None, dtype=nw.Float64)).to_native()

    assert math.isnan(measure.score(test))


def test_predict_cell_level_and_nulls_preserved(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange("temp").fit(train)

    test = make_frame({"temp": [50.0, 150.0, None]}, backend)
    cells = measure.predict(test)

    # Returned series is in the caller's backend, one entry per input row.
    col = nw.from_native(cells, series_only=True).to_list()
    assert col[0] == 1.0  # 50 is in range
    assert col[1] == 0.0  # 150 is out of range
    assert col[2] is None or math.isnan(col[2])  # null in -> missing out


def test_inclusive_flag(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)

    inclusive = DataAccuracyRange("temp", inclusive=True).fit(train)
    exclusive = DataAccuracyRange("temp", inclusive=False).fit(train)

    test = make_frame({"temp": [0.0, 100.0]}, backend)  # both values are the bounds
    assert inclusive.score(test) == pytest.approx(1.0)
    assert exclusive.score(test) == pytest.approx(0.0)


def test_specified_bounds_skip_fit(backend):
    # Bounds given by an expert: the measure is usable without fit.
    measure = DataAccuracyRange("temp", low=0.0, high=100.0)
    values = [50.0] * 95 + [-10.0, 150.0, 200.0, 999.0, -1.0]
    test = make_frame({"temp": values}, backend)

    assert measure.score(test) == pytest.approx(0.95)
    assert measure.low_ == 0.0
    assert measure.high_ == 100.0


def test_partial_spec_learns_rest(backend):
    # low is specified, high is left to be learned from the clean data at fit.
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    measure = DataAccuracyRange("temp", low=-10.0).fit(train)

    assert measure.low_ == -10.0
    assert measure.high_ == 100.0


def test_missing_column_raises(backend):
    train = make_frame({"temp": [0.0, 100.0]}, backend)
    with pytest.raises(ValueError, match="not found"):
        DataAccuracyRange("pressure").fit(train)


def test_non_numeric_column_raises(backend):
    train = make_frame({"label": ["a", "b"]}, backend)
    with pytest.raises(ValueError, match="numeric"):
        DataAccuracyRange("label").fit(train)


def test_not_fitted_raises(backend):
    test = make_frame({"temp": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        DataAccuracyRange("temp").predict(test)
    with pytest.raises(NotResolvedError):
        DataAccuracyRange("temp", low=0.0).score(test)  # high still unresolved
