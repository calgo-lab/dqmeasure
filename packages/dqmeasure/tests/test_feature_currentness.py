from __future__ import annotations

import math
from datetime import datetime, timedelta

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from dqmeasure import FeatureCurrentness
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

REF = datetime(2024, 1, 10)


def test_fit_learns_age_range(backend):
    train = make_frame({"updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)]}, backend)
    measure = FeatureCurrentness("updated_at", reference_time=REF).fit(train)

    assert measure.min_age_ == timedelta(days=1)
    assert measure.max_age_ == timedelta(days=2)


def test_score_is_iso_ratio(backend):
    train = make_frame({"updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)]}, backend)
    measure = FeatureCurrentness("updated_at", reference_time=REF).fit(train)

    # Learned age range [1d, 2d]: 95 in range, 3 too old, 2 too fresh -> 95/100.
    values = [datetime(2024, 1, 8, 12)] * 95 + [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 9, 18)] * 2
    test = make_frame({"updated_at": values}, backend)

    assert measure.score(test) == pytest.approx(0.95)


def test_predict_cell_level_and_nulls_preserved(backend):
    train = make_frame({"updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)]}, backend)
    measure = FeatureCurrentness("updated_at", reference_time=REF).fit(train)

    test = make_frame(
        {"updated_at": [datetime(2024, 1, 8, 12), datetime(2024, 1, 1), datetime(2024, 1, 9, 18), None]},
        backend,
    )
    cells = measure.predict(test)

    col = nw.from_native(cells, series_only=True).to_list()
    assert col[0] == 1.0  # in the age range
    assert col[1] == 0.0  # too old
    assert col[2] == 0.0  # too fresh
    assert col[3] is None or math.isnan(col[3])  # null in -> missing out


def test_specified_age_range_skips_fit(backend):
    measure = FeatureCurrentness("updated_at", min_age=timedelta(0), max_age=timedelta(days=2), reference_time=REF)
    test = make_frame({"updated_at": [datetime(2024, 1, 9), datetime(2024, 1, 1)]}, backend)

    assert measure.score(test) == pytest.approx(0.5)
    assert measure.min_age_ == timedelta(0)
    assert measure.max_age_ == timedelta(days=2)


def test_partial_spec_learns_rest(backend):
    train = make_frame({"updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)]}, backend)
    measure = FeatureCurrentness("updated_at", min_age=timedelta(0), reference_time=REF).fit(train)

    assert measure.min_age_ == timedelta(0)
    assert measure.max_age_ == timedelta(days=2)


def test_wall_clock_reference_time(backend):
    # Without a pinned reference_time, ages are taken against "now" at each call. Hour-scale margins make
    # this immune to the clock advancing between the fit-time and score-time now().
    now = datetime.now()
    train = make_frame({"updated_at": [now - timedelta(hours=2), now - timedelta(hours=1)]}, backend)
    measure = FeatureCurrentness("updated_at").fit(train)

    test = make_frame({"updated_at": [now - timedelta(hours=1, minutes=30), now - timedelta(days=1)]}, backend)
    assert measure.score(test) == pytest.approx(0.5)


def test_future_timestamps_fail_the_range(backend):
    measure = FeatureCurrentness("updated_at", min_age=timedelta(0), max_age=timedelta(days=2), reference_time=REF)
    test = make_frame({"updated_at": [datetime(2024, 1, 11)]}, backend)  # after the reference time

    assert measure.score(test) == pytest.approx(0.0)


def all_null_frame(n: int, backend: str):
    """A frame whose ``updated_at`` column is all-null but still datetime-typed."""
    if backend == "polars":
        return pl.DataFrame({"updated_at": pl.Series([None] * n, dtype=pl.Datetime)})
    return pd.DataFrame({"updated_at": pd.Series([pd.NaT] * n)})


def test_all_null_test_frame_scores_nan(backend):
    measure = FeatureCurrentness("updated_at", min_age=timedelta(0), max_age=timedelta(days=2), reference_time=REF)

    assert math.isnan(measure.score(all_null_frame(2, backend)))


def test_all_null_train_raises(backend):
    with pytest.raises(ValueError, match="no non-null timestamps"):
        FeatureCurrentness("updated_at", reference_time=REF).fit(all_null_frame(1, backend))


def test_missing_column_raises(backend):
    train = make_frame({"updated_at": [datetime(2024, 1, 8)]}, backend)
    with pytest.raises(ValueError, match="not found"):
        FeatureCurrentness("created_at", reference_time=REF).fit(train)


def test_non_temporal_column_raises(backend):
    train = make_frame({"updated_at": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="datetime"):
        FeatureCurrentness("updated_at", reference_time=REF).fit(train)


def test_not_fitted_raises(backend):
    test = make_frame({"updated_at": [datetime(2024, 1, 8)]}, backend)
    with pytest.raises(NotResolvedError):
        FeatureCurrentness("updated_at", reference_time=REF).predict(test)
    with pytest.raises(NotResolvedError):
        # max_age still unresolved
        FeatureCurrentness("updated_at", min_age=timedelta(0), reference_time=REF).score(test)
