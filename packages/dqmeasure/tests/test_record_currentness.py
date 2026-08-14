from __future__ import annotations

import math
from datetime import datetime, timedelta

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from dqmeasure import RecordCurrentness
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

REF = datetime(2024, 1, 10)


def test_takes_no_column():
    with pytest.raises(TypeError):
        RecordCurrentness("created_at")  # type: ignore[misc, arg-type]


def test_fit_learns_age_range_per_temporal_column(backend):
    train = make_frame(
        {
            "created_at": [datetime(2024, 1, 5), datetime(2024, 1, 7)],
            "updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)],
            "v": [1, 2],
        },
        backend,
    )
    measure = RecordCurrentness(reference_time=REF).fit(train)

    assert measure.age_ranges_ == {
        "created_at": (timedelta(days=3), timedelta(days=5)),
        "updated_at": (timedelta(days=1), timedelta(days=2)),
    }


def test_score_is_iso_ratio(backend):
    # Learned ranges: created_at [3d, 5d], updated_at [1d, 2d]. A record conforms only when every
    # timestamp is of the right age: rows 2 (created_at too old) and 3 (updated_at too fresh) fail -> 2/4.
    train = make_frame(
        {
            "created_at": [datetime(2024, 1, 5), datetime(2024, 1, 7)],
            "updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9)],
        },
        backend,
    )
    measure = RecordCurrentness(reference_time=REF).fit(train)

    test = make_frame(
        {
            "created_at": [datetime(2024, 1, 6), datetime(2024, 1, 5), datetime(2024, 1, 1), datetime(2024, 1, 7)],
            "updated_at": [datetime(2024, 1, 8), datetime(2024, 1, 9), datetime(2024, 1, 8), datetime(2024, 1, 9, 18)],
        },
        backend,
    )
    assert measure.score(test) == pytest.approx(0.5)


def test_predict_record_level_and_null_handling(backend):
    measure = RecordCurrentness(age_ranges={"created_at": (timedelta(0), timedelta(days=2))}, reference_time=REF)

    test = make_frame({"created_at": [datetime(2024, 1, 9), datetime(2024, 1, 1), None], "v": [1, 2, 3]}, backend)
    records = measure.predict(test)

    col = nw.from_native(records, series_only=True).to_list()
    assert col[0] == 1.0  # in the age range
    assert col[1] == 0.0  # too old
    assert col[2] is None or math.isnan(col[2])  # no timestamp -> no unit in scope


def test_partially_null_record_judged_on_remaining_timestamps(backend):
    measure = RecordCurrentness(
        age_ranges={"created_at": (timedelta(0), timedelta(days=2)), "updated_at": (timedelta(0), timedelta(days=2))},
        reference_time=REF,
    )
    # created_at is null but updated_at is in range: the record is judged on its one non-null timestamp.
    if backend == "polars":
        test = pl.DataFrame({"created_at": pl.Series([None], dtype=pl.Datetime), "updated_at": [datetime(2024, 1, 9)]})
    else:
        test = pd.DataFrame({"created_at": pd.Series([pd.NaT]), "updated_at": [datetime(2024, 1, 9)]})

    assert measure.score(test) == pytest.approx(1.0)


def test_specified_age_ranges_skip_fit(backend):
    measure = RecordCurrentness(age_ranges={"created_at": (timedelta(0), timedelta(days=2))}, reference_time=REF)
    test = make_frame({"created_at": [datetime(2024, 1, 9), datetime(2024, 1, 1)]}, backend)

    assert measure.score(test) == pytest.approx(0.5)
    assert measure.age_ranges_ == {"created_at": (timedelta(0), timedelta(days=2))}


def test_not_fitted_raises(backend):
    test = make_frame({"created_at": [datetime(2024, 1, 9)]}, backend)
    with pytest.raises(NotResolvedError):
        RecordCurrentness(reference_time=REF).score(test)


def test_frame_without_temporal_column_raises(backend):
    test = make_frame({"a": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="datetime or date"):
        RecordCurrentness(reference_time=REF).fit(test)


def test_all_null_temporal_column_at_fit_raises(backend):
    if backend == "polars":
        train = pl.DataFrame({"created_at": pl.Series([None], dtype=pl.Datetime), "v": [1]})
    else:
        train = pd.DataFrame({"created_at": pd.Series([pd.NaT]), "v": [1]})
    with pytest.raises(ValueError, match="no non-null timestamps"):
        RecordCurrentness(reference_time=REF).fit(train)


def test_uncovered_temporal_column_raises(backend):
    train = make_frame({"created_at": [datetime(2024, 1, 8)]}, backend)
    measure = RecordCurrentness(reference_time=REF).fit(train)

    test = make_frame({"created_at": [datetime(2024, 1, 8)], "updated_at": [datetime(2024, 1, 9)]}, backend)
    with pytest.raises(ValueError, match="not covered"):
        measure.score(test)


def test_covered_column_missing_from_frame_raises(backend):
    measure = RecordCurrentness(age_ranges={"created_at": (timedelta(0), timedelta(days=2))}, reference_time=REF)
    test = make_frame({"updated_at": [datetime(2024, 1, 9)]}, backend)

    with pytest.raises(ValueError, match="not covered"):
        measure.score(test)
