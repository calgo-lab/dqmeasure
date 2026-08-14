from __future__ import annotations

import math
from datetime import datetime, timedelta

import narwhals as nw
import pytest

from dqmeasure import UpdateFrequency
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

T0 = datetime(2024, 1, 1)


def minutes(*offsets: float) -> list[datetime]:
    return [T0 + timedelta(minutes=m) for m in offsets]


def test_fit_learns_max_gap(backend):
    train = make_frame({"tick": minutes(0, 1, 3)}, backend)
    measure = UpdateFrequency("tick").fit(train)

    assert measure.max_interval_ == timedelta(minutes=2)


def test_score_is_iso_ratio(backend):
    # 100 events -> 99 gaps; 5 of them are 2 minutes, the rest 1 minute. With a required interval of one
    # minute, 94 of 99 events keep up the frequency.
    offsets = [0.0]
    for i in range(99):
        offsets.append(offsets[-1] + (2.0 if i < 5 else 1.0))
    test = make_frame({"tick": minutes(*offsets)}, backend)
    measure = UpdateFrequency("tick", max_interval=timedelta(minutes=1))

    assert measure.score(test) == pytest.approx(94 / 99)


def test_predict_preserves_original_row_order(backend):
    # Unsorted input with a null and a duplicate timestamp. Ordered by time: 1/1 (first event, out of
    # scope), 1/2 (gap 1d, conforms), 1/2 (gap 0, conforms), 1/5 (gap 3d, violates); the null is out of
    # scope. predict returns these in the input's row order.
    test = make_frame(
        {"tick": [datetime(2024, 1, 5), datetime(2024, 1, 1), None, datetime(2024, 1, 2), datetime(2024, 1, 2)]},
        backend,
    )
    measure = UpdateFrequency("tick", max_interval=timedelta(days=1))

    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 0.0
    assert col[1] is None or math.isnan(col[1])
    assert col[2] is None or math.isnan(col[2])
    assert col[3] == 1.0
    assert col[4] == 1.0


def test_specified_interval_skips_fit(backend):
    measure = UpdateFrequency("tick", max_interval=timedelta(minutes=1))
    test = make_frame({"tick": minutes(0, 1, 3)}, backend)

    assert measure.score(test) == pytest.approx(0.5)  # gaps 1min (ok), 2min (late)
    assert measure.max_interval_ == timedelta(minutes=1)


def test_single_event_scores_nan(backend):
    measure = UpdateFrequency("tick", max_interval=timedelta(minutes=1))
    test = make_frame({"tick": minutes(0)}, backend)

    assert math.isnan(measure.score(test))


def test_fit_needs_two_timestamps(backend):
    train = make_frame({"tick": minutes(0)}, backend)
    with pytest.raises(ValueError, match="at least two"):
        UpdateFrequency("tick").fit(train)


def test_missing_column_raises(backend):
    train = make_frame({"tick": minutes(0, 1)}, backend)
    with pytest.raises(ValueError, match="not found"):
        UpdateFrequency("tock").fit(train)


def test_non_temporal_column_raises(backend):
    train = make_frame({"tick": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="datetime"):
        UpdateFrequency("tick").fit(train)


def test_not_fitted_raises(backend):
    test = make_frame({"tick": minutes(0, 1)}, backend)
    with pytest.raises(NotResolvedError):
        UpdateFrequency("tick").predict(test)
    with pytest.raises(NotResolvedError):
        UpdateFrequency("tick").score(test)
