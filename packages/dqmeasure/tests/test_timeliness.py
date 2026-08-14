from __future__ import annotations

import math
from datetime import datetime, timedelta

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from dqmeasure import TimelinessOfDataItems
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

EVENT = datetime(2024, 1, 1)


def test_fit_learns_max_latency(backend):
    train = make_frame(
        {
            "recorded_at": [EVENT + timedelta(minutes=5), EVENT + timedelta(minutes=45), None],
            "event_at": [EVENT, EVENT, EVENT],
        },
        backend,
    )
    measure = TimelinessOfDataItems("recorded_at", event_column="event_at").fit(train)

    # Learned from the rows with both timestamps set; the never-recorded row does not contribute.
    assert measure.max_latency_ == timedelta(minutes=45)


def test_score_is_iso_ratio(backend):
    # Data measured >= 60 min after the event is outdated; 60 items are timely, 40 outdated
    # (the roadmap example): X = 60/100.
    recorded = [EVENT + timedelta(minutes=30)] * 60 + [EVENT + timedelta(minutes=90)] * 40
    test = make_frame({"recorded_at": recorded, "event_at": [EVENT] * 100}, backend)
    measure = TimelinessOfDataItems("recorded_at", event_column="event_at", max_latency=timedelta(minutes=59))

    assert measure.score(test) == pytest.approx(0.6)


def test_predict_conditions(backend):
    test = make_frame(
        {
            "recorded_at": [
                EVENT + timedelta(minutes=30),  # timely
                EVENT + timedelta(hours=2),  # outdated
                None,  # never became available
                EVENT + timedelta(hours=2),  # no event time: out of scope regardless
                EVENT - timedelta(minutes=5),  # available before the event: timely
            ],
            "event_at": [EVENT, EVENT, EVENT, None, EVENT],
        },
        backend,
    )
    measure = TimelinessOfDataItems("recorded_at", event_column="event_at", max_latency=timedelta(hours=1))

    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 1.0
    assert col[1] == 0.0
    assert col[2] == 0.0
    assert col[3] is None or math.isnan(col[3])
    assert col[4] == 1.0


def test_specified_max_latency_skips_fit(backend):
    measure = TimelinessOfDataItems("recorded_at", event_column="event_at", max_latency=timedelta(hours=1))
    test = make_frame({"recorded_at": [EVENT + timedelta(minutes=30)], "event_at": [EVENT]}, backend)

    assert measure.score(test) == pytest.approx(1.0)
    assert measure.max_latency_ == timedelta(hours=1)


def test_all_events_null_scores_nan(backend):
    measure = TimelinessOfDataItems("recorded_at", event_column="event_at", max_latency=timedelta(hours=1))
    if backend == "polars":
        test = pl.DataFrame(
            {
                "recorded_at": pl.Series([EVENT, EVENT], dtype=pl.Datetime),
                "event_at": pl.Series([None, None], dtype=pl.Datetime),
            }
        )
    else:
        test = pd.DataFrame({"recorded_at": [EVENT, EVENT], "event_at": pd.Series([pd.NaT, pd.NaT])})

    assert math.isnan(measure.score(test))


def test_fit_needs_complete_pairs(backend):
    if backend == "polars":
        train = pl.DataFrame(
            {
                "recorded_at": pl.Series([None, EVENT], dtype=pl.Datetime),
                "event_at": pl.Series([EVENT, None], dtype=pl.Datetime),
            }
        )
    else:
        train = pd.DataFrame({"recorded_at": [pd.NaT, EVENT], "event_at": [EVENT, pd.NaT]})
    with pytest.raises(ValueError, match="no rows with both"):
        TimelinessOfDataItems("recorded_at", event_column="event_at").fit(train)


def test_missing_columns_raise(backend):
    train = make_frame({"recorded_at": [EVENT], "event_at": [EVENT]}, backend)
    with pytest.raises(ValueError, match="not found"):
        TimelinessOfDataItems("available_at", event_column="event_at").fit(train)
    with pytest.raises(ValueError, match="not found"):
        TimelinessOfDataItems("recorded_at", event_column="occurred_at").fit(train)


def test_non_temporal_columns_raise(backend):
    with pytest.raises(ValueError, match="datetime"):
        TimelinessOfDataItems("recorded_at", event_column="event_at").fit(
            make_frame({"recorded_at": [1.0], "event_at": [EVENT]}, backend)
        )
    with pytest.raises(ValueError, match="datetime"):
        TimelinessOfDataItems("recorded_at", event_column="event_at").fit(
            make_frame({"recorded_at": [EVENT], "event_at": [1.0]}, backend)
        )


def test_not_fitted_raises(backend):
    test = make_frame({"recorded_at": [EVENT], "event_at": [EVENT]}, backend)
    with pytest.raises(NotResolvedError):
        TimelinessOfDataItems("recorded_at", event_column="event_at").predict(test)
    with pytest.raises(NotResolvedError):
        TimelinessOfDataItems("recorded_at", event_column="event_at").score(test)
