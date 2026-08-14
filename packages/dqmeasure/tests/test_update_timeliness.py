from __future__ import annotations

import math
from datetime import datetime, timedelta

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from dqmeasure import TimelinessOfUpdate
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

DUE = datetime(2024, 1, 1)


def test_fit_learns_max_delay(backend):
    train = make_frame(
        {
            "updated_at": [DUE + timedelta(hours=1), DUE + timedelta(hours=3), None],
            "due_at": [DUE, DUE, DUE],
        },
        backend,
    )
    measure = TimelinessOfUpdate("updated_at", due_column="due_at").fit(train)

    # Learned from the rows with both timestamps set; the never-updated row does not contribute.
    assert measure.sla_ == timedelta(hours=3)


def test_score_is_iso_ratio(backend):
    # 200 records require updating; 180 were updated within the SLA (the roadmap example).
    updated = [DUE + timedelta(hours=1)] * 180 + [DUE + timedelta(days=2)] * 20
    test = make_frame({"updated_at": updated, "due_at": [DUE] * 200}, backend)
    measure = TimelinessOfUpdate("updated_at", due_column="due_at", sla=timedelta(hours=2))

    assert measure.score(test) == pytest.approx(0.9)


def test_predict_conditions(backend):
    test = make_frame(
        {
            "updated_at": [
                DUE + timedelta(hours=1),  # timely
                DUE + timedelta(days=2),  # late
                None,  # needed updating, never updated
                DUE + timedelta(days=2),  # no due time: out of scope regardless of the update
                DUE - timedelta(hours=1),  # updated before it was due: timely
            ],
            "due_at": [DUE, DUE, DUE, None, DUE],
        },
        backend,
    )
    measure = TimelinessOfUpdate("updated_at", due_column="due_at", sla=timedelta(hours=2))

    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 1.0
    assert col[1] == 0.0
    assert col[2] == 0.0
    assert col[3] is None or math.isnan(col[3])
    assert col[4] == 1.0


def test_specified_sla_skips_fit(backend):
    measure = TimelinessOfUpdate("updated_at", due_column="due_at", sla=timedelta(hours=2))
    test = make_frame({"updated_at": [DUE + timedelta(hours=1)], "due_at": [DUE]}, backend)

    assert measure.score(test) == pytest.approx(1.0)
    assert measure.sla_ == timedelta(hours=2)


def test_all_due_null_scores_nan(backend):
    measure = TimelinessOfUpdate("updated_at", due_column="due_at", sla=timedelta(hours=2))
    if backend == "polars":
        test = pl.DataFrame(
            {
                "updated_at": pl.Series([DUE, DUE], dtype=pl.Datetime),
                "due_at": pl.Series([None, None], dtype=pl.Datetime),
            }
        )
    else:
        test = pd.DataFrame({"updated_at": [DUE, DUE], "due_at": pd.Series([pd.NaT, pd.NaT])})

    assert math.isnan(measure.score(test))


def test_fit_needs_complete_pairs(backend):
    if backend == "polars":
        train = pl.DataFrame(
            {
                "updated_at": pl.Series([None, DUE], dtype=pl.Datetime),
                "due_at": pl.Series([DUE, None], dtype=pl.Datetime),
            }
        )
    else:
        train = pd.DataFrame({"updated_at": [pd.NaT, DUE], "due_at": [DUE, pd.NaT]})
    with pytest.raises(ValueError, match="no rows with both"):
        TimelinessOfUpdate("updated_at", due_column="due_at").fit(train)


def test_missing_columns_raise(backend):
    train = make_frame({"updated_at": [DUE], "due_at": [DUE]}, backend)
    with pytest.raises(ValueError, match="not found"):
        TimelinessOfUpdate("modified_at", due_column="due_at").fit(train)
    with pytest.raises(ValueError, match="not found"):
        TimelinessOfUpdate("updated_at", due_column="deadline").fit(train)


def test_non_temporal_columns_raise(backend):
    with pytest.raises(ValueError, match="datetime"):
        TimelinessOfUpdate("updated_at", due_column="due_at").fit(
            make_frame({"updated_at": [1.0], "due_at": [DUE]}, backend)
        )
    with pytest.raises(ValueError, match="datetime"):
        TimelinessOfUpdate("updated_at", due_column="due_at").fit(
            make_frame({"updated_at": [DUE], "due_at": [1.0]}, backend)
        )


def test_not_fitted_raises(backend):
    test = make_frame({"updated_at": [DUE], "due_at": [DUE]}, backend)
    with pytest.raises(NotResolvedError):
        TimelinessOfUpdate("updated_at", due_column="due_at").predict(test)
    with pytest.raises(NotResolvedError):
        TimelinessOfUpdate("updated_at", due_column="due_at").score(test)
