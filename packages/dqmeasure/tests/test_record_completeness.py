from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import RecordCompleteness

from ._helpers import make_frame


def test_takes_no_column():
    with pytest.raises(TypeError):
        RecordCompleteness("a")  # type: ignore[call-arg]


def test_score_is_iso_ratio(backend):
    # 4 records, 2 without any empty data item -> 2/4.
    test = make_frame({"a": [1, None, 3, 4], "b": ["x", "y", None, "w"]}, backend)
    assert RecordCompleteness().score(test) == pytest.approx(0.5)


def test_predict_record_level_no_nulls(backend):
    # A record with nulls scores 0.0; nulls are the measurand, so no record falls out of scope.
    test = make_frame({"a": [1, None, 3], "b": ["x", "y", None]}, backend)
    records = RecordCompleteness().predict(test)

    col = nw.from_native(records, series_only=True).to_list()
    assert col == [1.0, 0.0, 0.0]


def test_all_records_holed_scores_zero(backend):
    test = make_frame({"a": [None, 1], "b": ["x", None]}, backend)
    assert RecordCompleteness().score(test) == pytest.approx(0.0)


def test_empty_frame_scores_nan(backend):
    test = make_frame({"a": [], "b": []}, backend)
    assert math.isnan(RecordCompleteness().score(test))
