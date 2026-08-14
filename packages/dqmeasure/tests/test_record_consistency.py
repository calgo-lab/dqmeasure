from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import DataRecordConsistency

from ._helpers import make_frame


def test_takes_no_column():
    with pytest.raises(TypeError):
        DataRecordConsistency("a")  # type: ignore[call-arg]


def test_score_counts_every_occurrence(backend):
    # Records a, b, a: both occurrences of the repeated record count, so the standard scores 2/3
    # duplicates and we report the complement, the single unique record.
    test = make_frame({"a": [1, 2, 1], "b": ["x", "y", "x"]}, backend)
    assert DataRecordConsistency().score(test) == pytest.approx(1 / 3)


def test_duplication_is_not_cell_separable(backend):
    # Every value is duplicated within its own column, but no full record repeats: each row agrees with
    # some other row in each column separately, never with the same row in all of them.
    test = make_frame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"]}, backend)
    records = DataRecordConsistency().predict(test)

    col = nw.from_native(records, series_only=True).to_list()
    assert col == [1.0, 1.0, 1.0, 1.0]
    assert DataRecordConsistency().score(test) == pytest.approx(1.0)


def test_null_bearing_records_out_of_scope(backend):
    # Two identical null-bearing records are not duplicates of each other: they are out of scope.
    test = make_frame({"a": [1, 1, None, None], "b": ["x", "x", "y", "y"]}, backend)
    records = DataRecordConsistency().predict(test)

    col = nw.from_native(records, series_only=True).to_list()
    assert col[0] == 0.0
    assert col[1] == 0.0
    assert col[2] is None or math.isnan(col[2])
    assert col[3] is None or math.isnan(col[3])
    # B counts only the two in-scope records, and both are duplicates of each other.
    assert DataRecordConsistency().score(test) == pytest.approx(0.0)


def test_unique_frame_scores_one(backend):
    test = make_frame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, backend)
    assert DataRecordConsistency().score(test) == pytest.approx(1.0)


def test_constant_frame_scores_zero(backend):
    test = make_frame({"a": [7, 7, 7], "b": ["x", "x", "x"]}, backend)
    assert DataRecordConsistency().score(test) == pytest.approx(0.0)


def test_all_records_null_bearing_scores_nan(backend):
    test = make_frame({"a": [None, None], "b": ["x", "x"]}, backend)
    assert math.isnan(DataRecordConsistency().score(test))


def test_empty_frame_scores_nan(backend):
    test = make_frame({"a": [], "b": []}, backend)
    assert math.isnan(DataRecordConsistency().score(test))
