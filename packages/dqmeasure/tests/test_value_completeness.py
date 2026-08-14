from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import FeatureCompleteness, ValueCompleteness

from ._helpers import make_frame


def test_takes_no_column():
    with pytest.raises(TypeError):
        ValueCompleteness("a")  # type: ignore[call-arg]


def test_design_doc_example_is_exact(backend):
    # An address table with 3 columns and 80 records where 39 country values are missing: X = 201/240.
    # The integer-count score path makes this exact, not approximate.
    test = make_frame(
        {
            "ID": list(range(80)),
            "country_code": ["DE"] * 80,
            "country": [None] * 39 + ["Germany"] * 41,
        },
        backend,
    )
    assert ValueCompleteness().score(test) == 201 / 240


def test_score_is_cell_ratio(backend):
    # 3 rows x 4 columns = 12 cells, 2 null -> 10/12.
    test = make_frame(
        {
            "a": [1, None, 3],
            "b": ["x", "y", "z"],
            "c": [1.0, 2.0, 3.0],
            "d": [None, "v", "w"],
        },
        backend,
    )
    assert ValueCompleteness().score(test) == pytest.approx(10 / 12)


def test_predict_is_per_record_completeness(backend):
    # predict() reports Com-I-1 record completeness: the fraction of non-null cells per row.
    test = make_frame({"a": [1, None, None], "b": ["x", "y", None], "c": [1.0, None, None]}, backend)
    records = ValueCompleteness().predict(test)

    col = nw.from_native(records, series_only=True).to_list()
    assert col == [1.0, pytest.approx(1 / 3), 0.0]


def test_score_is_mean_of_feature_completeness(backend):
    test = make_frame({"a": [1, None, 3, None], "b": ["x", "y", None, "w"]}, backend)
    per_column = [FeatureCompleteness(c).score(test) for c in ("a", "b")]

    assert ValueCompleteness().score(test) == pytest.approx(sum(per_column) / len(per_column))


def test_fit_is_a_noop(backend):
    train = make_frame({"a": [1]}, backend)
    test = make_frame({"a": [1, None]}, backend)
    assert ValueCompleteness().fit(train).score(test) == pytest.approx(0.5)


def test_empty_frame_scores_nan(backend):
    test = make_frame({"a": [], "b": []}, backend)
    assert math.isnan(ValueCompleteness().score(test))
