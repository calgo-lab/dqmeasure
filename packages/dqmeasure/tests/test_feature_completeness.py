from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import FeatureCompleteness

from ._helpers import make_frame


def test_score_is_iso_ratio(backend):
    # ISO example: "Email" should be populated for all 1000 customers; 150 are null -> X = 850/1000.
    values = ["a@example.com"] * 850 + [None] * 150
    test = make_frame({"email": values}, backend)

    assert FeatureCompleteness("email").score(test) == pytest.approx(0.85)


def test_works_without_fit_and_fit_is_noop(backend):
    frame = make_frame({"country": ["DE", None, "FR", None]}, backend)
    measure = FeatureCompleteness("country")

    assert measure.score(frame) == pytest.approx(0.5)  # no fit needed
    assert measure.fit(frame) is measure  # fit is allowed and learns nothing
    assert measure.score(frame) == pytest.approx(0.5)


def test_predict_counts_every_cell(backend):
    test = make_frame({"email": ["a", None, "b"]}, backend)
    cells = FeatureCompleteness("email").predict(test)

    # Every cell is a unit: the series is null-free, 1.0 = present, 0.0 = null.
    assert nw.from_native(cells, series_only=True).to_list() == [1.0, 0.0, 1.0]


def test_all_null_column_scores_zero(backend):
    test = make_frame({"email": [None, None, None]}, backend)
    assert FeatureCompleteness("email").score(test) == pytest.approx(0.0)


def test_empty_frame_scores_nan(backend):
    test = make_frame({"email": []}, backend)
    assert math.isnan(FeatureCompleteness("email").score(test))


def test_missing_column_raises(backend):
    frame = make_frame({"email": ["a"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        FeatureCompleteness("phone").fit(frame)
    with pytest.raises(ValueError, match="not found"):
        FeatureCompleteness("phone").score(frame)
    with pytest.raises(ValueError, match="not found"):
        FeatureCompleteness("phone").predict(frame)
