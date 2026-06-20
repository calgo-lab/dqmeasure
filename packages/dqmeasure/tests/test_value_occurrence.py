"""Tests for ValueOccurrenceCompleteness, parametrised over Polars and pandas backends."""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
import pytest

from dqmeasure import ValueOccurrenceCompleteness
from dqmeasure.base import NotResolvedError

# Backends to run every test against. pandas is optional; skip if not installed.
BACKENDS = ["polars"]
try:  # pragma: no cover - import guard
    import pandas as pd

    BACKENDS.append("pandas")
except ImportError:  # pragma: no cover
    pass


def make_frame(data: dict[str, Sequence[object]], backend: str):
    """Build a native frame in the requested backend."""
    if backend == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def test_measure_metadata():
    assert ValueOccurrenceCompleteness.iso_id == "Com-ML-2"
    assert ValueOccurrenceCompleteness.orientation == "higher-is-better"


def test_score_only_measure_has_no_predict():
    assert not hasattr(ValueOccurrenceCompleteness(), "predict")


def test_identical_instance_scores_one(backend):
    frame = make_frame({"color": ["red", "red", "blue", "green"]}, backend)
    measure = ValueOccurrenceCompleteness().fit(frame)
    assert measure.score(frame)["color"] == pytest.approx(1.0)


def test_missing_domain_value_lowers_score(backend):
    train = make_frame({"color": ["red", "red", "blue", "blue"]}, backend)
    # "blue" disappeared; its 2 expected occurrences are missing. A=2, B=4.
    test = make_frame({"color": ["red", "red", "red", "red"]}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.score(test)["color"] == pytest.approx(0.5)


def test_overrepresentation_is_capped(backend):
    train = make_frame({"color": ["red", "blue"]}, backend)
    # "red" is over-represented (3 observed, 2 expected) and may not compensate for the
    # missing "blue": A = min(3,2) + min(1,2) = 3, B = 4 — and never X > 1.
    test = make_frame({"color": ["red", "red", "red", "blue"]}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.score(test)["color"] == pytest.approx(0.75)


def test_out_of_domain_value_adds_nothing(backend):
    train = make_frame({"color": ["red", "blue"]}, backend)
    # "purple" is outside the learned domain: it contributes to neither A nor B.
    test = make_frame({"color": ["red", "purple"]}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.score(test)["color"] == pytest.approx(0.5)


def test_expectations_scale_with_frame_size(backend):
    train = make_frame({"color": ["red", "red", "blue", "blue"]}, backend)
    # Same proportions at twice the size still score 1.0 — expectations are proportions
    # scaled to the measured frame, not raw clean counts.
    test = make_frame({"color": ["red", "red", "red", "red", "blue", "blue", "blue", "blue"]}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.score(test)["color"] == pytest.approx(1.0)


def test_nulls_count_as_missing_occurrences(backend):
    train = make_frame({"color": ["red", "blue"]}, backend)
    # A null is not a domain value: one of "blue"'s expected occurrences is missing.
    test = make_frame({"color": ["red", None]}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.score(test)["color"] == pytest.approx(0.5)


def test_empty_frame_scores_nan(backend):
    train = make_frame({"color": ["red", "blue"]}, backend)
    test = make_frame({"color": []}, backend)

    measure = ValueOccurrenceCompleteness().fit(train)
    assert math.isnan(measure.score(test)["color"])


def test_selects_categorical_columns_by_default(backend):
    train = make_frame({"color": ["red", "blue"], "size": [1.0, 2.0]}, backend)
    measure = ValueOccurrenceCompleteness().fit(train)
    assert measure.columns_ == ["color"]


def test_specified_expected_skips_fit(backend):
    # Expert-specified domain and proportions: usable without fit.
    measure = ValueOccurrenceCompleteness(columns=["color"], expected={"color": {"red": 0.5, "blue": 0.5}})
    test = make_frame({"color": ["red", "red", "red", "red"]}, backend)

    # expected per value = 0.5 * 4 = 2; observed red capped at 2, blue missing. A=2, B=4.
    assert measure.score(test)["color"] == pytest.approx(0.5)


def test_not_fitted_raises(backend):
    test = make_frame({"color": ["red"]}, backend)
    with pytest.raises(NotResolvedError):
        ValueOccurrenceCompleteness().score(test)
