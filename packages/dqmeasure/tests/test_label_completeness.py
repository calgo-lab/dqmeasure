from __future__ import annotations

import narwhals as nw
import pytest

from dqmeasure import FeatureCompleteness, LabelCompleteness

from ._helpers import make_frame


def test_score_is_one_minus_unlabelled_ratio(backend):
    # 10 samples, 3 unlabelled (null): A = 3, X = 1 - 3/10 = 0.7.
    values = ["cat"] * 5 + ["dog"] * 2 + [None, None, None]
    test = make_frame({"label": values}, backend)

    assert LabelCompleteness("label").score(test) == pytest.approx(0.7)


def test_sentinels_are_ordinary_labels(backend):
    # Missing labels are assumed null-encoded (model doc, §8): "" and "?" count as labels.
    test = make_frame({"label": ["cat", "", "?", None]}, backend)

    assert LabelCompleteness("label").score(test) == pytest.approx(0.75)


def test_predict_counts_every_sample(backend):
    test = make_frame({"label": ["cat", None, "dog"]}, backend)
    samples = LabelCompleteness("label").predict(test)

    # Every sample is a unit: the series is null-free, 1.0 = labelled.
    assert nw.from_native(samples, series_only=True).to_list() == [1.0, 0.0, 1.0]


def test_works_without_fit_and_fit_is_noop(backend):
    frame = make_frame({"label": ["cat", None]}, backend)
    measure = LabelCompleteness("label")

    assert measure.score(frame) == pytest.approx(0.5)  # no fit needed
    assert measure.fit(frame) is measure  # fit is allowed and learns nothing


def test_coincides_with_feature_completeness(backend):
    # Numerically equal to Com-ML-3 on the label column (see the model doc, §8).
    frame = make_frame({"label": ["cat", "dog", None, "cat", None]}, backend)

    assert LabelCompleteness("label").score(frame) == FeatureCompleteness("label").score(frame)


def test_missing_column_raises(backend):
    frame = make_frame({"label": ["cat"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        LabelCompleteness("target").score(frame)
