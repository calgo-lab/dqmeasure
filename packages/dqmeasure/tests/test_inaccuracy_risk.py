from __future__ import annotations

import math

import pytest

from dqmeasure import RiskOfDataSetInaccuracy
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame


def test_fit_learns_median_and_mad(backend):
    train = make_frame({"weight": [1.0, 2.0, 3.0, 4.0, 5.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight").fit(train)

    assert measure.center_ == 3.0
    assert measure.scale_ == pytest.approx(1.4826)  # MAD = 1, sigma-scaled


def test_score_is_iso_ratio(backend):
    # The ISO 25024 example distribution: 2000 is the outlier.
    train = make_frame({"weight": [100.0, 105.0, 120.0, 80.0, 75.0, 60.0, 130.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight").fit(train)

    test = make_frame({"weight": [100.0, 105.0, 120.0, 80.0, 2000.0]}, backend)
    # The standard reports the risk 1/5; we report its complement, the ratio of non-outliers.
    assert measure.score(test) == pytest.approx(4 / 5)


def test_robust_to_outliers_in_clean_data(backend):
    # An outlier in the clean data must not widen the reference enough to mask real outliers.
    train = make_frame({"weight": [100.0, 105.0, 120.0, 80.0, 75.0, 60.0, 130.0, 2000.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight").fit(train)

    test = make_frame({"weight": [100.0, 2000.0]}, backend)
    assert measure.score(test) == pytest.approx(0.5)


def test_predict_cell_level_and_nulls_preserved(backend):
    train = make_frame({"weight": [100.0, 105.0, 120.0, 80.0, 75.0, 60.0, 130.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight").fit(train)

    test = make_frame({"weight": [100.0, 2000.0, None]}, backend)
    import narwhals as nw

    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 1.0  # unremarkable value
    assert col[1] == 0.0  # outlier
    assert col[2] is None or math.isnan(col[2])  # null in -> missing out


def test_specified_reference_skips_fit(backend):
    measure = RiskOfDataSetInaccuracy("weight", center=100.0, scale=10.0)
    test = make_frame({"weight": [100.0, 120.0, 200.0]}, backend)

    # |200 - 100| = 100 > 3.5 * 10; the others are within the threshold.
    assert measure.score(test) == pytest.approx(2 / 3)
    assert measure.center_ == 100.0
    assert measure.scale_ == 10.0


def test_partial_spec_learns_rest(backend):
    train = make_frame({"weight": [1.0, 2.0, 3.0, 4.0, 5.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight", center=3.0).fit(train)

    assert measure.center_ == 3.0
    assert measure.scale_ == pytest.approx(1.4826)


def test_threshold_tightens_criterion(backend):
    train = make_frame({"weight": [1.0, 2.0, 3.0, 4.0, 5.0]}, backend)
    test = make_frame({"weight": [3.0, 5.0]}, backend)

    default = RiskOfDataSetInaccuracy("weight").fit(train)
    tight = RiskOfDataSetInaccuracy("weight", threshold=1.0).fit(train)

    assert default.score(test) == pytest.approx(1.0)  # 5 is within 3.5 robust sigmas
    assert tight.score(test) == pytest.approx(0.5)  # but not within 1


def test_constant_clean_column_flags_any_deviation(backend):
    train = make_frame({"weight": [5.0, 5.0, 5.0, 5.0]}, backend)
    measure = RiskOfDataSetInaccuracy("weight").fit(train)

    assert measure.scale_ == 0.0
    test = make_frame({"weight": [5.0, 6.0]}, backend)
    assert measure.score(test) == pytest.approx(0.5)


def test_missing_column_raises(backend):
    train = make_frame({"weight": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="not found"):
        RiskOfDataSetInaccuracy("height").fit(train)


def test_non_numeric_column_raises(backend):
    train = make_frame({"label": ["a", "b"]}, backend)
    with pytest.raises(ValueError, match="numeric"):
        RiskOfDataSetInaccuracy("label").fit(train)


def test_unsupported_method_raises(backend):
    train = make_frame({"weight": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="Unsupported method"):
        RiskOfDataSetInaccuracy("weight", method="iqr").fit(train)  # type: ignore[arg-type]


def test_not_fitted_raises(backend):
    test = make_frame({"weight": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        RiskOfDataSetInaccuracy("weight").predict(test)
    with pytest.raises(NotResolvedError):
        RiskOfDataSetInaccuracy("weight").score(test)
