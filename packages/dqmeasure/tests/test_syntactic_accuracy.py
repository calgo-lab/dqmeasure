from __future__ import annotations

import math

import pytest

from dqmeasure import SyntacticDataAccuracy
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame


def test_fit_learns_observed_domain(backend):
    train = make_frame({"breed": ["Labrador", "Poodle", "Labrador", None]}, backend)
    measure = SyntacticDataAccuracy("breed").fit(train)

    assert measure.domain_ == {"Labrador", "Poodle"}


def test_score_is_iso_ratio(backend):
    # The ISO 25024 example: "Laborador" is a misspelling of "Labrador"; 9 of 10 values are accurate.
    train = make_frame({"breed": ["Labrador", "Poodle", "Beagle"]}, backend)
    measure = SyntacticDataAccuracy("breed").fit(train)

    values = ["Labrador"] * 5 + ["Poodle", "Beagle", "Poodle", "Beagle", "Laborador"]
    test = make_frame({"breed": values}, backend)

    assert measure.score(test) == pytest.approx(0.9)


def test_predict_cell_level_and_nulls_preserved(backend):
    train = make_frame({"breed": ["Labrador", "Poodle"]}, backend)
    measure = SyntacticDataAccuracy("breed").fit(train)

    test = make_frame({"breed": ["Labrador", "Laborador", None]}, backend)
    import narwhals as nw

    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 1.0  # "Labrador" is a domain member
    assert col[1] == 0.0  # "Laborador" is not
    assert col[2] is None or math.isnan(col[2])  # null in -> missing out


def test_any_dtype_works(backend):
    # A numeric code column: any dtype can carry a domain.
    train = make_frame({"zip": [10115, 20095]}, backend)
    measure = SyntacticDataAccuracy("zip").fit(train)

    test = make_frame({"zip": [10115, 99999]}, backend)
    assert measure.score(test) == pytest.approx(0.5)


def test_specified_domain_skips_fit(backend):
    # The domain given by an expert: the measure is usable without fit.
    measure = SyntacticDataAccuracy("breed", domain=["Labrador", "Poodle"])
    test = make_frame({"breed": ["Labrador", "Laborador", "Poodle", "Beagle"]}, backend)

    assert measure.score(test) == pytest.approx(0.5)
    assert measure.domain_ == ["Labrador", "Poodle"]


def test_all_null_column_scores_nan(backend):
    # No units in scope (B = 0): X is undefined and reported as NaN.
    measure = SyntacticDataAccuracy("breed", domain=["Labrador"])
    test = make_frame({"breed": [None, None]}, backend)

    assert math.isnan(measure.score(test))


def test_missing_column_raises(backend):
    train = make_frame({"breed": ["Labrador"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        SyntacticDataAccuracy("color").fit(train)


def test_unsupported_method_raises(backend):
    train = make_frame({"breed": ["Labrador"]}, backend)
    with pytest.raises(ValueError, match="Unsupported method"):
        SyntacticDataAccuracy("breed", method="dtype").fit(train)  # type: ignore[arg-type]


def test_not_fitted_raises(backend):
    test = make_frame({"breed": ["Labrador"]}, backend)
    with pytest.raises(NotResolvedError):
        SyntacticDataAccuracy("breed").predict(test)
    with pytest.raises(NotResolvedError):
        SyntacticDataAccuracy("breed").score(test)
