from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import RiskOfDataInconsistency

from ._helpers import make_frame


def test_score_matches_iso_example(backend):
    frame = make_frame(
        {
            "tournament": ["Indiana Invitational", "Cleveland Open", "Des Moines Masters", "Indiana Invitational"],
            "year": [1998, 1999, 1999, 1999],
            "winner": ["Al Fredrickson", "Bob Albertson", "Al Fredrickson", "Ship Masterson"],
            "winner_date_of_birth": ["21 July 1975", "28 September 1968", "21 July 1975", "14 March 1977"],
        },
        backend,
    )
    # The standard's duplication ratios are 2/4, 3/4, 2/4, 2/4; we report their complements.
    expected = {"tournament": 2 / 4, "year": 1 / 4, "winner": 2 / 4, "winner_date_of_birth": 2 / 4}

    scores = {column: RiskOfDataInconsistency(column).score(frame) for column in expected}

    assert scores == pytest.approx(expected)
    # The standard's total for k=1: 9 duplications over the 4x4 table, i.e. 16 - 9 = 7 unique cells.
    assert sum(scores.values()) * 4 == pytest.approx(16 - 9)


def test_works_without_fit_and_fit_is_noop(backend):
    frame = make_frame({"id": ["a", "b", "a", "c"]}, backend)
    measure = RiskOfDataInconsistency("id")

    assert measure.score(frame) == pytest.approx(0.5)  # no fit needed; "b" and "c" are unique
    assert measure.fit(frame) is measure  # fit is allowed and learns nothing
    assert measure.score(frame) == pytest.approx(0.5)


def test_predict_flags_every_occurrence_and_skips_nulls(backend):
    test = make_frame({"id": ["a", "b", "a", None]}, backend)
    cells = nw.from_native(RiskOfDataInconsistency("id").predict(test), series_only=True).to_list()

    # Both occurrences of "a" are duplications, "b" is unique, the null is out of scope
    # (None on Polars, NaN on pandas).
    assert cells[:3] == [0.0, 1.0, 0.0]
    assert cells[3] is None or math.isnan(cells[3])


def test_nulls_are_not_duplicates_of_each_other(backend):
    test = make_frame({"id": ["a", None, None]}, backend)
    assert RiskOfDataInconsistency("id").score(test) == pytest.approx(1.0)


def test_unique_column_scores_one_and_constant_column_scores_zero(backend):
    unique = make_frame({"id": [1, 2, 3]}, backend)
    constant = make_frame({"id": [7, 7, 7]}, backend)

    assert RiskOfDataInconsistency("id").score(unique) == pytest.approx(1.0)
    assert RiskOfDataInconsistency("id").score(constant) == pytest.approx(0.0)


def test_no_units_in_scope_scores_nan(backend):
    empty = make_frame({"id": []}, backend)
    all_null = make_frame({"id": [None, None]}, backend)

    assert math.isnan(RiskOfDataInconsistency("id").score(empty))
    assert math.isnan(RiskOfDataInconsistency("id").score(all_null))


def test_missing_column_raises(backend):
    frame = make_frame({"id": ["a"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        RiskOfDataInconsistency("key").fit(frame)
    with pytest.raises(ValueError, match="not found"):
        RiskOfDataInconsistency("key").score(frame)
    with pytest.raises(ValueError, match="not found"):
        RiskOfDataInconsistency("key").predict(frame)
