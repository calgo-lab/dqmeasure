from __future__ import annotations

import math
from datetime import datetime

import narwhals as nw
import pytest

from dqmeasure import SemanticConsistency
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame

# -- specified rules -------------------------------------------------------------------------


def test_iso_example_ninety_five_of_hundred(backend):
    # ISO/IEC 25024, Table 3: recruitment must be after birth; 95 of 100 records satisfy -> X = 95/100.
    frame = make_frame(
        {
            "born": [datetime(1970, 1, 1)] * 100,
            "recruited": [datetime(1990, 1, 1)] * 95 + [datetime(1960, 1, 1)] * 5,
        },
        backend,
    )
    measure = SemanticConsistency("recruited", rules=[nw.col("recruited") > nw.col("born")])
    assert measure.score(frame) == pytest.approx(0.95)


def test_specified_rules_conjunction(backend):
    # Both rules must hold: rows failing either count as violations.
    frame = make_frame({"a": [5.0, 5.0, 5.0], "b": [1.0, 6.0, 1.0], "c": [10.0, 10.0, 1.0]}, backend)
    measure = SemanticConsistency("a", rules=[nw.col("a") > nw.col("b"), nw.col("a") < nw.col("c")])
    assert measure.score(frame) == pytest.approx(1 / 3)


# -- mined rules: single-column fds ---------------------------------------


def test_mines_lookup_and_flags_violations(backend):
    clean = make_frame({"zip": ["12345", "12345", "67890"], "city": ["Berlin", "Berlin", "Hamburg"]}, backend)
    measure = SemanticConsistency("city").fit(clean)
    assert measure.rule_descriptions_ == ["zip -> city"]

    # The middle row keeps a zip the clean data pairs with Berlin, but claims Hamburg.
    dirty = make_frame({"zip": ["12345", "12345", "67890"], "city": ["Berlin", "Hamburg", "Hamburg"]}, backend)
    cells = nw.from_native(measure.predict(dirty), series_only=True).to_list()
    assert cells == [1.0, 0.0, 1.0]
    assert measure.score(dirty) == pytest.approx(2 / 3)


def test_unseen_key_is_out_of_scope(backend):
    # A key the clean data never showed has no expected value, so the rule says nothing rather than 0.
    clean = make_frame({"zip": ["12345"], "city": ["Berlin"]}, backend)
    measure = SemanticConsistency("city").fit(clean)

    dirty = make_frame({"zip": ["12345", "99999"], "city": ["Berlin", "Munich"]}, backend)
    cells = nw.from_native(measure.predict(dirty), series_only=True).to_list()
    assert cells[0] == 1.0
    assert cells[1] is None or math.isnan(cells[1])
    assert measure.score(dirty) == pytest.approx(1.0)


def test_null_subject_or_context_is_out_of_scope(backend):
    clean = make_frame({"zip": ["12345", "67890"], "city": ["Berlin", "Hamburg"]}, backend)
    measure = SemanticConsistency("city").fit(clean)

    dirty = make_frame({"zip": ["12345", None, "67890"], "city": ["Berlin", "Berlin", None]}, backend)
    cells = nw.from_native(measure.predict(dirty), series_only=True).to_list()
    assert cells[0] == 1.0
    assert all(cell is None or math.isnan(cell) for cell in cells[1:])


def test_key_with_two_values_is_not_a_dependency(backend):
    # One zip carries two cities, so it determines nothing and no rule is left to measure against.
    clean = make_frame({"zip": ["12345", "12345", "67890"], "city": ["Berlin", "Kiel", "Hamburg"]}, backend)
    with pytest.warns(UserWarning, match="no rule survived"):
        measure = SemanticConsistency("city").fit(clean)
    assert measure.rule_descriptions_ == []
    assert math.isnan(measure.score(clean))


def test_without_context_column_scores_nan(backend):
    clean = make_frame({"city": ["Berlin", "Hamburg"]}, backend)
    with pytest.warns(UserWarning, match="no rule survived"):
        measure = SemanticConsistency("city").fit(clean)
    assert math.isnan(measure.score(clean))


# edgecases


def test_missing_column_raises(backend):
    frame = make_frame({"a": [1.0]}, backend)
    with pytest.raises(ValueError, match="not found"):
        SemanticConsistency("z").fit(frame)


def test_not_fitted_raises(backend):
    frame = make_frame({"a": [1.0], "b": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        SemanticConsistency("a").score(frame)
