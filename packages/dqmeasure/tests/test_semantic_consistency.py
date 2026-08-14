from __future__ import annotations

import math
from datetime import datetime

import narwhals as nw
import pytest

from dqmeasure import SemanticConsistency
from dqmeasure.base import NotResolvedError
from dqmeasure.measures import semantic_consistency

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


# -- method="dc": approximate single-tuple denial-constraint mining ---------------------------


def test_dc_mines_difference_bound_and_flags_violations(backend):
    clean = make_frame({"a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]}, backend)
    measure = SemanticConsistency("a", confidence=1.0).fit(clean)

    # The surviving order relation a > b is tightened to the learned bound a - b >= 9 (the minimum
    # clean difference), which subsumes it.
    assert measure.rule_descriptions_ == ["a - b >= 9.0"]

    dirty = make_frame({"a": [15.0, 0.5, None], "b": [1.0, 1.0, 1.0]}, backend)
    cells = nw.from_native(measure.predict(dirty), series_only=True).to_list()
    assert cells[0] == 1.0
    assert cells[1] == 0.0
    assert cells[2] is None or math.isnan(cells[2])  # null subject -> out of scope
    assert measure.score(dirty) == pytest.approx(0.5)


def test_dc_mines_temporal_rule(backend):
    clean = make_frame(
        {
            "born": [datetime(1970, 1, 1), datetime(1980, 6, 1), datetime(1960, 1, 1)],
            "recruited": [datetime(1995, 1, 1), datetime(2000, 6, 1), datetime(1986, 1, 1)],
        },
        backend,
    )
    measure = SemanticConsistency("recruited", confidence=1.0).fit(clean)
    assert measure.rule_descriptions_ == ["recruited - born >= 7305 days, 0:00:00"]

    dirty = make_frame(
        {
            "born": [datetime(1970, 1, 1), datetime(1990, 1, 1), None],
            "recruited": [datetime(1995, 1, 1), datetime(1989, 1, 1), datetime(2000, 1, 1)],
        },
        backend,
    )
    cells = nw.from_native(measure.predict(dirty), series_only=True).to_list()
    assert cells[0] == 1.0
    assert cells[1] == 0.0  # recruited before born
    assert cells[2] is None or math.isnan(cells[2])  # null context -> out of scope
    assert measure.score(dirty) == pytest.approx(0.5)


def test_dc_mines_equality_rule_for_strings(backend):
    clean = make_frame({"city": ["Berlin", "Kiel"], "city_check": ["Berlin", "Kiel"]}, backend)
    measure = SemanticConsistency("city", confidence=1.0).fit(clean)
    assert "city == city_check" in measure.rule_descriptions_

    dirty = make_frame({"city": ["Berlin", "Hamburg"], "city_check": ["Berlin", "Kiel"]}, backend)
    assert measure.score(dirty) == pytest.approx(0.5)


def test_dc_confidence_tolerates_clean_noise(backend):
    # Two noise rows (one reversed, one equal) put every candidate below 100%: nothing survives at
    # confidence 1.0, while a > b survives at 0.98 and is tightened into its difference bound.
    clean = make_frame({"a": [10.0] * 98 + [0.0, 1.0], "b": [1.0] * 100}, backend)
    tolerant = SemanticConsistency("a", confidence=0.98).fit(clean)
    assert tolerant.rule_descriptions_ == ["a - b >= 9.0"]
    with pytest.warns(UserWarning, match="no rule survived"):
        strict = SemanticConsistency("a", confidence=1.0).fit(clean)
    assert strict.rule_descriptions_ == []


def test_dc_without_compatible_context_scores_nan(backend):
    clean = make_frame({"a": [1.0, 2.0]}, backend)
    with pytest.warns(UserWarning, match="no rule survived"):
        measure = SemanticConsistency("a").fit(clean)
    assert math.isnan(measure.score(clean))


# -- method="llm": language-model rule proposals ----------------------------------------------


def stub_llm(monkeypatch, response: str) -> list[str]:
    """Replace the chat-completions transport with a canned response; returns the prompts it receives."""
    prompts: list[str] = []

    def fake_openai_completion(model: str, url: str, api_key: str | None):
        def complete(prompt: str) -> str:
            prompts.append(prompt)
            return response

        return complete

    monkeypatch.setattr(semantic_consistency, "openai_completion", fake_openai_completion)
    return prompts


def test_llm_rules_validated_on_clean_data(backend, monkeypatch):
    clean = make_frame({"a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]}, backend)
    proposals = "[\"col('a') > col('b')\", \"col('a') < col('b')\", \"import os\", \"col('a')\"]"
    stub_llm(monkeypatch, proposals)
    measure = SemanticConsistency("a", method="llm", confidence=1.0).fit(clean)

    # The contradicted, the non-compiling, and the non-boolean proposal are discarded.
    assert measure.rule_descriptions_ == ["col('a') > col('b')"]

    dirty = make_frame({"a": [15.0, 0.5], "b": [1.0, 1.0]}, backend)
    assert measure.score(dirty) == pytest.approx(0.5)


def test_llm_called_once_at_fit_never_at_score(backend, monkeypatch):
    prompts = stub_llm(monkeypatch, "[\"col('a') > col('b')\"]")

    clean = make_frame({"a": [10.0, 20.0], "b": [1.0, 2.0]}, backend)
    measure = SemanticConsistency("a", method="llm", confidence=1.0).fit(clean)
    measure.score(clean)
    measure.score(clean)
    assert len(prompts) == 1


def test_llm_prompt_contains_schema_examples_and_column(backend, monkeypatch):
    prompts = stub_llm(monkeypatch, "[\"col('a') > col('b')\"]")

    clean = make_frame({"a": [10.0, 20.0], "b": [1.0, 2.0]}, backend)
    SemanticConsistency("a", method="llm", n_examples=2).fit(clean)
    assert "a" in prompts[0]
    assert "b" in prompts[0]
    assert "10" in prompts[0]  # a sampled record
    assert "'a'" in prompts[0]  # the scoped column


# edgecases


def test_unsupported_method_raises(backend):
    clean = make_frame({"a": [1.0], "b": [1.0]}, backend)
    with pytest.raises(ValueError, match="Unsupported method"):
        SemanticConsistency("a", method="tree").fit(clean)  # type: ignore[arg-type]


def test_missing_column_raises(backend):
    frame = make_frame({"a": [1.0]}, backend)
    with pytest.raises(ValueError, match="not found"):
        SemanticConsistency("z").fit(frame)


def test_not_fitted_raises(backend):
    frame = make_frame({"a": [1.0], "b": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        SemanticConsistency("a").score(frame)
