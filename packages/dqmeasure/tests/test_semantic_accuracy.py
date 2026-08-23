from __future__ import annotations

import json
import random
import urllib.request
import warnings
from email.message import Message
from typing import Any
from urllib.error import HTTPError

import pytest

from dqmeasure import SemanticDataAccuracy
from dqmeasure.base import NotResolvedError
from dqmeasure.measures import _llm
from dqmeasure.measures._llm import complete_many, is_missing, render_record

from ._helpers import make_frame


def cell_values(native_series: Any) -> list[Any]:
    import narwhals as nw

    return list(nw.from_native(native_series, series_only=True).to_list())


def is_null(value: Any) -> bool:
    """pandas represents a null float as `nan`, polars as `None`; treat both as missing."""
    return value is None or value != value


def test_missing_column_raises(backend):
    train = make_frame({"a": ["x"], "b": ["y"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        SemanticDataAccuracy("c").fit(train)


def test_not_fitted_raises(backend):
    test = make_frame({"city": ["Berlin"], "country": ["Germany"]}, backend)
    # The reference cannot come from the constructor: fit is mandatory.
    with pytest.raises(NotResolvedError):
        SemanticDataAccuracy("country").predict(test)
    with pytest.raises(NotResolvedError):
        SemanticDataAccuracy("country").score(test)


def test_llm_examples_sampled_at_fit(backend):
    train = make_frame({"breed": [f"breed-{i}" for i in range(50)], "size": ["large"] * 50}, backend)
    measure = SemanticDataAccuracy("breed", n_examples=3).fit(train)

    assert len(measure.examples_) == 3
    assert all(set(example) == {"breed", "size"} for example in measure.examples_)


# -- prompt rendering -------------------------------------------------------------------------


def test_prompt_lists_column_names_once_and_matches_value_field_count(backend):
    train = make_frame({"city": ["Berlin"], "country": ["Germany"], "size": ["large"]}, backend)
    measure = SemanticDataAccuracy("city", n_examples=1).fit(train)
    prompt = measure._prompt({"city": "Munich", "country": "Germany", "size": "large"})

    assert prompt.count("city") == 2  # the header, and "Column: city"
    assert prompt.count("country") == 1  # only in the header; not repeated per record
    header_line = next(line for line in prompt.splitlines() if line.startswith("Columns: "))
    n_columns = len(header_line.removeprefix("Columns: ").split(" | "))
    record_line = next(line for line in prompt.splitlines() if line.startswith("Record: "))
    n_fields = len(record_line.removeprefix("Record: ").split(" | "))
    assert n_fields == n_columns


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_missing_value_rendered_as_missing_token(backend, missing_value):
    train = make_frame({"city": ["Berlin"], "country": ["Germany"]}, backend)
    measure = SemanticDataAccuracy("city", n_examples=1).fit(train)
    prompt = measure._prompt({"city": missing_value, "country": "Germany"})

    assert "Value: <missing>" in prompt


def test_render_record_treats_all_missing_encodings_as_missing_token():
    import pandas as pd

    row = {"a": None, "b": float("nan"), "c": pd.NA, "d": pd.NaT, "e": "ok"}
    assert render_record(row) == "<missing> | <missing> | <missing> | <missing> | ok"


def test_is_missing_pd_na_does_not_raise_typeerror():
    import pandas as pd

    assert is_missing(pd.NA) is True


# -- complete_many ------------------------------------------------------------------------------


def test_complete_many_preserves_order():
    def complete(prompt: str) -> str:
        return prompt.upper()

    prompts = [f"p{i}" for i in range(10)]
    assert complete_many(complete, prompts, n_jobs=1) == [p.upper() for p in prompts]
    assert complete_many(complete, prompts, n_jobs=4) == [p.upper() for p in prompts]
