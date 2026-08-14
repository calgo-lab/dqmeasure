from __future__ import annotations

from typing import Any

import pytest

from dqmeasure import SemanticDataAccuracy
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame


def cell_values(native_series: Any) -> list[Any]:
    import narwhals as nw

    return list(nw.from_native(native_series, series_only=True).to_list())


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
