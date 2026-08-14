from __future__ import annotations

import math

import narwhals as nw
import pytest

from dqmeasure import EmptyRecords

from ._helpers import make_frame


def test_takes_no_column():
    with pytest.raises(TypeError):
        EmptyRecords("a")  # type: ignore[call-arg]


def test_only_all_null_records_count_as_empty(backend):
    # 5 records, 2 fully empty; partially-null records carry data -> X = 1 - 2/5 = 0.6.
    test = make_frame(
        {"a": [1, None, 3, None, None], "b": ["x", None, None, None, "w"]},
        backend,
    )
    measure = EmptyRecords()

    records = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert records == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert measure.score(test) == pytest.approx(0.6)


def test_empty_frame_scores_nan(backend):
    test = make_frame({"a": [], "b": []}, backend)
    assert math.isnan(EmptyRecords().score(test))
