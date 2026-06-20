"""Tests for DataRecordConsistency, parametrised over Polars and pandas backends."""

from __future__ import annotations

from collections.abc import Sequence

import narwhals as nw
import polars as pl
import pytest

from dqmeasure import DataRecordConsistency
from dqmeasure.base import TABLE_SUBJECT, NotResolvedError

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
    assert DataRecordConsistency.iso_id == "Con-ML-1"
    assert DataRecordConsistency.orientation == "lower-is-better"


def test_all_unique_scores_zero(backend):
    train = make_frame({"a": [1, 2], "b": ["x", "y"]}, backend)
    test = make_frame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, backend)

    measure = DataRecordConsistency().fit(train)
    assert measure.score(test)[TABLE_SUBJECT] == pytest.approx(0.0)


def test_excess_copies_counted(backend):
    train = make_frame({"a": [1], "b": ["x"]}, backend)
    # A group of 3 identical records contributes 2 to A; 1 unique record. A=2, B=4.
    test = make_frame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]}, backend)

    measure = DataRecordConsistency().fit(train)
    assert measure.score(test)[TABLE_SUBJECT] == pytest.approx(0.5)


def test_key_column_subset(backend):
    train = make_frame({"id": [1], "noise": [0.0]}, backend)
    # Identical on "id" but different on "noise": duplicates only w.r.t. the key subset.
    test = make_frame({"id": [7, 7], "noise": [0.1, 0.2]}, backend)

    full = DataRecordConsistency().fit(train)
    keyed = DataRecordConsistency(columns=["id"]).fit(train)

    assert full.score(test)[TABLE_SUBJECT] == pytest.approx(0.0)
    assert keyed.score(test)[TABLE_SUBJECT] == pytest.approx(0.5)


def test_null_keyed_records_are_duplicates(backend):
    train = make_frame({"a": [1.0]}, backend)
    test = make_frame({"a": [None, None, 1.0]}, backend)

    measure = DataRecordConsistency().fit(train)
    # The two null records are identical: one excess copy out of three records.
    assert measure.score(test)[TABLE_SUBJECT] == pytest.approx(1 / 3)


def test_predict_shape_and_flags(backend):
    train = make_frame({"a": [1]}, backend)
    test = make_frame({"a": [5, 5, 6]}, backend)

    measure = DataRecordConsistency().fit(train)
    units = measure.predict(test)

    # Returned frame is in the caller's backend: one subject column, one row per record.
    assert type(units) is type(test)
    col = nw.from_native(units, eager_only=True)[TABLE_SUBJECT].to_list()
    assert col == [0.0, 1.0, 0.0]  # only the second occurrence is an excess copy


def test_explicit_columns_skip_fit(backend):
    # No reference to learn and columns given: usable without fit.
    test = make_frame({"a": [5, 5, 6]}, backend)
    measure = DataRecordConsistency(columns=["a"])
    assert measure.score(test)[TABLE_SUBJECT] == pytest.approx(1 / 3)


def test_not_fitted_raises(backend):
    test = make_frame({"a": [1]}, backend)
    with pytest.raises(NotResolvedError):
        DataRecordConsistency().predict(test)
    with pytest.raises(NotResolvedError):
        DataRecordConsistency().score(test)
