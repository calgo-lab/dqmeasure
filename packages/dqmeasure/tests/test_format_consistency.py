from __future__ import annotations

import math

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from dqmeasure import DataFormatConsistency
from dqmeasure.base import NotResolvedError

from ._helpers import make_frame


def test_iso_example_ten_of_fifteen(backend):
    # Example from ISO/IEC 25024, Table 3: canonical format yyyymm;
    # 10 of 15 date strings conform -> X = 10/15.
    dates = ["202401"] * 10 + ["2401"] * 5
    measure = DataFormatConsistency("date", formats={"dddddd"})
    assert measure.score(make_frame({"date": dates}, backend)) == pytest.approx(10 / 15)


def test_fit_learns_shapes_from_clean_data(backend):
    clean = make_frame({"date": ["202401", "202502"]}, backend)
    measure = DataFormatConsistency("date").fit(clean)
    assert measure.formats_ == {"dddddd"}

    dirty = make_frame({"date": ["202401"] * 10 + ["2401"] * 5}, backend)
    assert measure.score(dirty) == pytest.approx(10 / 15)


def test_shape_alphabet(backend):
    # Digits -> d, letters -> a, everything else literal: separators and casing distinguish formats.
    clean = make_frame({"code": ["AB-12"]}, backend)
    measure = DataFormatConsistency("code").fit(clean)
    assert measure.formats_ == {"aa-dd"}

    test = make_frame({"code": ["xy-99", "xy/99", "x-99", "AB-12"]}, backend)
    assert measure.score(test) == pytest.approx(2 / 4)


def test_multiple_learned_shapes_all_admissible(backend):
    clean = make_frame({"v": ["2024-01", "202401"]}, backend)
    measure = DataFormatConsistency("v").fit(clean)
    test = make_frame({"v": ["2025-12", "202512", "25-12"]}, backend)
    assert measure.score(test) == pytest.approx(2 / 3)


def test_predict_cell_level_and_nulls_preserved(backend):
    measure = DataFormatConsistency("v", formats={"dddd"})
    test = make_frame({"v": ["2024", "24", None]}, backend)
    col = nw.from_native(measure.predict(test), series_only=True).to_list()
    assert col[0] == 1.0
    assert col[1] == 0.0
    assert col[2] is None or math.isnan(col[2])  # null in -> missing out


def test_integer_column_allowed(backend):
    measure = DataFormatConsistency("v", formats={"i"})
    test = make_frame({"v": [10]}, backend)

    col = nw.from_native(measure.predict(test), series_only=True).to_list()

    assert col == [1.0]


def test_all_null_column_scores_nan(backend):
    measure = DataFormatConsistency("v", formats={"dddd"})
    # An all-null column needs an explicit dtype; inference has nothing to go on.
    if backend == "polars":
        frame = pl.DataFrame({"v": [None, None]}, schema={"v": pl.String})
    else:
        frame = pd.DataFrame({"v": [None, None]}, dtype="string")
    assert math.isnan(measure.score(frame))


def test_non_string_column_raises(backend):
    frame = make_frame({"x": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="string"):
        DataFormatConsistency("x").fit(frame)


def test_missing_column_raises(backend):
    frame = make_frame({"v": ["a"]}, backend)
    with pytest.raises(ValueError, match="not found"):
        DataFormatConsistency("w").fit(frame)


def test_not_fitted_raises(backend):
    frame = make_frame({"v": ["a"]}, backend)
    with pytest.raises(NotResolvedError):
        DataFormatConsistency("v").score(frame)
