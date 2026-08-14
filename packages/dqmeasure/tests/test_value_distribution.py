from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from dqmeasure import DataValueDistribution
from dqmeasure.base import NotResolvedError
from dqmeasure.measures.value_distribution import _ks_statistic

from ._helpers import make_frame

# -- ordered columns: Kolmogorov-Smirnov ----------------------------------------------------


def test_ks_identical_distributions_score_one(backend):
    frame = make_frame({"x": [1.0, 2.0, 3.0, 4.0]}, backend)
    assert DataValueDistribution("x").fit(frame).score(frame) == pytest.approx(1.0)


def test_ks_disjoint_distributions_score_zero(backend):
    clean = make_frame({"x": [1.0, 2.0, 3.0]}, backend)
    dirty = make_frame({"x": [10.0, 11.0]}, backend)
    assert DataValueDistribution("x").fit(clean).score(dirty) == pytest.approx(0.0)


def test_ks_hand_computed_shift(backend):
    # Reference {1,2,3,4}, observed {1,2,3,100}: the ECDFs disagree most just past 4 (1 vs 3/4),
    # a distance of 0.25 that we report as its complement.
    clean = make_frame({"x": [1.0, 2.0, 3.0, 4.0]}, backend)
    dirty = make_frame({"x": [1.0, 2.0, 3.0, 100.0]}, backend)
    assert DataValueDistribution("x").fit(clean).score(dirty) == pytest.approx(0.75)


def test_ks_matches_scipy_reference_implementation():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = random.Random(0)
    cases: list[tuple[list[float], list[float]]] = [
        # adversarial: constants, heavy ties, tiny and integer-grid samples
        ([1.0] * 10, [1.0] * 7),
        ([1.0] * 10, [2.0] * 3),
        ([1.0, 1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 2.0, 2.0]),
        ([0.0], [0.0]),
        ([0.0], [1.0]),
        ([float(rng.randint(0, 3)) for _ in range(30)], [float(rng.randint(0, 3)) for _ in range(25)]),
    ]
    for _ in range(20):
        a = [rng.gauss(0, 1) for _ in range(rng.randint(1, 50))]
        b = [rng.gauss(rng.uniform(-2, 2), rng.uniform(0.5, 2)) for _ in range(rng.randint(1, 50))]
        cases.append((a, b))
    for a, b in cases:
        ours = _ks_statistic(sorted(a), sorted(b))
        theirs = float(scipy_stats.ks_2samp(a, b).statistic)
        assert ours == pytest.approx(theirs)


def test_ks_on_datetime_column(backend):
    clean = make_frame({"d": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}, backend)
    dirty = make_frame({"d": [datetime(2021, 1, 1), datetime(2021, 1, 2)]}, backend)
    assert DataValueDistribution("d").fit(clean).score(dirty) == pytest.approx(0.0)


def test_ks_on_date_column_polars():
    # pandas has no plain date dtype; the Date branch is polars-only.
    clean = pl.DataFrame({"d": [date(2020, 1, 1), date(2020, 1, 2)]})
    dirty = pl.DataFrame({"d": [date(2021, 1, 1), date(2021, 1, 2)]})
    assert DataValueDistribution("d").fit(clean).score(dirty) == pytest.approx(0.0)


def test_ks_nulls_dropped_on_both_sides(backend):
    clean = make_frame({"x": [1.0, 2.0, None]}, backend)
    dirty = make_frame({"x": [1.0, None, 2.0]}, backend)
    assert DataValueDistribution("x").fit(clean).score(dirty) == pytest.approx(1.0)


def test_specified_reference_sample_skips_fit(backend):
    measure = DataValueDistribution("x", expected=[1.0, 2.0, 3.0, 4.0])
    dirty = make_frame({"x": [1.0, 2.0, 3.0, 100.0]}, backend)
    assert measure.score(dirty) == pytest.approx(0.75)


# -- unordered columns: total variation distance --------------------------------------------


def test_tvd_hand_computed_with_novel_category(backend):
    # Reference a:1/2, b:1/2; observed a:1/4, b:1/2, z:1/4 -> TVD = (1/4 + 0 + 1/4) / 2 = 1/4,
    # which we report as its complement.
    clean = make_frame({"c": ["a", "a", "b", "b"]}, backend)
    dirty = make_frame({"c": ["a", "b", "b", "z"]}, backend)
    assert DataValueDistribution("c").fit(clean).score(dirty) == pytest.approx(0.75)


def test_tvd_identical_and_disjoint(backend):
    clean = make_frame({"c": ["a", "b"]}, backend)
    assert DataValueDistribution("c").fit(clean).score(clean) == pytest.approx(1.0)
    disjoint = make_frame({"c": ["y", "z"]}, backend)
    assert DataValueDistribution("c").fit(clean).score(disjoint) == pytest.approx(0.0)


def test_tvd_on_boolean_column(backend):
    clean = make_frame({"f": [True, True, False, False]}, backend)
    dirty = make_frame({"f": [True, True, True, False]}, backend)
    assert DataValueDistribution("f").fit(clean).score(dirty) == pytest.approx(0.75)


def test_specified_mapping_is_normalized(backend):
    # Counts work as weights: {a: 2, b: 2} means the same reference as {a: 0.5, b: 0.5}.
    dirty = make_frame({"c": ["a", "b", "b", "z"]}, backend)
    assert DataValueDistribution("c", expected={"a": 2, "b": 2}).score(dirty) == pytest.approx(0.75)


def test_mismatched_reference_kind_raises(backend):
    ordered = make_frame({"x": [1.0, 2.0]}, backend)
    with pytest.raises(ValueError, match="reference sample"):
        DataValueDistribution("x", expected={"a": 0.5}).score(ordered)
    unordered = make_frame({"c": ["a", "b"]}, backend)
    with pytest.raises(ValueError, match="mapping"):
        DataValueDistribution("c", expected=["a", "b"]).score(unordered)


# -- edges ----------------------------------------------------------------------------------


def test_all_null_column_scores_nan(backend):
    clean = make_frame({"x": [1.0, 2.0]}, backend)
    # An all-null column needs an explicit dtype; inference has nothing to go on.
    if backend == "polars":
        dirty = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Float64})
    else:
        dirty = pd.DataFrame({"x": [None, None]}, dtype="float64")
    assert math.isnan(DataValueDistribution("x").fit(clean).score(dirty))


def test_empty_reference_scores_nan(backend):
    dirty = make_frame({"x": [1.0, 2.0]}, backend)
    assert math.isnan(DataValueDistribution("x", expected=[]).score(dirty))
    dirty_cat = make_frame({"c": ["a"]}, backend)
    assert math.isnan(DataValueDistribution("c", expected={}).score(dirty_cat))


def test_unsupported_dtype_raises(backend):
    frame = make_frame({"t": [timedelta(days=1), timedelta(days=2)]}, backend)
    with pytest.raises(ValueError, match="dtype"):
        DataValueDistribution("t").fit(frame)


def test_missing_column_raises(backend):
    frame = make_frame({"x": [1.0]}, backend)
    with pytest.raises(ValueError, match="not found"):
        DataValueDistribution("y").fit(frame)


def test_not_fitted_raises(backend):
    frame = make_frame({"x": [1.0]}, backend)
    with pytest.raises(NotResolvedError):
        DataValueDistribution("x").score(frame)
