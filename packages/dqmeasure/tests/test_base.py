"""Tests for the BaseMeasure parameter protocol."""

from __future__ import annotations

from dqmeasure import BaseMeasure, DataAccuracyRange


def test_get_params_introspects_init_signature():
    measure = DataAccuracyRange(columns=["a"], inclusive=False)
    assert measure.get_params() == {
        "columns": ["a"],
        "low": None,
        "high": None,
        "method": "minmax",
        "inclusive": False,
    }


def test_get_params_on_base_subclass_without_extra_params():
    class Minimal(BaseMeasure):
        iso_id = "Tst-0"
        orientation = "higher-is-better"

    assert Minimal().get_params() == {"columns": None}


def test_set_params_roundtrip():
    measure = DataAccuracyRange().set_params(inclusive=False)
    assert measure.get_params()["inclusive"] is False
