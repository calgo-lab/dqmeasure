from pathlib import Path

import pytest

from experiments.data import Dataset, load_hospital
from experiments.errors import inject

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The scenarios HOSP can run; SynTabFall's scenario names its columns.
CORRUPTING = ["missing", "typo", "outlier", "swap", "blank_record", "wrong_unit"]


@pytest.fixture(scope="module")
def hospital() -> Dataset:
    return load_hospital(DATA_DIR)


@pytest.mark.parametrize("scenario", CORRUPTING)
def test_same_seed_corrupts_identically(scenario: str, hospital: Dataset) -> None:
    first, _ = inject(hospital.frame, scenario, 0.25, seed=0)
    second, _ = inject(hospital.frame, scenario, 0.25, seed=0)
    assert first.equals(second)


@pytest.mark.parametrize("scenario", CORRUPTING)
def test_different_seed_corrupts_differently(scenario: str, hospital: Dataset) -> None:
    first, _ = inject(hospital.frame, scenario, 0.25, seed=0)
    other, _ = inject(hospital.frame, scenario, 0.25, seed=1)
    assert not first.equals(other)
