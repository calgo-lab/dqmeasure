from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# HOSP spells a missing value as this word rather than leaving the field empty.
MISSING_LITERAL = "empty"


@dataclass(frozen=True)
class Dataset:
    """A prepared df, its target column, and the target's positive class if it is binary."""

    name: str
    frame: pd.DataFrame
    target: str
    positive_label: str | None = None


def load_hospital(data_dir: Path) -> Dataset:
    """HOSP, clean, 1000 records."""
    raw = pd.read_csv(
        data_dir / "hospital" / "clean.csv",
        dtype=str,  # no dtype inference: identifiers keep their leading zeros etc.
        na_values=[MISSING_LITERAL],
    )

    # some wrangling as explained in the paper
    frame = raw.drop(columns=["index", "Address2", "Address3"]).rename(columns={"StateAverage": "Stateavg"})
    frame["Score"] = frame["Score"].str.removesuffix("%").astype(float)
    frame["Sample"] = frame["Sample"].str.removesuffix(" patients").astype(float)

    return Dataset(name="hospital", frame=frame.reset_index(drop=True), target="Condition")


def load_syntabfall(data_dir: Path) -> Dataset:
    """SynTabFall, 745,380 records."""
    raw = pd.read_csv(
        data_dir / "syntabfall" / "SynTabFall.csv",
        dtype=str,  # no dtype inference: ICD-10 and OPS codes keep their exact form
    )
    frame = raw.drop(columns=["id"])
    frame["age"] = frame["age"].astype(float)

    return Dataset(
        name="syntabfall",
        frame=frame.reset_index(drop=True),
        target="fallen",
        positive_label="True",
    )


_LOADERS = {"hospital": load_hospital, "syntabfall": load_syntabfall}


def load_dataset(name: str, data_dir: Path) -> Dataset:
    """Load a dataset by name."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r} (known: {sorted(_LOADERS)})")
    return _LOADERS[name](data_dir)
