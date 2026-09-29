"""Error scenarios injected into the test fold.

HOSP gets the six generic scenarios: `blank_record` empties whole rows, the other five apply one error type under
ECAR to every column of matching dtype. SynTabFall gets a scenario of its own, "ward_intake". `inject` returns the\
corrupted frame and the mask of the cells whose value changed.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from tab_err import ErrorModel, ErrorType, error_type
from tab_err.api import MidLevelConfig, mid_level
from tab_err.error_mechanism import EAR, ECAR, ENAR
from tab_err.error_type import ErrorTypeConfig

SCENARIOS = ("clean", "missing", "typo", "outlier", "swap", "blank_record", "wrong_unit", "ward_intake")

WRONG_UNIT = ErrorTypeConfig(wrong_unit_scaling=lambda value: value / 100)
TYPE_SEED_SHIFT = 500
MISSING_SENTINELS = ["<NA>", "nan", "NaN", "None", "NaT"]
NULL_PLACEHOLDER = "\x00missing"
NEEDS_NON_NULL_STRINGS = (error_type.Typo, error_type.Replace, error_type.Mojibake, error_type.Permutate)

# Instantiated in `build_error_config`, which knows the seeds.
ErrorTypeSpec = tuple[type[ErrorType], ErrorTypeConfig | None]


def _error_types(scenario: str, frame: pd.DataFrame) -> dict[str, list[ErrorTypeSpec]]:
    """The error types each column receives under a scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r} (known: {list(SCENARIOS)})")

    types: defaultdict[str, list[ErrorTypeSpec]] = defaultdict(list)
    numeric = list(frame.select_dtypes(include="number").columns)
    strings = [column for column in frame.columns if column not in numeric]

    if scenario in ("missing", "blank_record"):
        for column in frame.columns:
            types[column].append((error_type.MissingValue, None))
    if scenario == "typo":
        for column in strings:
            types[column].append((error_type.Typo, None))
    if scenario == "outlier":
        for column in numeric:
            types[column].append((error_type.Outlier, None))
    if scenario == "swap":
        # `tab_err` raises on a column with a single value.
        for column in strings:
            if frame[column].nunique(dropna=True) > 1:
                types[column].append((error_type.CategorySwap, None))
    if scenario == "wrong_unit":
        for column in numeric:
            types[column].append((error_type.WrongUnit, WRONG_UNIT))
    return dict(types)


def build_error_config(scenario: str, frame: pd.DataFrame, error_rate: float, seed: int) -> MidLevelConfig:
    if scenario == "ward_intake":
        return _syntabfall_config(frame, error_rate, seed)

    types = _error_types(scenario, frame)
    # One shared EAR seed makes every column blank the same rows.
    blanking = scenario == "blank_record"
    condition = frame.columns[0]
    columns: dict[str | int, list[ErrorModel]] = {}
    for offset, (column, column_types) in enumerate(types.items()):
        share = error_rate / len(column_types)
        # CategorySwap first: it needs a categorical column, and inserted missing values make it object again.
        ordered = sorted(column_types, key=lambda spec: spec[0] is not error_type.CategorySwap)
        columns[column] = [
            ErrorModel(
                error_mechanism=(
                    EAR(condition_to_column=condition, seed=seed)
                    if blanking
                    else ECAR(seed=seed + 1000 * offset + index)
                ),
                error_type=column_type(config, seed=seed + 1000 * offset + TYPE_SEED_SHIFT + index),
                error_rate=share,
            )
            for index, (column_type, config) in enumerate(ordered)
        ]
    return MidLevelConfig(columns)


def _syntabfall_join(frame: pd.DataFrame) -> tuple[str, ...]:
    """Story 1, a failed nightly join: `procedure` and the medical-items block."""
    block = [column for column in frame.columns if column == "medical_items" or column.startswith("medical_items-")]
    return ("procedure", *block)


PAPER_NURSING_CARE = (
    "decubitus-admission",
    "decubitus-at_the_moment",
    "decubitus-risk",
    "bed_mobility-impairment",
    "bed_mobility-jones",
    "bed_mobility-skin_condition",
    "transfer-impairment",
    "transfer",
    "excretions-impairment",
    "excretions-incontinence",
    "excretions-nykturie",
)
PAPER_COGNITIVE = (
    "cognition-impairment",
    "cognition-agitated",
    "cognition-confused",
    "cognition-disoriented_time",
    "cognition-disoriented_location",
    "cognition-disoriented_own_person",
    "psychotropic_or_sedatives_drugs",
)
PAPER_WALKING = (
    "walk-impairment",
    "walk-jones",
    "walk-balance_and_gait_impaired",
    "walking_aid",
)
ASSESSMENT = (*PAPER_NURSING_CARE, *PAPER_COGNITIVE, *PAPER_WALKING)
GRADE = ErrorTypeConfig(mislabel_weighing="uniform")
ROLLOVER = ErrorTypeConfig(add_delta_value=-100.0)


def _syntabfall_config(frame: pd.DataFrame, error_rate: float, seed: int) -> MidLevelConfig:
    join_columns = _syntabfall_join(frame)
    required = {"procedure", *ASSESSMENT, "fall-risk", "age", "diagnosis"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"The SynTabFall scenario needs columns this frame does not have: {missing}")

    columns: dict[str | int, list[ErrorModel]] = {}
    join = ECAR(seed=seed)
    for index, column in enumerate(join_columns):
        columns[column] = [ErrorModel(join, error_type.MissingValue(seed=seed + TYPE_SEED_SHIFT + index), error_rate)]
    assessment = EAR(condition_to_column="age", seed=seed + 1000)
    for index, column in enumerate(ASSESSMENT):
        columns[column] = [
            ErrorModel(assessment, error_type.MissingValue(seed=seed + 1000 + TYPE_SEED_SHIFT + index), error_rate)
        ]
    columns["fall-risk"] = [
        ErrorModel(
            ECAR(seed=seed + 2000),
            error_type.CategorySwap(GRADE, seed=seed + 2000 + TYPE_SEED_SHIFT),
            error_rate,
        )
    ]
    columns["age"] = [
        ErrorModel(
            ENAR(seed=seed + 3000),
            error_type.AddDelta(ROLLOVER, seed=seed + 3000 + TYPE_SEED_SHIFT),
            error_rate,
        )
    ]
    columns["diagnosis"] = [
        ErrorModel(
            ENAR(seed=seed + 4000),
            error_type.Typo(seed=seed + 4000 + TYPE_SEED_SHIFT),
            error_rate,
        )
    ]
    return MidLevelConfig(columns)


def inject(
    frame: pd.DataFrame,
    scenario: str,
    error_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrupt a frame under a scenario. Returns the corrupted frame and the mask of the cells that changed."""
    if scenario == "clean" or error_rate == 0:
        return frame.copy(), _false_mask(frame)

    config = build_error_config(scenario, frame, error_rate, seed)
    if not config.columns:
        return frame.copy(), _false_mask(frame)

    working = frame.copy()
    swapped = [
        column
        for column, models in config.columns.items()
        if any(isinstance(model.error_type, error_type.CategorySwap) for model in models)
    ]
    for column in swapped:
        working[column] = working[column].astype("category")

    placeheld = {
        column: frame[column].isna()
        for column, models in config.columns.items()
        if any(isinstance(model.error_type, NEEDS_NON_NULL_STRINGS) for model in models)
    }
    placeheld = {column: mask for column, mask in placeheld.items() if mask.any()}
    for column in placeheld:
        working[column] = working[column].fillna(NULL_PLACEHOLDER).astype(object)

    dirty, mask = mid_level.create_errors(working, config)

    for column in swapped:
        dirty[column] = dirty[column].astype(object)
    for column, was_missing in placeheld.items():
        # A typo of a value that was never there is not an error.
        dirty.loc[was_missing, column] = None
    for column in dirty.columns:
        if dirty[column].dtype == object:
            dirty[column] = dirty[column].replace(MISSING_SENTINELS, None)

    return dirty, _really_changed(frame, dirty, mask)


def _false_mask(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(False, index=frame.index, columns=frame.columns)


def _really_changed(clean: pd.DataFrame, dirty: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Restrict the mask to cells whose value differs. Two missing values are equal."""
    both_missing = clean.isna() & dirty.isna()
    differs = (clean != dirty) & ~both_missing
    return mask.astype(bool) & differs
