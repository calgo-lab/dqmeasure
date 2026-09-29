from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from dqmeasure import BaseMeasure
from experiments import DATA_QUALITY_MEASUREMENT, EXPERIMENTS
from experiments.data import Dataset, load_dataset
from experiments.errors import inject
from experiments.measures import LLMConfig, candidate_measures

logger = logging.getLogger("experiments.runner")

# For the classifier, a missing string is its own category; numeric columns keep their NaNs.
STRING_MISSING = "__missing__"

FINISHED = "FINISHED"

# Spelled out so that every run writes the same schema, even where a column is all null.
RESULT_DTYPES: dict[str, str] = {
    "experiment": "string",
    "dataset": "string",
    "scenario": "string",
    "error_rate": "Float64",
    "seed": "Int64",
    "fold": "Int64",
    "realized_error_rate": "Float64",
    "threshold": "Float64",
    "auc_clean": "Float64",
    "auc_dirty": "Float64",
    "tp_clean": "Int64",
    "fp_clean": "Int64",
    "tn_clean": "Int64",
    "fn_clean": "Int64",
    "tp_dirty": "Int64",
    "fp_dirty": "Int64",
    "tn_dirty": "Int64",
    "fn_dirty": "Int64",
    "n_rows_test": "Int64",
    "measure": "string",
    "scope": "string",
    "column": "string",
    "column_error_rate": "Float64",
    "dq_score": "Float64",
    "fit_seconds": "Float64",
    "score_seconds": "Float64",
    "status": "string",
}
CLASSIFIER_COLUMNS = ["threshold", "auc_clean", "auc_dirty"] + [
    f"{count}_{side}" for side in ("clean", "dirty") for count in ("tp", "fp", "tn", "fn")
]


@dataclass(frozen=True)
class RunSpec:
    """The coordinates of one experiment run."""

    dataset: str = "hospital"
    scenario: str = "clean"
    error_rate: float = 0.0
    seed: int = 0
    n_folds: int = 5
    experiment: str = DATA_QUALITY_MEASUREMENT
    data_dir: Path = Path("/app/data")
    results_dir: Path = Path("/results")
    llm: LLMConfig | None = None

    def __post_init__(self) -> None:
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {self.experiment!r} (known: {list(EXPERIMENTS)})")

    @property
    def corrupts_target(self) -> bool:
        """Whether this run corrupts the target column and skips the classifier."""
        return self.experiment == DATA_QUALITY_MEASUREMENT

    @property
    def out_dir(self) -> Path:
        return self.results_dir / self.experiment / self.dataset / self.scenario / str(self.error_rate) / str(self.seed)


def spec_from_env(env: Mapping[str, str]) -> RunSpec:
    """Read a run's coordinates from the environment; an unset `LLM_URL` skips `SemanticDataAccuracy`."""
    default = RunSpec()
    return RunSpec(
        dataset=env.get("DATASET", default.dataset),
        scenario=env.get("SCENARIO", default.scenario),
        error_rate=float(env.get("ERROR_RATE", default.error_rate)),
        seed=int(env.get("SEED", default.seed)),
        n_folds=int(env.get("N_FOLDS", default.n_folds)),
        experiment=env.get("EXPERIMENT", default.experiment),
        data_dir=Path(env.get("DATA_DIR", default.data_dir)),
        results_dir=Path(env.get("RESULTS_DIR", default.results_dir)),
        llm=LLMConfig.from_env(env),
    )


## the downstream machine-learning task

# Log-spaced: on a 0.98%-positive target the useful thresholds sit near zero.
THRESHOLDS = np.geomspace(1e-3, 0.9, 100)


def _fill_missing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.fillna(STRING_MISSING).astype(str)


def _downstream_classifier(x_train: pd.DataFrame, seed: int) -> TunedThresholdClassifierCV:
    """Ordinal-encoded strings and gradient boosting, its threshold tuned for macro-F1 by 5-fold CV, unfit.

    A 0.5 threshold predicts almost no positives on this imbalanced target.
    """
    numeric_columns = list(x_train.select_dtypes(include="number").columns)
    string_columns = [column for column in x_train.columns if column not in numeric_columns]

    encode = make_pipeline(
        FunctionTransformer(_fill_missing),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    )
    model = Pipeline(
        [
            ("encode", ColumnTransformer([("strings", encode, string_columns)], remainder="passthrough")),
            ("classify", HistGradientBoostingClassifier(max_iter=200, random_state=seed)),
        ]
    )
    return TunedThresholdClassifierCV(model, scoring="f1_macro", thresholds=THRESHOLDS)


def _downstream_metrics(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    tests: dict[str, pd.DataFrame],
    y_test: pd.Series,
    positive_label: str,
    seed: int,
) -> dict[str, Any]:
    """Fit on the clean train fold; `tp/fp/tn/fn` and AUC on each named test frame, plus the tuned threshold."""
    (negative_label,) = set(y_train.unique()) - {positive_label}

    model = _downstream_classifier(x_train, seed).fit(x_train, y_train)
    threshold = float(model.best_threshold_)
    positive_column = list(model.classes_).index(positive_label)

    result: dict[str, Any] = {"threshold": threshold}
    for name, frame in tests.items():
        proba = model.predict_proba(frame)[:, positive_column]
        predicted = model.predict(frame)
        (tn, fp), (fn, tp) = confusion_matrix(y_test, predicted, labels=[negative_label, positive_label])
        result[f"auc_{name}"] = float(roc_auc_score(y_test == positive_label, proba))
        result[f"tp_{name}"] = int(tp)
        result[f"fp_{name}"] = int(fp)
        result[f"tn_{name}"] = int(tn)
        result[f"fn_{name}"] = int(fn)
    return result


def _measure_row(
    label: str, column: str | None, measure: BaseMeasure, x_train: pd.DataFrame, x_dirty: pd.DataFrame
) -> dict[str, Any] | None:
    """Fit and score one measure; None if the measure rejects the column, a status if it fails."""
    row: dict[str, Any] = {
        "measure": label,
        "column": column,
        "dq_score": None,
        "fit_seconds": None,
        "score_seconds": None,
    }
    started = time.perf_counter()
    try:
        measure.fit(x_train)
    except ValueError:
        return None
    except Exception as error:  # one broken measure must not take the whole run down
        logger.warning("%s on %s failed to fit: %s", label, column, error)
        return {**row, "status": f"fit_failed:{type(error).__name__}"}
    row["fit_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        row["dq_score"] = measure.score(x_dirty)
        row["status"] = "ok"
    except Exception as error:
        logger.warning("%s on %s failed to score: %s", label, column, error)
        row["status"] = f"score_failed:{type(error).__name__}"
    row["score_seconds"] = time.perf_counter() - started
    return row


def execute(spec: RunSpec) -> pd.DataFrame:
    """Run every fold of one run and return the tidy results."""
    dataset: Dataset = load_dataset(spec.dataset, spec.data_dir)

    folds = KFold(n_splits=spec.n_folds, shuffle=True, random_state=spec.seed)
    rows: list[dict[str, Any]] = []
    frame = dataset.frame
    corrupts_target = spec.corrupts_target
    for fold, (train_index, test_index) in enumerate(folds.split(frame)):
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)

        # Downstream performance leaves the target out: a corrupted y_test would make its metrics meaningless.
        subject = test if corrupts_target else test.drop(columns=[dataset.target])
        reference = train if corrupts_target else train.drop(columns=[dataset.target])

        dirty, mask = inject(subject, spec.scenario, spec.error_rate, spec.seed + fold)
        realized = float(mask.to_numpy().mean())

        metrics: dict[str, Any] = dict.fromkeys(CLASSIFIER_COLUMNS)
        if not corrupts_target:
            assert dataset.positive_label is not None, "downstream performance needs a binary target"
            metrics = _downstream_metrics(
                reference,
                train[dataset.target],
                {"clean": subject, "dirty": dirty},
                test[dataset.target],
                dataset.positive_label,
                spec.seed,
            )
        logger.info("fold %d: realized_error_rate=%.4f %s", fold, realized, "" if corrupts_target else metrics)

        shared = {
            "experiment": spec.experiment,
            "dataset": spec.dataset,
            "scenario": spec.scenario,
            "error_rate": spec.error_rate,
            "seed": spec.seed,
            "fold": fold,
            "realized_error_rate": realized,
            **metrics,
            "n_rows_test": len(subject),
        }

        target = dataset.target if corrupts_target else None
        for label, column, measure in candidate_measures(list(subject.columns), target, spec.llm):
            row = _measure_row(label, column, measure, reference, dirty)
            if row is None:
                continue
            column_error_rate = float((mask[column] if column is not None else mask.any(axis=1)).mean())
            rows.append({**shared, "scope": measure.scope, "column_error_rate": column_error_rate, **row})

    return pd.DataFrame(rows, columns=list(RESULT_DTYPES)).astype(RESULT_DTYPES)


def run_and_write(spec: RunSpec) -> bool:
    """Execute one run and write its results unless its `FINISHED` file exists. Returns whether it ran."""
    if (spec.out_dir / FINISHED).exists():
        logger.info("already finished, nothing to do: %s", spec.out_dir)
        return False

    results = execute(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(spec.out_dir / "results.parquet", index=False)
    (spec.out_dir / FINISHED).touch()
    logger.info("wrote %d rows to %s", len(results), spec.out_dir)
    return True


def main() -> None:
    """Execute the run described by the environment, unless its results are already there."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = spec_from_env(os.environ)
    logger.info("run %s", spec)
    run_and_write(spec)


if __name__ == "__main__":
    main()
