from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from experiments import SCALABILITY

if TYPE_CHECKING:
    from experiments.measures import LLMConfig

logger = logging.getLogger("experiments.scalability")

DATASET = "nyc_taxi"
SEED = 0
REPETITIONS = 10
WARM_UP_ROWS = 1_000
# Just after the month ends. The currentness measures otherwise read the wall clock.
REFERENCE_TIME = dt.datetime(2024, 2, 1)
FULL_ROWS = 41_169_720
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, FULL_ROWS]
LARGE_SIZES = [size for size in SIZES if size >= 30_000_000]
SWEEP_SIZES = [size for size in SIZES if size < 30_000_000]
# The smallest size from which the row sweep's runtimes grow in proportion to the rows, so the fixed cost per call
# does not flatten the column curve.
COLUMN_SWEEP_ROWS = 10_000_000
COLUMN_COUNTS = [5, 10, 15, 20, 25]

LLM_COLUMN = "PU_Zone"
LLM_SIZES = [300, 500, 700, 900]
# Two rounds of 32 requests in flight, untimed, so connection setup and a cold provider are not timed.
LLM_WARM_UP = 64

ZONE_LOOKUP = "taxi_zone_lookup.csv"
PREPARED = "prepared.parquet"
MANIFEST = "manifest.json"
FINISHED = "FINISHED"
MONTHS = [f"yellow_tripdata_2024-{month:02d}.parquet" for month in range(1, 13)]
CODES = ["VendorID", "RatecodeID", "payment_type", "PULocationID", "DOLocationID"]
ZONE_FIELDS = ["Borough", "Zone", "service_zone"]
TIMESTAMPS = ("tpep_dropoff_datetime", "tpep_pickup_datetime")


def prepare_nyc_taxi(data_dir: Path, months: Sequence[str] = MONTHS) -> Path:
    """Write the downloaded months, shuffled and joined with the zone table, to `prepared.parquet`."""
    directory = data_dir / DATASET
    missing = [name for name in (ZONE_LOOKUP, *months) if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory} lacks {missing}; run `get-datasets` first")
    zones = pl.read_csv(directory / ZONE_LOOKUP).with_columns(pl.col("LocationID").cast(pl.String))

    parts, month_rows = [], {}
    for month in months:
        parts.append(pl.read_parquet(directory / month).with_columns(pl.col(*TIMESTAMPS).cast(pl.Datetime("us"))))
        month_rows[month] = parts[-1].height
    frame = pl.concat(parts).with_columns(pl.col(*CODES).cast(pl.String))
    del parts
    if list(months) == MONTHS and frame.height != FULL_ROWS:
        raise ValueError(f"the twelve months hold {frame.height} rows, expected {FULL_ROWS}")
    # Shuffled once, so every size is a prefix of the same sample.
    frame = frame.sample(fraction=1.0, shuffle=True, seed=SEED)
    for side in ("PU", "DO"):
        lookup = zones.rename({f: f"{side}_{f}" for f in ZONE_FIELDS})
        frame = frame.join(lookup, left_on=f"{side}LocationID", right_on="LocationID", how="left")

    prepared = directory / PREPARED
    frame.write_parquet(prepared)
    rows, columns = frame.height, frame.columns
    del frame
    manifest = {
        "rows": rows,
        "columns": columns,
        "months": month_rows,
        "requests": {str(n): llm_requests(pl.scan_parquet(prepared), n) for n in SIZES if n <= rows},
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s: %d rows, %d columns", prepared, rows, len(columns))
    return prepared


def load_nyc_taxi(data_dir: Path, n_rows: int) -> pl.DataFrame:
    frame = pl.scan_parquet(data_dir / DATASET / PREPARED).head(n_rows).collect()
    # A short file would otherwise be timed under the size the run asked for.
    if frame.height < n_rows:
        raise ValueError(f"{PREPARED} holds {frame.height} rows, the run needs {n_rows}; prepare more months")
    return frame


def llm_requests(frame: pl.DataFrame | pl.LazyFrame, n_rows: int) -> int:
    """How many requests `SemanticDataAccuracy` sends for the first `n_rows`: one per distinct record."""
    distinct = frame.lazy().head(n_rows).filter(pl.col(LLM_COLUMN).is_not_null()).unique()
    return int(distinct.select(pl.len()).collect(engine="streaming").item())


def runs(sweeps: Sequence[str] = ("rows", "columns", "llm")) -> list[tuple[str, int, int]]:
    """Every run as `(sweep, size, repetition)`."""
    listed: list[tuple[str, int, int]] = []
    # Repetition-major, so anything that drifts during the run hits every configuration alike.
    for r in range(REPETITIONS):
        # `large` selects the largest `rows` runs, which then run in their own Job.
        if "rows" in sweeps:
            listed += [("rows", size, r) for size in SWEEP_SIZES]
        if "large" in sweeps:
            listed += [("rows", size, r) for size in LARGE_SIZES]
        if "columns" in sweeps:
            listed += [("columns", k, r) for k in COLUMN_COUNTS]
        if "llm" in sweeps:
            listed.append(("llm", max(LLM_SIZES), r))
    return listed


def run_name(sweep: str, size: int, repetition: int) -> str:
    return f"{sweep}-{size}-r{repetition}"


def run_rows(sweep: str, size: int) -> int:
    if sweep == "rows":
        return size
    if sweep == "llm":
        return max(LLM_SIZES)
    return COLUMN_SWEEP_ROWS


def _time_suite(x: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Fit and score every measure once, each timed separately."""
    from dqmeasure import TimelinessOfDataItems, TimelinessOfUpdate
    from experiments.measures import candidate_measures

    measures = list(candidate_measures(list(columns)))
    if set(TIMESTAMPS) <= set(columns):
        measures += [
            ("TimelinessOfDataItems", TIMESTAMPS[0], TimelinessOfDataItems(*TIMESTAMPS)),
            ("TimelinessOfUpdate", TIMESTAMPS[0], TimelinessOfUpdate(*TIMESTAMPS)),
        ]
    rows: list[dict[str, Any]] = []
    for label, column, measure in measures:
        if hasattr(measure, "reference_time"):
            measure.reference_time = REFERENCE_TIME
        row: dict[str, Any] = {"measure": label, "column": column, "fit_seconds": None, "score_seconds": None}
        started = time.perf_counter()
        try:
            measure.fit(x)
        except ValueError:
            rows.append({**row, "status": "rejected"})
            continue
        row["fit_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        try:
            measure.score(x)
            row["status"] = "ok"
        except Exception as error:
            logger.warning("%s on %s failed to score: %s", label, column, error)
            row["status"] = f"score_failed:{type(error).__name__}"
        row["score_seconds"] = time.perf_counter() - started
        rows.append(row)
    return rows


def run_timing(frame: pl.DataFrame, sweep: str, size: int, repetition: int) -> pd.DataFrame:
    """Time every measure on `frame`, which `load_nyc_taxi` cut to the run's rows."""
    columns = frame.columns
    if sweep == "columns":
        # A new order per repetition: columns differ in cost, so the spread covers which columns, not only how many.
        order = pl.Series(frame.columns).sample(fraction=1.0, shuffle=True, seed=repetition).to_list()
        columns = order[:size]
    x = frame.select(columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _time_suite(frame.head(WARM_UP_ROWS), frame.columns)
        logger.info("%s %d: repetition %d", sweep, size, repetition)
        records = _time_suite(x, columns)
    shared = {"sweep": sweep, "n_rows": frame.height, "n_columns": len(columns), "repetition": repetition}
    return pd.DataFrame.from_records([{**shared, **r} for r in records])


def run_llm(frame: pl.DataFrame, llm: LLMConfig, repetition: int) -> pd.DataFrame:
    """Time `SemanticDataAccuracy` for each of `LLM_SIZES`."""
    from dqmeasure import SemanticDataAccuracy

    requests = {n: llm_requests(frame, n) for n in LLM_SIZES}
    warm_up = frame.head(LLM_WARM_UP)
    SemanticDataAccuracy(LLM_COLUMN, **llm.settings()).fit(warm_up).score(warm_up)

    records: list[dict[str, Any]] = []
    for n_rows in LLM_SIZES:
        x = frame.head(n_rows)
        measure = SemanticDataAccuracy(LLM_COLUMN, **llm.settings())
        started = time.perf_counter()
        measure.fit(x)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = measure.score(x)
        records.append(
            {
                "n_rows": n_rows,
                "repetition": repetition,
                "requests": requests[n_rows],
                "fit_seconds": fit_seconds,
                "score_seconds": time.perf_counter() - started,
                "dq_score": score,
                "warnings": "; ".join(str(w.message) for w in caught),
            }
        )
        logger.info("llm %d rows: %s", n_rows, records[-1])
    return pd.DataFrame.from_records(records)


def run_and_write(
    sweep: str, size: int, repetition: int, data_dir: Path, results_dir: Path, llm: LLMConfig | None
) -> bool:
    """Execute one run unless it is already finished. Returns whether it ran."""
    target = results_dir / SCALABILITY / DATASET / sweep / str(size) / str(repetition)
    if (target / FINISHED).exists():
        logger.info("already finished, nothing to do: %s", target)
        return False
    frame = load_nyc_taxi(data_dir, run_rows(sweep, size))
    if sweep == "llm":
        if llm is None:
            raise ValueError("the llm run needs LLM_URL")
        results = run_llm(frame, llm, repetition)
    else:
        results = run_timing(frame, sweep, size, repetition)
    target.mkdir(parents=True, exist_ok=True)
    results.to_parquet(target / "results.parquet", index=False)
    (target / FINISHED).touch()
    logger.info("wrote %s", target)
    return True


def main() -> None:
    """Execute the run an Indexed Job's pod stands for: `runs(SWEEPS)[JOB_COMPLETION_INDEX]`."""
    from experiments.measures import LLMConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = os.environ
    sweeps = env.get("SWEEPS", "rows,columns,llm").split(",")
    sweep, size, repetition = runs(sweeps)[int(env["JOB_COMPLETION_INDEX"])]
    llm = LLMConfig.from_env(env)
    logger.info("run %s", run_name(sweep, size, repetition))
    data_dir, results_dir = Path(env.get("DATA_DIR", "/app/data")), Path(env.get("RESULTS_DIR", "/results"))
    run_and_write(sweep, size, repetition, data_dir, results_dir, llm)


def prepare_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the downloaded NYC taxi trips.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "experiments/data")))
    parser.add_argument("--months", type=int, default=len(MONTHS), help=f"how many of the {len(MONTHS)} to use")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_nyc_taxi(args.data_dir, MONTHS[: args.months])


def local_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scalability experiment locally.")
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--sweep", action="append", choices=["rows", "columns", "llm", "large"])
    parser.add_argument("--max-size", type=int, help="skip row runs above this many rows")
    parser.add_argument("--repetitions", type=int, help=f"only the first N of {REPETITIONS}")
    parser.add_argument("--dry-run", action="store_true", help="list the runs instead of executing them")
    args = parser.parse_args(argv)

    selected = [
        (sweep, size, r)
        for sweep, size, r in runs(args.sweep or ("rows", "columns", "llm"))
        if not (sweep == "rows" and args.max_size and size > args.max_size)
        and (args.repetitions is None or r < args.repetitions)
    ]
    if args.dry_run:
        print("\n".join(run_name(*run) for run in selected))
        return

    from experiments.measures import LLMConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    llm = LLMConfig()
    for i, run in enumerate(selected, start=1):
        logger.info("run %d/%d: %s", i, len(selected), run_name(*run))
        run_and_write(*run, args.data_dir, args.results_dir, llm)


if __name__ == "__main__":
    main()
