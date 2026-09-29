"""The sweep definition, shared by `run-sweep` (local, in this process) and `k8s/generate_values_yaml.py`."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from itertools import product
from pathlib import Path

from experiments import DATA_QUALITY_MEASUREMENT, DOWNSTREAM_PERFORMANCE, EXPERIMENTS

logger = logging.getLogger("experiments.sweep")

# Each fold experiment runs on one dataset. SynTabFall's scenario names its columns, the others pick
# theirs by dtype.
SWEEPS = {
    DATA_QUALITY_MEASUREMENT: ("hospital", ["missing", "typo", "outlier", "swap", "blank_record", "wrong_unit"]),
    DOWNSTREAM_PERFORMANCE: ("syntabfall", ["ward_intake"]),
}
ERROR_RATES = [0.01, 0.05, 0.1, 0.25, 0.5]
SEEDS = [0]


def runs(experiment: str = DATA_QUALITY_MEASUREMENT) -> list[tuple[str, float, int]]:
    """Every run as `(scenario, error_rate, seed)`: one `clean` reference per seed, then the grid."""
    _dataset, scenarios = SWEEPS[experiment]
    return [("clean", 0.0, seed) for seed in SEEDS] + list(product(scenarios, ERROR_RATES, SEEDS))


def run_name(scenario: str, error_rate: float, seed: int) -> str:
    """A Kubernetes Job name, which allows neither `.` nor `_`."""
    return f"{scenario}-e{error_rate}-s{seed}".replace(".", "-").replace("_", "-")


def main(argv: Sequence[str] | None = None) -> None:
    """Execute every run of the sweep in this process, skipping the ones already finished."""
    parser = argparse.ArgumentParser(description="Run the experiment sweep locally.")
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=DATA_QUALITY_MEASUREMENT)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--scenario", action="append", help="only these scenarios; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="list the runs instead of executing them")
    parser.add_argument("--llm", action="store_true", help="also run SemanticDataAccuracy, with the pinned LLMConfig")
    args = parser.parse_args(argv)

    selected = [c for c in runs(args.experiment) if not args.scenario or c[0] in args.scenario]
    if args.dry_run:
        for scenario, error_rate, seed in selected:
            print(run_name(scenario, error_rate, seed))
        return
    if args.llm and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--llm needs OPENAI_API_KEY; run with `uv run --env-file .env run-sweep --llm`")

    log_file = args.results_dir / args.experiment / "sweep.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    from experiments.measures import LLMConfig
    from experiments.runner import RunSpec, run_and_write

    dataset, _scenarios = SWEEPS[args.experiment]
    failed = []
    for index, (scenario, error_rate, seed) in enumerate(selected, start=1):
        name = run_name(scenario, error_rate, seed)
        logger.info("run %d/%d: %s", index, len(selected), name)
        spec = RunSpec(
            dataset=dataset,
            scenario=scenario,
            error_rate=error_rate,
            seed=seed,
            n_folds=args.n_folds,
            experiment=args.experiment,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            llm=LLMConfig() if args.llm else None,
        )
        try:
            run_and_write(spec)
        except Exception:
            # One bad run should not cost the others; re-running the sweep picks up where this left off.
            logger.exception("run failed: %s", name)
            failed.append(name)

    if failed:
        raise SystemExit(f"{len(failed)} of {len(selected)} runs failed: {', '.join(failed)}")
    logger.info("%d runs done", len(selected))


if __name__ == "__main__":
    main()
