from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments import DATA_QUALITY_MEASUREMENT, SCALABILITY
from experiments.sweep import SWEEPS, run_name, runs

VALUES = Path(__file__).parent / "values.yaml"
# Sized for the largest run each Job executes: 10M rows in the sweep, all 41M rows in `large`.
SWEEP_MEMORY = "32Gi"
LARGE_MEMORY = "96Gi"


def build_runs(experiment: str) -> list[dict[str, Any]]:
    """One Job per run. The scalability experiment has none; it runs as Indexed Jobs."""
    if experiment == SCALABILITY:
        return []
    return [
        {"name": run_name(scenario, error_rate, seed), "scenario": scenario, "errorRate": error_rate, "seed": seed}
        for scenario, error_rate, seed in runs(experiment)
    ]


def build_indexed(parallelism: int, large_parallelism: int) -> list[dict[str, Any]]:
    """The scalability experiment's two Indexed Jobs.

    An Indexed Job has one pod template, so the largest sizes run in their own Job instead of every run
    asking for their memory.
    """
    from experiments import scalability  # imports pandas, which the other experiments do not need here

    jobs = [
        ("sweep", ("rows", "columns", "llm"), parallelism, SWEEP_MEMORY),
        ("large", ("large",), large_parallelism, LARGE_MEMORY),
    ]
    return [
        {
            "name": name,
            "sweeps": ",".join(sweeps),
            "count": len(scalability.runs(sweeps)),
            "parallelism": n_parallel,
            "resources": {"requests": {"cpu": 4, "memory": memory}, "limits": {"cpu": 4, "memory": memory}},
        }
        for name, sweeps, n_parallel, memory in jobs
    ]


def main() -> None:
    values = yaml.safe_load(VALUES.read_text()) if VALUES.exists() else {}
    experiment = values.get("experiment", DATA_QUALITY_MEASUREMENT)
    # The dataset follows from the experiment; scalability runs on the taxi trips.
    dataset = values["dataset"] = SWEEPS[experiment][0] if experiment in SWEEPS else "nyc_taxi"
    values["runs"] = build_runs(experiment)
    if experiment == SCALABILITY:
        values["indexed"] = build_indexed(values.get("parallelism", 12), values.get("largeParallelism", 4))
    else:
        values.pop("indexed", None)
    VALUES.write_text(yaml.safe_dump(values, sort_keys=False))
    jobs = [f"{job['name']}: {job['count']}" for job in values.get("indexed", [])]
    print(f"wrote {experiment!r} on {dataset!r} to {VALUES}: {len(values['runs'])} runs", *jobs, sep="; ")


if __name__ == "__main__":
    main()
