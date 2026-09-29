"""The experiments' names; `sweep` imports them without pulling in the runner's dependencies."""

DATA_QUALITY_MEASUREMENT = "data-quality-measurement"
DOWNSTREAM_PERFORMANCE = "downstream-performance"
EXPERIMENTS = (DATA_QUALITY_MEASUREMENT, DOWNSTREAM_PERFORMANCE)
# Not among `EXPERIMENTS`: those are the fold experiments a `RunSpec` and `run-sweep` accept.
SCALABILITY = "scalability"
