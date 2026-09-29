"""The DQMs a run evaluates: table-scoped measures once, column-scoped ones per column.

Measures that reject a column (a date measure on HOSP, say) are dropped by the runner.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dqmeasure
from dqmeasure import (
    BaseMeasure,
    DataAccuracyRange,
    DataFormatConsistency,
    DataRecordConsistency,
    DataValueDistribution,
    EmptyRecords,
    FeatureCompleteness,
    FeatureCurrentness,
    LabelCompleteness,
    RecordCompleteness,
    RecordCurrentness,
    RiskOfDataInconsistency,
    RiskOfDataSetInaccuracy,
    SemanticConsistency,
    SemanticDataAccuracy,
    SyntacticDataAccuracy,
    UpdateFrequency,
    ValueCompleteness,
    ValueOccurrenceCompleteness,
)
from experiments.cache import CachedSemanticDataAccuracy

COLUMN_MEASURES: tuple[type[BaseMeasure], ...] = (
    SyntacticDataAccuracy,
    DataAccuracyRange,
    RiskOfDataSetInaccuracy,
    FeatureCompleteness,
    ValueOccurrenceCompleteness,
    DataValueDistribution,
    DataFormatConsistency,
    RiskOfDataInconsistency,
    SemanticConsistency,
    FeatureCurrentness,
    UpdateFrequency,
)

TABLE_MEASURES: tuple[type[BaseMeasure], ...] = (
    ValueCompleteness,
    RecordCompleteness,
    EmptyRecords,
    DataRecordConsistency,
    RecordCurrentness,
)

_DIMENSION_BY_ISO_PREFIX = {
    "Acc": "Accuracy",
    "Com": "Completeness",
    "Con": "Consistency",
    "Cur": "Currentness",
    "Tml": "Timeliness",
}


def measure_dimension(measure: str) -> str:
    cls = getattr(dqmeasure, measure)
    iso_id = cls.iso_25024_id or cls.iso_5259_id
    return _DIMENSION_BY_ISO_PREFIX[iso_id.split("-", 1)[0]]


@dataclass(frozen=True)
class LLMConfig:
    """Where `SemanticDataAccuracy` sends its prompts; the defaults are the published run's pins.

    The provider is pinned because OpenRouter serves one model from upstreams with different quantizations.
    """

    url: str = "https://openrouter.ai/api/v1"
    model: str = "deepseek/deepseek-v4-flash-0731"
    n_jobs: int = 32
    provider: str | None = "baidu/fp8"
    cache_path: Path | None = Path("experiments/cache/verdicts.sqlite")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LLMConfig | None:
        """The pins, overridden by the `LLM_*` variables; None if `LLM_URL` is unset."""
        url = env.get("LLM_URL", "").strip()
        if not url:
            return None
        cache_path = env.get("LLM_CACHE", "").strip()
        return cls(
            url=url,
            model=env.get("LLM_MODEL") or cls.model,
            n_jobs=int(env.get("LLM_N_JOBS") or cls.n_jobs),
            provider=env.get("LLM_PROVIDER", "").strip() or None,
            cache_path=Path(cache_path) if cache_path else None,
        )

    def settings(self) -> dict[str, Any]:
        """The keyword arguments of `SemanticDataAccuracy`."""
        return {"llm_model": self.model, "llm_url": self.url, "n_jobs": self.n_jobs, "provider": self.provider}


def candidate_measures(
    columns: list[str],
    target: str | None = None,
    llm: LLMConfig | None = None,
) -> Iterator[tuple[str, str | None, BaseMeasure]]:
    """Yield `(label, column, measure)` for every measure to attempt.

    `LabelCompleteness` runs on `target` only; with `llm.cache_path`, `SemanticDataAccuracy` answers from the cache.
    """
    for measure_class in TABLE_MEASURES:
        yield measure_class.__name__, None, measure_class()  # type: ignore[call-arg]

    for column in columns:
        for measure_class in COLUMN_MEASURES:
            yield measure_class.__name__, column, measure_class(column)
        if column == target:
            yield LabelCompleteness.__name__, column, LabelCompleteness(column)
        if llm is not None:
            measure = (
                SemanticDataAccuracy(column, **llm.settings())
                if llm.cache_path is None
                else CachedSemanticDataAccuracy(column, llm.cache_path, **llm.settings())
            )
            yield "SemanticDataAccuracy", column, measure
