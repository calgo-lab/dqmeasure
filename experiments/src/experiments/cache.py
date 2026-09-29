"""`SemanticDataAccuracy` with a persistent SQLite cache of its verdicts, keyed by prompt, model and provider.

A stored NULL, an unreadable answer, counts as a hit.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any

import narwhals as nw

from dqmeasure import SemanticDataAccuracy
from dqmeasure.measures._llm import is_missing

logger = logging.getLogger("experiments.cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    prompt_hash TEXT NOT NULL,
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    verdict     INTEGER,
    PRIMARY KEY (prompt_hash, model, provider)
)
"""


class CachedSemanticDataAccuracy(SemanticDataAccuracy):
    """`SemanticDataAccuracy` that asks the LLM only for the prompts missing from the cache at `cache_path`."""

    def __init__(self, column: str, cache_path: Path, **kwargs: Any) -> None:
        super().__init__(column, **kwargs)
        self.cache_path = cache_path

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        rows = list(frame.iter_rows(named=True))
        askable = [i for i, null in enumerate(frame[self.column].is_null().to_list()) if not null]
        keys = {
            i: (hashlib.sha256(self._prompt(rows[i]).encode()).hexdigest(), self.llm_model, self.provider or "")
            for i in askable
        }

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(_SCHEMA)
            verdicts: dict[int, float | None] = {}
            for i in askable:
                found = conn.execute(
                    "SELECT verdict FROM verdicts WHERE prompt_hash = ? AND model = ? AND provider = ?", keys[i]
                ).fetchone()
                if found is not None:
                    verdicts[i] = None if found[0] is None else float(found[0])

            misses = [i for i in askable if i not in verdicts]
            logger.info("%s: %d cached, %d asked", self.column, len(askable) - len(misses), len(misses))
            if misses:
                units = super()._measure_units(frame[misses, :]).to_list()
                answered = {i: None if is_missing(v) else float(v) for i, v in zip(misses, units, strict=True)}
                conn.executemany(
                    "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?)",
                    [(*keys[i], None if v is None else int(v)) for i, v in answered.items()],
                )
                verdicts |= answered

        data = [verdicts.get(i) for i in range(len(rows))]
        return nw.new_series(self.column, data, nw.Float64, backend=frame.implementation)
