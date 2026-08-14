from __future__ import annotations

import re
import warnings
from typing import Any, Self

import narwhals as nw
from narwhals.typing import IntoDataFrame

from dqmeasure.base import NotResolvedError, PositionalMeasure
from dqmeasure.measures._llm import openai_completion, render_record


def _parse_verdict(response: str) -> float:
    """Map an LLM response to the condition result: leading "yes" -> 1.0, "no" -> 0.0."""
    match = re.search(r"[A-Za-z]+", response)
    word = match.group().lower() if match else ""
    if word == "yes":
        return 1.0
    if word == "no":
        return 0.0
    warnings.warn(f"Could not parse LLM verdict {response!r}; counting the value as accurate.", stacklevel=2)
    return 1.0


class SemanticDataAccuracy(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-2` "Semantic data accuracy" (`Acc-ML-2` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column. The measure is
    scoped to one column, but its condition reads the whole row, which goes into the prompt as context.

    The standard compares each value against reality. A language model stands in for reality here: it judges
    whether each value is semantically accurate given the rest of its record and real-world knowledge,
    prompted with example records sampled from the clean data (few-shot serialization inspired by mimir's
    ``llm_master``, see https://github.com/calgo-lab/mimir). A value that is wrong but plausible (the name
    "George" stored where "John" was true) passes, so ``X`` is an upper bound on true semantic accuracy.

    The measure talks to any OpenAI-compatible chat-completions endpoint and sends one request per distinct
    record, which suits small tables or samples rather than bulk scoring.

    The reference (the sampled example records) cannot be specified in the constructor:
    [`fit`][dqmeasure.base.BaseMeasure.fit] on clean data is always required.

    Parameters
    ----------
    column:
        The column the measure applies to. All other columns of the frame go into the prompt as context.
    llm_model, llm_url:
        Model name and base URL (up to and including ``/v1``) of an OpenAI-compatible chat-completions
        endpoint. The defaults target a local Ollama server with a model small enough for a laptop; point
        ``llm_url`` at LM Studio, vLLM, llama.cpp server, or a hosted API to swap the backend.
    llm_api_key:
        Bearer token for the endpoint. ``None`` (default) falls back to the ``OPENAI_API_KEY`` environment
        variable; local servers typically need none.
    n_examples:
        Number of clean records sampled at fit time as few-shot examples in the prompt.
    random_state:
        Seed for the example sampling, making the measurement procedure reproducible.
    """

    iso_5259_id = "Acc-ML-2"
    iso_25024_id = "Acc-I-2"

    examples_: list[dict[str, Any]]

    def __init__(
        self,
        column: str,
        llm_model: str = "llama3.2:3b",
        llm_url: str = "http://localhost:11434/v1",
        llm_api_key: str | None = None,
        n_examples: int = 5,
        random_state: int = 0,
    ) -> None:
        super().__init__(column=column)
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.n_examples = n_examples
        self.random_state = random_state

    def fit(self, X: IntoDataFrame) -> Self:
        """Sample the few-shot example records from a clean (training) dataframe. Mandatory for this measure."""
        frame = nw.from_native(X, eager_only=True)
        self._validate(frame)
        k = min(self.n_examples, len(frame))
        self.examples_ = list(frame.sample(n=k, seed=self.random_state).iter_rows(named=True))
        self._resolved = True
        return self

    def _check_is_resolved(self) -> None:
        if not getattr(self, "_resolved", False):
            raise NotResolvedError(
                f"{type(self).__name__}: the reference cannot be specified in the constructor; "
                "call fit() on clean data first."
            )

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell condition: 1.0 if the model judges the value accurate, 0.0 if not, null if the value is
        # missing (the same null-preserving float encoding as DataAccuracyRange, for backend parity).
        llm = openai_completion(self.llm_model, self.llm_url, self.llm_api_key)
        nulls = frame[self.column].is_null().to_list()
        cache: dict[str, float] = {}  # identical records ask identical questions; judge each prompt once
        data: list[float | None] = []
        for i, row in enumerate(frame.iter_rows(named=True)):
            if nulls[i]:
                data.append(None)
                continue
            prompt = self._prompt(row)
            if prompt not in cache:
                cache[prompt] = _parse_verdict(llm(prompt))
            data.append(cache[prompt])
        return nw.new_series(self.column, data, nw.Float64, backend=frame.implementation)

    def _prompt(self, row: dict[str, Any]) -> str:
        examples = "\n".join(render_record(example) for example in self.examples_)
        return (
            "You judge whether a value in a table record is semantically accurate: whether the value makes "
            "sense for its column, given the rest of the record and real-world knowledge.\n"
            "Answer with a single word: yes or no.\n\n"
            f"Records from the same table that are known to be accurate:\n{examples}\n\n"
            f"Record: {render_record(row)}\n"
            f"Column: {self.column}\n"
            f"Value: {row[self.column]}\n"
            "Is the value semantically accurate? Answer yes or no:"
        )
