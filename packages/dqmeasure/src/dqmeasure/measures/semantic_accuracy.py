from __future__ import annotations

import json
import warnings
from typing import Any, Self

import narwhals as nw
from narwhals.typing import IntoDataFrame

from dqmeasure.base import NotResolvedError, PositionalMeasure
from dqmeasure.measures._llm import complete_many, is_missing, openai_completion, render_record


def _parse_verdict(response: str) -> float | None:
    try:
        parsed = json.loads(response)
        return 1.0 if bool(parsed["accurate"]) else 0.0
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


class SemanticDataAccuracy(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-2` "Semantic data accuracy" (`Acc-ML-2` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column. The measure is
    scoped to one column, but its condition reads the whole row, which goes into the prompt as context.

    An LLM judges whether each value is semantically accurate given the rest of its record and its
    real-world knowledge. The LLM is prompted with example records sampled from the clean data
    (few-shot serialization inspired by mimir's ``llm_master``, see https://github.com/calgo-lab/mimir).

    The measure sends requests to any OpenAI-compatible chat-completions endpoint, sending one request per record.

    The reference (the sampled example records) cannot be specified in the constructor:
    [`fit`][dqmeasure.base.BaseMeasure.fit] on clean data is always required.

    Parameters
    ----------
    column:
        The column the measure applies to. All other columns of the frame go into the prompt as context.
    llm_model, llm_url:
        Model name and base URL (up to and including ``/v1``) of an OpenAI-compatible chat-completions
        endpoint. The defaults target a local Ollama server with a model small enough for a laptop.
    llm_api_key:
        Bearer token for the endpoint. ``None`` (default) falls back to the ``OPENAI_API_KEY`` environment
        variable.
    n_examples:
        Number of clean records sampled at fit time as few-shot examples in the prompt.
    random_state:
        Seed for the example sampling, making the measurement procedure reproducible.
    n_jobs:
        Number of concurrent requests. ``1`` (default) sends them sequentially.
    provider:
        Pin every request to one upstream provider (routers such as OpenRouter otherwise pick per request,
        which harms reproducibility). ``None`` (default) leaves routing to the endpoint.
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
        n_jobs: int = 1,
        provider: str | None = None,
    ) -> None:
        super().__init__(column=column)
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.n_examples = n_examples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.provider = provider

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
        # missing or the model's response could not be parsed.
        llm = openai_completion(self.llm_model, self.llm_url, self.llm_api_key, provider=self.provider)
        nulls = frame[self.column].is_null().to_list()
        rows = list(frame.iter_rows(named=True))
        row_prompts: list[str | None] = [None if nulls[i] else self._prompt(row) for i, row in enumerate(rows)]

        # Every distinct prompt is asked once
        distinct_prompts = list(dict.fromkeys(prompt for prompt in row_prompts if prompt is not None))
        responses = complete_many(llm, distinct_prompts, n_jobs=self.n_jobs)

        verdicts: dict[str, float | None] = {}
        failures: list[str] = []
        for prompt, response in zip(distinct_prompts, responses, strict=True):
            verdict = _parse_verdict(response)
            verdicts[prompt] = verdict
            if verdict is None:
                failures.append(response)
        if failures:
            warnings.warn(
                f"Could not parse {len(failures)} of {len(distinct_prompts)} LLM responses as a verdict "
                f"(example: {failures[0]!r}). Counting them as missing, not as accurate.",
                stacklevel=2,
            )

        data: list[float | None] = [None if prompt is None else verdicts[prompt] for prompt in row_prompts]
        return nw.new_series(self.column, data, nw.Float64, backend=frame.implementation)

    def _prompt(self, row: dict[str, Any]) -> str:
        header = " | ".join(row.keys())
        examples = "\n".join(render_record(example) for example in self.examples_)
        value = "<missing>" if is_missing(row[self.column]) else row[self.column]
        return (
            "You judge whether a value in a table record is semantically accurate: whether the value makes "
            "sense for its column, given the rest of the record and real-world knowledge.\n"
            'Fields are pipe-separated in the order given by the header. A missing value is '
            "shown as <missing>.\n"
            'Respond with JSON only, in the form {"accurate": true} or {"accurate": false}.\n\n'
            f"Columns: {header}\n\n"
            f"Records from the same table that are known to be accurate:\n{examples}\n\n"
            f"Record: {render_record(row)}\n"
            f"Column: {self.column}\n"
            f"Value: {value}\n"
            "Is the value (not the row) semantically accurate? Respond with JSON only:"
        )
