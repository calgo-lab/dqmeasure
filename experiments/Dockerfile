FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
COPY packages/dqmeasure/pyproject.toml packages/dqmeasure/
COPY experiments/pyproject.toml experiments/
RUN uv sync --frozen --no-dev --package experiments --no-install-workspace

COPY packages/dqmeasure packages/dqmeasure
COPY experiments/src experiments/src
# HOSP's 300KB CSV only. The taxi trips are 700MB written once onto their own volume, and SynTabFall is
# downloaded from Zenodo.
COPY experiments/data /app/data
RUN uv sync --frozen --no-dev --package experiments

ENV PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    RESULTS_DIR=/results \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["run-one"]
