from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import polars as pl


def make_frame(data: dict[str, Sequence[object]], backend: str):
    """Build a native frame in the requested backend."""
    if backend == "polars":
        return pl.DataFrame(data)
    return pd.DataFrame(data)
