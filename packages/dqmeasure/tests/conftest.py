from __future__ import annotations

import pytest

# Backends every test runs against. Both are installed (polars is a runtime dependency, pandas a dev one),
# so a missing one is a broken environment, not a reason to silently shrink the matrix.
BACKENDS = ["polars", "pandas"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
