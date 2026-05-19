"""tests/runtime_validation/ 固有 fixture (ADR 0001 amended)。

`tests/conftest.py:32` の `reset_global_state` autouse fixture は test teardown 時に
`_MODEL_CLASS_OBJ_REGISTRY.clear()` / `ModelLoad._instance_cache.clear()` /
`config_registry._config_cache.clear()` を実行する。これにより runtime_validation lane の
2 件目以降の test で `KeyError: "Model '...' not found in registry"` が発生する。

本 lane は ADR 0001 amended (2026-05-18) の責務範囲セクションで「public API
`annotate()` を必ず経由」と明文化されており、registry / config / ModelLoad cache を
test 間で持続させる必要がある。そのため同名 fixture を override し teardown を no-op に
する。

scope:
    `tests/runtime_validation/` 配下のみ (pytest の conftest はディレクトリ scope)。
    他 directory の reset 動作には影響しない。
"""

from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def reset_global_state(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """`tests/conftest.py` の同名 autouse fixture を override (no-op)。

    runtime_validation lane では registry / ModelLoad / config を test 間で持続させる。
    """
    yield
