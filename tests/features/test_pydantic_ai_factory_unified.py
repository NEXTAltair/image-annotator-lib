"""
PydanticAI Model Factory 統一テスト

pytest-bdd による BDD シナリオ実行。
"""

import pytest
from pytest_bdd import scenarios

from .step_definitions.pydantic_ai_factory_unified_steps import *  # noqa: F403

# フィーチャーファイルから自動的にシナリオを読み込み
scenarios("pydantic_ai_factory_unified.feature")


# テストマーク設定
pytestmark = [pytest.mark.bdd, pytest.mark.pydantic_ai_factory]
