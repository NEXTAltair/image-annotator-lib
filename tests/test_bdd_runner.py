# tests/test_bdd_runner.py
"""pytest-bdd E2E テストランナーファイル

PydanticAI統一実装対応のBDDテストを実行します。
このファイルは pytest によってテストとして検出され、
`scenarios()` 関数を呼び出すことで、関連する .feature ファイルと
ステップ定義を結びつけて E2E BDD テストを実行します。
"""

import pytest
from pytest_bdd import scenarios

# PydanticAI統一実装対応のステップ定義をインポート
from .features.step_definitions.common_steps import *  # noqa: F403
from .features.step_definitions.webapi_annotate_steps import *  # noqa: F403
from .features.step_definitions.pydantic_ai_provider_level_steps import *  # noqa: F403

# WebAPI アノテーション E2E テストのメインシナリオ
scenarios("features/webapi_annotate.feature")

# 共通アノテーション機能の基本的なテストシナリオ  
scenarios("features/annotation_common.feature")

# マークを追加してBDDテストとして識別
pytestmark = pytest.mark.bdd
