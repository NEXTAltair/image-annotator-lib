"""
PydanticAI 公式テスト戦略に基づく共有フィクスチャ
"""
import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel

# グローバル設定で実APIへのリクエストを全面的に禁止
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def mock_pydantic_ai_model() -> TestModel:
    """
    PydanticAI TestModel を使用した標準テストフィクスチャ。
    呼び出されると、適切な型のダミーデータを自動生成する。
    """
    return TestModel()


@pytest.fixture
def mock_pydantic_ai_function_model() -> FunctionModel:
    """
    カスタムロジックを注入できる FunctionModel のフィクスチャ。
    特定のレスポンスをシミュレートするのに使用する。
    """

    def custom_response_logic(messages, info) -> ModelResponse:
        """テスト用のカスタムレスポンスを生成する関数"""
        # ここでテストシナリオに応じたレスポンスを生成できる
        # 例: 特定のタグを含むレスポンス
        return ModelResponse(parts=[TextPart('{"tags": ["custom_function_tag"]}')])

    return FunctionModel(custom_response_logic)
