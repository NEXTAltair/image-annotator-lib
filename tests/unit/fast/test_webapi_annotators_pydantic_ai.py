"""
PydanticAI 公式テスト戦略に基づく WebAPI Annotator のユニットテスト
"""
import pytest
from PIL import Image
from unittest.mock import MagicMock

# テスト対象のクラス
from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

# PydanticAI テストフィクスチャ
from tests.unit.fixtures.pydantic_ai_fixtures import mock_pydantic_ai_model, mock_pydantic_ai_function_model

# テスト対象の全WebAPIアノテータークラス
WEBAPI_ANNOTATOR_CLASSES = [
    AnthropicApiAnnotator,
    GoogleApiAnnotator,
    OpenAIApiAnnotator,
    OpenRouterApiAnnotator,
]


@pytest.mark.fast
@pytest.mark.parametrize("annotator_class", WEBAPI_ANNOTATOR_CLASSES)
def test_webapi_annotator_initialization(annotator_class):
    """各WebAPIアノテーターが正常に初期化されることをテスト"""
    annotator = annotator_class("test_model")
    assert annotator.model_name == "test_model"
    assert hasattr(annotator, "_agent"), "Annotator should have a _agent attribute"


@pytest.mark.fast
@pytest.mark.parametrize("annotator_class", WEBAPI_ANNOTATOR_CLASSES)
def test_webapi_annotator_predict_with_test_model(annotator_class, mock_pydantic_ai_model):
    """
    TestModel を使用して、各WebAPIアノテーターの predict メソッドが
    適切な形式のデータを返すことをテストする。
    """
    annotator = annotator_class("test_model")
    
    # テスト用のダミー画像
    dummy_image = MagicMock(spec=Image.Image)
    dummy_phash = "dummy_phash_123"

    # Agent.override を使用して実APIコールを TestModel に差し替える
    with annotator._agent.override(model=mock_pydantic_ai_model):
        results = annotator.predict([dummy_image], [dummy_phash])

    # 検証
    assert isinstance(results, list)
    assert len(results) == 1
    
    result = results[0]
    assert result["phash"] == dummy_phash
    assert result["error"] is None
    assert "tags" in result
    assert isinstance(result["tags"], list)
    # TestModel はデフォルトで空のリストまたはダミーデータを生成する
    if result["tags"]:
        assert all(isinstance(tag, str) for tag in result["tags"])


@pytest.mark.fast
@pytest.mark.parametrize("annotator_class", WEBAPI_ANNOTATOR_CLASSES)
def test_webapi_annotator_context_manager(annotator_class, mock_pydantic_ai_model):
    """
    各WebAPIアノテーターのコンテキストマネージャーが正常に動作することをテスト
    """
    annotator = annotator_class("test_model")
    
    with annotator._agent.override(model=mock_pydantic_ai_model):
        with annotator as managed_annotator:
            # __enter__ が self を返すことを確認
            assert managed_annotator is annotator
            # コンテキスト内で predict が動作することを確認
            results = managed_annotator.predict([MagicMock(spec=Image.Image)], ["dummy_phash"])
            assert len(results) == 1
    
    # __exit__ がエラーなく完了したことを確認


@pytest.mark.fast
def test_openrouter_custom_logic_with_function_model(mock_pydantic_ai_function_model):
    """
    FunctionModel を使用して OpenRouter のようなカスタムロジックをテスト
    """
    annotator = OpenRouterApiAnnotator("openrouter_test_model")
    
    # FunctionModel を使用してカスタムレスポンスを注入
    with annotator._agent.override(model=mock_pydantic_ai_function_model):
        results = annotator.predict([MagicMock(spec=Image.Image)], ["phash1"])

    assert len(results) == 1
    result = results[0]
    assert result["error"] is None
    assert result["tags"] == ["custom_function_tag"]
