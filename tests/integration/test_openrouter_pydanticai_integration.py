#!/usr/bin/env python3
"""
OpenRouter API PydanticAI統合テスト (pytest形式)

PydanticAI版OpenRouter APIアノテーターの実動作を検証する
pytest形式に準拠したテスト実装
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

pytestmark = [pytest.mark.integration, pytest.mark.webapi]


@pytest.fixture
def test_image() -> Image.Image:
    """テスト用の小さな画像を作成するフィクスチャ"""
    img = Image.new("RGB", (64, 64), color="green")
    return img


@pytest.fixture
def multiple_test_images() -> list[Image.Image]:
    """複数のテスト画像を作成するフィクスチャ"""
    return [
        Image.new("RGB", (64, 64), color="green"),
        Image.new("RGB", (64, 64), color="blue")
    ]


class TestOpenRouterPydanticAIStructure:
    """OpenRouter PydanticAI実装の構造テスト"""

    def test_required_methods_exist(self):
        """必要メソッドの存在確認"""
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        # 必要メソッドの存在確認 (新しいProvider-level実装)
        required_methods = [
            "__init__",
            "__enter__",
            "__exit__",
            "_run_inference",
            "run_with_model",
            "_handle_api_error",
        ]

        for method in required_methods:
            assert hasattr(OpenRouterApiAnnotator, method), f"Method {method} not found"

    def test_base_class_inheritance(self):
        """基底クラス継承確認"""
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from image_annotator_lib.core.base import WebApiBaseAnnotator

        assert issubclass(OpenRouterApiAnnotator, WebApiBaseAnnotator), "WebApiBaseAnnotator inheritance failed"


class TestImagePreprocessing:
    """画像前処理のテスト"""

    def test_image_preprocessing_to_binary(self, multiple_test_images):
        """画像前処理のテスト"""
        from pydantic_ai.messages import BinaryContent
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        # Mixinインスタンス作成
        mixin = PydanticAIAnnotatorMixin("test-model")

        # 前処理実行
        processed = mixin._preprocess_images_to_binary(multiple_test_images)

        # 結果検証
        assert isinstance(processed, list)
        assert len(processed) == 2

        for item in processed:
            assert isinstance(item, BinaryContent)
            assert item.media_type == "image/webp"
            assert len(item.data) > 0


class TestConfigHashGeneration:
    """設定ハッシュ生成テスト"""

    def test_config_hash_generation(self):
        """設定ハッシュ生成テスト"""
        from image_annotator_lib.core.webapi_agent_cache import create_config_hash

        # テスト用設定データ
        config_data = {
            "model_id": "anthropic/claude-3.5-sonnet",
            "temperature": 0.7,
            "max_tokens": 1800,
            "referer": "https://example.com",
            "app_name": "TestApp",
        }

        # ハッシュ生成
        config_hash = create_config_hash(config_data)
        config_hash2 = create_config_hash(config_data)

        # 結果検証
        assert isinstance(config_hash, str)
        assert len(config_hash) > 0
        assert config_hash == config_hash2  # 同一データで同一ハッシュ


class TestAgentCreation:
    """Provider Factory OpenRouter Agent作成のモックテスト"""

    @patch("image_annotator_lib.core.pydantic_ai_factory.infer_model")
    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    def test_agent_creation_mock(self, mock_agent_class, mock_infer_model):
        """Agent作成モックテスト"""
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

        with patch.object(PydanticAIProviderFactory, "get_provider") as mock_get_provider:
            # Mock model
            mock_model = MagicMock()
            mock_model.system = "openai"
            mock_infer_model.return_value = mock_model

            # Mock provider
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            # Mock agent
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # OpenRouter用config_data
            config_data = {
                "model_id": "anthropic/claude-3.5-sonnet",
                "referer": "https://example.com",
                "app_name": "TestApp",
            }

            # Provider Factory でOpenRouter Agent作成
            agent = PydanticAIProviderFactory.create_openrouter_agent(
                model_name="test-model",
                api_model_id="openrouter:anthropic/claude-3.5-sonnet",
                api_key="test-api-key",
                config_data=config_data,
            )

            # 検証
            assert agent is not None
            mock_infer_model.assert_called_once_with("openai:anthropic/claude-3.5-sonnet")
            mock_get_provider.assert_called_once()

            # get_provider呼び出し引数の確認
            call_args = mock_get_provider.call_args
            assert call_args[0][0] == "openai"  # provider_name
            provider_kwargs = call_args[1]
            assert provider_kwargs["api_key"] == "test-api-key"
            assert provider_kwargs["base_url"] == "https://openrouter.ai/api/v1"
            assert "default_headers" in provider_kwargs

            # ヘッダーの確認
            headers = provider_kwargs["default_headers"]
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "TestApp"

            mock_agent_class.assert_called_once()


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_authentication_error(self):
        """認証エラーテスト"""
        from image_annotator_lib.exceptions.errors import ApiAuthenticationError
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")
        
        with pytest.raises(ApiAuthenticationError):
            annotator._handle_api_error(Exception("401 authentication failed"))

    def test_rate_limit_error(self):
        """レート制限エラーテスト"""
        from image_annotator_lib.exceptions.errors import ApiRateLimitError
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")
        
        with pytest.raises(ApiRateLimitError):
            annotator._handle_api_error(Exception("429 rate limit exceeded"))

    def test_timeout_error(self):
        """タイムアウトエラーテスト"""
        from image_annotator_lib.exceptions.errors import ApiTimeoutError
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")
        
        with pytest.raises(ApiTimeoutError):
            annotator._handle_api_error(Exception("timeout occurred"))

    def test_server_error(self):
        """サーバーエラーテスト"""
        from image_annotator_lib.exceptions.errors import ApiServerError
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")
        
        with pytest.raises(ApiServerError):
            annotator._handle_api_error(Exception("500 server error"))

    def test_general_error(self):
        """一般エラーテスト"""
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")
        error_msg = annotator._handle_api_error(Exception("general error"))
        assert "OpenRouter API Error: general error" in error_msg


class TestInferencePipeline:
    """Provider Manager OpenRouter推論パイプラインのモックテスト"""

    @patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model')
    def test_inference_pipeline_mock(self, mock_inference, test_image):
        """推論パイプライン モックテスト"""
        from image_annotator_lib.core.provider_manager import ProviderManager
        from image_annotator_lib.core.types import AnnotationSchema

        # テストデータ準備
        expected_result = [
            {
                "response": AnnotationSchema(
                    tags=["test", "openrouter"], 
                    captions=["Mock test image for OpenRouter"], 
                    score=0.88
                ),
                "error": None,
            }
        ]

        mock_inference.return_value = expected_result

        # 推論実行
        results = ProviderManager.run_inference_with_model(
            model_name="test-model", 
            images=[test_image], 
            api_model_id="anthropic/claude-3.5-sonnet"
        )

        # 結果検証
        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, dict)
        assert "response" in result
        assert isinstance(result["response"], AnnotationSchema)
        assert result.get("error") is None

        # モック呼び出し確認
        mock_inference.assert_called_once_with(
            model_name="test-model", 
            images=[test_image], 
            api_model_id="anthropic/claude-3.5-sonnet"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])