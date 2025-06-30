#!/usr/bin/env python3
"""
統一Provider-level PydanticAI統合テスト (pytest形式)

全WebAPIプロバイダー(OpenAI, Anthropic, Google, OpenRouter)の
Provider-level実装の統一性とデザインパターンを検証する
"""

import pytest
from PIL import Image

pytestmark = [pytest.mark.integration, pytest.mark.webapi]


@pytest.fixture
def test_image() -> Image.Image:
    """テスト用の小さな画像を作成するフィクスチャ"""
    img = Image.new("RGB", (64, 64), color="red")
    return img


@pytest.fixture
def provider_test_data():
    """プロバイダーテストデータ"""
    return [
        (
            "OpenAI",
            "image_annotator_lib.model_class.annotator_webapi.openai_api_response",
            "OpenAIApiAnnotator",
        ),
        (
            "Anthropic",
            "image_annotator_lib.model_class.annotator_webapi.anthropic_api",
            "AnthropicApiAnnotator",
        ),
        ("Google", "image_annotator_lib.model_class.annotator_webapi.google_api", "GoogleApiAnnotator"),
        (
            "OpenRouter",
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat",
            "OpenRouterApiAnnotator",
        ),
    ]


class TestAllProvidersStructure:
    """全プロバイダーのProvider-level構造統一性テスト"""

    def test_required_methods_exist(self, provider_test_data):
        """全プロバイダーの必要メソッド存在確認"""
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        # 共通の必要メソッド
        required_methods = ["__init__", "__enter__", "__exit__", "run_with_model", "_handle_api_error"]

        for provider_name, module_path, class_name in provider_test_data:
            # モジュールとクラスをインポート
            module = __import__(module_path, fromlist=[class_name])
            annotator_class = getattr(module, class_name)

            # 必要メソッドの存在確認
            for method in required_methods:
                assert hasattr(annotator_class, method), f"{provider_name}: Method {method} not found"

            # PydanticAIAnnotatorMixin継承確認
            assert issubclass(annotator_class, PydanticAIAnnotatorMixin), (
                f"{provider_name}: PydanticAIAnnotatorMixin inheritance failed"
            )


class TestProviderManagerIntegration:
    """ProviderManager統合テスト"""

    def test_provider_determination(self):
        """プロバイダー判定テスト"""
        from image_annotator_lib.core.provider_manager import ProviderManager

        test_cases = [
            ("gpt-4o", "openai"),
            ("claude-3-5-sonnet", "anthropic"),
            ("gemini-pro", "google"),
            ("anthropic:claude-3-5-sonnet", "anthropic"),
            ("openai:gpt-4o", "openai"),
            ("google:gemini-pro", "google"),
            ("openrouter:meta-llama/llama-3", "openrouter"),
        ]

        for api_model_id, expected_provider in test_cases:
            determined_provider = ProviderManager._determine_provider("test-model", api_model_id)
            assert determined_provider == expected_provider, (
                f"Provider determination failed for {api_model_id}: expected {expected_provider}, got {determined_provider}"
            )

    def test_provider_manager_required_methods(self):
        """ProviderManager必要メソッド確認"""
        from image_annotator_lib.core.provider_manager import ProviderManager

        required_manager_methods = [
            "get_provider_instance",
            "run_inference_with_model",
            "_determine_provider",
        ]

        for method in required_manager_methods:
            assert hasattr(ProviderManager, method), f"ProviderManager method {method} not found"


class TestProviderFactoryIntegration:
    """PydanticAIProviderFactory統合テスト"""

    def test_factory_required_methods(self):
        """Factory必要メソッド確認"""
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

        required_factory_methods = [
            "get_provider",
            "create_agent",
            "get_cached_agent",
            "create_openrouter_agent",
            "_extract_provider_name",
        ]

        for method in required_factory_methods:
            assert hasattr(PydanticAIProviderFactory, method), f"Factory method {method} not found"

    def test_mixin_required_methods(self):
        """Mixin必要メソッド確認"""
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        required_mixin_methods = [
            "__init__",
            "_preprocess_images_to_binary",
            "_run_inference_with_model",
            "_setup_agent",
            "_get_provider_name",
        ]

        for method in required_mixin_methods:
            assert hasattr(PydanticAIAnnotatorMixin, method), f"Mixin method {method} not found"


class TestUnifiedImagePreprocessing:
    """統一画像前処理テスト"""

    def test_image_preprocessing(self, test_image):
        """統一画像前処理テスト"""
        from pydantic_ai.messages import BinaryContent

        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        # テスト画像作成
        test_images = [test_image, test_image]

        # Mixinインスタンス作成
        mixin = PydanticAIAnnotatorMixin("test-model")

        # 前処理実行
        processed = mixin._preprocess_images_to_binary(test_images)

        # 結果検証
        assert isinstance(processed, list)
        assert len(processed) == 2

        for item in processed:
            assert isinstance(item, BinaryContent)
            assert item.media_type == "image/webp"
            assert len(item.data) > 0


class TestAPIWrapperBackwardCompatibility:
    """API Wrapper後方互換性テスト"""

    def test_pydantic_ai_detection(self):
        """PydanticAI annotator検出テスト"""
        from image_annotator_lib.api import _is_pydantic_ai_webapi_annotator
        from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

        # 全プロバイダーのPydanticAI判定テスト
        pydantic_annotators = [
            ("Anthropic", AnthropicApiAnnotator),
            ("Google", GoogleApiAnnotator),
            ("OpenAI", OpenAIApiAnnotator),
        ]

        for provider_name, annotator_class in pydantic_annotators:
            assert _is_pydantic_ai_webapi_annotator(annotator_class), (
                f"{provider_name}: PydanticAI WebAPI annotator detection failed"
            )

    def test_wrapper_methods(self):
        """ラッパー必要メソッドテスト"""
        from image_annotator_lib.api import PydanticAIWebAPIWrapper
        from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator

        # ラッパーインスタンス作成テスト
        wrapper = PydanticAIWebAPIWrapper("test-model", AnthropicApiAnnotator)

        # 必要メソッドの確認
        required_wrapper_methods = ["__init__", "__enter__", "__exit__", "predict"]

        for method in required_wrapper_methods:
            assert hasattr(wrapper, method), f"Wrapper method {method} not found"


class TestUnifiedErrorHandling:
    """統一エラーハンドリングテスト"""

    def test_unified_error_handling(self, managed_config_registry, api_key_manager):
        """統一エラーハンドリングテスト"""
        from image_annotator_lib.exceptions.errors import (
            ApiAuthenticationError,
            ApiRateLimitError,
            ApiServerError,
            ApiTimeoutError,
        )

        # エラーハンドリング対象プロバイダー(OpenRouterは同一実装なのでOpenAIと同じ)
        providers_with_modules = [
            (
                "OpenAI",
                "image_annotator_lib.model_class.annotator_webapi.openai_api_response",
                "OpenAIApiAnnotator",
            ),
            (
                "Anthropic",
                "image_annotator_lib.model_class.annotator_webapi.anthropic_api",
                "AnthropicApiAnnotator",
            ),
            ("Google", "image_annotator_lib.model_class.annotator_webapi.google_api", "GoogleApiAnnotator"),
        ]

        error_test_cases = [
            ("authentication failed", ApiAuthenticationError),
            ("rate limit exceeded", ApiRateLimitError),
            ("timeout occurred", ApiTimeoutError),
            ("server error", ApiServerError),
        ]

        for provider_name, module_path, class_name in providers_with_modules:
            # モジュールとクラスをインポート
            module = __import__(module_path, fromlist=[class_name])
            annotator_class = getattr(module, class_name)

            # テスト用の設定をセットアップ
            test_model_name = f"test-{provider_name.lower()}-model"
            managed_config_registry.set(
                test_model_name,
                {
                    "api_key": api_key_manager.get_key(provider_name.lower()),
                    "api_model_id": f"{provider_name.lower()}-test-model",
                    "class": class_name,
                },
            )

            annotator = annotator_class(test_model_name)

            # 各エラータイプのテスト
            for error_message, expected_exception in error_test_cases:
                with pytest.raises(expected_exception):
                    annotator._handle_api_error(Exception(error_message))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
