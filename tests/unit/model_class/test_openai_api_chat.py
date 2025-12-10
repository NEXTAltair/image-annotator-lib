"""Unit tests for OpenAI API Chat annotator (openai_api_chat.py)

Phase C Task: Achieve 17% → 70%+ coverage for openai_api_chat.py

Test Strategy:
- REAL components: UnifiedAnnotationResult conversion, error handling, config loading
- MOCKED: PydanticAI Agent.run(), external API calls, config_registry
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from image_annotator_lib.core.types import AnnotationSchema, UnifiedAnnotationResult
from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import (
    OpenRouterApiAnnotator,
)

# ==============================================================================
# Priority 1A: OpenAI WebAPI Models - Test 1-5
# ==============================================================================


class TestOpenAIChatContextManager:
    """Context manager initialization tests for OpenAI/OpenRouter annotators."""

    @pytest.mark.unit
    def test_openai_chat_context_manager_initialization(self, managed_config_registry):
        """Test __enter__ setup with OpenRouter-specific configuration.

        Coverage: Lines 34-56 (__enter__ setup)

        REAL components:
        - Real OpenRouter prefix handling ("openrouter:" addition)
        - Real referer/app_name header configuration
        - Real config loading from config_registry

        MOCKED:
        - PydanticAI Agent creation
        - config_registry.get() for model-specific settings

        Scenario:
        1. Configure OpenRouter model with referer/app_name
        2. Call __enter__() to initialize annotator
        3. Verify Agent created with correct OpenRouter prefix
        4. Verify custom headers passed to Agent config

        Assertions:
        - Agent created via PydanticAIProviderFactory.get_cached_agent
        - api_model_id prefixed with "openrouter:"
        - referer and app_name in config_data
        """
        # Setup: Configure OpenRouter model (without referer/app_name in config dict)
        config = {
            "class": "OpenRouterApiAnnotator",
            "model_name_on_provider": "meta-llama/llama-3.1-8b-instruct",
            "api_model_id": "meta-llama/llama-3.1-8b-instruct",
            "api_key": "test_openrouter_key_123",
            "timeout": 30,
            "retry_count": 1,
            "temperature": 0.7,
            "max_output_tokens": 1800,
        }
        managed_config_registry.set("test_openrouter_model", config)

        # Set OpenRouter-specific headers directly in config_registry
        # These are retrieved via config_registry.get() during __enter__()
        from image_annotator_lib.core.config import config_registry

        config_registry.set("test_openrouter_model", "referer", "https://test-app.example.com")
        config_registry.set("test_openrouter_model", "app_name", "TestApp")

        # Mock PydanticAI Agent creation
        with patch(
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Act: Enter context manager
            annotator = OpenRouterApiAnnotator(model_name="test_openrouter_model")
            result = annotator.__enter__()

            # Assert: Context manager returns annotator instance
            assert result is annotator, "__enter__はannotatorインスタンスを返す"

            # Assert: Agent created with correct parameters
            assert mock_get_agent.called, "PydanticAIProviderFactory.get_cached_agentが呼ばれた"
            call_args = mock_get_agent.call_args

            # Verify OpenRouter prefix added
            assert call_args[1]["api_model_id"].startswith("openrouter:"), (
                "api_model_idに'openrouter:'プレフィックス追加"
            )

            # Verify custom headers in config_data
            config_data = call_args[1]["config_data"]
            assert config_data["referer"] == "https://test-app.example.com", "refererヘッダー設定"
            assert config_data["app_name"] == "TestApp", "app_nameヘッダー設定"

            # Assert: Agent assigned to instance
            assert annotator.agent is mock_agent, "Agentがannotatorインスタンスに割り当て"


class TestOpenAIChatInference:
    """Inference execution tests for OpenAI/OpenRouter annotators."""

    @pytest.mark.unit
    def test_openai_chat_run_with_model_success(self, managed_config_registry, lightweight_test_images):
        """Test successful inference with UnifiedAnnotationResult conversion.

        Coverage: Lines 63-140 (run_with_model core logic)

        REAL components:
        - Real UnifiedAnnotationResult conversion
        - Real capabilities handling
        - Real raw_output serialization

        MOCKED:
        - Agent.run() returns AnnotationSchema
        - Agent instance

        Scenario:
        1. Setup annotator with mocked Agent
        2. Call run_with_model() with test images
        3. Mock Agent.run() returns AnnotationSchema
        4. Verify UnifiedAnnotationResult created correctly

        Assertions:
        - Response parsed successfully
        - Tags extracted from AnnotationSchema
        - raw_output contains serialized response
        - No error in result
        """
        # Setup: Configure model (use unique name to avoid system config conflicts)
        config = {
            "class": "OpenRouterApiAnnotator",
            "model_name_on_provider": "test-model",
            "api_model_id": "test-model",
            "api_key": "test_key",
            "capabilities": ["tags", "captions", "scores"],
        }
        managed_config_registry.set("test_openai_model", config)

        # Mock Agent response
        mock_schema = AnnotationSchema(
            tags=["test_tag_1", "test_tag_2"], captions=["test caption"], score=0.95
        )

        with patch(
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Mock _run_inference_with_model to return AnnotationSchema
            annotator = OpenRouterApiAnnotator(model_name="test_openai_model")
            annotator.__enter__()

            with patch.object(annotator, "_run_inference_with_model", return_value=mock_schema):
                with patch.object(annotator, "_wait_for_rate_limit"):
                    # Act: Run inference
                    results = annotator.run_with_model(
                        images=lightweight_test_images[:1], model_id="test-model"
                    )

                    # Assert: Results returned
                    assert len(results) == 1, "1つの結果が返される"
                    result = results[0]

                    # Assert: UnifiedAnnotationResult structure
                    assert isinstance(result, UnifiedAnnotationResult), "UnifiedAnnotationResult型"
                    assert result.tags == ["test_tag_1", "test_tag_2"], "タグ正しく抽出"
                    assert result.captions == ["test caption"], "キャプション正しく抽出"
                    assert result.scores == {"score": 0.95}, "スコア正しく抽出"
                    assert result.error is None, "エラーなし"
                    assert result.provider_name == "openrouter", "プロバイダー名正しい"
                    assert result.framework == "api", "フレームワーク正しい"

                    # Assert: raw_output serialized
                    assert result.raw_output is not None, "raw_output存在"

    @pytest.mark.unit
    def test_openai_chat_error_handling_http_errors(self, managed_config_registry, lightweight_test_images):
        """Test HTTP error handling with ModelHTTPError.

        Coverage: Lines 141-168 (ModelHTTPError path)

        REAL components:
        - Real error wrapping to UnifiedAnnotationResult
        - Real error message formatting

        MOCKED:
        - ModelHTTPError exception from Agent.run()

        Scenario:
        1. Setup annotator
        2. Mock Agent.run() to raise ModelHTTPError
        3. Call run_with_model()
        4. Verify error captured in UnifiedAnnotationResult

        Assertions:
        - Error message formatted correctly
        - Result contains error field
        - No exception propagated
        """
        # Setup
        config = {
            "class": "OpenRouterApiAnnotator",
            "model_name_on_provider": "test-model",
            "api_model_id": "test-model",
            "api_key": "test_key",
            "capabilities": ["tags"],
        }
        managed_config_registry.set("test_model_error", config)

        with patch(
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            annotator = OpenRouterApiAnnotator(model_name="test_model_error")
            annotator.__enter__()

            # Mock HTTP error (ModelHTTPError needs model_name argument)

            http_error = ModelHTTPError(
                status_code=429,
                body="Rate limit exceeded",
                model_name="test-model",
            )

            with patch.object(annotator, "_run_inference_with_model", side_effect=http_error):
                with patch.object(annotator, "_wait_for_rate_limit"):
                    # Act: Run inference (should catch error)
                    results = annotator.run_with_model(
                        images=lightweight_test_images[:1], model_id="test-model"
                    )

                    # Assert: Error captured in result
                    assert len(results) == 1, "1つの結果が返される"
                    result = results[0]

                    assert isinstance(result, UnifiedAnnotationResult), "UnifiedAnnotationResult型"
                    assert result.error is not None, "エラーフィールド存在"
                    assert "429" in result.error, "ステータスコード含まれる"
                    assert "Rate limit exceeded" in result.error, "エラーメッセージ含まれる"
                    assert result.tags is None, "タグなし（エラー時）"

    @pytest.mark.unit
    def test_openai_chat_batch_processing(self, managed_config_registry, lightweight_test_images):
        """Test batch processing with rate limiting.

        Coverage: Lines 71-83 (batch loop iteration)

        REAL components:
        - Real loop iteration
        - Real BinaryContent conversion per image

        MOCKED:
        - Multiple Agent.run() calls
        - _wait_for_rate_limit() calls

        Scenario:
        1. Setup annotator with 3 test images
        2. Mock successful inference for each image
        3. Verify loop processes all images
        4. Verify _wait_for_rate_limit() called before each inference

        Assertions:
        - All images processed (loop iterations match input count)
        - _wait_for_rate_limit() called once per image
        - No exceptions during batch processing
        """
        # Setup
        config = {
            "class": "OpenRouterApiAnnotator",
            "model_name_on_provider": "test-model",
            "api_model_id": "test-model",
            "api_key": "test_key",
            "capabilities": ["tags"],
        }
        managed_config_registry.set("test_batch_model", config)

        mock_schema = AnnotationSchema(tags=["batch_tag"], captions=[], score=0.8)

        with patch(
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            annotator = OpenRouterApiAnnotator(model_name="test_batch_model")
            annotator.__enter__()

            with patch.object(annotator, "_run_inference_with_model", return_value=mock_schema):
                with patch.object(annotator, "_wait_for_rate_limit") as mock_wait:
                    # Act: Run inference on 3 images
                    results = annotator.run_with_model(
                        images=lightweight_test_images, model_id="test-model"
                    )

                    # Assert: All images processed
                    assert len(results) == len(lightweight_test_images), (
                        "全画像が処理される（ループ反復数一致）"
                    )

                    # Assert: Rate limiting called before each inference
                    assert mock_wait.call_count == len(lightweight_test_images), (
                        "_wait_for_rate_limit()が各画像前に呼ばれる"
                    )

                    # Assert: No exceptions occurred
                    for result in results:
                        assert result.error is None, "バッチ処理中に例外なし"

    @pytest.mark.unit
    def test_openrouter_custom_headers(self, managed_config_registry):
        """Test OpenRouter-specific header configuration.

        Coverage: Lines 34-56, 185-220 (__enter__ + _run_inference)

        REAL components:
        - Real "openrouter:" prefix addition
        - Real headers in config_data

        MOCKED:
        - config_registry.get for referer/app_name
        - Agent creation

        Scenario:
        1. Configure model with referer and app_name
        2. Enter context manager
        3. Verify "openrouter:" prefix added to api_model_id
        4. Verify referer/app_name passed to Agent config

        Assertions:
        - Correct prefix applied
        - referer in config_data
        - app_name in config_data
        """
        # Setup: OpenRouter with custom headers
        config = {
            "class": "OpenRouterApiAnnotator",
            "model_name_on_provider": "custom-model",
            "api_model_id": "custom-model",
            "api_key": "test_key",
            "temperature": 0.5,
            "max_output_tokens": 2000,
        }
        managed_config_registry.set("custom_model", config)

        # Set OpenRouter-specific headers directly in config_registry
        from image_annotator_lib.core.config import config_registry

        config_registry.set("custom_model", "referer", "https://custom-app.com")
        config_registry.set("custom_model", "app_name", "CustomApp")

        with patch(
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Act: Initialize and enter context
            annotator = OpenRouterApiAnnotator(model_name="custom_model")
            annotator.__enter__()

            # Assert: get_cached_agent called with correct parameters
            call_kwargs = mock_get_agent.call_args[1]

            # Verify OpenRouter prefix
            assert call_kwargs["api_model_id"] == "openrouter:custom-model", "OpenRouterプレフィックス追加"

            # Verify custom headers
            config_data = call_kwargs["config_data"]
            assert config_data["referer"] == "https://custom-app.com", "refererヘッダー設定"
            assert config_data["app_name"] == "CustomApp", "app_nameヘッダー設定"

            # Verify other config parameters
            assert config_data["temperature"] == 0.5, "temperature設定"
            assert config_data["max_tokens"] == 2000, "max_tokens設定"
