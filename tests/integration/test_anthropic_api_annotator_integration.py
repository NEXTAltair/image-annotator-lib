# tests/integration/test_anthropic_api_annotator_integration.py
"""
Integration tests for Anthropic API annotator.
Tests Provider-level PydanticAI integration, error handling, and Claude-specific functionality.
"""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.exceptions.errors import (
    WebApiError,
)
from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator


class TestAnthropicApiAnnotatorIntegration:
    """Integration tests for Anthropic API annotator."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        PydanticAIProviderFactory.clear_cache()
        yield
        PydanticAIProviderFactory.clear_cache()

    @pytest.fixture
    def anthropic_annotator_config(self, managed_config_registry):
        """Setup Anthropic annotator configuration."""
        config = {
            "class": "AnthropicApiAnnotator",
            "model_name_on_provider": "claude-3-5-sonnet-20241022",
            "api_model_id": "claude-3-5-sonnet-20241022",  # Kept in config_registry, filtered in WebAPIModelConfig
            "api_key": "test-anthropic-api-key",  # Kept in config_registry, filtered in WebAPIModelConfig
            "capabilities": ["tags", "captions", "scores"],  # Kept in config_registry, filtered in WebAPIModelConfig
            "timeout": 30,
            "retry_count": 3,
            "min_request_interval": 1.0,
        }
        managed_config_registry.set("anthropic_test_model", config)
        return config

    @pytest.fixture
    def anthropic_annotator(self, anthropic_annotator_config):
        """Create an Anthropic API annotator instance."""
        return AnthropicApiAnnotator("anthropic_test_model")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_annotator_initialization(self, anthropic_annotator):
        """Test proper initialization of Anthropic API annotator."""
        assert anthropic_annotator.model_name == "anthropic_test_model"
        assert anthropic_annotator.api_model_id == "claude-3-5-sonnet-20241022"
        assert anthropic_annotator.api_key.get_secret_value() == "test-anthropic-api-key"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_context_manager_integration(self, anthropic_annotator):
        """Test context manager setup and teardown."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Test context manager entry
            with anthropic_annotator as annotator:
                assert annotator is anthropic_annotator
                assert annotator.agent is not None
                mock_get_agent.assert_called_once()

            # Test context manager exit (should not raise exceptions)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_success(
        self, anthropic_annotator, lightweight_test_images, pydantic_ai_test_model
    ):
        """Test successful inference with Anthropic API using UnifiedAnnotationResult."""
        from image_annotator_lib.core.types import AnnotationSchema, UnifiedAnnotationResult

        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["test_model_tag"], captions=["A test caption."], score=0.95
            )
            mock_run_inference.return_value = mock_annotation

            with anthropic_annotator as annotator:
                results = annotator.run_with_model(lightweight_test_images[:2], "anthropic:claude-3-5-sonnet")

        # Verify results structure - should now only be UnifiedAnnotationResult format
        assert len(results) == 2
        for result in results:
            assert isinstance(result, UnifiedAnnotationResult)
            assert result.error is None
            assert result.tags == ["test_model_tag"]
            assert result.captions == ["A test caption."]
            if result.scores:
                assert result.scores.get("score") == 0.95

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_different_claude_version(
        self, anthropic_annotator, lightweight_test_images, pydantic_ai_test_model
    ):
        """Test model ID override with different Claude versions."""
        from image_annotator_lib.core.types import AnnotationSchema

        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["claude-3-haiku-tag"], captions=["Haiku model caption."], score=0.88
            )
            mock_run_inference.return_value = mock_annotation

            with anthropic_annotator as annotator:
                # Test with different Claude model
                results = annotator.run_with_model(
                    lightweight_test_images[:1],
                    "anthropic:claude-3-haiku",  # Different from config
                )

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].tags == ["claude-3-haiku-tag"]
        assert results[0].captions == ["Haiku model caption."]
        if results[0].scores:
            assert results[0].scores.get("score") == 0.88

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_authentication_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API authentication errors."""
        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("authentication failed")

            with anthropic_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0].error is not None
            assert "authentication failed" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_rate_limit_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API rate limiting."""
        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("rate limit exceeded")

            with anthropic_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0].error is not None
            assert "rate limit exceeded" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_model_not_found_error(
        self, anthropic_annotator, lightweight_test_images, pydantic_ai_test_model
    ):
        """Test handling of model not found errors (404)."""
        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("モデル未検出エラー: Model not found")

            with anthropic_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-invalid-model")

            assert len(results) == 1
            assert results[0].error is not None
            assert "モデル未検出エラー: Model not found" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_timeout_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API timeouts."""
        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("timeout occurred")

            with anthropic_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0].error is not None
            assert "timeout occurred" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_server_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API server errors."""
        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("500 server error")

            with anthropic_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0].error is not None
            assert "500 server error" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.skip(reason="TODO: Fix mock setup - currently makes real API calls when ALLOW_MODEL_REQUESTS=True")
    def test_anthropic_generic_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of generic Anthropic API errors."""
        from pydantic_ai import models
        # Temporarily enable ALLOW_MODEL_REQUESTS so mock can work
        original_value = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        try:
            with patch(
                "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
            ) as mock_get_agent:
                mock_agent = MagicMock()
                mock_agent.run = AsyncMock(side_effect=Exception("unknown error"))
                mock_get_agent.return_value = mock_agent

                anthropic_annotator._setup_agent()

                # Should handle generic errors gracefully
                results = anthropic_annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")
        finally:
            models.ALLOW_MODEL_REQUESTS = original_value

        assert len(results) == 1
        assert results[0].error is not None
        assert "Anthropic API Error" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_with_different_formats(self, anthropic_annotator, lightweight_test_images):
        """Test image preprocessing with different input formats."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            captured_inputs = []

            def mock_run(user_prompt=None, message_history=None, model_settings=None, **kwargs):
                if message_history:
                    captured_inputs.append(message_history[0])  # binary_content is in message_history[0]
                mock_response = MagicMock()
                mock_response.tags = ["preprocessing_test"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Test with PIL Images
            results = anthropic_annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")
            assert len(results) == 1
            assert results[0].error is None

            # Test preprocessing of different formats in _run_inference
            test_image = lightweight_test_images[0]

            # Convert to base64
            buffer = BytesIO()
            test_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            # Test with base64 string
            results_base64 = anthropic_annotator._run_inference([base64_data])
            assert len(results_base64) == 1

            # Test with bytes
            results_bytes = anthropic_annotator._run_inference([image_bytes])
            assert len(results_bytes) == 1

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_not_initialized_error(self, anthropic_annotator, lightweight_test_images):
        """Test error handling when agent is not initialized."""
        # Don't setup agent - should raise WebApiError
        with pytest.raises(WebApiError) as exc_info:
            anthropic_annotator.run_with_model(lightweight_test_images[:1], "anthropic:claude-3-5-sonnet")

        assert "Agent が初期化されていません" in str(exc_info.value)
        assert exc_info.value.provider_name == "Anthropic"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.skip(reason="TODO: Fix mock setup - currently makes real API calls when ALLOW_MODEL_REQUESTS=True")
    def test_batch_processing_resilience(self, anthropic_annotator, lightweight_test_images):
        """Test batch processing with individual failures."""
        from pydantic_ai import models
        # Temporarily enable ALLOW_MODEL_REQUESTS so mock can work
        original_value = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        try:
            with patch(
                "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
            ) as mock_get_agent:
                call_count = 0

                def mock_run(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1

                    if call_count % 3 == 0:
                        # Every third call fails
                        raise Exception("intermittent failure")
                    else:
                        # Successful calls
                        mock_response = MagicMock()
                        mock_response.tags = [f"batch_tag_{call_count}"]
                        return MagicMock(data=mock_response)

                mock_agent = MagicMock()
                mock_agent.run = AsyncMock(side_effect=mock_run)
                mock_get_agent.return_value = mock_agent

                anthropic_annotator._setup_agent()

                # Test with 6 images (2 will fail, 4 will succeed)
                test_images = lightweight_test_images * 2  # 6 images total
                results = anthropic_annotator.run_with_model(test_images, "anthropic:claude-3-5-sonnet")
        finally:
            models.ALLOW_MODEL_REQUESTS = original_value

        assert len(results) == 6

        # Count successes and failures
        success_count = sum(1 for r in results if r.error is None)
        error_count = sum(1 for r in results if r.error is not None)

        assert success_count == 4  # Should have 4 successes
        assert error_count == 2  # Should have 2 failures

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_integration(self, anthropic_annotator_config, lightweight_test_images):
        """Test integration with Provider Manager."""

        # Test environment detection should work, and we mock the provider instance
        from unittest.mock import MagicMock

        from image_annotator_lib.core.types import AnnotationSchema

        # Create mock response matching the expected structure
        mock_response_data = AnnotationSchema(
            tags=["test_model_tag"], captions=["A test caption."], score=0.95
        )

        # Mock the entire provider instance to avoid PydanticAI complexity
        mock_provider_instance = MagicMock()
        mock_provider_instance.run_with_model.return_value = [
            {"response": mock_response_data, "error": None}
        ]

        with patch.object(
            ProviderManager, "get_provider_instance", return_value=mock_provider_instance
        ) as mock_get_provider:
            # Test through Provider Manager
            result = ProviderManager.run_inference_with_model(
                "anthropic_test_model", lightweight_test_images[:1], api_model_id="anthropic:claude-3-5-sonnet"
            )

            # Verify that get_provider_instance was called
            mock_get_provider.assert_called_once_with("anthropic")
            # Verify that the provider's run_with_model was called
            mock_provider_instance.run_with_model.assert_called_once()

        assert result is not None
        assert len(result) > 0

        # Verify ProviderManager properly handled the request
        for _image_hash, annotation_result in result.items():
            # AnnotationResult is a TypedDict, so access like a dictionary
            assert annotation_result.get("error") is None
            assert annotation_result.get("tags") == ["test_model_tag"]
            assert annotation_result.get("formatted_output") == mock_response_data

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_integration(self, anthropic_annotator, lightweight_test_images):
        """Test rate limiting functionality."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            with patch.object(anthropic_annotator, "_wait_for_rate_limit") as mock_rate_limit:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["rate_limit_test"]
                mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
                mock_get_agent.return_value = mock_agent

                anthropic_annotator._setup_agent()

                # Test multiple images (should trigger rate limiting)
                results = anthropic_annotator.run_with_model(lightweight_test_images, "anthropic:claude-3-5-sonnet")

                # Verify rate limiting was called for each image
                assert mock_rate_limit.call_count == len(lightweight_test_images)
                assert len(results) == len(lightweight_test_images)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_error_handling(self, anthropic_annotator):
        """Test error handling in image preprocessing."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Test with invalid image data
            invalid_data = ["invalid_base64_data", b"invalid_bytes"]

            results = anthropic_annotator._run_inference(invalid_data)

            # Should handle preprocessing errors gracefully
            assert len(results) == len(invalid_data)
            for result in results:
                from image_annotator_lib.core.types import UnifiedAnnotationResult

                assert isinstance(result, UnifiedAnnotationResult)
                assert result.error is not None
                assert "画像前処理エラー" in result.error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_validation(self, managed_config_registry):
        """Test configuration validation for Anthropic annotator."""
        from unittest.mock import patch
        from image_annotator_lib.exceptions.errors import ConfigurationError

        # Test missing model_name_on_provider (should raise ConfigurationError)
        invalid_config = {
            "class": "AnthropicApiAnnotator",
            "api_model_id": "claude-3-5-sonnet",
            # Missing model_name_on_provider field
        }
        managed_config_registry.set("invalid_anthropic", invalid_config)

        # This should raise ConfigurationError due to missing model_name_on_provider
        with pytest.raises(ConfigurationError) as exc_info:
            AnthropicApiAnnotator("invalid_anthropic")

        assert "model_name_on_provider" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_model_id_override_functionality(
        self, anthropic_annotator, lightweight_test_images, pydantic_ai_test_model
    ):
        """Test model ID override in various Claude models."""
        from image_annotator_lib.core.types import AnnotationSchema

        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["override_test_tag"], captions=["Override test caption."], score=0.92
            )
            mock_run_inference.return_value = mock_annotation

            with anthropic_annotator as annotator:
                # Test with different Claude models
                claude_models = ["anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku", "anthropic:claude-3-opus"]

                for model_id in claude_models:
                    results = annotator.run_with_model(lightweight_test_images[:1], model_id)
                    assert len(results) == 1
                    assert results[0].error is None
                    assert results[0].tags == ["override_test_tag"]
                    assert results[0].captions == ["Override test caption."]
                    if results[0].scores:
                        assert results[0].scores.get("score") == 0.92

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.skip(reason="TODO: Fix mock setup - currently makes real API calls when ALLOW_MODEL_REQUESTS=True")
    def test_sequential_claude_requests(self, anthropic_annotator, lightweight_test_images):
        """Test handling of sequential requests to Claude (per specification)."""
        from pydantic_ai import models
        # Temporarily enable ALLOW_MODEL_REQUESTS so mock can work
        original_value = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        try:
            with patch(
                "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
            ) as mock_get_agent:
                sequential_calls = []

                def mock_run(*args, **kwargs):
                    call_id = len(sequential_calls) + 1
                    sequential_calls.append(call_id)
                    mock_response = MagicMock()
                    mock_response.tags = [f"sequential_claude_{call_id}"]
                    mock_response.captions = [f"Sequential caption {call_id}"]
                    mock_response.score = 0.85
                    return MagicMock(data=mock_response)

                mock_agent = MagicMock()
                mock_agent.run = AsyncMock(side_effect=mock_run)
                mock_get_agent.return_value = mock_agent

                anthropic_annotator._setup_agent()

                # Test sequential processing (per specification)
                results = anthropic_annotator.run_with_model(lightweight_test_images, "anthropic:claude-3-5-sonnet")
        finally:
            models.ALLOW_MODEL_REQUESTS = original_value

        # Verify all requests were processed sequentially
        assert len(results) == len(lightweight_test_images)
        assert len(sequential_calls) == len(lightweight_test_images)

        # Verify all successful with UnifiedAnnotationResult format
        for result in results:
            from image_annotator_lib.core.types import UnifiedAnnotationResult

            assert isinstance(result, UnifiedAnnotationResult)
            assert result.error is None
            assert result.tags is not None
            assert result.captions is not None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_large_image_handling(self, anthropic_annotator):
        """Test handling of large images with Claude."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["large_image_test"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Create a larger test image
            large_image = Image.new("RGB", (1024, 1024), "blue")

            results = anthropic_annotator.run_with_model([large_image], "anthropic:claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0].error is None
            # Check that we have tags in the result
            assert results[0].tags is not None
