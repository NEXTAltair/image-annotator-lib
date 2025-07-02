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
            "api_model_id": "claude-3-5-sonnet",
            "api_key": "test-anthropic-api-key",
            "timeout": 30,
            "max_retries": 3,
            "rate_limit_requests_per_minute": 40,
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
        assert anthropic_annotator.api_model_id == "claude-3-5-sonnet"
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
        """Test successful inference with Anthropic API using TestModel."""
        from image_annotator_lib.core.types import AnnotationSchema

        with patch.object(
            anthropic_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["test_model_tag"], captions=["A test caption."], score=0.95
            )
            mock_run_inference.return_value = mock_annotation

            with anthropic_annotator as annotator:
                results = annotator.run_with_model(lightweight_test_images[:2], "claude-3-5-sonnet")

        # Verify results
        assert len(results) == 2
        for result in results:
            assert result["error"] is None
            response = result["response"]
            assert response.tags == ["test_model_tag"]
            assert response.captions == ["A test caption."]
            assert response.score == 0.95

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
                    "claude-3-haiku",  # Different from config
                )

        assert len(results) == 1
        assert results[0]["error"] is None
        response = results[0]["response"]
        assert response.tags == ["claude-3-haiku-tag"]
        assert response.captions == ["Haiku model caption."]
        assert response.score == 0.88

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
                results = annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "authentication failed" in results[0]["error"]
            assert results[0]["response"] is None

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
                results = annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "rate limit exceeded" in results[0]["error"]
            assert results[0]["response"] is None

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
                results = annotator.run_with_model(lightweight_test_images[:1], "claude-invalid-model")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "モデル未検出エラー: Model not found" in results[0]["error"]
            assert results[0]["response"] is None

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
                results = annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "timeout occurred" in results[0]["error"]
            assert results[0]["response"] is None

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
                results = annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "500 server error" in results[0]["error"]
            assert results[0]["response"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_generic_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of generic Anthropic API errors."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("unknown error"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should handle generic errors gracefully
            results = anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "Anthropic API Error" in results[0]["error"]
            assert results[0]["response"] is None

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
            results = anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")
            assert len(results) == 1
            assert results[0]["error"] is None

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
            anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

        assert "Agent が初期化されていません" in str(exc_info.value)
        assert exc_info.value.provider_name == "Anthropic"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_resilience(self, anthropic_annotator, lightweight_test_images):
        """Test batch processing with individual failures."""
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
            results = anthropic_annotator.run_with_model(test_images, "claude-3-5-sonnet")

            assert len(results) == 6

            # Count successes and failures
            success_count = sum(1 for r in results if r["error"] is None)
            error_count = sum(1 for r in results if r["error"] is not None)

            assert success_count == 4  # Should have 4 successes
            assert error_count == 2  # Should have 2 failures

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_integration(self, anthropic_annotator_config, lightweight_test_images):
        """Test integration with Provider Manager."""

        # Mock the underlying run_with_model method
        mock_response = {
            "response": {"tags": ["test_model_tag"], "caption": "A test caption."},
            "error": None,
        }

        with patch.object(
            AnthropicApiAnnotator, "run_with_model", return_value=[mock_response]
        ) as mock_run:
            # Test through Provider Manager
            result = ProviderManager.run_inference_with_model(
                "anthropic_test_model", lightweight_test_images[:1], api_model_id="claude-3-5-sonnet"
            )

            # Verify that run_with_model was called
            mock_run.assert_called_once()

        assert result is not None
        assert len(result) > 0

        # Verify ProviderManager properly handled the request
        for image_hash, annotation_result in result.items():
            assert annotation_result["error"] is None
            assert "tags" in annotation_result["formatted_output"]
            assert annotation_result["formatted_output"]["tags"] == ["test_model_tag"]

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
                results = anthropic_annotator.run_with_model(lightweight_test_images, "claude-3-5-sonnet")

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
                assert result["error"] is not None
                assert "画像前処理エラー" in result["error"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_validation(self, managed_config_registry):
        """Test configuration validation for Anthropic annotator."""
        from unittest.mock import patch

        # Test missing API key
        invalid_config = {
            "class": "AnthropicApiAnnotator",
            "api_model_id": "claude-3-5-sonnet",
            # Missing api_key
        }
        managed_config_registry.set("invalid_anthropic", invalid_config)

        # Temporarily disable test environment detection to test validation
        with patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False):
            with pytest.raises((WebApiError, ValueError)) as exc_info:
                AnthropicApiAnnotator("invalid_anthropic")

            assert "キーが設定されていません" in str(exc_info.value)

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
                claude_models = ["claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus"]

                for model_id in claude_models:
                    results = annotator.run_with_model(lightweight_test_images[:1], model_id)
                    assert len(results) == 1
                    assert results[0]["error"] is None
                    response = results[0]["response"]
                    assert response.tags == ["override_test_tag"]
                    assert response.captions == ["Override test caption."]
                    assert response.score == 0.92

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_claude_requests(self, anthropic_annotator, lightweight_test_images):
        """Test handling of concurrent requests to Claude."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            concurrent_calls = []

            def mock_run(*args, **kwargs):
                call_id = len(concurrent_calls) + 1
                concurrent_calls.append(call_id)
                mock_response = MagicMock()
                mock_response.tags = [f"concurrent_claude_{call_id}"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Simulate concurrent processing
            results = anthropic_annotator.run_with_model(lightweight_test_images, "claude-3-5-sonnet")

            # Verify all requests were processed
            assert len(results) == len(lightweight_test_images)
            assert len(concurrent_calls) == len(lightweight_test_images)

            # Verify all successful
            for result in results:
                assert result["error"] is None

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

            results = anthropic_annotator.run_with_model([large_image], "claude-3-5-sonnet")

            assert len(results) == 1
            assert results[0]["error"] is None
            # Check that we have a response rather than comparing MagicMock objects
            assert results[0]["response"] is not None
            assert hasattr(results[0]["response"], "tags")
