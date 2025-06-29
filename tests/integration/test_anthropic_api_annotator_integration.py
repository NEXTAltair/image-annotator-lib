# tests/integration/test_anthropic_api_annotator_integration.py
"""
Integration tests for Anthropic API annotator.
Tests Provider-level PydanticAI integration, error handling, and Claude-specific functionality.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image
import base64
from io import BytesIO

from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiServerError,
    ApiTimeoutError,
    ModelNotFoundError,
    WebApiError
)


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
            "rate_limit_requests_per_minute": 40
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
    def test_run_with_model_success(self, anthropic_annotator, lightweight_test_images):
        """Test successful inference with Anthropic API."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Mock successful API response
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["anthropic_tag_1", "anthropic_tag_2"]
            mock_response.caption = "Claude generated caption"
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            # Setup agent
            anthropic_annotator._setup_agent()

            # Test inference
            results = anthropic_annotator.run_with_model(
                lightweight_test_images[:2], 
                "claude-3-5-sonnet"
            )

            # Verify results
            assert len(results) == 2
            for result in results:
                assert result["error"] is None
                assert result["response"] == mock_response
                assert mock_agent.run.called

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_different_claude_version(self, anthropic_annotator, lightweight_test_images):
        """Test model ID override with different Claude versions."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["claude_haiku_tag"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Test with different Claude model
            results = anthropic_annotator.run_with_model(
                lightweight_test_images[:1],
                "claude-3-haiku"  # Different from config
            )

            assert len(results) == 1
            assert results[0]["error"] is None
            assert results[0]["response"] == mock_response

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_authentication_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API authentication errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("authentication failed"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should raise ApiAuthenticationError
            with pytest.raises(ApiAuthenticationError) as exc_info:
                anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert "Anthropic API 認証エラー" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_rate_limit_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API rate limiting."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should raise ApiRateLimitError
            with pytest.raises(ApiRateLimitError) as exc_info:
                anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert "Anthropic API レート制限" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_model_not_found_error(self, anthropic_annotator, lightweight_test_images):
        """Test handling of model not found errors (404)."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("404 not_found_error model: claude-invalid-model"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should raise ModelNotFoundError
            with pytest.raises(ModelNotFoundError) as exc_info:
                anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-invalid-model")

            assert "claude-invalid-model" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_timeout_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API timeouts."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("timeout occurred"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should raise ApiTimeoutError
            with pytest.raises(ApiTimeoutError) as exc_info:
                anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert "Anthropic API タイムアウト" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_server_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of Anthropic API server errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("500 server error"))
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Should raise ApiServerError
            with pytest.raises(ApiServerError) as exc_info:
                anthropic_annotator.run_with_model(lightweight_test_images[:1], "claude-3-5-sonnet")

            assert "Anthropic API サーバーエラー" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_anthropic_generic_error_handling(self, anthropic_annotator, lightweight_test_images):
        """Test handling of generic Anthropic API errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
            test_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
            assert error_count == 2    # Should have 2 failures

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_integration(self, anthropic_annotator_config, lightweight_test_images):
        """Test integration with Provider Manager."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["provider_manager_claude"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            # Test through Provider Manager
            result = ProviderManager.run_inference_with_model(
                "anthropic_test_model",
                lightweight_test_images[:1],
                api_model_id="claude-3-5-sonnet"
            )

            assert result is not None
            assert len(result) > 0

            # Verify ProviderManager properly handled the request
            for image_hash, annotation_result in result.items():
                assert annotation_result.error is None
                assert annotation_result.tags == ["provider_manager_claude"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_integration(self, anthropic_annotator, lightweight_test_images):
        """Test rate limiting functionality."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch.object(anthropic_annotator, '_wait_for_rate_limit') as mock_rate_limit:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["rate_limit_test"]
                mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
                mock_get_agent.return_value = mock_agent

                anthropic_annotator._setup_agent()

                # Test multiple images (should trigger rate limiting)
                results = anthropic_annotator.run_with_model(
                    lightweight_test_images, 
                    "claude-3-5-sonnet"
                )

                # Verify rate limiting was called for each image
                assert mock_rate_limit.call_count == len(lightweight_test_images)
                assert len(results) == len(lightweight_test_images)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_error_handling(self, anthropic_annotator):
        """Test error handling in image preprocessing."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
        # Test missing API key
        invalid_config = {
            "class": "AnthropicApiAnnotator",
            "api_model_id": "claude-3-5-sonnet"
            # Missing api_key
        }
        managed_config_registry.set("invalid_anthropic", invalid_config)

        with pytest.raises(Exception):  # Should raise configuration error
            AnthropicApiAnnotator("invalid_anthropic")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_model_id_override_functionality(self, anthropic_annotator, lightweight_test_images):
        """Test model ID override in various Claude models."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            captured_model_calls = []
            
            def mock_run(user_prompt=None, message_history=None, model_settings=None, **kwargs):
                captured_model_calls.append(model_settings)  # model info is in model_settings
                mock_response = MagicMock()
                mock_response.tags = ["model_override_test"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            anthropic_annotator._setup_agent()

            # Test with different Claude models
            claude_models = ["claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus"]
            
            for model_id in claude_models:
                results = anthropic_annotator.run_with_model(
                    lightweight_test_images[:1], 
                    model_id
                )
                assert len(results) == 1
                assert results[0]["error"] is None

            # Verify different models were used
            assert len(captured_model_calls) == len(claude_models)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_claude_requests(self, anthropic_annotator, lightweight_test_images):
        """Test handling of concurrent requests to Claude."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
            results = anthropic_annotator.run_with_model(
                lightweight_test_images,
                "claude-3-5-sonnet"
            )

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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
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
            assert results[0]["response"] == mock_response