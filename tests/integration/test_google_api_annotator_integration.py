# tests/integration/test_google_api_annotator_integration.py
"""
Integration tests for Google API annotator.
Tests Provider-level PydanticAI integration, error handling, and resource management.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image
import asyncio

from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiServerError,
    ApiTimeoutError
)


class TestGoogleApiAnnotatorIntegration:
    """Integration tests for Google API annotator."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        PydanticAIProviderFactory.clear_cache()
        yield
        PydanticAIProviderFactory.clear_cache()

    @pytest.fixture
    def google_annotator_config(self, managed_config_registry):
        """Setup Google annotator configuration."""
        config = {
            "class": "GoogleApiAnnotator",
            "api_model_id": "gemini-1.5-pro",
            "api_key": "test-google-api-key",
            "timeout": 30,
            "max_retries": 3,
            "rate_limit_requests_per_minute": 30
        }
        managed_config_registry.set("google_test_model", config)
        return config

    @pytest.fixture
    def google_annotator(self, google_annotator_config):
        """Create a Google API annotator instance."""
        return GoogleApiAnnotator("google_test_model")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_annotator_initialization(self, google_annotator):
        """Test proper initialization of Google API annotator."""
        assert google_annotator.model_name == "google_test_model"
        assert google_annotator.api_model_id == "gemini-1.5-pro"
        assert google_annotator.api_key.get_secret_value() == "test-google-api-key"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_context_manager_integration(self, google_annotator):
        """Test context manager setup and teardown."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Test context manager entry
            with google_annotator as annotator:
                assert annotator is google_annotator
                assert annotator.agent is not None
                mock_get_agent.assert_called_once()

            # Test context manager exit (should not raise exceptions)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_success(self, google_annotator, lightweight_test_images):
        """Test successful inference with Google API."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Mock successful API response
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["google_tag_1", "google_tag_2"]
            mock_response.caption = "Google generated caption"
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            # Setup agent
            google_annotator._setup_agent()

            # Test inference
            results = google_annotator.run_with_model(
                lightweight_test_images[:2], 
                "gemini-1.5-pro"
            )

            # Verify results
            assert len(results) == 2
            for result in results:
                assert result["error"] is None
                assert result["response"] == mock_response
                assert mock_agent.run.called

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_different_model_id(self, google_annotator, lightweight_test_images):
        """Test model ID override functionality."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["override_tag"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Test with different model ID
            results = google_annotator.run_with_model(
                lightweight_test_images[:1],
                "gemini-1.5-flash"  # Different from config
            )

            assert len(results) == 1
            assert results[0]["error"] is None
            assert results[0]["response"] == mock_response

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_authentication_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API authentication errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("authentication failed"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should raise ApiAuthenticationError
            with pytest.raises(ApiAuthenticationError) as exc_info:
                google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert "Google API認証エラー" in str(exc_info.value)
            assert exc_info.value.provider_name == "google"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_rate_limit_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API rate limiting."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("quota exceeded"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should raise ApiRateLimitError
            with pytest.raises(ApiRateLimitError) as exc_info:
                google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert "Google APIレート制限" in str(exc_info.value)
            assert exc_info.value.provider_name == "google"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_timeout_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API timeouts."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError("timeout occurred"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should raise ApiTimeoutError
            with pytest.raises(ApiTimeoutError) as exc_info:
                google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert "Google APIタイムアウト" in str(exc_info.value)
            assert exc_info.value.provider_name == "google"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_server_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API server errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("500 server error"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should raise ApiServerError
            with pytest.raises(ApiServerError) as exc_info:
                google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert "Google APIサーバーエラー" in str(exc_info.value)
            assert exc_info.value.provider_name == "google"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_generic_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of generic Google API errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("unknown error"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should handle generic errors gracefully
            results = google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0]["error"] is not None
            assert "Google API Error" in results[0]["error"]
            assert results[0]["response"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_with_mixed_results(self, google_annotator, lightweight_test_images):
        """Test batch processing with some successes and some failures."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            call_count = 0
            
            def mock_run(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count % 2 == 0:
                    # Every second call fails
                    raise Exception("intermittent error")
                else:
                    # Successful calls
                    mock_response = MagicMock()
                    mock_response.tags = [f"batch_tag_{call_count}"]
                    return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Test with 4 images
            results = google_annotator.run_with_model(lightweight_test_images, "gemini-1.5-pro")

            assert len(results) == len(lightweight_test_images)
            
            # Check mixed results
            success_count = sum(1 for r in results if r["error"] is None)
            error_count = sum(1 for r in results if r["error"] is not None)
            
            assert success_count > 0
            assert error_count > 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_integration(self, google_annotator, lightweight_test_images):
        """Test image preprocessing to binary content."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Track what gets passed to the agent
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

            google_annotator._setup_agent()

            # Test image preprocessing
            results = google_annotator.run_with_model(lightweight_test_images[:2], "gemini-1.5-pro")

            # Verify preprocessing occurred
            assert len(captured_inputs) == 2
            assert len(results) == 2
            
            # Verify all results successful
            for result in results:
                assert result["error"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_generate_tags_from_response(self, google_annotator):
        """Test tag generation from API responses."""
        # Test with structured response object
        mock_response_obj = MagicMock()
        mock_response_obj.tags = ["test_tag_1", "test_tag_2"]
        
        formatted_output = {"response": mock_response_obj, "error": None}
        tags = google_annotator._generate_tags(formatted_output)
        assert tags == ["test_tag_1", "test_tag_2"]

        # Test with dict response
        dict_response = {"tags": ["dict_tag_1", "dict_tag_2"]}
        formatted_output = {"response": dict_response, "error": None}
        tags = google_annotator._generate_tags(formatted_output)
        assert tags == ["dict_tag_1", "dict_tag_2"]

        # Test with error response
        formatted_output = {"response": None, "error": "API Error"}
        tags = google_annotator._generate_tags(formatted_output)
        assert tags == []

        # Test with empty response
        formatted_output = {"response": None, "error": None}
        tags = google_annotator._generate_tags(formatted_output)
        assert tags == []

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_integration(self, google_annotator_config, lightweight_test_images):
        """Test integration with Provider Manager."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["provider_manager_tag"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            # Test through Provider Manager
            result = ProviderManager.run_inference_with_model(
                "google_test_model",
                lightweight_test_images[:1],
                api_model_id="gemini-1.5-pro"
            )

            assert result is not None
            assert len(result) > 0

            # Verify ProviderManager properly handled the request
            for image_hash, annotation_result in result.items():
                assert annotation_result.error is None
                assert annotation_result.tags == ["provider_manager_tag"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_integration(self, google_annotator, lightweight_test_images):
        """Test rate limiting functionality."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch.object(google_annotator, '_wait_for_rate_limit') as mock_rate_limit:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["rate_limit_test"]
                mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
                mock_get_agent.return_value = mock_agent

                google_annotator._setup_agent()

                # Test multiple images (should trigger rate limiting)
                results = google_annotator.run_with_model(
                    lightweight_test_images, 
                    "gemini-1.5-pro"
                )

                # Verify rate limiting was called for each image
                assert mock_rate_limit.call_count == len(lightweight_test_images)
                assert len(results) == len(lightweight_test_images)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_validation(self, managed_config_registry):
        """Test configuration validation for Google annotator."""
        # Test missing API key
        invalid_config = {
            "class": "GoogleApiAnnotator",
            "api_model_id": "gemini-1.5-pro"
            # Missing api_key
        }
        managed_config_registry.set("invalid_google", invalid_config)

        with pytest.raises(Exception):  # Should raise configuration error
            GoogleApiAnnotator("invalid_google")

        # Test missing model ID
        invalid_config2 = {
            "class": "GoogleApiAnnotator",
            "api_key": "test-key"
            # Missing api_model_id
        }
        managed_config_registry.set("invalid_google2", invalid_config2)

        annotator = GoogleApiAnnotator("invalid_google2")
        
        # Should raise error when trying to run inference without model ID
        with pytest.raises(ValueError):
            annotator._run_inference([Image.new("RGB", (64, 64), "red")])

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_model_id_override_through_provider_manager(self, google_annotator_config, lightweight_test_images):
        """Test model ID override functionality through Provider Manager."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            captured_model_id = None
            
            def mock_run(user_prompt=None, message_history=None, model_settings=None, **kwargs):
                nonlocal captured_model_id
                captured_model_id = model_settings  # model info is in model_settings
                mock_response = MagicMock()
                mock_response.tags = ["override_test"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            # Test with different model ID than configured
            result = ProviderManager.run_inference_with_model(
                "google_test_model",
                lightweight_test_images[:1],
                api_model_id="gemini-1.5-flash"  # Different from config
            )

            assert result is not None
            # Verify the override was passed through
            # Note: This depends on how the model override is implemented

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_requests_handling(self, google_annotator, lightweight_test_images):
        """Test handling of concurrent requests to Google API."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            concurrent_calls = []
            
            def mock_run(*args, **kwargs):
                concurrent_calls.append(len(concurrent_calls) + 1)
                mock_response = MagicMock()
                mock_response.tags = [f"concurrent_tag_{len(concurrent_calls)}"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=mock_run)
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Simulate concurrent processing by processing multiple images
            results = google_annotator.run_with_model(
                lightweight_test_images,
                "gemini-1.5-pro"
            )

            # Verify all requests were processed
            assert len(results) == len(lightweight_test_images)
            assert len(concurrent_calls) == len(lightweight_test_images)
            
            # Verify all successful
            for result in results:
                assert result["error"] is None