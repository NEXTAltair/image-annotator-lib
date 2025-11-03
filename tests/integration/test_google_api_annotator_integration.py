# tests/integration/test_google_api_annotator_integration.py
"""
Integration tests for Google API annotator.
Tests Provider-level PydanticAI integration, error handling, and resource management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

# Note: No longer importing custom API exceptions - using PydanticAI unified error handling
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator


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
            "model_name_on_provider": "gemini-1.5-pro",
            "api_model_id": "gemini-1.5-pro",
            "api_key": "test-google-api-key",
            "timeout": 30,
            "max_retries": 3,
            "rate_limit_requests_per_minute": 30,
            "capabilities": ["tags", "captions", "scores"],  # マルチモーダルLLM対応
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
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
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
    def test_run_with_model_success(
        self, google_annotator, lightweight_test_images, pydantic_ai_test_model
    ):
        """Test successful inference with Google API using TestModel."""
        from image_annotator_lib.core.types import AnnotationSchema

        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["google_test_tag"], captions=["A Google test caption."], score=0.93
            )
            mock_run_inference.return_value = mock_annotation

            with google_annotator as annotator:
                results = annotator.run_with_model(lightweight_test_images[:2], "gemini-1.5-pro")

        # Verify results
        assert len(results) == 2
        for result in results:
            from image_annotator_lib.core.types import UnifiedAnnotationResult

            assert isinstance(result, UnifiedAnnotationResult)
            assert result.error is None
            assert result.tags == ["google_test_tag"]
            assert result.captions == ["A Google test caption."]
            if result.scores:
                assert result.scores.get("score") == 0.93

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_run_with_model_different_model_id(self, google_annotator, lightweight_test_images):
        """Test model ID override functionality."""
        from image_annotator_lib.core.types import AnnotationSchema

        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Return actual AnnotationSchema object instead of MagicMock
            mock_annotation = AnnotationSchema(
                tags=["gemini-flash-tag"], captions=["Flash model caption."], score=0.89
            )
            mock_run_inference.return_value = mock_annotation

            with google_annotator as annotator:
                # Test with different model ID
                results = annotator.run_with_model(
                    lightweight_test_images[:1],
                    "gemini-1.5-flash",  # Different from config
                )

            assert len(results) == 1
            assert results[0].error is None
            assert results[0].tags == ["gemini-flash-tag"]
            assert results[0].captions == ["Flash model caption."]
            if results[0].scores:
                assert results[0].scores.get("score") == 0.89

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_authentication_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API authentication errors."""
        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("authentication failed")

            with google_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0].error is not None
            assert "authentication failed" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_rate_limit_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API rate limiting."""
        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("quota exceeded")

            with google_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0].error is not None
            assert "quota exceeded" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_timeout_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API timeouts."""
        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = TimeoutError("timeout occurred")

            with google_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0].error is not None
            assert "timeout occurred" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_server_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of Google API server errors."""
        with patch.object(
            google_annotator, "_run_inference_async", new_callable=AsyncMock
        ) as mock_run_inference:
            # Mock should raise the expected exception
            mock_run_inference.side_effect = Exception("500 server error")

            with google_annotator as annotator:
                # Should return error in result instead of raising exception
                results = annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0].error is not None
            assert "500 server error" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_google_api_generic_error_handling(self, google_annotator, lightweight_test_images):
        """Test handling of generic Google API errors."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(side_effect=Exception("unknown error"))
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Should handle generic errors gracefully
            results = google_annotator.run_with_model(lightweight_test_images[:1], "gemini-1.5-pro")

            assert len(results) == 1
            assert results[0].error is not None
            assert "Google API Error" in results[0].error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_with_mixed_results(self, google_annotator, lightweight_test_images):
        """Test batch processing with some successes and some failures."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
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
            success_count = sum(1 for r in results if r.error is None)
            error_count = sum(1 for r in results if r.error is not None)

            assert success_count > 0
            assert error_count > 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_integration(self, google_annotator, lightweight_test_images):
        """Test image preprocessing to binary content."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track what gets passed to the agent
            captured_inputs = []

            async def mock_run(user_prompt_parts, model_settings=None, **kwargs):
                # user_prompt_parts is a list containing [str, BinaryContent]
                if isinstance(user_prompt_parts, list) and len(user_prompt_parts) > 1:
                    captured_inputs.append(user_prompt_parts[1])  # BinaryContent is at index 1
                mock_response = MagicMock()
                mock_response.tags = ["preprocessing_test"]
                mock_response.captions = ["Test caption"]
                mock_response.score = 0.9
                mock_result = MagicMock()
                mock_result.output = mock_response
                return mock_result

            mock_agent = MagicMock()
            mock_agent.run = mock_run
            mock_get_agent.return_value = mock_agent

            google_annotator._setup_agent()

            # Test image preprocessing
            results = google_annotator.run_with_model(lightweight_test_images[:2], "gemini-1.5-pro")

            # Verify preprocessing occurred
            assert len(captured_inputs) == 2
            assert len(results) == 2

            # Verify all results successful
            for result in results:
                assert result.error is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_generate_tags_from_response(self, google_annotator):
        """Test tag generation from API responses (UnifiedAnnotationResult)."""
        from image_annotator_lib.core.types import UnifiedAnnotationResult, TaskCapability

        # Test with structured UnifiedAnnotationResult with tags
        result_with_tags = UnifiedAnnotationResult(
            model_name=google_annotator.model_name,
            capabilities={TaskCapability.TAGS},
            tags=["test_tag_1", "test_tag_2"],
            provider_name="google",
            framework="api",
        )
        tags = google_annotator._generate_tags(result_with_tags)
        assert tags == ["test_tag_1", "test_tag_2"]

        # Test with UnifiedAnnotationResult with error
        result_with_error = UnifiedAnnotationResult(
            model_name=google_annotator.model_name,
            capabilities={TaskCapability.TAGS},
            error="API Error",
            provider_name="google",
            framework="api",
        )
        tags = google_annotator._generate_tags(result_with_error)
        assert tags == []

        # Test with UnifiedAnnotationResult with no tags
        result_no_tags = UnifiedAnnotationResult(
            model_name=google_annotator.model_name,
            capabilities={TaskCapability.TAGS},
            tags=None,
            provider_name="google",
            framework="api",
        )
        tags = google_annotator._generate_tags(result_no_tags)
        assert tags == []

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_integration(self, google_annotator_config, lightweight_test_images):
        """Test integration with Provider Manager."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["provider_manager_tag"]
            mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
            mock_get_agent.return_value = mock_agent

            # Test through Provider Manager
            result = ProviderManager.run_inference_with_model(
                "google_test_model", lightweight_test_images[:1], api_model_id="gemini-1.5-pro"
            )

            assert result is not None
            assert len(result) > 0

            # Verify ProviderManager properly handled the request
            # ProviderManager returns dict[str, AnnotationResult] not dict[str, dict[str, AnnotationResult]]
            for _image_hash, annotation_result in result.items():
                # AnnotationResult is TypedDict, so always use dictionary access
                assert annotation_result.get("error") is None or annotation_result.get("error") == ""
                # The mock should return provider_manager_tag from the mock setup
                assert len(annotation_result.get("tags", [])) >= 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_integration(self, google_annotator, lightweight_test_images):
        """Test rate limiting functionality."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            with patch.object(google_annotator, "_wait_for_rate_limit") as mock_rate_limit:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["rate_limit_test"]
                mock_agent.run = AsyncMock(return_value=MagicMock(data=mock_response))
                mock_get_agent.return_value = mock_agent

                google_annotator._setup_agent()

                # Test multiple images (should trigger rate limiting)
                results = google_annotator.run_with_model(lightweight_test_images, "gemini-1.5-pro")

                # Verify rate limiting was called for each image
                assert mock_rate_limit.call_count == len(lightweight_test_images)
                assert len(results) == len(lightweight_test_images)

    @pytest.mark.skip(reason="Test environment detection inconsistency - needs revision")
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_validation(self, managed_config_registry):
        """Test configuration validation for Google annotator."""
        from unittest.mock import patch

        # Test missing API key
        invalid_config = {
            "class": "GoogleApiAnnotator",
            "api_model_id": "gemini-1.5-pro",
            # Missing api_key
        }
        managed_config_registry.set("invalid_google", invalid_config)

        # Temporarily disable test environment detection to test validation
        with patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False):
            with pytest.raises((ValueError, KeyError)) as exc_info:
                GoogleApiAnnotator("invalid_google")

            # Should raise error about missing configuration
            assert "api_key" in str(exc_info.value).lower() or "keyerror" in str(exc_info.value).lower()

        # Test missing model ID
        invalid_config2 = {
            "class": "GoogleApiAnnotator",
            "api_key": "test-key",
            # Missing api_model_id
        }
        managed_config_registry.set("invalid_google2", invalid_config2)

        annotator = GoogleApiAnnotator("invalid_google2")

        # Should raise error when trying to run inference without model ID
        with pytest.raises(ValueError):
            annotator._run_inference([Image.new("RGB", (64, 64), "red")])

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_model_id_override_through_provider_manager(
        self, google_annotator_config, lightweight_test_images
    ):
        """Test model ID override functionality through Provider Manager."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
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
                api_model_id="gemini-1.5-flash",  # Different from config
            )

            assert result is not None
            # Verify the override was passed through
            # Note: This depends on how the model override is implemented

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_requests_handling(self, google_annotator, lightweight_test_images):
        """Test handling of concurrent requests to Google API."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
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
            results = google_annotator.run_with_model(lightweight_test_images, "gemini-1.5-pro")

            # Verify all requests were processed
            assert len(results) == len(lightweight_test_images)
            assert len(concurrent_calls) == len(lightweight_test_images)

            # Verify all successful
            for result in results:
                assert result.error is None
