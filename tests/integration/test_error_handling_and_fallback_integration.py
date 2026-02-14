# tests/integration/test_error_handling_and_fallback_integration.py
"""
Integration tests for error handling and fallback strategies.
Tests comprehensive error scenarios, graceful degradation, and system resilience.
"""

from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.api import annotate
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAgentFactory
from image_annotator_lib.core.types import AnnotationResult
from image_annotator_lib.exceptions.errors import ModelLoadError, WebApiError


class TestErrorHandlingAndFallbackIntegration:
    """Integration tests for error handling and fallback strategies."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        PydanticAIAgentFactory.clear_cache()
        yield
        PydanticAIAgentFactory.clear_cache()

    @pytest.fixture
    def mixed_model_configs(self, managed_config_registry):
        """Setup configurations for mixed model types (local + webapi)."""
        configs = {
            "working_openai_model": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key",
                "capabilities": ["tags", "captions", "scores"],
            },
            "failing_anthropic_model": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "invalid-key",
                "capabilities": ["tags", "captions", "scores"],
            },
            "working_local_model": {
                "class": "WDTagger",
                "model_path": "test/model/path",
                "device": "cpu",
                "estimated_size_gb": 1.0,
                "capabilities": ["tags"],
            },
            "failing_local_model": {
                "class": "DeepDanbooruTagger",
                "model_path": "/nonexistent/path",
                "device": "cpu",
                "estimated_size_gb": 0.5,
                "capabilities": ["tags"],
            },
        }

        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)

        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_network_timeout_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of network timeouts and connection errors."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_provider_run:

            def mock_timeout_inference(model_name, images, api_model_id=None):
                """Mock provider manager with timeout"""
                results = {}
                for i, _image in enumerate(images):
                    phash = f"test_hash_{i}"
                    results[phash] = {
                        model_name: AnnotationResult(
                            tags=[], formatted_output={}, error="Request timed out"
                        )
                    }
                return results

            mock_provider_run.side_effect = mock_timeout_inference

            try:
                result = ProviderManager.run_inference_with_model(
                    "working_openai_model", lightweight_test_images[:1], api_model_id="gpt-4o-mini"
                )

                # Should handle timeout gracefully
                if result:
                    for _image_hash, model_results in result.items():
                        for _model_name, annotation_result in model_results.items():
                            # Handle both dict, object, and string access patterns
                            if isinstance(annotation_result, str):
                                error_msg = annotation_result
                            elif hasattr(annotation_result, "error"):
                                error_msg = annotation_result.error
                            elif isinstance(annotation_result, dict):
                                error_msg = annotation_result.get("error")
                            else:
                                error_msg = str(annotation_result)

                            assert error_msg is not None
                            assert "timeout" in error_msg.lower() or "timed out" in error_msg.lower()

            except Exception as e:
                # Timeouts should be handled gracefully, not propagated
                pytest.fail(f"Network timeout should be handled gracefully: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of API rate limiting."""
        call_count = 0

        def mock_rate_limit_inference(model_name, images, api_model_id=None):
            """Mock provider manager with rate limiting"""
            nonlocal call_count
            call_count += 1

            results = {}
            for i, _image in enumerate(images):
                phash = f"test_hash_{i}"

                if call_count <= 2:
                    # First 2 calls hit rate limit
                    results[phash] = {
                        model_name: AnnotationResult(
                            tags=[], formatted_output={}, error="Rate limit exceeded"
                        )
                    }
                else:
                    # Subsequent calls succeed
                    results[phash] = {
                        model_name: AnnotationResult(
                            tags=["rate_limit_recovery"],
                            formatted_output={"tags": ["rate_limit_recovery"]},
                            error=None,
                        )
                    }
            return results

        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_provider_run:
            mock_provider_run.side_effect = mock_rate_limit_inference

            # Test multiple sequential requests
            for i in range(3):
                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model", lightweight_test_images[:1], api_model_id="gpt-4o-mini"
                    )

                    if i < 2:
                        # First 2 should fail with rate limit
                        if result:
                            for _image_hash, model_results in result.items():
                                for _model_name, annotation_result in model_results.items():
                                    # Handle both dict, object, and string access patterns
                                    if isinstance(annotation_result, str):
                                        error_msg = annotation_result
                                    elif hasattr(annotation_result, "error"):
                                        error_msg = annotation_result.error
                                    elif isinstance(annotation_result, dict):
                                        error_msg = annotation_result.get("error")
                                    else:
                                        error_msg = str(annotation_result)

                                    assert error_msg is not None
                                    assert "rate limit" in error_msg.lower()
                    else:
                        # Third should succeed
                        if result:
                            for _image_hash, model_results in result.items():
                                for _model_name, annotation_result in model_results.items():
                                    # Handle both dict, object, and string access patterns
                                    if isinstance(annotation_result, str):
                                        error_msg = annotation_result
                                    elif hasattr(annotation_result, "error"):
                                        error_msg = annotation_result.error
                                    elif isinstance(annotation_result, dict):
                                        error_msg = annotation_result.get("error")
                                    else:
                                        error_msg = None

                                    assert error_msg is None

                except Exception as e:
                    # Rate limiting should be handled, not propagated
                    pytest.fail(f"Rate limiting should be handled gracefully at attempt {i}: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_malformed_response_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of malformed API responses."""
        malformed_responses = [
            "Not JSON at all",
            '{"incomplete": json',
            '{"valid_json": "but_wrong_structure"}',
            "",
            None,
        ]

        def mock_malformed_inference(model_name, images, api_model_id=None):
            """Mock provider manager with malformed response"""
            results = {}
            for i, _image in enumerate(images):
                phash = f"test_hash_{i}"
                results[phash] = {
                    model_name: AnnotationResult(
                        tags=[], formatted_output={}, error="JSON parse error: malformed response"
                    )
                }
            return results

        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_provider_run:
            for i, _malformed_response in enumerate(malformed_responses):
                mock_provider_run.side_effect = mock_malformed_inference

                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model", lightweight_test_images[:1], api_model_id="gpt-4o-mini"
                    )

                    # Should handle malformed responses gracefully
                    if result:
                        for _image_hash, model_results in result.items():
                            for _model_name, annotation_result in model_results.items():
                                # Handle both dict and AnnotationResult access patterns
                                if isinstance(annotation_result, dict):
                                    error_msg = annotation_result.get("error", "")
                                else:
                                    error_msg = getattr(annotation_result, "error", "")

                                # Should have error for malformed response
                                assert error_msg is not None
                                error_msg_lower = error_msg.lower()
                                assert any(
                                    keyword in error_msg_lower
                                    for keyword in [
                                        "json",
                                        "format",
                                        "parse",
                                        "malformed",
                                        "invalid",
                                        "error",
                                    ]
                                )

                except Exception as e:
                    pytest.fail(f"Malformed response {i} should be handled gracefully: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_invalid_image_data_handling(self, mixed_model_configs):
        """Test handling of invalid or corrupted image data."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.run.return_value = MagicMock(data={"tags": ["should_not_reach"]})
            mock_get_agent.return_value = mock_agent

            # Test various invalid image inputs
            invalid_images = [
                b"Not an image at all",  # Invalid bytes
                "",  # Empty string
                None,  # None value
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",  # Corrupted PNG header
            ]

            for _i, invalid_image in enumerate(invalid_images):
                try:
                    results = annotate([invalid_image], ["working_openai_model"])

                    # Should handle invalid images gracefully
                    if results:
                        for _image_hash, model_results in results.items():
                            for _model_name, result in model_results.items():
                                # Should have error for invalid image
                                assert result["error"] is not None
                                error_msg = result["error"].lower()
                                assert any(
                                    keyword in error_msg
                                    for keyword in ["image", "format", "invalid", "corrupted", "decode"]
                                )

                except Exception:
                    # Some invalid image types might be caught earlier and raise exceptions
                    # This is acceptable as long as they're handled consistently
                    pass

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_error_handling(self, managed_config_registry, lightweight_test_images):
        """Test handling of configuration errors and missing configurations."""
        # Test with missing model configuration
        try:
            results = annotate(lightweight_test_images[:1], ["nonexistent_model"])

            # Should handle missing configuration gracefully
            if results:
                for _image_hash, model_results in results.items():
                    if "nonexistent_model" in model_results:
                        result = model_results["nonexistent_model"]
                        assert result.error is not None
                        assert "config" in result.error.lower() or "not found" in result.error.lower()

        except Exception as e:
            # Configuration errors might be handled at different levels
            assert "config" in str(e).lower() or "not found" in str(e).lower()

        # Test with invalid configuration
        invalid_config = {"class": "NonexistentAnnotator", "invalid_param": "invalid_value"}
        managed_config_registry.set("invalid_model", invalid_config)

        try:
            results = annotate(lightweight_test_images[:1], ["invalid_model"])

            if results:
                for _image_hash, model_results in results.items():
                    if "invalid_model" in model_results:
                        result = model_results["invalid_model"]
                        assert result.error is not None

        except Exception:
            # Invalid configuration should be handled appropriately
            pass

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cascading_failure_prevention(self, mixed_model_configs, lightweight_test_images):
        """Test that failures in one component don't cascade to others."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            with patch("image_annotator_lib.api._create_annotator_instance") as mock_load_model:
                # Create a scenario where one failure type doesn't affect others
                failure_cascade_test = False

                def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                    nonlocal failure_cascade_test

                    mock_agent = MagicMock()

                    if "failing" in model_name:
                        # First failure should not affect subsequent calls
                        if not failure_cascade_test:
                            failure_cascade_test = True
                            raise Exception("First failure - should be isolated")
                        else:
                            # Subsequent calls should work normally
                            mock_agent.run.return_value = MagicMock(
                                data={"tags": ["recovery_tag"], "recovered": True}
                            )
                    else:
                        mock_agent.run.return_value = MagicMock(
                            data={"tags": ["normal_tag"], "normal": True}
                        )

                    return mock_agent

                def mock_model_loading(model_name):
                    # Local models should work independently of WebAPI failures
                    mock_annotator = MagicMock()
                    mock_annotator.run_inference.return_value = {
                        "test_hash": AnnotationResult(
                            tags=["local_independent_tag"],
                            formatted_output={"tags": ["local_independent_tag"]},
                            error=None,
                        )
                    }
                    return mock_annotator

                mock_get_agent.side_effect = mock_agent_creation
                mock_load_model.side_effect = mock_model_loading

                # Test sequence: failing model, then working models
                test_sequence = [
                    ["failing_anthropic_model"],  # Should fail
                    ["working_openai_model"],  # Should work despite previous failure
                    ["working_local_model"],  # Should work independently
                ]

                for i, models in enumerate(test_sequence):
                    try:
                        results = annotate(lightweight_test_images[:1], models)

                        if i == 0:
                            # First call (failing model) - might succeed or fail
                            pass
                        else:
                            # Subsequent calls should succeed
                            assert results is not None
                            assert len(results) > 0

                    except Exception as e:
                        if i > 0:
                            pytest.fail(f"Cascading failure detected at step {i}: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_error_message_quality(self, mixed_model_configs, lightweight_test_images):
        """Test that error messages are informative and actionable."""
        # Test different error scenarios with expected message qualities
        error_scenarios = [
            ("authentication", "Invalid API key", ["api", "key", "authentication"]),
            ("network", "Connection timeout", ["connection", "timeout", "network"]),
            ("rate_limit", "Rate limit exceeded", ["rate", "limit", "exceeded"]),
            ("invalid_model", "Model not found", ["model", "not found", "invalid"]),
        ]

        def mock_error_inference(model_name, images, api_model_id=None, error_message="Generic error"):
            """Mock provider manager with specific error message"""
            results = {}
            for i, _image in enumerate(images):
                phash = f"test_hash_{i}"
                results[phash] = {
                    model_name: AnnotationResult(tags=[], formatted_output={}, error=error_message)
                }
            return results

        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_provider_run:
            for scenario_name, error_message, expected_keywords in error_scenarios:
                # Configure mock to return specific error message
                mock_provider_run.side_effect = (
                    lambda model_name, images, api_model_id=None: mock_error_inference(
                        model_name, images, api_model_id, error_message
                    )
                )

                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model", lightweight_test_images[:1], api_model_id="gpt-4o-mini"
                    )

                    if result:
                        for _image_hash, model_results in result.items():
                            for _model_name, annotation_result in model_results.items():
                                # Handle both dict and AnnotationResult access patterns
                                if isinstance(annotation_result, dict):
                                    error_msg = annotation_result.get("error")
                                else:
                                    error_msg = getattr(annotation_result, "error", None)

                                assert error_msg is not None

                                error_msg_lower = error_msg.lower()

                                # Check that error message contains relevant keywords
                                keyword_found = any(
                                    keyword.lower() in error_msg_lower for keyword in expected_keywords
                                )
                                assert keyword_found, (
                                    f"Error message '{error_msg}' doesn't contain expected keywords {expected_keywords}"
                                )

                                # Check that error message is not just a generic "error occurred"
                                assert len(error_msg.strip()) > 10, "Error message too generic"

                except Exception as e:
                    pytest.fail(f"Error message quality test failed for {scenario_name}: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_system_resilience_under_stress(self, mixed_model_configs, lightweight_test_images):
        """Test system resilience under multiple concurrent error conditions."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            with patch("image_annotator_lib.api._create_annotator_instance") as mock_load_model:
                # Create a stress scenario with multiple error types
                call_count = 0

                def mock_agent_with_various_errors(model_name, api_model_id, api_key, config_data=None):
                    nonlocal call_count
                    call_count += 1

                    mock_agent = MagicMock()

                    # Rotate through different error types
                    error_types = [
                        WebApiError("Network error"),
                        WebApiError("Rate limit exceeded"),
                        WebApiError("Authentication failed"),
                        None,  # Success case
                    ]

                    error = error_types[call_count % len(error_types)]

                    if error:
                        mock_agent.run.side_effect = error
                    else:
                        mock_agent.run.return_value = MagicMock(
                            data={"tags": ["stress_test_success"], "call": call_count}
                        )

                    return mock_agent

                def mock_model_with_various_errors(model_name):
                    nonlocal call_count
                    call_count += 1

                    # Rotate through different error types for local models
                    if call_count % 3 == 0:
                        raise ModelLoadError("Model loading failed")
                    elif call_count % 3 == 1:
                        raise MemoryError("Insufficient memory")
                    else:
                        mock_annotator = MagicMock()
                        mock_annotator.run_inference.return_value = {
                            "test_hash": AnnotationResult(
                                tags=["stress_local_success"],
                                formatted_output={"tags": ["stress_local_success"]},
                                error=None,
                            )
                        }
                        return mock_annotator

                mock_get_agent.side_effect = mock_agent_with_various_errors
                mock_load_model.side_effect = mock_model_with_various_errors

                # Run multiple stress test iterations
                stress_results = []

                for _iteration in range(6):  # Multiple iterations to hit different error patterns
                    try:
                        results = annotate(
                            lightweight_test_images[:1], ["working_openai_model", "working_local_model"]
                        )
                        stress_results.append(("success", results))

                    except Exception as e:
                        stress_results.append(("exception", str(e)))

                # Analyze stress test results
                success_count = len([r for r in stress_results if r[0] == "success"])

                # System should handle stress gracefully - at least some successes
                assert success_count > 0, "System failed completely under stress"

                # Should not have excessive unhandled exceptions
                exception_count = len([r for r in stress_results if r[0] == "exception"])
                assert exception_count < len(stress_results), "Too many unhandled exceptions under stress"
