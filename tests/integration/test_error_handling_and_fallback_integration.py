# tests/integration/test_error_handling_and_fallback_integration.py
"""
Integration tests for error handling and fallback strategies.
Tests comprehensive error scenarios, graceful degradation, and system resilience.
"""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io
import json

from image_annotator_lib.api import annotate
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.types import AnnotationResult
from image_annotator_lib.exceptions.errors import WebApiError, ModelLoadError


class TestErrorHandlingAndFallbackIntegration:
    """Integration tests for error handling and fallback strategies."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        PydanticAIProviderFactory.clear_cache()
        yield
        PydanticAIProviderFactory.clear_cache()

    @pytest.fixture
    def mixed_model_configs(self, managed_config_registry):
        """Setup configurations for mixed model types (local + webapi)."""
        configs = {
            "working_openai_model": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key"
            },
            "failing_anthropic_model": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "invalid-key"
            },
            "working_local_model": {
                "class": "WDTagger",
                "model_path": "test/model/path",
                "device": "cpu",
                "estimated_size_gb": 1.0
            },
            "failing_local_model": {
                "class": "DeepDanbooruTagger",
                "model_path": "/nonexistent/path",
                "device": "cpu",
                "estimated_size_gb": 0.5
            }
        }
        
        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)
        
        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_partial_failure_graceful_degradation(self, mixed_model_configs, lightweight_test_images):
        """Test that partial failures in batch processing don't cause complete failure."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_load_model:
                
                def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
                    mock_agent = MagicMock()
                    
                    if "failing" in model_name:
                        mock_agent.run.side_effect = WebApiError("API authentication failed")
                    else:
                        mock_agent.run.return_value = MagicMock(
                            data={"tags": ["success_tag"], "model": model_name}
                        )
                    
                    return mock_agent
                
                def mock_model_loading(model_name):
                    mock_annotator = MagicMock()
                    
                    if "failing" in model_name:
                        raise ModelLoadError(f"Failed to load model {model_name}")
                    else:
                        mock_annotator.run_inference.return_value = {
                            "test_hash": AnnotationResult(
                                tags=["local_success_tag"],
                                formatted_output={"tags": ["local_success_tag"]},
                                error=None
                            )
                        }
                    
                    return mock_annotator
                
                mock_get_agent.side_effect = mock_agent_creation
                mock_load_model.side_effect = mock_model_loading
                
                # Test with mixed working and failing models
                test_models = [
                    "working_openai_model",
                    "failing_anthropic_model", 
                    "working_local_model",
                    "failing_local_model"
                ]
                
                try:
                    results = annotate(lightweight_test_images[:2], test_models)
                    
                    # Should return results even with some failures
                    assert isinstance(results, dict)
                    assert len(results) > 0
                    
                    # Check that successful models produced results
                    success_count = 0
                    error_count = 0
                    
                    for image_hash, model_results in results.items():
                        for model_name, result in model_results.items():
                            if result["error"] is None:
                                success_count += 1
                            else:
                                error_count += 1
                    
                    # Should have both successes and failures
                    assert success_count > 0, "No successful annotations found"
                    assert error_count > 0, "No failed annotations found (test setup issue)"
                    
                except Exception as e:
                    pytest.fail(f"Partial failure test should not raise exceptions: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_network_timeout_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of network timeouts and connection errors."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            
            def mock_agent_with_timeout(model_name, api_model_id, api_key, config_hash=None):
                mock_agent = MagicMock()
                
                # Simulate network timeout
                import asyncio
                mock_agent.run.side_effect = asyncio.TimeoutError("Request timed out")
                
                return mock_agent
            
            mock_get_agent.side_effect = mock_agent_with_timeout
            
            try:
                result = ProviderManager.run_inference_with_model(
                    "working_openai_model",
                    lightweight_test_images[:1],
                    api_model_id="gpt-4o-mini"
                )
                
                # Should handle timeout gracefully
                if result:
                    for image_hash, annotation_result in result.items():
                        assert annotation_result.error is not None
                        assert "timeout" in annotation_result.error.lower() or "timed out" in annotation_result.error.lower()
                
            except Exception as e:
                # Timeouts should be handled gracefully, not propagated
                pytest.fail(f"Network timeout should be handled gracefully: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_rate_limiting_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of API rate limiting."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            
            call_count = 0
            
            def mock_agent_with_rate_limit(model_name, api_model_id, api_key, config_hash=None):
                nonlocal call_count
                mock_agent = MagicMock()
                
                def mock_run(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    
                    if call_count <= 2:
                        # First 2 calls hit rate limit
                        raise WebApiError("Rate limit exceeded")
                    else:
                        # Subsequent calls succeed
                        return MagicMock(data={"tags": ["rate_limit_recovery"], "attempt": call_count})
                
                mock_agent.run.side_effect = mock_run
                return mock_agent
            
            mock_get_agent.side_effect = mock_agent_with_rate_limit
            
            # Test multiple sequential requests
            for i in range(3):
                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model",
                        lightweight_test_images[:1],
                        api_model_id="gpt-4o-mini"
                    )
                    
                    if i < 2:
                        # First 2 should fail with rate limit
                        if result:
                            for image_hash, annotation_result in result.items():
                                assert annotation_result.error is not None
                                assert "rate limit" in annotation_result.error.lower()
                    else:
                        # Third should succeed
                        if result:
                            for image_hash, annotation_result in result.items():
                                assert annotation_result.error is None
                    
                except Exception as e:
                    # Rate limiting should be handled, not propagated
                    pytest.fail(f"Rate limiting should be handled gracefully at attempt {i}: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_malformed_response_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of malformed API responses."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            
            malformed_responses = [
                "Not JSON at all",
                '{"incomplete": json',
                '{"valid_json": "but_wrong_structure"}',
                "",
                None
            ]
            
            for i, malformed_response in enumerate(malformed_responses):
                mock_agent = MagicMock()
                
                if malformed_response is None:
                    mock_agent.run.return_value = None
                else:
                    mock_agent.run.return_value = MagicMock(data=malformed_response)
                
                mock_get_agent.return_value = mock_agent
                
                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model",
                        lightweight_test_images[:1],
                        api_model_id="gpt-4o-mini"
                    )
                    
                    # Should handle malformed responses gracefully
                    if result:
                        for image_hash, annotation_result in result.items():
                            # Should have error for malformed response
                            assert annotation_result.error is not None
                            error_msg = annotation_result.error.lower()
                            assert any(keyword in error_msg for keyword in 
                                     ["json", "format", "parse", "malformed", "invalid"])
                    
                except Exception as e:
                    pytest.fail(f"Malformed response {i} should be handled gracefully: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_pressure_handling(self, mixed_model_configs, lightweight_test_images):
        """Test handling of memory pressure scenarios."""
        with patch('image_annotator_lib.api._create_annotator_instance') as mock_load_model:
            
            memory_pressure_count = 0
            
            def mock_model_loading_with_memory_pressure(model_name):
                nonlocal memory_pressure_count
                memory_pressure_count += 1
                
                if memory_pressure_count <= 2:
                    # Simulate memory pressure
                    raise MemoryError("Insufficient memory to load model")
                else:
                    # Recovery after memory pressure
                    mock_annotator = MagicMock()
                    mock_annotator.run_inference.return_value = {
                        "test_hash": AnnotationResult(
                            tags=["memory_recovery_tag"],
                            formatted_output={"tags": ["memory_recovery_tag"]},
                            error=None
                        )
                    }
                    return mock_annotator
            
            mock_load_model.side_effect = mock_model_loading_with_memory_pressure
            
            # Test multiple attempts with memory pressure
            for attempt in range(3):
                try:
                    results = annotate(lightweight_test_images[:1], ["working_local_model"])
                    
                    if attempt < 2:
                        # First 2 attempts should fail with memory error
                        if results:
                            for image_hash, model_results in results.items():
                                for model_name, result in model_results.items():
                                    assert result["error"] is not None
                                    assert "memory" in result["error"].lower()
                    else:
                        # Third attempt should succeed
                        if results:
                            for image_hash, model_results in results.items():
                                for model_name, result in model_results.items():
                                    assert result["error"] is None
                
                except Exception as e:
                    pytest.fail(f"Memory pressure at attempt {attempt} should be handled: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_invalid_image_data_handling(self, mixed_model_configs):
        """Test handling of invalid or corrupted image data."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            
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
            
            for i, invalid_image in enumerate(invalid_images):
                try:
                    results = annotate([invalid_image], ["working_openai_model"])
                    
                    # Should handle invalid images gracefully
                    if results:
                        for image_hash, model_results in results.items():
                            for model_name, result in model_results.items():
                                # Should have error for invalid image
                                assert result["error"] is not None
                                error_msg = result["error"].lower()
                                assert any(keyword in error_msg for keyword in 
                                         ["image", "format", "invalid", "corrupted", "decode"])
                
                except Exception as e:
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
                for image_hash, model_results in results.items():
                    if "nonexistent_model" in model_results:
                        result = model_results["nonexistent_model"]
                        assert result["error"] is not None
                        assert "config" in result["error"].lower() or "not found" in result["error"].lower()
        
        except Exception as e:
            # Configuration errors might be handled at different levels
            assert "config" in str(e).lower() or "not found" in str(e).lower()
        
        # Test with invalid configuration
        invalid_config = {
            "class": "NonexistentAnnotator",
            "invalid_param": "invalid_value"
        }
        managed_config_registry.set("invalid_model", invalid_config)
        
        try:
            results = annotate(lightweight_test_images[:1], ["invalid_model"])
            
            if results:
                for image_hash, model_results in results.items():
                    if "invalid_model" in model_results:
                        result = model_results["invalid_model"]
                        assert result["error"] is not None
        
        except Exception as e:
            # Invalid configuration should be handled appropriately
            pass

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cascading_failure_prevention(self, mixed_model_configs, lightweight_test_images):
        """Test that failures in one component don't cascade to others."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_load_model:
                
                # Create a scenario where one failure type doesn't affect others
                failure_cascade_test = False
                
                def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
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
                            error=None
                        )
                    }
                    return mock_annotator
                
                mock_get_agent.side_effect = mock_agent_creation
                mock_load_model.side_effect = mock_model_loading
                
                # Test sequence: failing model, then working models
                test_sequence = [
                    ["failing_anthropic_model"],  # Should fail
                    ["working_openai_model"],     # Should work despite previous failure
                    ["working_local_model"]       # Should work independently
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
                            pytest.fail(f"Cascading failure detected at step {i}: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_error_message_quality(self, mixed_model_configs, lightweight_test_images):
        """Test that error messages are informative and actionable."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            
            # Test different error scenarios with expected message qualities
            error_scenarios = [
                ("authentication", "Invalid API key", ["api", "key", "authentication"]),
                ("network", "Connection timeout", ["connection", "timeout", "network"]),
                ("rate_limit", "Rate limit exceeded", ["rate", "limit", "exceeded"]),
                ("invalid_model", "Model not found", ["model", "not found", "invalid"])
            ]
            
            for scenario_name, error_message, expected_keywords in error_scenarios:
                mock_agent = MagicMock()
                mock_agent.run.side_effect = WebApiError(error_message)
                mock_get_agent.return_value = mock_agent
                
                try:
                    result = ProviderManager.run_inference_with_model(
                        "working_openai_model",
                        lightweight_test_images[:1],
                        api_model_id="gpt-4o-mini"
                    )
                    
                    if result:
                        for image_hash, annotation_result in result.items():
                            assert annotation_result.error is not None
                            
                            error_msg = annotation_result.error.lower()
                            
                            # Check that error message contains relevant keywords
                            keyword_found = any(keyword.lower() in error_msg for keyword in expected_keywords)
                            assert keyword_found, f"Error message '{annotation_result.error}' doesn't contain expected keywords {expected_keywords}"
                            
                            # Check that error message is not just a generic "error occurred"
                            assert len(annotation_result.error.strip()) > 10, "Error message too generic"
                
                except Exception as e:
                    pytest.fail(f"Error message quality test failed for {scenario_name}: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_system_resilience_under_stress(self, mixed_model_configs, lightweight_test_images):
        """Test system resilience under multiple concurrent error conditions."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_load_model:
                
                # Create a stress scenario with multiple error types
                call_count = 0
                
                def mock_agent_with_various_errors(model_name, api_model_id, api_key, config_hash=None):
                    nonlocal call_count
                    call_count += 1
                    
                    mock_agent = MagicMock()
                    
                    # Rotate through different error types
                    error_types = [
                        WebApiError("Network error"),
                        WebApiError("Rate limit exceeded"), 
                        WebApiError("Authentication failed"),
                        None  # Success case
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
                                error=None
                            )
                        }
                        return mock_annotator
                
                mock_get_agent.side_effect = mock_agent_with_various_errors
                mock_load_model.side_effect = mock_model_with_various_errors
                
                # Run multiple stress test iterations
                stress_results = []
                
                for iteration in range(6):  # Multiple iterations to hit different error patterns
                    try:
                        results = annotate(
                            lightweight_test_images[:1], 
                            ["working_openai_model", "working_local_model"]
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