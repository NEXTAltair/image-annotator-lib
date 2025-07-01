# tests/integration/test_memory_management_integration.py
"""
Integration tests for memory management scenarios.
Tests ModelLoad caching, Provider-level resource sharing, and memory pressure handling.
"""

import gc
import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel

from image_annotator_lib.api import annotate
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.types import AnnotationResult


class TestMemoryManagementIntegration:
    """Integration tests for memory management scenarios."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear all caches before each test
        PydanticAIProviderFactory.clear_cache()
        
        # Disable real API requests for PydanticAI models
        models.ALLOW_MODEL_REQUESTS = False

        # Force garbage collection
        gc.collect()

        yield

        # Clean up after each test
        PydanticAIProviderFactory.clear_cache()
        gc.collect()

    @pytest.fixture
    def memory_test_configs(self, managed_config_registry):
        """Setup configurations for memory management testing."""
        from tests.integration.conftest import _ensure_test_class_mapping
        
        configs = {
            "memory_openai_1": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key-1",
            },
            "memory_openai_2": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-3.5-turbo",
                "api_key": "test-openai-key-1",  # Same key as above
            },
            "memory_anthropic": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "test-anthropic-key",
            },
            "memory_local_large": {
                "class": "WDTagger",
                "model_path": "test/large/model",
                "base_model": "wd-v1-4-tags",  # Add required base_model
                "device": "cpu",
                "estimated_size_gb": 2.5,
            },
            "memory_local_small": {
                "class": "ImprovedAesthetic",
                "model_path": "test/small/model",
                "base_model": "improved-aesthetic",  # Add required base_model
                "device": "cpu",
                "estimated_size_gb": 0.5,
            },
        }

        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)
            # Ensure class mapping exists
            _ensure_test_class_mapping(model_name, config)

        return configs

    @pytest.fixture
    def mock_test_model(self):
        """Create a TestModel that returns consistent mock responses."""
        test_model = TestModel()
        # Set a realistic annotation response
        test_model.response = ModelResponse(
            parts=[TextPart('{"tags": ["realistic_tag1", "realistic_tag2"],"formatted_output": {"tags": ["realistic_tag1", "realistic_tag2"]}}')]
        )
        return test_model

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing_memory_efficiency(
        self, memory_test_configs, lightweight_test_images
    ):
        """Test that Provider instances are shared efficiently to save memory."""
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
        from pydantic_ai.messages import ModelResponse, TextPart
        import json

        # Test multiple models that should share providers
        models_to_test = ["memory_openai_1", "memory_openai_2"]  # Same API key

        # Mock the Agent's run method to return a valid ModelResponse
        with patch("pydantic_ai.Agent.run") as mock_run:
            # The model is expected to return a JSON string inside a TextPart
            annotation_data = {"tags": ["mocked_tag1", "mocked_tag2"], "formatted_output": "mocked_output"}
            response_json = json.dumps(annotation_data)
            
            # The `run` method returns a ModelResponse object
            mock_run.return_value = ModelResponse(parts=[TextPart(content=response_json)])
            
            # Use standard annotation function
            results = annotate(images_list=lightweight_test_images[:1], model_name_list=models_to_test)

            # Verify results are successful
            assert len(results) == 1
            image_results = list(results.values())[0]

            for model_name in models_to_test:
                assert model_name in image_results
                assert image_results[model_name]["error"] is None, f"Model {model_name} failed unexpectedly."
                
            # Verify that provider was shared (only one provider instance for the same API key)
            assert len(PydanticAIProviderFactory._providers) == 1

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_local_model_memory_management(self, memory_test_configs, lightweight_test_images):
        """Test memory management for local ML models."""
        from image_annotator_lib.core.utils import calculate_phash

        with patch("image_annotator_lib.api._create_annotator_instance") as mock_create_annotator:
            # Track model loading and memory usage
            loaded_models = {}
            memory_usage = []

            def mock_annotator_creation(model_name, **kwargs):
                if model_name not in loaded_models:
                    # Simulate memory allocation
                    config = memory_test_configs[model_name]
                    memory_size = config.get("estimated_size_gb", 1.0)
                    memory_usage.append(memory_size)

                    mock_annotator = MagicMock()
                    mock_annotator.model_name = model_name
                    mock_annotator.memory_size = memory_size
                    
                    # Mock the context manager methods
                    mock_annotator.__enter__.return_value = mock_annotator
                    mock_annotator.__exit__.return_value = None

                    # Mock the predict method to return a valid result
                    image = lightweight_test_images[0]
                    phash = calculate_phash(image)
                    mock_annotator.predict.return_value = [
                        AnnotationResult(
                            phash=phash,
                            tags=[f"local_tag_{model_name}"],
                            formatted_output={"tags": [f"local_tag_{model_name}"]},
                            error=None,
                        )
                    ]
                    loaded_models[model_name] = mock_annotator
                
                return loaded_models[model_name]

            mock_create_annotator.side_effect = mock_annotator_creation

            # Test sequential loading of different sized models
            sequential_models = ["memory_local_small", "memory_local_large"]

            for model_name in sequential_models:
                results = annotate(images_list=lightweight_test_images[:1], model_name_list=[model_name])

                # Verify model was loaded and used
                assert len(results) == 1
                image_results = list(results.values())[0]
                assert model_name in image_results
                assert image_results[model_name]["error"] is None

            # Verify memory tracking
            assert len(loaded_models) == 2
            assert sum(memory_usage) == 3.0  # 0.5 + 2.5 GB

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_lru_cache_memory_management(self, memory_test_configs, lightweight_test_images):
        """Test LRU cache behavior for memory management."""
        from image_annotator_lib.core.webapi_agent_cache import WebApiAgentCache
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
        from pydantic_ai.messages import ModelResponse, TextPart
        import json

        with patch("pydantic_ai.Agent.run") as mock_run:
            annotation_data = {"tags": ["lru_tag"], "formatted_output": "lru_output"}
            response_json = json.dumps(annotation_data)
            mock_run.return_value = ModelResponse(parts=[TextPart(content=response_json)])
            
            # We need to clear caches to properly test LRU
            PydanticAIProviderFactory.clear_cache()
            WebApiAgentCache.clear_cache()
            
            # Configure cache size for the test
            original_max_size = WebApiAgentCache._MAX_CACHE_SIZE
            WebApiAgentCache.set_max_cache_size(2)

            try:
                lru_sequence = [
                    "memory_openai_1",    # -> cache: [oa1]
                    "memory_anthropic",   # -> cache: [oa1, an]
                    "memory_openai_2",    # -> cache: [an, oa2] (oa1 evicted)
                    "memory_openai_1",    # -> cache: [oa2, oa1] (an evicted)
                ]

                for model_name in lru_sequence:
                    annotate(images_list=lightweight_test_images[:1], model_name_list=[model_name])

                cache_info = WebApiAgentCache.get_cache_info()
                assert cache_info["cache_size"] <= 2
                
                # Verify the correct items are in the cache
                cached_keys = cache_info["cached_agents"]
                assert any("memory_openai_2" in key for key in cached_keys)
                assert any("memory_openai_1" in key for key in cached_keys)
                assert not any("memory_anthropic" in key for key in cached_keys)

            finally:
                # Reset MAX_SIZE to avoid affecting other tests
                WebApiAgentCache.set_max_cache_size(original_max_size)
                PydanticAIProviderFactory.clear_cache()
                WebApiAgentCache.clear_cache()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_pressure_handling(self, memory_test_configs, lightweight_test_images):
        """Test handling of memory pressure scenarios."""
        # This test checks error handling in the `annotate` function when a model fails to load.
        # We mock `_create_annotator_instance` to simulate this failure.
        with patch("image_annotator_lib.api._create_annotator_instance", side_effect=MemoryError("Test MemoryError")):
            results = annotate(
                images_list=lightweight_test_images[:1], model_name_list=["memory_local_large"]
            )

            # Verify that the error was caught and reported in the results
            assert len(results) == 1
            image_results = list(results.values())[0]
            assert "memory_local_large" in image_results
            error = image_results["memory_local_large"]["error"]
            assert error is not None
            assert "MemoryError" in error
            assert "Test MemoryError" in error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_memory_usage(self, memory_test_configs, lightweight_test_images):
        """Test memory usage during concurrent operations."""
        from image_annotator_lib.core.utils import calculate_phash
        from pydantic_ai.messages import ModelResponse, TextPart
        import json

        # Mock local model creation and WebAPI's Agent.run method
        with patch("image_annotator_lib.api._create_annotator_instance") as mock_create_instance, \
             patch("pydantic_ai.Agent.run") as mock_run:

            # This mock will handle creation for both local and webapi models
            def mock_creation_logic(model_name):
                if "local" in model_name:
                    mock_annotator = MagicMock()
                    mock_annotator.model_name = model_name
                    mock_annotator.__enter__.return_value = mock_annotator
                    mock_annotator.__exit__.return_value = None
                    # Predict must return a list of AnnotationResult for each image
                    results = [AnnotationResult(phash=calculate_phash(img), tags=[f"local_{model_name}"]) for img in lightweight_test_images[:2]]
                    mock_annotator.predict.return_value = results
                    return mock_annotator
                else:
                    # For WebAPI models, we let the original PydanticAIWebAPIWrapper be created
                    from image_annotator_lib.api import PydanticAIWebAPIWrapper
                    # The class passed to the wrapper is not used as its `run` method is mocked
                    return PydanticAIWebAPIWrapper(model_name, MagicMock())

            mock_create_instance.side_effect = mock_creation_logic
            
            # Setup mock for WebAPI models' Agent.run method
            web_annotation = {"tags": ["web_api_tag"], "formatted_output": "web_output"}
            mock_run.return_value = ModelResponse(parts=[TextPart(content=json.dumps(web_annotation))])
            
            # Test concurrent usage with mixed model types
            mixed_models = [
                "memory_openai_1",
                "memory_local_small",
                "memory_anthropic",
                "memory_local_large",
            ]

            results = annotate(
                images_list=lightweight_test_images[:2],  # Multiple images
                model_name_list=mixed_models,
            )

            # Verify that creator was called for all models
            assert mock_create_instance.call_count == len(mixed_models)

            # Verify results for all images and models
            assert len(results) == 2
            for image_hash, model_results in results.items():
                assert len(model_results) == len(mixed_models)
                for model_name in mixed_models:
                    assert model_name in model_results
                    assert model_results[model_name]["error"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_invalidation_memory_cleanup(self, memory_test_configs, lightweight_test_images):
        """Test that cache invalidation properly cleans up memory."""
        from image_annotator_lib.core.webapi_agent_cache import WebApiAgentCache
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
        from pydantic_ai.messages import ModelResponse, TextPart
        import json

        with patch("pydantic_ai.Agent.run") as mock_run:
            annotation_data = {"tags": ["cache_tag"], "formatted_output": "cache_output"}
            mock_run.return_value = ModelResponse(parts=[TextPart(content=json.dumps(annotation_data))])
            
            # Build up cache
            models_to_cache = ["memory_openai_1", "memory_anthropic"]

            for model_name in models_to_cache:
                annotate(images_list=lightweight_test_images[:1], model_name_list=[model_name])

            # Verify cache is populated
            assert WebApiAgentCache.get_cache_info()["cache_size"] == 2
            assert len(PydanticAIProviderFactory._providers) > 0

            # Trigger cache clear
            PydanticAIProviderFactory.clear_cache()

            # Verify cache was cleared
            assert WebApiAgentCache.get_cache_info()["cache_size"] == 0
            assert len(PydanticAIProviderFactory._providers) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_mixed_workload_memory_efficiency(self, memory_test_configs, lightweight_test_images):
        """Test memory efficiency under mixed workload conditions."""
        from image_annotator_lib.core.utils import calculate_phash
        from pydantic_ai.messages import ModelResponse, TextPart
        import json

        with patch("image_annotator_lib.api._create_annotator_instance") as mock_create_instance, \
             patch("pydantic_ai.Agent.run") as mock_run:

            def mock_creation_logic(model_name):
                if "local" in model_name:
                    mock_annotator = MagicMock()
                    mock_annotator.model_name = model_name
                    mock_annotator.__enter__.return_value = mock_annotator
                    mock_annotator.__exit__.return_value = None
                    mock_annotator.predict.return_value = [AnnotationResult(phash=calculate_phash(lightweight_test_images[0]), tags=[f"local_{model_name}"])]
                    return mock_annotator
                else:
                    from image_annotator_lib.api import PydanticAIWebAPIWrapper
                    return PydanticAIWebAPIWrapper(model_name, MagicMock())

            mock_create_instance.side_effect = mock_creation_logic
            
            web_annotation = {"tags": ["web_api_tag"], "formatted_output": "web_output"}
            mock_run.return_value = ModelResponse(parts=[TextPart(content=json.dumps(web_annotation))])

            workload_sequence = [
                ["memory_openai_1"],
                ["memory_local_small"],
                ["memory_anthropic"],
                ["memory_local_large"],
                ["memory_openai_2"],
            ]

            for model_batch in workload_sequence:
                results = annotate(images_list=lightweight_test_images[:1], model_name_list=model_batch)
                assert len(results) == 1
                image_results = list(results.values())[0]
                for model_name in model_batch:
                    assert model_name in image_results
                    assert image_results[model_name]["error"] is None
        
            assert mock_create_instance.call_count == len(workload_sequence)
            assert mock_run.call_count == 3

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_monitoring_integration(self, memory_test_configs, lightweight_test_images):
        """Test integration with memory monitoring systems."""
        with patch("image_annotator_lib.api._create_annotator_instance") as mock_create_annotator:
            # Mock memory monitoring
            memory_snapshots = []

            def mock_model_loading_with_monitoring(model_name, **kwargs):
                # Simulate memory monitoring
                config = memory_test_configs[model_name]
                estimated_size = config.get("estimated_size_gb", 1.0)

                memory_snapshots.append(
                    {"model": model_name, "estimated_size_gb": estimated_size, "timestamp": time.time()}
                )

                mock_annotator = MagicMock()
                mock_annotator.__enter__.return_value = mock_annotator
                mock_annotator.__exit__.return_value = None
                mock_annotator.predict.return_value = [
                    AnnotationResult(
                        phash="test_hash",
                        tags=[f"monitored_{model_name}"],
                        formatted_output={"tags": [f"monitored_{model_name}"]},
                        error=None,
                    )
                ]
                return mock_annotator

            mock_create_annotator.side_effect = mock_model_loading_with_monitoring

            # Test memory monitoring during model operations
            models_to_monitor = ["memory_local_small", "memory_local_large"]

            for model_name in models_to_monitor:
                annotate(images_list=lightweight_test_images[:1], model_name_list=[model_name])

            # Verify memory monitoring data
            assert len(memory_snapshots) == len(models_to_monitor)

            total_estimated_memory = sum(snap["estimated_size_gb"] for snap in memory_snapshots)
            assert total_estimated_memory == 3.0  # 0.5 + 2.5 GB

            # Verify chronological order
            timestamps = [snap["timestamp"] for snap in memory_snapshots]
            assert timestamps == sorted(timestamps)
