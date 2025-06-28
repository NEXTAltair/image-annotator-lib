# tests/integration/test_memory_management_integration.py
"""
Integration tests for memory management scenarios.
Tests ModelLoad caching, Provider-level resource sharing, and memory pressure handling.
"""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import gc
import time

from image_annotator_lib.api import annotate
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.types import AnnotationResult


class TestMemoryManagementIntegration:
    """Integration tests for memory management scenarios."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear all caches before each test
        PydanticAIProviderFactory.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        yield
        
        # Clean up after each test
        PydanticAIProviderFactory.clear_cache()
        gc.collect()

    @pytest.fixture
    def memory_test_configs(self, managed_config_registry):
        """Setup configurations for memory management testing."""
        configs = {
            "memory_openai_1": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key-1"
            },
            "memory_openai_2": {
                "class": "OpenAIApiChatAnnotator", 
                "api_model_id": "gpt-3.5-turbo",
                "api_key": "test-openai-key-1"  # Same key as above
            },
            "memory_anthropic": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "test-anthropic-key"
            },
            "memory_local_large": {
                "class": "WDTagger",
                "model_path": "test/large/model",
                "device": "cpu",
                "estimated_size_gb": 2.5
            },
            "memory_local_small": {
                "class": "ImprovedAesthetic",
                "model_path": "test/small/model", 
                "device": "cpu",
                "estimated_size_gb": 0.5
            }
        }
        
        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)
        
        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing_memory_efficiency(self, memory_test_configs, lightweight_test_images):
        """Test that Provider instances are shared efficiently to save memory."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Track provider instance creation
            provider_instances = {}
            call_count = 0
            
            def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
                nonlocal call_count
                call_count += 1
                
                # Simulate provider sharing based on API key
                provider_key = f"{api_key}_{api_model_id.split(':')[0] if ':' in api_model_id else 'openai'}"
                
                if provider_key not in provider_instances:
                    provider_instances[provider_key] = f"provider_instance_{len(provider_instances)}"
                
                mock_agent = MagicMock()
                mock_agent.provider_instance = provider_instances[provider_key]
                mock_response = MagicMock()
                mock_response.tags = [f"shared_tag_{call_count}"]
                mock_agent.run.return_value = MagicMock(data=mock_response)
                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test multiple models that should share providers
            models_to_test = ["memory_openai_1", "memory_openai_2"]  # Same API key
            
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=models_to_test
            )

            # Verify provider sharing occurred
            assert len(provider_instances) == 1  # Should share provider instance
            assert call_count == 2  # But should have 2 agent calls
            
            # Verify results
            assert len(results) == 1
            image_results = list(results.values())[0]
            
            for model_name in models_to_test:
                assert model_name in image_results
                assert image_results[model_name]["error"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_local_model_memory_management(self, memory_test_configs, lightweight_test_images):
        """Test memory management for local ML models."""
        with patch('image_annotator_lib.core.model_factory.ModelLoad') as mock_model_load_class:
            # Track model loading and memory usage
            loaded_models = {}
            memory_usage = []
            
            def mock_model_loading():
                mock_model_load = MagicMock()
                
                def mock_load_model(model_name):
                    if model_name not in loaded_models:
                        # Simulate memory allocation
                        config = memory_test_configs[model_name]
                        memory_size = config.get("estimated_size_gb", 1.0)
                        memory_usage.append(memory_size)
                        
                        mock_annotator = MagicMock()
                        mock_annotator.model_name = model_name
                        mock_annotator.memory_size = memory_size
                        mock_annotator.run_inference.return_value = {
                            "test_hash": AnnotationResult(
                                tags=[f"local_tag_{model_name}"],
                                formatted_output={"tags": [f"local_tag_{model_name}"]},
                                error=None
                            )
                        }
                        loaded_models[model_name] = mock_annotator
                    
                    return loaded_models[model_name]
                
                mock_model_load.load_model.side_effect = mock_load_model
                mock_model_load._loaded_models = loaded_models  # Track for inspection
                return mock_model_load

            mock_model_load_class.return_value = mock_model_loading()

            # Test sequential loading of different sized models
            sequential_models = ["memory_local_small", "memory_local_large"]
            
            for model_name in sequential_models:
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=[model_name]
                )
                
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory') as mock_factory:
            # Mock cache with limited size
            cache_contents = {}
            max_cache_size = 2
            
            def mock_get_cached_agent(model_name, api_model_id, api_key, config_hash=None):
                cache_key = f"{model_name}_{api_model_id}_{api_key}"
                
                if cache_key in cache_contents:
                    # Move to end (LRU behavior)
                    agent = cache_contents.pop(cache_key)
                    cache_contents[cache_key] = agent
                    return agent
                
                # Create new agent
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = [f"lru_tag_{len(cache_contents)}"]
                mock_agent.run.return_value = MagicMock(data=mock_response)
                
                # LRU eviction if needed
                if len(cache_contents) >= max_cache_size:
                    # Remove least recently used (first item)
                    oldest_key = next(iter(cache_contents))
                    cache_contents.pop(oldest_key)
                
                cache_contents[cache_key] = mock_agent
                return mock_agent

            def mock_clear_cache():
                cache_contents.clear()

            mock_factory.get_cached_agent.side_effect = mock_get_cached_agent
            mock_factory.clear_cache.side_effect = mock_clear_cache

            # Test LRU behavior with multiple models
            test_sequence = [
                "memory_openai_1",
                "memory_anthropic",
                "memory_openai_2",  # Should cause eviction of memory_openai_1
                "memory_openai_1"   # Should be recreated
            ]

            for model_name in test_sequence:
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=[model_name]
                )
                
                assert len(results) == 1
                
                # Verify cache size doesn't exceed limit
                assert len(cache_contents) <= max_cache_size

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_pressure_handling(self, memory_test_configs, lightweight_test_images):
        """Test handling of memory pressure scenarios."""
        with patch('image_annotator_lib.core.model_factory.image_annotator_lib.api._create_annotator_instance') as mock_load_model:
            # Simulate memory pressure
            memory_pressure_count = 0
            
            def mock_load_with_memory_pressure(model_name):
                nonlocal memory_pressure_count
                memory_pressure_count += 1
                
                # First few attempts fail due to memory pressure
                if memory_pressure_count <= 2:
                    raise MemoryError(f"Insufficient memory to load {model_name}")
                
                # Later attempts succeed (after cleanup)
                mock_annotator = MagicMock()
                mock_annotator.run_inference.return_value = {
                    "test_hash": AnnotationResult(
                        tags=[f"memory_recovery_{model_name}"],
                        formatted_output={"tags": [f"memory_recovery_{model_name}"]},
                        error=None
                    )
                }
                return mock_annotator

            mock_load_model.side_effect = mock_load_with_memory_pressure

            # Test memory pressure recovery
            for attempt in range(3):
                try:
                    results = annotate(
                        images_list=lightweight_test_images[:1],
                        model_name_list=["memory_local_large"]
                    )
                    
                    if attempt < 2:
                        # First attempts should fail
                        assert len(results) == 1
                        image_results = list(results.values())[0]
                        assert image_results["memory_local_large"]["error"] is not None
                        assert "memory" in image_results["memory_local_large"]["error"].lower()
                    else:
                        # Final attempt should succeed
                        assert len(results) == 1
                        image_results = list(results.values())[0]
                        assert image_results["memory_local_large"]["error"] is None
                        
                except Exception as e:
                    if attempt < 2:
                        # Memory errors are acceptable for first attempts
                        assert "memory" in str(e).lower()
                    else:
                        pytest.fail(f"Memory pressure should be resolved by attempt {attempt}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_memory_usage(self, memory_test_configs, lightweight_test_images):
        """Test memory usage during concurrent operations."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.core.model_factory.image_annotator_lib.api._create_annotator_instance') as mock_load_model:
                
                # Track concurrent resource usage
                concurrent_webapi_calls = []
                concurrent_local_loads = []
                
                def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
                    concurrent_webapi_calls.append({
                        "time": time.time(),
                        "model": model_name,
                        "count": len(concurrent_webapi_calls) + 1
                    })
                    
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.tags = [f"concurrent_webapi_{len(concurrent_webapi_calls)}"]
                    mock_agent.run.return_value = MagicMock(data=mock_response)
                    return mock_agent

                def mock_model_loading(model_name):
                    concurrent_local_loads.append({
                        "time": time.time(),
                        "model": model_name,
                        "count": len(concurrent_local_loads) + 1
                    })
                    
                    mock_annotator = MagicMock()
                    mock_annotator.run_inference.return_value = {
                        "test_hash": AnnotationResult(
                            tags=[f"concurrent_local_{len(concurrent_local_loads)}"],
                            formatted_output={"tags": [f"concurrent_local_{len(concurrent_local_loads)}"]},
                            error=None
                        )
                    }
                    return mock_annotator

                mock_get_agent.side_effect = mock_agent_creation
                mock_load_model.side_effect = mock_model_loading

                # Test concurrent usage with mixed model types
                mixed_models = [
                    "memory_openai_1",
                    "memory_local_small", 
                    "memory_anthropic",
                    "memory_local_large"
                ]

                results = annotate(
                    images_list=lightweight_test_images[:2],  # Multiple images
                    model_name_list=mixed_models
                )

                # Verify concurrent operations occurred
                assert len(concurrent_webapi_calls) > 0
                assert len(concurrent_local_loads) > 0
                
                # Verify results
                assert len(results) == 2
                
                for image_hash, model_results in results.items():
                    for model_name in mixed_models:
                        assert model_name in model_results
                        assert model_results[model_name]["error"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_invalidation_memory_cleanup(self, memory_test_configs, lightweight_test_images):
        """Test that cache invalidation properly cleans up memory."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory') as mock_factory:
            # Track cache state
            cache_state = {"agents": {}, "cleared": False}
            
            def mock_get_cached_agent(model_name, api_model_id, api_key, config_hash=None):
                cache_key = f"{model_name}_{api_model_id}"
                
                if cache_key not in cache_state["agents"]:
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.tags = [f"cache_tag_{len(cache_state['agents'])}"]
                    mock_agent.run.return_value = MagicMock(data=mock_response)
                    cache_state["agents"][cache_key] = mock_agent
                
                return cache_state["agents"][cache_key]

            def mock_clear_cache():
                cache_state["agents"].clear()
                cache_state["cleared"] = True

            mock_factory.get_cached_agent.side_effect = mock_get_cached_agent
            mock_factory.clear_cache.side_effect = mock_clear_cache

            # Build up cache
            models_to_cache = ["memory_openai_1", "memory_anthropic"]
            
            for model_name in models_to_cache:
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=[model_name]
                )
                assert len(results) == 1

            # Verify cache is populated
            assert len(cache_state["agents"]) == 2
            assert not cache_state["cleared"]

            # Trigger cache clear
            PydanticAIProviderFactory.clear_cache()

            # Verify cache was cleared
            assert len(cache_state["agents"]) == 0
            assert cache_state["cleared"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_mixed_workload_memory_efficiency(self, memory_test_configs, lightweight_test_images):
        """Test memory efficiency under mixed workload conditions."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.core.model_factory.image_annotator_lib.api._create_annotator_instance') as mock_load_model:
                
                # Track resource allocation
                resource_usage = {
                    "webapi_agents": 0,
                    "local_models": 0,
                    "peak_webapi": 0,
                    "peak_local": 0
                }
                
                def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
                    resource_usage["webapi_agents"] += 1
                    resource_usage["peak_webapi"] = max(
                        resource_usage["peak_webapi"], 
                        resource_usage["webapi_agents"]
                    )
                    
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.tags = [f"mixed_webapi_{resource_usage['webapi_agents']}"]
                    mock_agent.run.return_value = MagicMock(data=mock_response)
                    return mock_agent

                def mock_model_loading(model_name):
                    resource_usage["local_models"] += 1
                    resource_usage["peak_local"] = max(
                        resource_usage["peak_local"],
                        resource_usage["local_models"]
                    )
                    
                    mock_annotator = MagicMock()
                    mock_annotator.run_inference.return_value = {
                        "test_hash": AnnotationResult(
                            tags=[f"mixed_local_{resource_usage['local_models']}"],
                            formatted_output={"tags": [f"mixed_local_{resource_usage['local_models']}"]},
                            error=None
                        )
                    }
                    return mock_annotator

                mock_get_agent.side_effect = mock_agent_creation
                mock_load_model.side_effect = mock_model_loading

                # Test mixed workload - alternating between WebAPI and local models
                workload_sequence = [
                    ["memory_openai_1"],
                    ["memory_local_small"],
                    ["memory_anthropic"], 
                    ["memory_local_large"],
                    ["memory_openai_2"]
                ]

                for model_batch in workload_sequence:
                    results = annotate(
                        images_list=lightweight_test_images[:1],
                        model_name_list=model_batch
                    )
                    
                    assert len(results) == 1
                    
                    # Verify successful processing
                    image_results = list(results.values())[0]
                    for model_name in model_batch:
                        assert model_name in image_results
                        assert image_results[model_name]["error"] is None

                # Verify resource usage patterns
                assert resource_usage["webapi_agents"] > 0
                assert resource_usage["local_models"] > 0
                assert resource_usage["peak_webapi"] <= resource_usage["webapi_agents"]
                assert resource_usage["peak_local"] <= resource_usage["local_models"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_monitoring_integration(self, memory_test_configs, lightweight_test_images):
        """Test integration with memory monitoring systems."""
        with patch('image_annotator_lib.core.model_factory.image_annotator_lib.api._create_annotator_instance') as mock_load_model:
            # Mock memory monitoring
            memory_snapshots = []
            
            def mock_model_loading_with_monitoring(model_name):
                # Simulate memory monitoring
                config = memory_test_configs[model_name]
                estimated_size = config.get("estimated_size_gb", 1.0)
                
                memory_snapshots.append({
                    "model": model_name,
                    "estimated_size_gb": estimated_size,
                    "timestamp": time.time()
                })
                
                mock_annotator = MagicMock()
                mock_annotator.model_size = estimated_size
                mock_annotator.run_inference.return_value = {
                    "test_hash": AnnotationResult(
                        tags=[f"monitored_{model_name}"],
                        formatted_output={"tags": [f"monitored_{model_name}"]},
                        error=None
                    )
                }
                return mock_annotator

            mock_load_model.side_effect = mock_model_loading_with_monitoring

            # Test memory monitoring during model operations
            models_to_monitor = ["memory_local_small", "memory_local_large"]
            
            for model_name in models_to_monitor:
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=[model_name]
                )
                
                assert len(results) == 1

            # Verify memory monitoring data
            assert len(memory_snapshots) == len(models_to_monitor)
            
            total_estimated_memory = sum(snap["estimated_size_gb"] for snap in memory_snapshots)
            assert total_estimated_memory == 3.0  # 0.5 + 2.5 GB
            
            # Verify chronological order
            timestamps = [snap["timestamp"] for snap in memory_snapshots]
            assert timestamps == sorted(timestamps)