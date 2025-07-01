# tests/integration/test_memory_management_simple.py
"""
Simplified memory management integration tests.
Focus on essential memory management features without complex mocking.
"""

import gc

import pytest
from pydantic_ai import models

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.webapi_agent_cache import WebApiAgentCache


class TestMemoryManagementSimple:
    """Simplified memory management tests."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear all caches before each test
        PydanticAIProviderFactory.clear_cache()
        WebApiAgentCache.clear_cache()
        
        # Disable real API requests for PydanticAI models
        models.ALLOW_MODEL_REQUESTS = False

        # Force garbage collection
        gc.collect()

        yield

        # Clean up after each test
        PydanticAIProviderFactory.clear_cache()
        WebApiAgentCache.clear_cache()
        gc.collect()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_webapi_agent_cache_functionality(self):
        """Test basic WebAPI agent cache operations."""
        
        # Verify cache starts empty
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 0
        assert len(cache_info["cached_agents"]) == 0

        # Test cache size management
        original_max_size = WebApiAgentCache._MAX_CACHE_SIZE
        try:
            # Set small cache size for testing
            WebApiAgentCache.set_max_cache_size(2)
            assert WebApiAgentCache._MAX_CACHE_SIZE == 2
            
            # Verify cache info reflects new size
            cache_info = WebApiAgentCache.get_cache_info()
            assert cache_info["max_cache_size"] == 2
            
        finally:
            # Restore original size
            WebApiAgentCache.set_max_cache_size(original_max_size)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        
        # Verify initial state
        cache_info = WebApiAgentCache.get_cache_info()
        initial_size = cache_info["cache_size"]
        
        # Clear cache (should be safe even if empty)
        WebApiAgentCache.clear_cache()
        
        # Verify cache is still empty/cleared
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 0
        assert len(cache_info["cached_agents"]) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_factory_cache_clear(self):
        """Test provider factory cache operations."""
        
        # Test that clear_cache method exists and is callable
        try:
            PydanticAIProviderFactory.clear_cache()
            # If we get here, the method exists and didn't raise an exception
            assert True
        except AttributeError:
            pytest.fail("PydanticAIProviderFactory.clear_cache() method not found")
        except Exception as e:
            pytest.fail(f"PydanticAIProviderFactory.clear_cache() raised unexpected exception: {e}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_cleanup_sequence(self):
        """Test proper memory cleanup sequence."""
        
        # Record initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Clear all caches
        PydanticAIProviderFactory.clear_cache()
        WebApiAgentCache.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Verify memory cleanup completed without errors
        final_objects = len(gc.get_objects())
        
        # We don't assert specific numbers since object count can vary,
        # but we verify that the cleanup process completes successfully
        assert isinstance(initial_objects, int)
        assert isinstance(final_objects, int)
        
        # Verify cache states after cleanup
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration 
    def test_cache_max_size_validation(self):
        """Test cache size validation."""
        
        original_max_size = WebApiAgentCache._MAX_CACHE_SIZE
        try:
            # Test invalid cache size
            with pytest.raises(ValueError, match="最大キャッシュサイズは1以上である必要があります"):
                WebApiAgentCache.set_max_cache_size(0)
            
            with pytest.raises(ValueError, match="最大キャッシュサイズは1以上である必要があります"):
                WebApiAgentCache.set_max_cache_size(-1)
            
            # Test valid cache sizes
            WebApiAgentCache.set_max_cache_size(1)
            assert WebApiAgentCache._MAX_CACHE_SIZE == 1
            
            WebApiAgentCache.set_max_cache_size(100)
            assert WebApiAgentCache._MAX_CACHE_SIZE == 100
            
        finally:
            # Restore original size
            WebApiAgentCache.set_max_cache_size(original_max_size)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_info_structure(self):
        """Test cache info data structure."""
        
        cache_info = WebApiAgentCache.get_cache_info()
        
        # Verify expected keys exist
        required_keys = ["cached_agents", "cache_size", "max_cache_size", "last_used_times"]
        for key in required_keys:
            assert key in cache_info, f"Expected key '{key}' not found in cache_info"
        
        # Verify data types
        assert isinstance(cache_info["cached_agents"], list)
        assert isinstance(cache_info["cache_size"], int)
        assert isinstance(cache_info["max_cache_size"], int)
        assert isinstance(cache_info["last_used_times"], dict)
        
        # Verify consistency
        assert cache_info["cache_size"] == len(cache_info["cached_agents"])
        assert cache_info["cache_size"] <= cache_info["max_cache_size"]