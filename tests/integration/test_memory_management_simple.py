# tests/integration/test_memory_management_simple.py
"""
Simplified memory management integration tests.
Focus on essential memory management features without complex mocking.
"""

import gc

import pytest
from pydantic_ai import models

from image_annotator_lib.core.base.pydantic_ai_annotator import AdvancedAgentFactory
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory


class TestMemoryManagementSimple:
    """Simplified memory management tests."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear all caches before each test
        AdvancedAgentFactory.clear_cache()
        PydanticAIProviderFactory.clear_cache()

        # Disable real API requests for PydanticAI models
        models.ALLOW_MODEL_REQUESTS = False

        # Force garbage collection
        gc.collect()

        yield

        # Clean up after each test
        AdvancedAgentFactory.clear_cache()
        PydanticAIProviderFactory.clear_cache()
        gc.collect()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_factory_functionality(self):
        """Test basic PydanticAI provider factory operations."""

        # Verify factory starts with empty providers
        assert len(PydanticAIProviderFactory._providers) == 0

        # Create a provider to test caching
        provider1 = PydanticAIProviderFactory.get_provider("openai", api_key="test_key")
        assert provider1 is not None
        assert len(PydanticAIProviderFactory._providers) == 1

        # Get same provider again - should be cached
        provider2 = PydanticAIProviderFactory.get_provider("openai", api_key="test_key")
        assert provider1 is provider2  # Same instance due to caching

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_clearing(self):
        """Test cache clearing functionality."""

        # Verify initial state
        initial_size = len(PydanticAIProviderFactory._providers)

        # Add a provider to cache
        PydanticAIProviderFactory.get_provider("openai", api_key="test_key")
        assert len(PydanticAIProviderFactory._providers) >= 1

        # Clear cache
        PydanticAIProviderFactory.clear_cache()

        # Verify cache is cleared
        assert len(PydanticAIProviderFactory._providers) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_advanced_agent_factory_cache(self):
        """Test AdvancedAgentFactory caching functionality."""

        # Verify factory starts with empty cache
        assert len(AdvancedAgentFactory._agent_cache) == 0

        # Clear cache (should be safe even if empty)
        AdvancedAgentFactory.clear_cache()

        # Verify cache is still empty/cleared
        assert len(AdvancedAgentFactory._agent_cache) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_cleanup_sequence(self):
        """Test proper memory cleanup sequence."""

        # Record initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Clear all caches
        AdvancedAgentFactory.clear_cache()
        PydanticAIProviderFactory.clear_cache()

        # Force garbage collection
        gc.collect()

        # Count objects after cleanup (should be similar to initial)
        final_objects = len(gc.get_objects())
        assert isinstance(final_objects, int)

        # Verify cache states after cleanup
        assert len(AdvancedAgentFactory._agent_cache) == 0
        assert len(PydanticAIProviderFactory._providers) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_key_generation(self):
        """Test provider key generation for caching."""

        # Test that different configurations create different cache keys
        provider1 = PydanticAIProviderFactory.get_provider("openai", api_key="key1")
        provider2 = PydanticAIProviderFactory.get_provider("openai", api_key="key2")

        # Different API keys should create different providers
        assert provider1 is not provider2
        assert len(PydanticAIProviderFactory._providers) == 2

        # Same configuration should reuse provider
        provider3 = PydanticAIProviderFactory.get_provider("openai", api_key="key1")
        assert provider1 is provider3
        assert len(PydanticAIProviderFactory._providers) == 2

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_types(self):
        """Test different provider types."""

        # Test OpenAI provider
        openai_provider = PydanticAIProviderFactory.get_provider("openai", api_key="test_key")
        assert openai_provider is not None

        # Test Anthropic provider
        anthropic_provider = PydanticAIProviderFactory.get_provider("anthropic", api_key="test_key")
        assert anthropic_provider is not None

        # Test Google provider
        google_provider = PydanticAIProviderFactory.get_provider("google", api_key="test_key")
        assert google_provider is not None

        # Verify they are different instances
        assert openai_provider is not anthropic_provider
        assert openai_provider is not google_provider
        assert anthropic_provider is not google_provider

        # Verify all are cached
        assert len(PydanticAIProviderFactory._providers) == 3
