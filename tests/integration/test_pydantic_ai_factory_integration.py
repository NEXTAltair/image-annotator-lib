# tests/integration/test_pydantic_ai_factory_integration.py
import pytest
from unittest.mock import patch, MagicMock
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.webapi_agent_cache import WebApiAgentCache

@pytest.fixture(autouse=True)
def manage_factory_cache():
    """Fixture to automatically clear factory caches before and after each test."""
    # Clear caches before the test
    PydanticAIProviderFactory._providers.clear()
    WebApiAgentCache.clear_cache()
    
    # Store original max size
    original_max_size = WebApiAgentCache._MAX_CACHE_SIZE

    yield

    # Restore original max size and clear caches after the test
    WebApiAgentCache.set_max_cache_size(original_max_size)
    PydanticAIProviderFactory._providers.clear()
    WebApiAgentCache.clear_cache()


class TestPydanticAIFactoryIntegration:
    """
    Integration tests for the PydanticAI Factory and its caching logic.
    """

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_caching_with_lru_strategy(self, MockAgent):
        """
        Tests that the Agent instances are cached using an LRU strategy.
        """
        # Make the mock agent return a new mock instance each time it's called
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        # Set a small cache size for testing
        WebApiAgentCache.set_max_cache_size(2)

        # --- Step 1: Fill the cache ---
        agent1 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "key1")
        agent2 = PydanticAIProviderFactory.get_cached_agent("model2", "openai:gpt-4", "key2")

        assert id(agent1) != id(agent2)
        
        # --- Step 2: Verify they are cached ---
        agent1_cached = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "key1")
        assert agent1_cached is agent1
        
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 2

        # --- Step 3: Access agent1 to make it recently used ---
        PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "key1")

        # --- Step 4: Add a new agent to trigger eviction ---
        agent3 = PydanticAIProviderFactory.get_cached_agent("model3", "openai:gpt-4", "key3")
        assert id(agent3) != id(agent1)
        
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 2
        
        # --- Step 5: Verify that the least recently used (agent2) was evicted ---
        agent2_new = PydanticAIProviderFactory.get_cached_agent("model2", "openai:gpt-4", "key2")
        assert id(agent2_new) != id(agent2)


    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_invalidation_on_config_change(self, MockAgent, managed_config_registry):
        """
        Tests that the cache is invalidated when model configuration changes.
        """
        # Make the mock agent return a new mock instance each time it's called
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        config1 = {"temperature": 0.5}
        config2 = {"temperature": 0.9}

        # --- Step 1: Get an agent with initial config ---
        agent1 = PydanticAIProviderFactory.get_cached_agent(
            "model1", "openai:gpt-4", "key1", config_data=config1
        )

        # --- Step 2: Get it again, should be cached ---
        agent1_cached = PydanticAIProviderFactory.get_cached_agent(
            "model1", "openai:gpt-4", "key1", config_data=config1
        )
        assert agent1_cached is agent1

        # --- Step 3: Get it with different config, should be a new instance ---
        agent2 = PydanticAIProviderFactory.get_cached_agent(
            "model1", "openai:gpt-4", "key1", config_data=config2
        )
        assert agent2 is not agent1


    @patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_openrouter_special_handling(self, mock_get_provider, managed_config_registry):
        """
        Tests special handling for OpenRouter, such as custom headers and base_url.
        """
        config = {
            "provider": "openrouter",
            "referer": "http://my-app.com",
            "app_name": "My Great App"
        }
        managed_config_registry.set("openrouter_model", config)

        PydanticAIProviderFactory.get_cached_agent(
            "openrouter_model", "openrouter:some/model", "key1", config_data=config
        )

        mock_get_provider.assert_called_once_with(
            "openrouter",
            api_key="key1",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://my-app.com",
                "X-Title": "My Great App",
            },
        )
