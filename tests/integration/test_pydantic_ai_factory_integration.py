# tests/integration/test_pydantic_ai_factory_integration.py
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory


@pytest.fixture(autouse=True)
def manage_factory_cache():
    """Fixture to automatically clear factory caches before and after each test."""
    import asyncio
    import gc

    from image_annotator_lib.core.base.pydantic_ai_annotator import AdvancedAgentFactory
    from image_annotator_lib.core.config import config_registry

    # 1. Clear all caches before each test
    PydanticAIProviderFactory.clear_cache()
    AdvancedAgentFactory.clear_cache()

    # 2. Close any existing event loops
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass  # No loop to close

    # 3. Set new event loop for clean state
    try:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
    except RuntimeError:
        pass

    # 4. Clear any test configurations from config registry
    test_configs = [
        key for key in getattr(config_registry, "_config", {}).keys() if key.startswith("test_")
    ]
    for key in test_configs:
        try:
            config_registry._config.pop(key, None)
        except (AttributeError, KeyError):
            pass

    # 5. Force garbage collection
    gc.collect()

    yield

    # Clean up after each test
    # 1. Clear all caches
    PydanticAIProviderFactory.clear_cache()
    AdvancedAgentFactory.clear_cache()

    # 2. Clean up test configurations again
    test_configs = [
        key for key in getattr(config_registry, "_config", {}).keys() if key.startswith("test_")
    ]
    for key in test_configs:
        try:
            config_registry._config.pop(key, None)
        except (AttributeError, KeyError):
            pass

    # 3. Close event loop if it exists
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass

    # 4. Force garbage collection
    gc.collect()


class TestPydanticAIFactoryIntegration:
    """
    Integration tests for the PydanticAI Factory and its provider caching logic.
    """

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_caching_functionality(self, mock_test_env, MockAgent):
        """
        Tests that the Provider instances are cached correctly.
        """
        # Make the mock agent return a new mock instance each time it's called
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        # Create agents with same provider - should reuse provider
        agent1 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "same_key")
        agent2 = PydanticAIProviderFactory.get_cached_agent("model2", "openai:gpt-4", "same_key")

        # Different agents but should share provider configuration
        assert agent1 is not agent2  # Different agents

        # Verify provider was created (at least one entry in providers cache)
        assert len(PydanticAIProviderFactory._providers) >= 1

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_different_provider_configs(self, mock_test_env, MockAgent):
        """
        Tests that different provider configurations create separate cached providers.
        """
        # Make the mock agent return a new mock instance each time it's called
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        # Create agents with different API keys - should create different providers
        agent1 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "key1")
        agent2 = PydanticAIProviderFactory.get_cached_agent("model2", "openai:gpt-4", "key2")

        # Should have multiple provider configurations cached
        assert len(PydanticAIProviderFactory._providers) >= 2

    @patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_openrouter_special_handling(self, mock_test_env, mock_get_provider):
        """
        Tests special handling for OpenRouter, such as custom headers and base_url.
        """
        config = {"referer": "http://my-app.com", "app_name": "My Great App"}

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

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_extract_provider_name(self):
        """
        Tests provider name extraction from model IDs.
        """
        assert PydanticAIProviderFactory._extract_provider_name("openai:gpt-4") == "openai"
        assert PydanticAIProviderFactory._extract_provider_name("anthropic:claude-3") == "anthropic"
        assert PydanticAIProviderFactory._extract_provider_name("google:gemini-pro") == "google"
        assert PydanticAIProviderFactory._extract_provider_name("openrouter:mistral/7b") == "openrouter"

        # Auto-detection tests
        assert PydanticAIProviderFactory._extract_provider_name("gpt-4") == "openai"
        assert PydanticAIProviderFactory._extract_provider_name("claude-3-sonnet") == "anthropic"
        assert PydanticAIProviderFactory._extract_provider_name("gemini-pro") == "google"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_clear_functionality(self):
        """
        Tests that cache clearing works correctly.
        """
        # Add some providers to cache
        PydanticAIProviderFactory.get_provider("openai", api_key="test_key")
        PydanticAIProviderFactory.get_provider("anthropic", api_key="test_key")

        # Verify providers are cached
        assert len(PydanticAIProviderFactory._providers) >= 2

        # Clear cache
        PydanticAIProviderFactory.clear_cache()

        # Verify cache is cleared
        assert len(PydanticAIProviderFactory._providers) == 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_types_support(self):
        """
        Tests that all supported provider types can be created.
        """
        supported_providers = ["openai", "anthropic", "google", "openrouter"]

        for provider_name in supported_providers:
            try:
                provider = PydanticAIProviderFactory.get_provider(provider_name, api_key="test_key")
                assert provider is not None
            except Exception as e:
                pytest.fail(f"Failed to create {provider_name} provider: {e}")

        # Verify all providers are cached
        assert len(PydanticAIProviderFactory._providers) == len(supported_providers)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_unsupported_provider_error(self):
        """
        Tests that unsupported provider names raise appropriate errors.
        """
        with pytest.raises(ValueError, match="Unsupported provider: invalid_provider"):
            PydanticAIProviderFactory.get_provider("invalid_provider", api_key="test_key")
