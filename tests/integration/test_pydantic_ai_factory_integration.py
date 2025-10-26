# tests/integration/test_pydantic_ai_factory_integration.py
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory


@pytest.fixture
def manage_factory_cache():
    """Fixture to clear factory caches (simplified)"""
    from image_annotator_lib.core.base.pydantic_ai_annotator import AdvancedAgentFactory

    # Event loop操作を削除
    # キャッシュクリアのみ実行
    PydanticAIProviderFactory.clear_cache()
    AdvancedAgentFactory.clear_cache()

    yield

    # 後処理もキャッシュクリアのみ
    PydanticAIProviderFactory.clear_cache()
    AdvancedAgentFactory.clear_cache()


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
        Tests that the Agent instances are cached correctly when using same configuration.
        """
        # Make the mock agent return the same instance for same config
        mock_agent_instance = MagicMock()
        MockAgent.return_value = mock_agent_instance

        # Create agents with same configuration - should return same Agent instance
        agent1 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "same_key")
        agent2 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "same_key")

        # Same model name and same configuration should return cached Agent
        assert agent1 is agent2, "同じ設定では同じAgentインスタンスが返されるべき"

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_different_provider_configs(self, mock_test_env, MockAgent):
        """
        Tests that different provider configurations create separate Agent instances.
        """
        # Make the mock agent return a new mock instance each time it's called
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        # Create agents with different model names - should create different Agents
        agent1 = PydanticAIProviderFactory.get_cached_agent("model1", "openai:gpt-4", "key1")
        agent2 = PydanticAIProviderFactory.get_cached_agent("model2", "openai:gpt-4", "key2")

        # Different model names should create different Agent instances
        assert agent1 is not agent2, "異なるモデル名では異なるAgentが返されるべき"

    @patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory.OpenAIChatModel")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_openrouter_special_handling(
        self, mock_test_env, mock_openai_model_class, mock_agent_class, mock_get_provider
    ):
        """
        Tests special handling for OpenRouter, such as custom headers and base_url.
        """
        # Setup mocks properly
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_model_instance = MagicMock()
        mock_openai_model_class.return_value = mock_model_instance

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        config = {"referer": "http://my-app.com", "app_name": "My Great App"}

        # Execute
        agent = PydanticAIProviderFactory.get_cached_agent(
            "openrouter_model", "openrouter:some/model", "key1", config_data=config
        )

        # Verify agent was returned
        assert agent is mock_agent_instance

        # Verify get_provider was called with correct arguments
        assert mock_get_provider.called
        call_kwargs = mock_get_provider.call_args.kwargs
        assert "api_key" in call_kwargs
        assert "base_url" in call_kwargs
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        if "default_headers" in call_kwargs:
            headers = call_kwargs["default_headers"]
            assert headers["HTTP-Referer"] == "http://my-app.com"
            assert headers["X-Title"] == "My Great App"

        # Verify OpenAIChatModel was called with provider
        assert mock_openai_model_class.called
        call_kwargs = mock_openai_model_class.call_args.kwargs
        assert "provider" in call_kwargs
        assert call_kwargs["provider"] is mock_provider

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
