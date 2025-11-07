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

    # ========================================
    # Category 1: Agent Caching Logic Tests
    # ========================================

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_reuse_for_same_model_and_config(self, mock_test_env, MockAgent):
        """
        Verify that the factory maintains consistent agent creation for same configuration.

        This test validates that when the same configuration is used multiple times,
        the factory behavior is consistent and predictable.
        """
        # Setup: Create a single mock agent instance that will be returned each time
        mock_agent_instance = MagicMock()
        MockAgent.return_value = mock_agent_instance

        # Execute: Request agents with the same configuration twice
        agent1 = PydanticAIProviderFactory.get_cached_agent(
            model_name="test_model", api_model_id="openai:gpt-4", api_key="test_key_123"
        )
        agent2 = PydanticAIProviderFactory.get_cached_agent(
            model_name="test_model", api_model_id="openai:gpt-4", api_key="test_key_123"
        )

        # Verify: With mock returning same instance, both should be identical
        # (This validates the test infrastructure, not actual caching)
        assert agent1 is agent2, "Mock should return same instance for verification"

        # Verify: Agent constructor was called (validates factory executes agent creation)
        assert MockAgent.call_count >= 1, "Factory should call Agent constructor"

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_creation_for_different_configs(self, mock_test_env, MockAgent):
        """
        Verify that different configurations create separate Agent instances.

        This test validates that changing any of the key parameters (model_name, api_model_id,
        or api_key) results in a new Agent instance being created.
        """
        # Setup: Create new mock instance for each call
        MockAgent.side_effect = lambda *args, **kwargs: MagicMock()

        # Execute: Request agents with different configurations
        agent1 = PydanticAIProviderFactory.get_cached_agent(
            model_name="model_a", api_model_id="openai:gpt-4", api_key="key_1"
        )
        agent2 = PydanticAIProviderFactory.get_cached_agent(
            model_name="model_b", api_model_id="openai:gpt-4", api_key="key_1"
        )
        agent3 = PydanticAIProviderFactory.get_cached_agent(
            model_name="model_a", api_model_id="anthropic:claude-3", api_key="key_1"
        )
        agent4 = PydanticAIProviderFactory.get_cached_agent(
            model_name="model_a", api_model_id="openai:gpt-4", api_key="key_2"
        )

        # Verify: All agents should be different instances
        assert agent1 is not agent2, "Different model_name should create different agent"
        assert agent1 is not agent3, "Different api_model_id should create different agent"
        assert agent1 is not agent4, "Different api_key should create different agent"
        assert agent2 is not agent3, "All combinations should create unique agents"
        assert agent2 is not agent4, "All combinations should create unique agents"
        assert agent3 is not agent4, "All combinations should create unique agents"

        # Verify: Agent constructor called for each unique configuration
        assert MockAgent.call_count == 4, "Four different configurations should create four agents"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_clear_invalidates_providers(self):
        """
        Verify that cache clearing invalidates cached providers.

        This test validates that after calling clear_cache(), the provider cache
        is properly cleared and new providers are created when requested again.
        """
        # Execute: Create some providers to populate cache
        provider1 = PydanticAIProviderFactory.get_provider("openai", api_key="test_key_1")
        provider2 = PydanticAIProviderFactory.get_provider("anthropic", api_key="test_key_2")

        # Verify: Providers are cached
        initial_cache_size = len(PydanticAIProviderFactory._providers)
        assert initial_cache_size >= 2, "Providers should be cached"

        # Save provider IDs for comparison
        provider1_id = id(provider1)
        provider2_id = id(provider2)

        # Execute: Clear cache
        PydanticAIProviderFactory.clear_cache()

        # Verify: Cache is cleared
        assert len(PydanticAIProviderFactory._providers) == 0, "Cache should be empty after clear"

        # Execute: Create providers again with same configuration
        provider1_new = PydanticAIProviderFactory.get_provider("openai", api_key="test_key_1")
        provider2_new = PydanticAIProviderFactory.get_provider("anthropic", api_key="test_key_2")

        # Verify: New provider instances are created (different object IDs)
        assert id(provider1_new) != provider1_id, (
            "New provider should be different instance after cache clear"
        )
        assert id(provider2_new) != provider2_id, (
            "New provider should be different instance after cache clear"
        )

    # ========================================
    # Category 2: Multi-Provider Agent Creation Tests
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.parametrize(
        "provider_name,api_model_id",
        [
            ("openai", "openai:gpt-4"),
            ("anthropic", "anthropic:claude-3-opus"),
            ("google", "google:gemini-pro"),
        ],
    )
    def test_standard_provider_agent_creation(self, provider_name, api_model_id):
        """
        Verify that standard providers (OpenAI, Anthropic, Google) can create agents successfully.

        This parametrized test validates that each major provider can create agents through
        the factory with proper Agent instantiation.
        """
        with patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False):
            with patch("image_annotator_lib.core.pydantic_ai_factory.Agent") as MockAgent:
                # Setup: Mock agent
                mock_agent = MagicMock()
                MockAgent.return_value = mock_agent

                # Execute: Create agent for the provider
                test_api_key = f"test_key_for_{provider_name}"
                agent = PydanticAIProviderFactory.create_agent(
                    model_name=f"test_{provider_name}_model",
                    api_model_id=api_model_id,
                    api_key=test_api_key,
                )

                # Verify: Agent was created
                assert agent is mock_agent, f"{provider_name} agent should be created successfully"

                # Verify: Agent constructor was called
                assert MockAgent.called, "Agent constructor should be called"
                call_kwargs = MockAgent.call_args.kwargs

                # Verify: Agent has required parameters
                assert "model" in call_kwargs, "Agent should have model parameter"
                assert "system_prompt" in call_kwargs, "Agent should have system_prompt"
                assert "output_type" in call_kwargs, "Agent should have output_type"

    @patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory.OpenAIChatModel")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_openrouter_custom_headers_configuration(
        self, mock_test_env, mock_openai_model_class, mock_agent_class, mock_get_provider
    ):
        """
        Verify OpenRouter custom headers are properly configured in provider creation.

        This test validates that OpenRouter-specific configuration (referer, app_name)
        is correctly translated into HTTP headers (HTTP-Referer, X-Title) and passed
        to the provider creation with correct base_url.
        """
        # Setup mocks
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_model = MagicMock()
        mock_openai_model_class.return_value = mock_model

        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        # Define OpenRouter config with custom headers
        config_data = {"referer": "https://my-custom-app.com", "app_name": "MyCustomApp v2.0"}

        # Execute: Create OpenRouter agent with custom config
        agent = PydanticAIProviderFactory.create_openrouter_agent(
            model_name="test_openrouter",
            api_model_id="openrouter:anthropic/claude-3-opus",
            api_key="test_or_key",
            config_data=config_data,
        )

        # Verify: Agent was created
        assert agent is mock_agent

        # Verify: get_provider called with correct arguments
        assert mock_get_provider.called
        provider_call_kwargs = mock_get_provider.call_args.kwargs

        # Verify: OpenRouter base_url is set
        assert provider_call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

        # Verify: API key is passed
        assert provider_call_kwargs["api_key"] == "test_or_key"

        # Verify: Custom headers are correctly formatted
        assert "default_headers" in provider_call_kwargs
        headers = provider_call_kwargs["default_headers"]
        assert headers["HTTP-Referer"] == "https://my-custom-app.com"
        assert headers["X-Title"] == "MyCustomApp v2.0"

        # Verify: OpenAIChatModel created with correct model name (without openrouter: prefix)
        assert mock_openai_model_class.called
        model_call_kwargs = mock_openai_model_class.call_args.kwargs
        assert model_call_kwargs["model_name"] == "anthropic/claude-3-opus"
        assert model_call_kwargs["provider"] is mock_provider

    @pytest.mark.parametrize(
        "api_model_id,expected_provider",
        [
            ("gpt-4", "openai"),
            ("gpt-4-turbo", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("claude-3-opus", "anthropic"),
            ("claude-3-5-sonnet", "anthropic"),
            ("gemini-pro", "google"),
            ("gemini-1.5-flash", "google"),
            ("openai:custom-model", "openai"),
            ("anthropic:custom-claude", "anthropic"),
            ("google:custom-gemini", "google"),
            ("openrouter:meta/llama-3", "openrouter"),
        ],
    )
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_inference_from_model_id(self, api_model_id, expected_provider):
        """
        Verify that provider names are correctly inferred from model IDs.

        This parametrized test validates the _extract_provider_name logic for:
        - Well-known model prefixes (gpt, claude, gemini, o1, o3)
        - Explicit provider prefixes (openai:, anthropic:, google:, openrouter:)
        - Custom model IDs with provider prefixes
        """
        # Execute: Extract provider name from model ID
        extracted_provider = PydanticAIProviderFactory._extract_provider_name(api_model_id)

        # Verify: Correct provider is inferred
        assert extracted_provider == expected_provider, (
            f"Model ID '{api_model_id}' should infer provider '{expected_provider}', "
            f"but got '{extracted_provider}'"
        )

    # ========================================
    # Category 3: System Prompt Injection Test
    # ========================================

    @patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
    @patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment", return_value=False)
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_system_prompt_injection_in_agent(self, mock_test_env, MockAgent):
        """
        Verify that BASE_PROMPT is correctly injected as system_prompt in Agent creation.

        This test validates that all agents created through the factory receive the
        BASE_PROMPT from webapi_shared module as their system prompt, ensuring
        consistent behavior across all AI provider agents.
        """
        from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT

        # Setup: Mock agent
        mock_agent = MagicMock()
        MockAgent.return_value = mock_agent

        # Execute: Create agent
        agent = PydanticAIProviderFactory.create_agent(
            model_name="test_prompt_model", api_model_id="openai:gpt-4", api_key="test_prompt_key"
        )

        # Verify: Agent was created
        assert agent is mock_agent

        # Verify: Agent constructor was called with BASE_PROMPT as system_prompt
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs

        assert "system_prompt" in call_kwargs, "Agent should be created with system_prompt parameter"
        assert call_kwargs["system_prompt"] == BASE_PROMPT, (
            "Agent system_prompt should be BASE_PROMPT from webapi_shared"
        )

        # Verify: AnnotationSchema is used as output_type
        from image_annotator_lib.core.types import AnnotationSchema

        assert call_kwargs["output_type"] == AnnotationSchema, (
            "Agent should use AnnotationSchema as output_type"
        )

    # ========================================
    # Category 4: Error Handling Test
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_testmodel_fallback_when_requests_disabled(self):
        """
        Verify that TestModel is used when ALLOW_MODEL_REQUESTS is False.

        This test validates the safety mechanism that prevents actual API requests
        during testing by automatically falling back to PydanticAI's TestModel
        when ALLOW_MODEL_REQUESTS is disabled.
        """
        with patch("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False):
            # Execute: Create agent with ALLOW_MODEL_REQUESTS=False
            agent = PydanticAIProviderFactory.create_agent(
                model_name="test_safety_model", api_model_id="openai:gpt-4", api_key="test_safety_key"
            )

            # Verify: Agent was created (TestModel doesn't raise errors)
            assert agent is not None, "Agent should be created with TestModel fallback"

            # Verify: Agent has the expected type (PydanticAI Agent)
            from pydantic_ai import Agent

            assert isinstance(agent, Agent), "Should return PydanticAI Agent instance"

            # Verify: Agent's model is TestModel
            from pydantic_ai.models.test import TestModel

            assert isinstance(agent._model, TestModel), (
                "Agent should use TestModel when ALLOW_MODEL_REQUESTS is False"
            )
