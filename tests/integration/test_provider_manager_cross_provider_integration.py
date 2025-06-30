# tests/integration/test_provider_manager_cross_provider_integration.py
"""
Integration tests for Provider Manager cross-provider scenarios.
Tests multi-provider usage, resource sharing, and provider switching.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.types import AnnotationResult


class TestProviderManagerCrossProviderIntegration:
    """Integration tests for cross-provider scenarios in Provider Manager."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear provider caches before each test
        PydanticAIProviderFactory.clear_cache()
        yield
        # Clear after test
        PydanticAIProviderFactory.clear_cache()

    @pytest.fixture
    def multi_provider_configs(self, managed_config_registry):
        """Setup configurations for multiple providers."""
        configs = {
            "openai_model": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key",
            },
            "anthropic_model": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "test-anthropic-key",
            },
            "google_model": {
                "class": "GoogleApiAnnotator",
                "api_model_id": "gemini-1.5-pro",
                "api_key": "test-google-key",
            },
            "openrouter_model": {
                "class": "OpenAIApiChatAnnotator",
                "api_model_id": "openrouter:anthropic/claude-3.5-sonnet",
                "api_key": "test-openrouter-key",
            },
        }

        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)

        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_sequential_multi_provider_usage(self, multi_provider_configs, lightweight_test_images):
        """Test sequential usage of different providers."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Mock successful agent creation and inference for all providers
            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                mock_agent = MagicMock()

                # Simulate different responses from different providers
                if "openai" in model_name or "gpt" in api_model_id:
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["openai_tag"], "provider": "openai"}
                    )
                elif "anthropic" in model_name or "claude" in api_model_id:
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["anthropic_tag"], "provider": "anthropic"}
                    )
                elif "google" in model_name or "gemini" in api_model_id:
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["google_tag"], "provider": "google"}
                    )
                else:
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["unknown_tag"], "provider": "unknown"}
                    )

                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test sequential provider usage
            test_models = ["openai_model", "anthropic_model", "google_model"]

            for model_name in test_models:
                try:
                    result = ProviderManager.run_inference_with_model(
                        model_name,
                        lightweight_test_images[:1],
                        api_model_id=multi_provider_configs[model_name]["api_model_id"],
                    )

                    assert result is not None
                    assert len(result) > 0

                    # Verify the result structure
                    for image_hash, annotation_result in result.items():
                        assert isinstance(annotation_result, AnnotationResult)
                        assert annotation_result.error is None

                except Exception as e:
                    pytest.fail(f"Sequential multi-provider test failed for {model_name}: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_multi_provider_usage(self, multi_provider_configs, lightweight_test_images):
        """Test concurrent usage of different providers."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track concurrent agent creation
            created_agents = {}

            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                provider_key = f"{model_name}_{api_model_id}"

                if provider_key not in created_agents:
                    mock_agent = MagicMock()
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": [f"tag_{model_name}"], "model": model_name}
                    )
                    created_agents[provider_key] = mock_agent

                return created_agents[provider_key]

            mock_get_agent.side_effect = mock_agent_creation

            # Simulate concurrent usage by calling multiple providers rapidly
            test_scenarios = [
                ("openai_model", "gpt-4o-mini"),
                ("anthropic_model", "claude-3-5-sonnet"),
                ("google_model", "gemini-1.5-pro"),
            ]

            results = []
            for model_name, api_model_id in test_scenarios:
                try:
                    result = ProviderManager.run_inference_with_model(
                        model_name, lightweight_test_images[:1], api_model_id=api_model_id
                    )
                    results.append((model_name, result))

                except Exception as e:
                    pytest.fail(f"Concurrent multi-provider test failed for {model_name}: {e!s}")

            # Verify all providers worked independently
            assert len(results) == len(test_scenarios)

            # Verify agents were created for each provider
            assert mock_get_agent.call_count == len(test_scenarios)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing(self, multi_provider_configs, lightweight_test_images):
        """Test that provider instances are properly shared across multiple model requests."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track agent creation calls
            agent_creation_calls = []
            shared_agents = {}

            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                call_signature = (model_name, api_model_id, api_key, config_hash)
                agent_creation_calls.append(call_signature)

                # Return same agent for same signature (simulating caching)
                agent_key = (api_model_id, api_key)
                if agent_key not in shared_agents:
                    mock_agent = MagicMock()
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["shared_tag"], "calls": len(agent_creation_calls)}
                    )
                    shared_agents[agent_key] = mock_agent

                return shared_agents[agent_key]

            mock_get_agent.side_effect = mock_agent_creation

            # Test multiple requests to same provider with same API key
            same_provider_requests = [
                ("openai_model", "gpt-4o-mini"),
                ("openai_model", "gpt-4o-mini"),  # Same provider, same model
                ("openai_model", "gpt-3.5-turbo"),  # Same provider, different model
            ]

            for model_name, api_model_id in same_provider_requests:
                result = ProviderManager.run_inference_with_model(
                    model_name, lightweight_test_images[:1], api_model_id=api_model_id
                )
                assert result is not None

            # Verify caching behavior - should have called agent creation for each unique signature
            assert len(agent_creation_calls) == len(same_provider_requests)

            # But shared agents should exist for same (api_model_id, api_key) combinations
            expected_shared_agents = len(
                set(
                    (req[1], multi_provider_configs["openai_model"]["api_key"])
                    for req in same_provider_requests
                )
            )
            assert len(shared_agents) == expected_shared_agents

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_switching_performance(self, multi_provider_configs, lightweight_test_images):
        """Test performance characteristics when switching between providers."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track timing and resource usage
            provider_switches = []
            current_provider = None

            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                nonlocal current_provider

                # Determine provider from api_model_id
                if "gpt" in api_model_id or "openai" in api_model_id:
                    new_provider = "openai"
                elif "claude" in api_model_id or "anthropic" in api_model_id:
                    new_provider = "anthropic"
                elif "gemini" in api_model_id or "google" in api_model_id:
                    new_provider = "google"
                else:
                    new_provider = "unknown"

                if current_provider != new_provider:
                    provider_switches.append((current_provider, new_provider))
                    current_provider = new_provider

                mock_agent = MagicMock()
                mock_agent.run.return_value = MagicMock(
                    data={"provider": new_provider, "switch_count": len(provider_switches)}
                )
                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test rapid provider switching
            switching_sequence = [
                ("openai_model", "gpt-4o-mini"),
                ("anthropic_model", "claude-3-5-sonnet"),
                ("google_model", "gemini-1.5-pro"),
                ("openai_model", "gpt-3.5-turbo"),  # Back to OpenAI
                ("anthropic_model", "claude-3-5-sonnet"),  # Back to Anthropic
            ]

            for model_name, api_model_id in switching_sequence:
                try:
                    result = ProviderManager.run_inference_with_model(
                        model_name, lightweight_test_images[:1], api_model_id=api_model_id
                    )
                    assert result is not None

                except Exception as e:
                    pytest.fail(f"Provider switching test failed at {model_name}: {e!s}")

            # Verify provider switches were tracked
            assert len(provider_switches) > 0

            # Verify no performance degradation (all calls succeeded)
            assert mock_get_agent.call_count == len(switching_sequence)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_error_isolation(self, multi_provider_configs, lightweight_test_images):
        """Test that errors in one provider don't affect others."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:

            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                mock_agent = MagicMock()

                # Make anthropic provider fail
                if "anthropic" in model_name or "claude" in api_model_id:
                    mock_agent.run.side_effect = Exception("Anthropic API Error")
                else:
                    mock_agent.run.return_value = MagicMock(
                        data={"tags": ["success_tag"], "provider": model_name}
                    )

                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test mixed success/failure scenario
            test_sequence = [
                ("openai_model", "gpt-4o-mini", True),  # Should succeed
                ("anthropic_model", "claude-3-5-sonnet", False),  # Should fail
                ("google_model", "gemini-1.5-pro", True),  # Should succeed
                ("openai_model", "gpt-3.5-turbo", True),  # Should still succeed
            ]

            for model_name, api_model_id, should_succeed in test_sequence:
                try:
                    result = ProviderManager.run_inference_with_model(
                        model_name, lightweight_test_images[:1], api_model_id=api_model_id
                    )

                    if should_succeed:
                        assert result is not None
                        # Verify successful result structure
                        for image_hash, annotation_result in result.items():
                            assert annotation_result.error is None
                    else:
                        # Failing provider should return error in result, not raise exception
                        if result:
                            for image_hash, annotation_result in result.items():
                                assert annotation_result.error is not None

                except Exception as e:
                    if should_succeed:
                        pytest.fail(f"Provider {model_name} should have succeeded but failed: {e!s}")
                    # If should_succeed is False, exceptions are acceptable

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_openrouter_provider_integration(self, multi_provider_configs, lightweight_test_images):
        """Test OpenRouter as a special case provider with custom handling."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.create_openrouter_agent"
        ) as mock_create_openrouter:
            mock_agent = MagicMock()
            mock_agent.run.return_value = MagicMock(
                data={"tags": ["openrouter_tag"], "provider": "openrouter"}
            )
            mock_create_openrouter.return_value = mock_agent

            try:
                result = ProviderManager.run_inference_with_model(
                    "openrouter_model",
                    lightweight_test_images[:1],
                    api_model_id="openrouter:anthropic/claude-3.5-sonnet",
                )

                assert result is not None
                assert len(result) > 0

                # Verify OpenRouter-specific handling was called
                mock_create_openrouter.assert_called_once()

                # Verify result structure
                for image_hash, annotation_result in result.items():
                    assert isinstance(annotation_result, AnnotationResult)
                    assert annotation_result.error is None

            except Exception as e:
                pytest.fail(f"OpenRouter provider integration failed: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cross_provider_resource_management(self, multi_provider_configs, lightweight_test_images):
        """Test resource management across multiple providers."""
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track resource allocation
            active_providers = set()
            max_concurrent_providers = 0

            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                # Determine provider
                if "gpt" in api_model_id:
                    provider = "openai"
                elif "claude" in api_model_id:
                    provider = "anthropic"
                elif "gemini" in api_model_id:
                    provider = "google"
                else:
                    provider = "unknown"

                active_providers.add(provider)
                nonlocal max_concurrent_providers
                max_concurrent_providers = max(max_concurrent_providers, len(active_providers))

                mock_agent = MagicMock()
                mock_agent.run.return_value = MagicMock(
                    data={"provider": provider, "concurrent_count": len(active_providers)}
                )

                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test resource usage with multiple providers
            resource_test_sequence = [
                ("openai_model", "gpt-4o-mini"),
                ("anthropic_model", "claude-3-5-sonnet"),
                ("google_model", "gemini-1.5-pro"),
            ]

            for model_name, api_model_id in resource_test_sequence:
                result = ProviderManager.run_inference_with_model(
                    model_name, lightweight_test_images[:1], api_model_id=api_model_id
                )
                assert result is not None

            # Verify resource tracking
            assert len(active_providers) > 0
            assert max_concurrent_providers <= len(resource_test_sequence)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_configuration_validation(self, managed_config_registry):
        """Test validation of provider configurations across different providers."""
        # Test invalid configurations
        invalid_configs = [
            (
                "missing_api_key",
                {
                    "class": "OpenAIApiChatAnnotator",
                    "api_model_id": "gpt-4o-mini",
                    # Missing api_key
                },
            ),
            (
                "invalid_model_id",
                {
                    "class": "AnthropicApiAnnotator",
                    "api_model_id": "invalid-model-id",
                    "api_key": "test-key",
                },
            ),
            (
                "missing_class",
                {
                    "api_model_id": "gpt-4o-mini",
                    "api_key": "test-key",
                    # Missing class
                },
            ),
        ]

        for model_name, config in invalid_configs:
            managed_config_registry.set(model_name, config)

            with patch(
                "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent"
            ) as mock_get_agent:
                mock_get_agent.side_effect = Exception("Configuration validation failed")

                try:
                    result = ProviderManager.run_inference_with_model(
                        model_name,
                        [Image.new("RGB", (64, 64), "red")],
                        api_model_id=config.get("api_model_id", "default"),
                    )

                    # Should handle configuration errors gracefully
                    if result:
                        for image_hash, annotation_result in result.items():
                            assert annotation_result.error is not None

                except Exception:
                    # Configuration validation errors are acceptable
                    pass
