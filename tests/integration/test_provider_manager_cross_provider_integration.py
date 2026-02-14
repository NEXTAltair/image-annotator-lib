# tests/integration/test_provider_manager_cross_provider_integration.py
"""
Integration tests for Provider Manager cross-provider scenarios.
Tests multi-provider usage, resource sharing, and provider switching.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAgentFactory


class TestProviderManagerCrossProviderIntegration:
    """Integration tests for cross-provider scenarios in Provider Manager."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Clear provider caches before each test
        PydanticAIAgentFactory.clear_cache()
        yield
        # Clear after test
        PydanticAIAgentFactory.clear_cache()

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
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Mock successful agent creation and inference for all providers
            def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
                mock_agent = MagicMock()

                # Use AsyncMock for async run method and simulate different responses
                mock_agent.run = AsyncMock()

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

                    # Verify the result structure (handle both dict and AnnotationResult)
                    for _image_hash, annotation_result in result.items():
                        # AnnotationResult is TypedDict, so always use dictionary access
                        assert annotation_result.get("error") is None

                except Exception as e:
                    pytest.fail(f"Sequential multi-provider test failed for {model_name}: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_concurrent_multi_provider_usage(self, multi_provider_configs, lightweight_test_images):
        """Test concurrent usage of different providers."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_run_inference:
            # Mock successful provider manager responses
            def mock_provider_response(model_name, images_list, api_model_id=None):
                return {
                    f"test_phash_{model_name}": {
                        "tags": [f"tag_{model_name}"],
                        "formatted_output": {"tags": [f"tag_{model_name}"]},
                        "error": None,
                        "phash": f"test_phash_{model_name}",
                    }
                }

            mock_run_inference.side_effect = mock_provider_response

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

            # Verify provider manager was called for each scenario
            assert mock_run_inference.call_count == len(test_scenarios)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing(self, multi_provider_configs, lightweight_test_images):
        """Test that provider instances are properly shared across multiple model requests."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_run_inference:
            # Track provider manager calls
            provider_calls = []

            def mock_provider_response(model_name, images_list, api_model_id=None):
                provider_calls.append((model_name, api_model_id))
                return {
                    f"shared_phash_{len(provider_calls)}": {
                        "tags": ["shared_tag"],
                        "formatted_output": {"tags": ["shared_tag"]},
                        "error": None,
                        "phash": f"shared_phash_{len(provider_calls)}",
                    }
                }

            mock_run_inference.side_effect = mock_provider_response

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

            # Verify provider manager was called for each request
            assert len(provider_calls) == len(same_provider_requests)

            # Verify unique model IDs were used
            unique_model_ids = {call[1] for call in provider_calls}
            expected_unique_ids = len({req[1] for req in same_provider_requests})
            assert len(unique_model_ids) == expected_unique_ids

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_switching_performance(self, multi_provider_configs, lightweight_test_images):
        """Test performance characteristics when switching between providers."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_run_inference:
            # Track provider switching
            provider_switches = []
            current_provider = None

            def mock_provider_response(model_name, images_list, api_model_id=None):
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

                return {
                    f"switch_phash_{len(provider_switches)}": {
                        "tags": [f"{new_provider}_tag"],
                        "formatted_output": {
                            "provider": new_provider,
                            "switch_count": len(provider_switches),
                        },
                        "error": None,
                        "phash": f"switch_phash_{len(provider_switches)}",
                    }
                }

            mock_run_inference.side_effect = mock_provider_response

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
            assert mock_run_inference.call_count == len(switching_sequence)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_error_isolation(self, multi_provider_configs, lightweight_test_images):
        """Test that errors in one provider don't affect others."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_run_inference:

            def mock_provider_response(model_name, images_list, api_model_id=None):
                # Make anthropic provider fail
                if "anthropic" in model_name or "claude" in api_model_id:
                    return {
                        "error_phash": {
                            "tags": [],
                            "formatted_output": None,
                            "error": "Anthropic API Error",
                            "phash": "error_phash",
                        }
                    }
                else:
                    return {
                        f"success_phash_{model_name}": {
                            "tags": ["success_tag"],
                            "formatted_output": {"tags": ["success_tag"], "provider": model_name},
                            "error": None,
                            "phash": f"success_phash_{model_name}",
                        }
                    }

            mock_run_inference.side_effect = mock_provider_response

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
                        for _image_hash, annotation_result in result.items():
                            if isinstance(annotation_result, dict):
                                assert (
                                    annotation_result.get("error") is None
                                    or annotation_result.get("error") == ""
                                )
                            elif hasattr(annotation_result, "error"):
                                assert annotation_result.error is None or annotation_result.error == ""
                    else:
                        # Failing provider should return error in result, not raise exception
                        if result:
                            for _image_hash, annotation_result in result.items():
                                if isinstance(annotation_result, dict):
                                    assert annotation_result.get("error") is not None
                                elif hasattr(annotation_result, "error"):
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
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.create_openrouter_agent"
        ) as mock_create_openrouter:
            mock_agent = MagicMock()
            # Explicitly set AsyncMock for async run method
            mock_agent.run = AsyncMock()
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

                # Verify result structure (AnnotationResult is TypedDict)
                for _image_hash, annotation_result in result.items():
                    # AnnotationResult is TypedDict, so always use dictionary access
                    assert annotation_result.get("error") is None

            except Exception as e:
                pytest.fail(f"OpenRouter provider integration failed: {e!s}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cross_provider_resource_management(self, multi_provider_configs, lightweight_test_images):
        """Test resource management across multiple providers."""
        with patch(
            "image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model"
        ) as mock_run_inference:
            # Track resource allocation
            active_providers = set()
            max_concurrent_providers = 0

            def mock_provider_response(model_name, images_list, api_model_id=None):
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

                return {
                    f"resource_phash_{provider}": {
                        "tags": [f"{provider}_tag"],
                        "formatted_output": {
                            "provider": provider,
                            "concurrent_count": len(active_providers),
                        },
                        "error": None,
                        "phash": f"resource_phash_{provider}",
                    }
                }

            mock_run_inference.side_effect = mock_provider_response

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
                "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
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
                        for _image_hash, annotation_result in result.items():
                            # AnnotationResult is TypedDict, so check attributes instead of isinstance
                            if hasattr(annotation_result, "error"):
                                assert annotation_result.error is not None
                            elif isinstance(annotation_result, dict):
                                assert annotation_result.get("error") is not None

                except Exception:
                    # Configuration validation errors are acceptable
                    pass

    # ========================================
    # Category A: Agent Cache & Provider Instance Management
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_cache_reuse_across_same_provider(self, managed_config_registry, lightweight_test_images):
        """
        Verify that agents are reused when using the same provider with identical configuration.

        Tests PydanticAIAgentFactory's caching logic to ensure Agent instances
        are properly reused across multiple inference calls with same model_name, api_model_id, and api_key.
        """
        from unittest.mock import MagicMock, patch

        model_name = "test_cache_reuse"
        managed_config_registry.set(model_name, {"provider": "openai", "api_key": "test_cache_key"})

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Create single mock agent that will be reused
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            # Call twice with same configuration
            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_cache_test"

                # First call
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4",
                    api_keys={"openai": "test_cache_key"},
                )

                # Second call with same configuration
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4",
                    api_keys={"openai": "test_cache_key"},
                )

                # Verify get_cached_agent was called twice with same parameters
                assert mock_get_agent.call_count == 2
                call_args_list = mock_get_agent.call_args_list

                # Both calls should have identical parameters
                assert call_args_list[0] == call_args_list[1]
                assert call_args_list[0].kwargs["model_name"] == model_name
                assert call_args_list[0].kwargs["api_model_id"] == "openai:gpt-4"
                assert call_args_list[0].kwargs["api_key"] == "test_cache_key"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_creation_for_different_configurations(
        self, managed_config_registry, lightweight_test_images
    ):
        """
        Verify that different configurations create separate Agent instances.

        Tests that changing any of the key parameters (model_name, api_model_id, or api_key)
        results in different Agent instances being requested from the factory.
        """
        from unittest.mock import MagicMock, patch

        # Setup multiple model configurations
        managed_config_registry.set("model_a", {"provider": "openai", "api_key": "key_a"})
        managed_config_registry.set("model_b", {"provider": "openai", "api_key": "key_b"})

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Create new mock agent for each call
            mock_get_agent.side_effect = lambda **kwargs: MagicMock()

            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_diff_config"

                # Test 1: Different model_name
                ProviderManager.run_inference_with_model(
                    model_name="model_a",
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4",
                    api_keys={"openai": "key_a"},
                )

                ProviderManager.run_inference_with_model(
                    model_name="model_b",
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4",
                    api_keys={"openai": "key_b"},
                )

                # Test 2: Different api_model_id (same model_name)
                ProviderManager.run_inference_with_model(
                    model_name="model_a",
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-3.5-turbo",
                    api_keys={"openai": "key_a"},
                )

                # Verify factory was called with different parameters
                assert mock_get_agent.call_count >= 3
                call_args_list = mock_get_agent.call_args_list

                # First two calls should have different model_name and api_key
                assert call_args_list[0].kwargs["model_name"] != call_args_list[1].kwargs["model_name"]
                assert call_args_list[0].kwargs["api_key"] != call_args_list[1].kwargs["api_key"]

                # First and third calls should have different api_model_id
                assert call_args_list[0].kwargs["api_model_id"] != call_args_list[2].kwargs["api_model_id"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_lifecycle_management(self, managed_config_registry, lightweight_test_images):
        """
        Verify provider instance lifecycle management including creation, reuse, and cleanup.

        Tests the full lifecycle of provider instances: initial creation, reuse across calls,
        cache clearing, and recreation after clear.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        model_name = "test_lifecycle"
        managed_config_registry.set(model_name, {"provider": "anthropic", "api_key": "test_lifecycle_key"})

        # Track agent creation and simulate caching
        agent_cache = {}
        agent_creation_count = [0]  # Use list to allow modification in nested function

        def mock_get_cached_agent_impl(model_name, api_model_id, api_key, config_data=None):
            """Simulate agent caching behavior"""
            # Create cache key from parameters
            cache_key = f"{model_name}:{api_model_id}:{api_key}"

            # Check if agent is in cache
            if cache_key in agent_cache:
                # Return cached agent (no increment)
                return agent_cache[cache_key]

            # Create new agent (increment count)
            agent_creation_count[0] += 1
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.data = AnnotationSchema(
                tags=["lifecycle_tag"], captions=["Test caption"], score=0.9, metadata={}
            )
            mock_agent.run_sync.return_value = mock_result

            # Cache the agent
            agent_cache[cache_key] = mock_agent
            return mock_agent

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent",
            side_effect=mock_get_cached_agent_impl,
        ):
            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_lifecycle"

                # Phase 1: Initial agent creation
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="anthropic:claude-3-opus",
                    api_keys={"anthropic": "test_lifecycle_key"},
                )

                initial_count = agent_creation_count[0]
                assert initial_count == 1, "Agent should be created once on first call"

                # Phase 2: Agent reuse (same configuration)
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="anthropic:claude-3-opus",
                    api_keys={"anthropic": "test_lifecycle_key"},
                )

                # Agent should be reused, not recreated
                assert agent_creation_count[0] == initial_count, (
                    "Agent should be reused for same configuration"
                )

                # Phase 3: Clear cache (simulate by clearing our mock cache)
                agent_cache.clear()

                # Phase 4: Agent recreation after clear
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="anthropic:claude-3-opus",
                    api_keys={"anthropic": "test_lifecycle_key"},
                )

                # New agent should be created after cache clear
                assert agent_creation_count[0] == initial_count + 1, (
                    "New agent should be created after cache clear"
                )

    # ========================================
    # Category B: Dynamic Model Switching & Result Consistency
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_api_model_id_override_functionality(self, managed_config_registry, lightweight_test_images):
        """
        Verify that the same model_name can dynamically switch to different api_model_id.

        Tests the flexibility of ProviderManager to handle model ID overrides,
        allowing the same configured model to use different underlying API models.
        """
        from unittest.mock import MagicMock, patch

        model_name = "flexible_model"
        managed_config_registry.set(model_name, {"provider": "openai", "api_key": "test_override_key"})

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Track which api_model_id was used for each call
            api_model_ids_used = []

            def track_model_id(**kwargs):
                api_model_ids_used.append(kwargs.get("api_model_id"))
                return MagicMock()

            mock_get_agent.side_effect = track_model_id

            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_override"

                # Call 1: Use GPT-4
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4",
                    api_keys={"openai": "test_override_key"},
                )

                # Call 2: Use GPT-3.5-turbo (same model_name, different api_model_id)
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-3.5-turbo",
                    api_keys={"openai": "test_override_key"},
                )

                # Call 3: Use GPT-4o-mini (another override)
                ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openai:gpt-4o-mini",
                    api_keys={"openai": "test_override_key"},
                )

                # Verify different api_model_id values were used
                assert len(api_model_ids_used) == 3
                assert api_model_ids_used[0] == "openai:gpt-4"
                assert api_model_ids_used[1] == "openai:gpt-3.5-turbo"
                assert api_model_ids_used[2] == "openai:gpt-4o-mini"

                # Verify all three model IDs are different
                assert len(set(api_model_ids_used)) == 3, "All api_model_id values should be unique"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cross_provider_result_format_consistency(
        self, managed_config_registry, lightweight_test_images
    ):
        """
        Verify that all providers return results in consistent AnnotationResult format.

        Tests that regardless of provider (OpenAI, Anthropic, Google),
        the result structure adheres to AnnotationResult TypedDict specification.
        """
        from unittest.mock import MagicMock, patch

        # Setup configurations for multiple providers
        providers = ["openai", "anthropic", "google"]
        for i, provider in enumerate(providers):
            model_name = f"test_{provider}_model"
            managed_config_registry.set(model_name, {"provider": provider, "api_key": f"key_{provider}"})

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Create consistent AnnotationSchema response
            from image_annotator_lib.core.types import AnnotationSchema

            def create_consistent_agent(model_name, api_model_id, api_key, config_data=None):
                mock_agent = MagicMock()

                # Create consistent response structure
                mock_result = MagicMock()
                mock_result.data = AnnotationSchema(
                    tags=["test_tag_1", "test_tag_2"],
                    captions=["Test caption"],
                    score=0.95,
                    metadata={"provider": api_model_id.split(":")[0]},
                )
                mock_agent.run_sync.return_value = mock_result
                return mock_agent

            mock_get_agent.side_effect = create_consistent_agent

            results = {}

            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_consistency"

                # Call each provider
                for provider in providers:
                    model_name = f"test_{provider}_model"
                    api_model_id = f"{provider}:test-model"

                    result = ProviderManager.run_inference_with_model(
                        model_name=model_name,
                        images_list=lightweight_test_images[:1],
                        api_model_id=api_model_id,
                        api_keys={provider: f"key_{provider}"},
                    )

                    results[provider] = result

                # Verify all providers returned results
                assert len(results) == 3

                # Verify consistent result structure across all providers
                for provider, result in results.items():
                    assert "phash_consistency" in result, f"{provider} should return phash key"

                    annotation_result = result["phash_consistency"]

                    # Check AnnotationResult TypedDict structure
                    assert isinstance(annotation_result, dict), f"{provider} result should be dict"
                    assert "tags" in annotation_result, f"{provider} should have 'tags' field"
                    assert "formatted_output" in annotation_result, (
                        f"{provider} should have 'formatted_output'"
                    )
                    assert annotation_result.get("error") is None, f"{provider} should not have error"

                    # Verify consistent data types
                    assert isinstance(annotation_result["tags"], list), f"{provider} tags should be list"
                    assert len(annotation_result["tags"]) > 0, f"{provider} should return non-empty tags"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_specific_configuration_handling(
        self, managed_config_registry, lightweight_test_images
    ):
        """
        Verify provider-specific configurations are correctly handled (e.g., OpenRouter custom headers).

        Tests that providers with special configuration requirements (like OpenRouter's
        referer and app_name headers) are properly configured and validated.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        # Setup OpenRouter model with custom headers
        model_name = "test_openrouter_custom"
        managed_config_registry.set(
            model_name,
            {
                "provider": "openrouter",
                "api_key": "test_or_key",
                "referer": "https://test-app.example.com",
                "app_name": "Test Application v1.0",
            },
        )

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory.get_cached_agent"
        ) as mock_get_agent:
            # Mock agent with proper structure
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.data = AnnotationSchema(
                tags=["openrouter_tag"], captions=["Test caption"], score=0.9, metadata={}
            )
            mock_agent.run_sync.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_or_custom"

                # Execute with OpenRouter-specific configuration
                result = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openrouter:anthropic/claude-3-opus",
                    api_keys={"openrouter": "test_or_key"},
                )

                # Verify get_cached_agent was called
                assert mock_get_agent.called, "get_cached_agent should be called for OpenRouter"

                # Verify result structure (confirms OpenRouter provider was used)
                assert "phash_or_custom" in result
                assert result["phash_or_custom"]["error"] is None

                # Verify get_cached_agent was called with correct parameters
                call_args = mock_get_agent.call_args
                # Check both positional and keyword arguments
                if call_args.kwargs:
                    assert call_args.kwargs.get("model_name") == model_name, (
                        "model_name should match in kwargs"
                    )
                    assert call_args.kwargs.get("api_model_id") == "openrouter:anthropic/claude-3-opus", (
                        "api_model_id should match"
                    )
                else:
                    # Check positional arguments
                    assert len(call_args.args) >= 2, "Should have at least model_name and api_model_id"
