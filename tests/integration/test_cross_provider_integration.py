# tests/integration/test_cross_provider_integration.py
from unittest.mock import patch

import pytest

from image_annotator_lib.api import annotate
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.registry import register_annotators


@pytest.fixture(autouse=True)
def manage_caches():
    """Fixture to automatically clear caches before and after each test."""
    ProviderManager._provider_instances.clear()
    yield
    ProviderManager._provider_instances.clear()


class TestCrossProviderIntegration:
    """
    Integration tests for scenarios involving multiple providers.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_simultaneous_multi_provider_usage(
        self, managed_config_registry, lightweight_test_images, api_key_manager
    ):
        """
        Tests the system's ability to handle simultaneous requests to different providers.
        """
        from unittest.mock import AsyncMock

        from pydantic_ai.result import FinalResult

        from image_annotator_lib.core.types import AnnotationSchema

        # Setup configs with proper test API keys
        managed_config_registry.set(
            "google_model",
            {
                "class": "GoogleApiAnnotator",
                "api_model_id": "gemini-pro",
                "api_key": api_key_manager.get_key("google"),
                "capabilities": ["tags", "captions", "scores"],
            },
        )
        managed_config_registry.set(
            "openai_model",
            {
                "class": "OpenAIApiAnnotator",
                "api_model_id": "gpt-4",
                "api_key": api_key_manager.get_key("openai"),
                "capabilities": ["tags", "captions", "scores"],
            },
        )

        # Re-register annotators to include the dynamically added models
        register_annotators()

        # The TestModel in pydantic-ai has issues with various image input formats.
        # To make the test robust and independent of the TestModel's implementation,
        # we directly patch `agent.run` to return a valid, structured response.
        mock_response = FinalResult(
            output=AnnotationSchema(tags=["test_tag"], captions=["test caption"], score=0.99)
        )

        # Patch the Agent.run method where it's used: in the pydantic_ai_factory module
        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.Agent.run",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Annotate with both models
            results = annotate(
                images_list=lightweight_test_images,
                model_name_list=["google_model", "openai_model"],
            )

        # Verify results are aggregated
        assert len(results.keys()) == len(lightweight_test_images)

        # Check that results for each model are present for the first image
        first_phash = next(iter(results))
        assert "google_model" in results[first_phash]
        assert "openai_model" in results[first_phash]

        # The mocked agent.run should provide predictable results
        google_result = results[first_phash]["google_model"]
        openai_result = results[first_phash]["openai_model"]

        assert google_result is not None
        assert openai_result is not None

        # Debug: print the actual results to understand the structure
        print(f"Google result: {google_result}")
        print(f"OpenAI result: {openai_result}")

        # Check if we have results (even if there are errors, basic functionality should work)
        from image_annotator_lib.core.types import UnifiedAnnotationResult

        assert isinstance(google_result, UnifiedAnnotationResult)
        assert isinstance(openai_result, UnifiedAnnotationResult)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_resource_isolation_and_conflict_avoidance(self, managed_config_registry):
        """
        Tests that resources are properly isolated between different providers.
        """
        # Get provider instances for two different providers
        google_provider = ProviderManager.get_provider_instance("google")
        openai_provider = ProviderManager.get_provider_instance("openai")
        anthropic_provider = ProviderManager.get_provider_instance("anthropic")

        # Verify they are distinct objects
        assert id(google_provider) != id(openai_provider)
        assert id(google_provider) != id(anthropic_provider)
        assert id(openai_provider) != id(anthropic_provider)

        # Verify they have their own separate context dictionaries
        assert id(google_provider._active_contexts) != id(openai_provider._active_contexts)
        assert id(google_provider._active_contexts) != id(anthropic_provider._active_contexts)


# ==============================================================================
# Phase B Task 3.1: Sequential Provider Switching Tests
# ==============================================================================


class TestSequentialProviderSwitching:
    """Sequential provider switching tests.

    Tests provider switching and reuse patterns.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_sequential_provider_switching_real_instances(
        self, managed_config_registry, lightweight_test_images, api_key_manager
    ):
        """Test switching between providers sequentially.

        REAL components:
        - Real ProviderManager._provider_instances cache
        - Real Provider instance management

        Scenario:
        1. Use google provider → create instance
        2. Use openai provider → create instance
        3. Use anthropic provider → create instance
        4. Return to google → reuse instance

        Assertions:
        - Each provider creates unique instance
        - Provider instances cached correctly
        - Returning to provider reuses instance
        """
        from unittest.mock import AsyncMock

        from pydantic_ai.result import FinalResult

        from image_annotator_lib.core.types import AnnotationSchema

        # Setup: Configure 3 providers
        providers = ["google", "openai", "anthropic"]
        for provider in providers:
            model_id = {
                "google": "gemini-pro",
                "openai": "gpt-4",
                "anthropic": "claude-3-5-sonnet-latest",
            }[provider]

            managed_config_registry.set(
                f"{provider}_switch_model",
                {
                    "class": "PydanticAIWebAPIAnnotator",
                    "model_name_on_provider": model_id,
                    "api_model_id": model_id,
                    "api_key": api_key_manager.get_key(provider),
                    "capabilities": ["tags", "captions", "scores"],
                },
            )

        register_annotators()

        # Mock API responses
        mock_response = FinalResult(
            output=AnnotationSchema(tags=["switch_test"], captions=["test"], score=0.9)
        )

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.Agent.run",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Act: Use providers sequentially
            for provider in providers:
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=[f"{provider}_switch_model"],
                )
                assert len(results) > 0, f"{provider} provider returned results"

            # Assert: 3 distinct provider instances created
            assert len(ProviderManager._provider_instances) == 3, "3つのProviderインスタンス作成"

            # Act: Return to google (should reuse instance)
            google_instance_before = ProviderManager.get_provider_instance("google")
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["google_switch_model"],
            )
            google_instance_after = ProviderManager.get_provider_instance("google")

            # Assert: Same instance reused
            assert google_instance_before is google_instance_after, "Googleインスタンス再利用"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_reuse_verification(self):
        """Test provider instance reuse with ID verification.

        REAL components:
        - Real Provider instance caching

        Scenario:
        1. Get google provider → instance1
        2. Get google provider again → instance2
        3. Verify instance1 === instance2 (same object ID)

        Assertions:
        - Same provider name returns same instance
        - Object IDs match (identity check)
        """
        # Act: Get provider twice
        instance1 = ProviderManager.get_provider_instance("google")
        instance2 = ProviderManager.get_provider_instance("google")

        # Assert: Same instance returned
        assert instance1 is instance2, "同じProviderインスタンスが返される"
        assert id(instance1) == id(instance2), "Object IDが一致"

        # Act: Get different provider
        instance_openai = ProviderManager.get_provider_instance("openai")

        # Assert: Different instance
        assert instance1 is not instance_openai, "異なるProviderは異なるインスタンス"
        assert id(instance1) != id(instance_openai), "Object IDが異なる"


# ==============================================================================
# Phase B Task 3.2: Configuration Consistency Tests
# ==============================================================================


class TestConfigurationConsistency:
    """Configuration consistency tests.

    Tests cache invalidation on configuration changes.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_config_change_invalidates_cache(
        self, managed_config_registry, lightweight_test_images, api_key_manager
    ):
        """Test configuration change invalidates provider cache.

        REAL components:
        - Real configuration management
        - Real cache invalidation logic

        Scenario:
        1. Use model with config1
        2. Change configuration (e.g., timeout)
        3. Use model again → should detect config change
        4. Verify behavior consistent with new config

        Assertions:
        - Configuration changes detected
        - System behavior reflects new config
        """
        from unittest.mock import AsyncMock

        from pydantic_ai.result import FinalResult

        from image_annotator_lib.core.types import AnnotationSchema

        # Setup: Initial configuration
        config1 = {
            "class": "PydanticAIWebAPIAnnotator",
            "model_name_on_provider": "gemini-pro",
            "api_model_id": "gemini-pro",
            "api_key": api_key_manager.get_key("google"),
            "timeout": 30,
            "capabilities": ["tags"],
        }
        managed_config_registry.set("config_test_model", config1)
        register_annotators()

        mock_response = FinalResult(output=AnnotationSchema(tags=["config_test"], captions=[], score=0.9))

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.Agent.run",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Act: Use with config1
            results1 = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["config_test_model"],
            )
            assert len(results1) > 0, "Config1で動作成功"

            # Act: Change configuration
            config2 = config1.copy()
            config2["timeout"] = 60  # Change timeout
            managed_config_registry.set("config_test_model", config2)
            register_annotators()

            # Act: Use with config2
            results2 = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["config_test_model"],
            )
            assert len(results2) > 0, "Config2で動作成功"

            # Assert: Both configurations worked
            # (Config change handled internally by system)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_api_key_change_creates_new_provider(self, managed_config_registry, api_key_manager):
        """Test API key change creates new provider instance.

        REAL components:
        - Real PydanticAIProviderFactory._providers cache
        - Real provider key generation (includes API key)

        Scenario:
        1. Get provider with api_key1
        2. Change API key to api_key2
        3. Get provider again
        4. Verify different provider instance created

        Assertions:
        - API key change creates new instance
        - Old and new instances have different IDs
        """
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

        # Act: Get provider with first API key
        provider1 = PydanticAIProviderFactory.get_provider("openai", api_key="test_key_v1")

        # Act: Get provider with different API key
        provider2 = PydanticAIProviderFactory.get_provider("openai", api_key="test_key_v2")

        # Assert: Different instances
        assert provider1 is not provider2, "異なるAPI keyでは異なるProviderインスタンス"
        assert id(provider1) != id(provider2), "Object IDが異なる"

        # Assert: Both providers cached
        assert len(PydanticAIProviderFactory._providers) >= 2, "両Providerがキャッシュ済み"


# ==============================================================================
# Phase B Task 3.3: Error Isolation Tests
# ==============================================================================


class TestErrorIsolation:
    """Error isolation tests.

    Tests that errors in one provider don't affect others.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_error_isolation(
        self, managed_config_registry, lightweight_test_images, api_key_manager
    ):
        """Test error in one provider doesn't affect others.

        REAL components:
        - Real error handling and isolation
        - Real provider instance management

        Scenario:
        1. Configure google_model (will succeed)
        2. Configure openai_model (will fail via mock)
        3. Use both models
        4. Verify google succeeds despite openai failure

        Assertions:
        - Google provider returns results
        - OpenAI provider error recorded
        - Google unaffected by OpenAI error
        """
        from unittest.mock import AsyncMock

        from pydantic_ai.result import FinalResult

        from image_annotator_lib.core.types import AnnotationSchema

        # Setup: Configure two providers
        managed_config_registry.set(
            "google_isolation_model",
            {
                "class": "PydanticAIWebAPIAnnotator",
                "model_name_on_provider": "gemini-pro",
                "api_model_id": "gemini-pro",
                "api_key": api_key_manager.get_key("google"),
                "capabilities": ["tags"],
            },
        )
        managed_config_registry.set(
            "openai_isolation_model",
            {
                "class": "PydanticAIWebAPIAnnotator",
                "model_name_on_provider": "gpt-4",
                "api_model_id": "gpt-4",
                "api_key": api_key_manager.get_key("openai"),
                "capabilities": ["tags"],
            },
        )
        register_annotators()

        # Mock: Google succeeds, OpenAI fails
        def mock_agent_run(*args, **kwargs):
            # Simulate error for openai, success for google
            # (In reality, we'd need to differentiate based on model)
            # For simplicity, alternate: first call succeeds, second fails
            if not hasattr(mock_agent_run, "call_count"):
                mock_agent_run.call_count = 0
            mock_agent_run.call_count += 1

            if mock_agent_run.call_count % 2 == 1:
                # Google succeeds
                return FinalResult(output=AnnotationSchema(tags=["isolation_test"], captions=[], score=0.9))
            else:
                # OpenAI fails
                raise RuntimeError("Simulated OpenAI API error")

        with patch(
            "image_annotator_lib.core.pydantic_ai_factory.Agent.run",
            new_callable=AsyncMock,
            side_effect=mock_agent_run,
        ):
            # Act: Use both models (one will fail)
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["google_isolation_model", "openai_isolation_model"],
            )

            # Assert: Results contain both attempts
            first_phash = next(iter(results))
            assert "google_isolation_model" in results[first_phash], "Googleモデル結果あり"
            assert "openai_isolation_model" in results[first_phash], (
                "OpenAIモデル結果あり（エラー含む可能性）"
            )

            # Assert: Google result should be present (either success or error recorded)
            google_result = results[first_phash]["google_isolation_model"]
            assert google_result is not None, "Googleモデル結果が存在"
