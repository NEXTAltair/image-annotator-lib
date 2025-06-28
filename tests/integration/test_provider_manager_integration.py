# tests/integration/test_provider_manager_integration.py
import pytest
from image_annotator_lib.core.provider_manager import ProviderManager

# A placeholder for checking API key availability
def api_key_available(provider: str):
    """Placeholder function to check if an API key is available."""
    return False

class TestProviderManagerIntegration:
    """
    Integration tests for the Provider Manager's core logic.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.parametrize(
        "model_name, api_model_id, config, expected_provider",
        [
            # 1. Explicit provider in config
            ("test_model_explicit", "any_id", {"provider": "google"}, "google"),
            ("test_model_explicit_case", "any_id", {"provider": "Anthropic"}, "anthropic"),
            # 2. Provider from model_id prefix
            ("test_model_prefix", "openai:gpt-4", {}, "openai"),
            ("test_model_prefix_edge", "google:gemini-pro", {"provider": "anthropic"}, "anthropic"), # explicit config wins
            # 3. Provider from model name pattern
            ("test_model_pattern", "gpt-4-turbo", {}, "openai"),
            ("test_model_pattern", "claude-3-opus-20240229", {}, "anthropic"),
            ("test_model_pattern", "gemini-1.5-pro-latest", {}, "google"),
            # 4. Fallback to API key in config
            ("test_model_fallback", "some_custom_model", {"google_api_key": "key"}, "google"),
            ("test_model_fallback", "some_custom_model", {"anthropic_api_key": "key"}, "anthropic"),
            ("test_model_fallback", "some_custom_model", {"openai_api_key": "key"}, "openai"),
            # 5. Default fallback
            ("test_model_default", "unknown_model", {}, "openai"),
        ],
    )
    def test_provider_determination_integration(
        self, managed_config_registry, model_name, api_model_id, config, expected_provider
    ):
        """
        Tests provider determination with actual configurations and model IDs.
        """
        # Setup config for the test case
        managed_config_registry.set(model_name, config)

        # Run the determination logic
        determined_provider = ProviderManager._determine_provider(model_name, api_model_id)

        # Assert the result
        assert determined_provider == expected_provider

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing_integration(self):
        """
        Tests that requests for different models using the same provider
        share the same underlying provider instance.
        """
        # Clear any cached instances before the test
        ProviderManager._provider_instances.clear()

        # Get instance for the first google model
        instance1 = ProviderManager.get_provider_instance("google")

        # Get instance for the second google model
        instance2 = ProviderManager.get_provider_instance("google")

        # Get a different provider instance
        instance3 = ProviderManager.get_provider_instance("openai")

        assert id(instance1) == id(instance2)
        assert id(instance1) != id(instance3)
        assert "google" in ProviderManager._provider_instances
        assert "openai" in ProviderManager._provider_instances
        assert len(ProviderManager._provider_instances) == 2

        # Cleanup after test
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.real_api
    @pytest.mark.skipif(not api_key_available("any"), reason="API key required for real API tests")
    def test_provider_instance_sharing_real_api(self, api_key_manager):
        """
        Tests provider instance sharing using real API calls.
        """
        pytest.skip("Test not yet implemented.")