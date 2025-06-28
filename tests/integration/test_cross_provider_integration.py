# tests/integration/test_cross_provider_integration.py
import pytest
from unittest.mock import patch, call
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.api import annotate

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

    @patch("image_annotator_lib.api._is_pydantic_ai_webapi_annotator", return_value=True)
    @patch("image_annotator_lib.core.provider_manager.GoogleProviderInstance.run_with_model")
    @patch("image_annotator_lib.core.provider_manager.OpenAIProviderInstance.run_with_model")
    @patch("image_annotator_lib.api.get_cls_obj_registry")
    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_simultaneous_multi_provider_usage(
        self, mock_get_cls_registry, mock_openai_run, mock_google_run, mock_is_webapi, managed_config_registry, lightweight_test_images
    ):
        """
        Tests the system's ability to handle simultaneous requests to different providers.
        """
        # Mock the class registry to include our test models
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
        
        mock_get_cls_registry.return_value = {
            "google_model": GoogleApiAnnotator,
            "openai_model": OpenAIApiAnnotator,
        }
        
        # Mock the return values to match the number of images
        num_images = len(lightweight_test_images)
        mock_google_run.return_value = [{"provider": "google", "tags": ["g-tag"], "response": {"tags": ["g-tag"]}}] * num_images
        mock_openai_run.return_value = [{"provider": "openai", "tags": ["o-tag"], "response": {"tags": ["o-tag"]}}] * num_images

        # Setup configs
        managed_config_registry.set("google_model", {"provider": "google", "api_model_id": "gemini-pro"})
        managed_config_registry.set("openai_model", {"provider": "openai", "api_model_id": "gpt-4"})

        # Annotate with both models
        results = annotate(
            images_list=lightweight_test_images,
            model_name_list=["google_model", "openai_model"],
        )

        # Verify that the correct provider instances were called
        mock_google_run.assert_called_once()
        mock_openai_run.assert_called_once()

        # Verify results are aggregated
        assert len(results.keys()) == len(lightweight_test_images)
        
        # Check that results for each model are present for the first image
        first_phash = next(iter(results))
        assert "google_model" in results[first_phash]
        assert "openai_model" in results[first_phash]
        assert results[first_phash]["google_model"]["tags"] == ["g-tag"]
        assert results[first_phash]["openai_model"]["tags"] == ["o-tag"]


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