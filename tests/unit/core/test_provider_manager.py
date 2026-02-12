"""Plan 1: Simplified ProviderManager unit tests.

Tests for the simplified ProviderManager that delegates
all model management to PydanticAIAgentFactory.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    ProviderManager.clear_cache()
    yield
    ProviderManager.clear_cache()


# ========================================================================
# _get_provider tests
# ========================================================================


@pytest.mark.unit
class TestGetProvider:
    """Tests for ProviderManager._get_provider()."""

    def test_colon_prefix_extracts_provider(self):
        """Model ID with colon prefix returns the prefix as provider."""
        assert ProviderManager._get_provider("openrouter:meta-llama") == "openrouter"
        assert ProviderManager._get_provider("openai:gpt-4") == "openai"

    def test_gpt_model_returns_openai(self):
        assert ProviderManager._get_provider("gpt-4-turbo") == "openai"
        assert ProviderManager._get_provider("gpt-4o") == "openai"

    def test_o1_o3_model_returns_openai(self):
        assert ProviderManager._get_provider("o1-preview") == "openai"
        assert ProviderManager._get_provider("o3-mini") == "openai"

    def test_claude_model_returns_anthropic(self):
        assert ProviderManager._get_provider("claude-3-opus") == "anthropic"
        assert ProviderManager._get_provider("claude-3-5-sonnet") == "anthropic"

    def test_gemini_model_returns_google(self):
        assert ProviderManager._get_provider("gemini-1.5-pro") == "google"
        assert ProviderManager._get_provider("gemini-2.0-flash") == "google"

    def test_unknown_model_returns_unknown(self):
        assert ProviderManager._get_provider("custom-model-xyz") == "unknown"
        assert ProviderManager._get_provider("llama-3") == "unknown"


# ========================================================================
# _get_api_key tests
# ========================================================================


@pytest.mark.unit
class TestGetApiKey:
    """Tests for ProviderManager._get_api_key()."""

    def test_injected_api_keys_preferred(self):
        """Injected API keys take precedence over config."""
        api_keys = {"anthropic": "injected_key"}
        result = ProviderManager._get_api_key("test_model", "claude-3-opus", api_keys)
        assert result == "injected_key"

    def test_openrouter_prefix_routes_to_openrouter_key(self):
        """OpenRouter model ID routes to 'openrouter' key in api_keys."""
        api_keys = {"openrouter": "or_key"}
        result = ProviderManager._get_api_key("test_model", "openrouter:meta-llama", api_keys)
        assert result == "or_key"

    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_fallback_to_config_registry(self, mock_config):
        """Falls back to config_registry when no injected keys."""
        mock_config.get.return_value = "config_api_key"
        result = ProviderManager._get_api_key("test_model", "gpt-4", None)
        assert result == "config_api_key"
        mock_config.get.assert_called_once_with("test_model", "api_key", default="")

    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_returns_none_when_no_key_found(self, mock_config):
        """Returns None when no API key is configured."""
        mock_config.get.return_value = ""
        result = ProviderManager._get_api_key("test_model", "gpt-4", None)
        assert result is None


# ========================================================================
# run_inference_with_model tests
# ========================================================================


@pytest.mark.unit
class TestRunInferenceWithModel:
    """Tests for ProviderManager.run_inference_with_model()."""

    @patch("image_annotator_lib.core.provider_manager.calculate_phash")
    @patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_successful_single_image_inference(
        self, mock_config, mock_factory, mock_preprocess, mock_phash
    ):
        """Single image inference returns correct AnnotationResult."""
        from image_annotator_lib.core.types import AnnotationSchema

        mock_config.get.return_value = "test_api_key"
        mock_phash.return_value = "phash_abc"

        mock_binary = MagicMock()
        mock_preprocess.return_value = [mock_binary]

        mock_agent = MagicMock()
        mock_factory.get_or_create_agent.return_value = mock_agent

        mock_response = MagicMock()
        mock_response.data = AnnotationSchema(tags=["tag1", "tag2"], captions=["test"], score=0.9)
        mock_agent.run_sync.return_value = mock_response

        test_image = Image.new("RGB", (100, 100))
        results = ProviderManager.run_inference_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="claude-3-opus",
        )

        assert len(results) == 1
        assert "phash_abc" in results
        result = results["phash_abc"]
        assert result["tags"] == ["tag1", "tag2"]
        assert result["error"] is None

    @patch("image_annotator_lib.core.provider_manager.calculate_phash")
    @patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_empty_response_returns_error(
        self, mock_config, mock_factory, mock_preprocess, mock_phash
    ):
        """Empty response returns error in AnnotationResult."""
        mock_config.get.return_value = "test_api_key"
        mock_phash.return_value = "phash_empty"

        mock_binary = MagicMock()
        mock_preprocess.return_value = [mock_binary]

        mock_agent = MagicMock()
        mock_factory.get_or_create_agent.return_value = mock_agent

        mock_response = MagicMock()
        mock_response.data = None
        mock_agent.run_sync.return_value = mock_response

        test_image = Image.new("RGB", (100, 100))
        results = ProviderManager.run_inference_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="gpt-4",
        )

        assert len(results) == 1
        result = results["phash_empty"]
        assert result["error"] == "Empty response from API"
        assert result["tags"] == []

    @patch("image_annotator_lib.core.provider_manager.calculate_phash")
    @patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_inference_exception_returns_error_result(
        self, mock_config, mock_factory, mock_preprocess, mock_phash
    ):
        """Exception during inference returns error result per image."""
        mock_config.get.return_value = "test_api_key"
        mock_phash.return_value = "phash_err"

        mock_binary = MagicMock()
        mock_preprocess.return_value = [mock_binary]

        mock_agent = MagicMock()
        mock_factory.get_or_create_agent.return_value = mock_agent
        mock_agent.run_sync.side_effect = RuntimeError("API timeout")

        test_image = Image.new("RGB", (100, 100))
        results = ProviderManager.run_inference_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="gpt-4",
        )

        assert len(results) == 1
        result = results["phash_err"]
        assert "API timeout" in result["error"]
        assert result["tags"] == []

    @patch("image_annotator_lib.core.provider_manager.calculate_phash")
    @patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_multiple_images_returns_all_results(
        self, mock_config, mock_factory, mock_preprocess, mock_phash
    ):
        """Multiple images each get separate results keyed by phash."""
        from image_annotator_lib.core.types import AnnotationSchema

        mock_config.get.return_value = "key"
        mock_phash.side_effect = ["p1", "p2", "p3"]

        mock_binaries = [MagicMock(), MagicMock(), MagicMock()]
        mock_preprocess.return_value = mock_binaries

        mock_agent = MagicMock()
        mock_factory.get_or_create_agent.return_value = mock_agent

        responses = []
        for tags in [["a"], ["b"], ["c"]]:
            resp = MagicMock()
            resp.data = AnnotationSchema(tags=tags, captions=[], score=0.8)
            responses.append(resp)
        mock_agent.run_sync.side_effect = responses

        test_images = [Image.new("RGB", (50, 50)) for _ in range(3)]
        results = ProviderManager.run_inference_with_model(
            model_name="m", images_list=test_images, api_model_id="gpt-4"
        )

        assert len(results) == 3
        assert results["p1"]["tags"] == ["a"]
        assert results["p2"]["tags"] == ["b"]
        assert results["p3"]["tags"] == ["c"]

    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    @patch("image_annotator_lib.core.provider_manager.config_registry")
    def test_factory_error_raises_webapi_error(self, mock_config, mock_factory):
        """Factory failure raises WebApiError."""
        from image_annotator_lib.exceptions.errors import WebApiError

        mock_config.get.return_value = "key"
        mock_factory.get_or_create_agent.side_effect = ValueError("Bad model")

        test_image = Image.new("RGB", (50, 50))
        with pytest.raises(WebApiError, match="Inference failed"):
            ProviderManager.run_inference_with_model(
                model_name="m", images_list=[test_image], api_model_id="bad"
            )

    @patch("image_annotator_lib.core.provider_manager.calculate_phash")
    @patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    def test_api_keys_injection_passes_to_factory(
        self, mock_factory, mock_preprocess, mock_phash
    ):
        """Injected api_keys are resolved and passed to the factory."""
        from image_annotator_lib.core.types import AnnotationSchema

        mock_phash.return_value = "ph"
        mock_preprocess.return_value = [MagicMock()]

        mock_agent = MagicMock()
        mock_factory.get_or_create_agent.return_value = mock_agent
        resp = MagicMock()
        resp.data = AnnotationSchema(tags=["ok"], captions=[], score=0.9)
        mock_agent.run_sync.return_value = resp

        test_image = Image.new("RGB", (50, 50))
        ProviderManager.run_inference_with_model(
            model_name="m",
            images_list=[test_image],
            api_model_id="claude-3-opus",
            api_keys={"anthropic": "injected_key_123"},
        )

        # Verify the factory was called with the injected key
        mock_factory.get_or_create_agent.assert_called_once_with("claude-3-opus", "injected_key_123")


# ========================================================================
# clear_cache tests
# ========================================================================


@pytest.mark.unit
class TestClearCache:
    """Tests for ProviderManager.clear_cache()."""

    @patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
    def test_clear_cache_delegates_to_factory(self, mock_factory):
        """clear_cache calls factory's clear_cache."""
        ProviderManager.clear_cache()
        mock_factory.clear_cache.assert_called_once()
