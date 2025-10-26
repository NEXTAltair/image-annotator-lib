from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.exceptions.errors import WebApiError


@pytest.fixture(autouse=True)
def clear_provider_instances():
    """Clear provider instances before each test."""
    ProviderManager._provider_instances.clear()
    yield
    ProviderManager._provider_instances.clear()


@pytest.mark.unit
def test_get_provider_instance_creates_anthropic_instance():
    """Test creating a new Anthropic provider instance."""
    instance = ProviderManager.get_provider_instance("anthropic")

    assert instance is not None
    assert "anthropic" in ProviderManager._provider_instances
    assert ProviderManager._provider_instances["anthropic"] is instance


@pytest.mark.unit
def test_get_provider_instance_creates_openai_instance():
    """Test creating a new OpenAI provider instance."""
    instance = ProviderManager.get_provider_instance("openai")

    assert instance is not None
    assert "openai" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_creates_google_instance():
    """Test creating a new Google provider instance."""
    instance = ProviderManager.get_provider_instance("google")

    assert instance is not None
    assert "google" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_creates_openrouter_instance():
    """Test creating a new OpenRouter provider instance."""
    instance = ProviderManager.get_provider_instance("openrouter")

    assert instance is not None
    assert "openrouter" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_returns_cached_instance():
    """Test that get_provider_instance returns cached instance."""
    instance1 = ProviderManager.get_provider_instance("anthropic")
    instance2 = ProviderManager.get_provider_instance("anthropic")

    assert instance1 is instance2


@pytest.mark.unit
def test_get_provider_instance_unsupported_provider_raises_error():
    """Test that unsupported provider raises WebApiError."""
    with pytest.raises(WebApiError, match="Unsupported provider: unknown_provider"):
        ProviderManager.get_provider_instance("unknown_provider")


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_explicit_config(mock_config):
    """Test provider determination from explicit configuration."""
    mock_config.get.return_value = "google"

    provider = ProviderManager._determine_provider("test_model", "any_model_id")

    assert provider == "google"
    mock_config.get.assert_called_once_with("test_model", "provider")


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_id_prefix(mock_config):
    """Test provider determination from model ID prefix."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "openai:gpt-4")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_gpt(mock_config):
    """Test provider determination from GPT model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "gpt-4-turbo")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_claude(mock_config):
    """Test provider determination from Claude model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "claude-3-opus")

    assert provider == "anthropic"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_gemini(mock_config):
    """Test provider determination from Gemini model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "gemini-1.5-pro")

    assert provider == "google"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_fallback_to_api_key(mock_config):
    """Test provider determination fallback to API key configuration."""

    def config_get_side_effect(model_name, key, default=None):
        if key == "provider":
            return None
        elif key == "google_api_key":
            return "test_key"
        return default

    mock_config.get.side_effect = config_get_side_effect

    provider = ProviderManager._determine_provider("test_model", "custom_model")

    assert provider == "google"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_default_fallback(mock_config):
    """Test provider determination default fallback to OpenAI."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "unknown_model")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_converts_results_to_phash_dict(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model converts results to phash-based dict."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock successful response
    from image_annotator_lib.core.types import AnnotationSchema

    mock_response = AnnotationSchema(tags=["test_tag"], captions=["test caption"], score=0.9)
    mock_provider_instance.run_with_model.return_value = [{"response": mock_response}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["phash"] == "test_phash_123"
    assert results["test_phash_123"]["tags"] == ["test_tag"]
    assert results["test_phash_123"]["error"] is None


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_handles_error_response(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model handles error responses."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock error response
    mock_provider_instance.run_with_model.return_value = [{"error": "API Error"}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["error"] == "API Error"
    assert results["test_phash_123"]["tags"] == []


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_handles_no_response(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model handles missing response."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock response with no "response" key
    mock_provider_instance.run_with_model.return_value = [{}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["error"] == "No response from provider"
    assert results["test_phash_123"]["tags"] == []
