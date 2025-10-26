from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib import api

# Import the real classes for the registry
from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.model_class.tagger_onnx import WDTagger

# --- Mock Data ---

# Mock class registry using REAL classes
MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES = {
    "Gemini 1.5 Pro": GoogleApiAnnotator,
    "GPT-4o": OpenAIApiAnnotator,
    "Claude 3 Opus": AnthropicApiAnnotator,
    "Gemini 1.5 Flash (OpenRouter)": OpenRouterApiAnnotator,
    "wd-v1-4-convnext-tagger-v2": WDTagger,  # Use a real local model name
}

# Mock data as returned by load_available_api_models using realistic values
MOCK_AVAILABLE_API_MODELS = {
    "google/gemini-1.5-pro-latest": {"provider": "google", "model_name_short": "Gemini 1.5 Pro"},
    "openai/gpt-4o": {"provider": "openai", "model_name_short": "GPT-4o"},
    "anthropic/claude-3-opus-20240229": {"provider": "anthropic", "model_name_short": "Claude 3 Opus"},
    "google/gemini-flash-1.5": {"provider": "google", "model_name_short": "Gemini 1.5 Flash (OpenRouter)"},
}

# --- Fixtures ---


@pytest.fixture(autouse=True)
def clear_instance_registry():  # Renamed fixture, removed mock resets
    """Clear the internal instance registry before each test."""
    api._MODEL_INSTANCE_REGISTRY.clear()
    yield
    api._MODEL_INSTANCE_REGISTRY.clear()


# --- Test Cases ---


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_web_api_instance_success(mock_find_model, mock_get_factory):
    """Test successful instantiation for a Web API annotator."""
    model_name_short = "Gemini 1.5 Pro"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return GoogleApiAnnotator
    mock_find_model.return_value = (model_name_short, GoogleApiAnnotator)

    instance = api._create_annotator_instance(model_name_short)

    # Test actual behavior
    assert isinstance(instance, api.PydanticAIWebAPIWrapper)
    assert instance.model_name == model_name_short


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_local_model_instance_success(mock_find_model, mock_get_factory):
    """Test successful instantiation for a local model annotator."""
    model_name = "wd-v1-4-convnext-tagger-v2"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return WDTagger
    mock_find_model.return_value = (model_name, WDTagger)

    with patch.object(WDTagger, "__init__", return_value=None):
        instance = api._create_annotator_instance(model_name)

    # Test actual behavior
    assert isinstance(instance, WDTagger)


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_web_api_instance_model_not_found_in_toml(mock_find_model, mock_get_factory):
    """Test KeyError when Web API model info is missing in available_api_models.toml."""
    model_name_short = "NonExistentWebModel"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_factory.get_available_models.return_value = []
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return None (not found)
    mock_find_model.return_value = None

    # Test actual error behavior
    with pytest.raises(KeyError, match=f"Model '{model_name_short}' not found in registry or available models."):
        api._create_annotator_instance(model_name_short)


@pytest.mark.unit
def test_get_annotator_instance_caches_instance():
    """Test that get_annotator_instance caches the created instance."""
    # Use a realistic local model name
    model_name = "wd-v1-4-convnext-tagger-v2"
    RealLocalAnnotatorClass = MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES[model_name]

    mock_instance = MagicMock(spec=RealLocalAnnotatorClass)
    with patch("image_annotator_lib.api._create_annotator_instance", return_value=mock_instance):
        # Test caching behavior through actual state
        instance1 = api.get_annotator_instance(model_name)
        assert instance1 is mock_instance
        assert model_name in api._MODEL_INSTANCE_REGISTRY
        assert api._MODEL_INSTANCE_REGISTRY[model_name] is mock_instance

        # Second call should return cached instance
        instance2 = api.get_annotator_instance(model_name)
        assert instance2 is mock_instance
        assert instance1 is instance2  # Same cached object

    # Fixture handles cleanup


@pytest.mark.unit
@patch("image_annotator_lib.core.simplified_agent_wrapper.SimplifiedAgentWrapper")
@patch("image_annotator_lib.api.get_agent_factory")
def test_create_annotator_with_simplified_agent_wrapper(mock_get_factory, mock_wrapper):
    """Test creating annotator using SimplifiedAgentWrapper for direct model_id."""
    model_id = "google/gemini-2.5-pro-preview-03-25"

    # Mock agent factory to return True (is a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = True
    mock_get_factory.return_value = mock_factory

    instance = api._create_annotator_instance(model_id)

    # Verify SimplifiedAgentWrapper was called
    mock_wrapper.assert_called_once_with(model_id, api_keys=None)
    assert instance == mock_wrapper.return_value


@pytest.mark.unit
@patch("image_annotator_lib.core.simplified_agent_wrapper.SimplifiedAgentWrapper")
@patch("image_annotator_lib.api.get_agent_factory")
def test_get_annotator_instance_with_api_keys_no_cache(mock_get_factory, mock_wrapper):
    """Test that get_annotator_instance doesn't use cache when api_keys are provided."""
    model_name = "test-model"
    api_keys = {"openai": "sk-test123"}

    # Mock agent factory
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = True
    mock_get_factory.return_value = mock_factory

    # First call with api_keys
    instance1 = api.get_annotator_instance(model_name, api_keys=api_keys)
    # Second call with api_keys
    instance2 = api.get_annotator_instance(model_name, api_keys=api_keys)

    # Should create new instances each time (no caching)
    assert mock_wrapper.call_count == 2
    assert model_name not in api._MODEL_INSTANCE_REGISTRY


@pytest.mark.unit
@patch("image_annotator_lib.core.config.config_registry")
def test_pydanticai_wrapper_predict_without_api_model_id(mock_config):
    """Test PydanticAIWebAPIWrapper.predict raises error when api_model_id is not configured."""
    from PIL import Image

    # Mock config to return None for api_model_id
    mock_config.get.return_value = None

    wrapper = api.PydanticAIWebAPIWrapper("test-model", MagicMock())

    # Enter context to trigger config loading
    with wrapper:
        # Test that predict raises ValueError
        with pytest.raises(ValueError, match="has no api_model_id configured"):
            wrapper.predict([Image.new("RGB", (100, 100))], ["test_phash"])


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
@patch("image_annotator_lib.core.config.config_registry")
def test_pydanticai_wrapper_predict_handles_provider_error(mock_config, mock_get_capabilities, mock_run_inference):
    """Test PydanticAIWebAPIWrapper.predict handles ProviderManager errors gracefully."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability

    # Mock config
    mock_config.get.return_value = "test/model-id"

    # Mock capabilities
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    # Mock ProviderManager to raise exception
    mock_run_inference.side_effect = Exception("API Error")

    wrapper = api.PydanticAIWebAPIWrapper("test-model", MagicMock())

    test_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    phash_list = ["phash1", "phash2"]

    with wrapper:
        results = wrapper.predict(test_images, phash_list)

        # Should return error results for all images
        assert len(results) == 2
        assert all(result.error is not None for result in results)
        assert all("Failed to run inference" in result.error for result in results)
