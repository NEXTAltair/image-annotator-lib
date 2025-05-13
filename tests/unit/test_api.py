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


@patch("image_annotator_lib.api.get_cls_obj_registry", return_value=MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES)
def test_create_web_api_instance_success(mock_get_cls_reg):
    """Test successful instantiation call for a Web API annotator, patching __init__."""
    model_name_short = "Gemini 1.5 Pro"
    expected_model_name = "Gemini 1.5 Pro"
    RealGoogleAnnotatorClass = MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES[model_name_short]

    with patch.object(RealGoogleAnnotatorClass, "__init__", return_value=None) as mock_init:
        instance = api._create_annotator_instance(model_name_short)

    mock_get_cls_reg.assert_called_once()
    mock_init.assert_called_once_with(model_name=expected_model_name)
    assert isinstance(instance, RealGoogleAnnotatorClass)


@patch("image_annotator_lib.api.get_cls_obj_registry", return_value=MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES)
def test_create_local_model_instance_success(mock_get_cls_reg):
    """Test successful instantiation call for a local model annotator, patching __init__."""
    model_name = "wd-v1-4-convnext-tagger-v2"
    RealLocalAnnotatorClass = MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES[model_name]

    with patch.object(RealLocalAnnotatorClass, "__init__", return_value=None) as mock_init:
        instance = api._create_annotator_instance(model_name)

    mock_get_cls_reg.assert_called_once()
    mock_init.assert_called_once_with(model_name=model_name)
    assert isinstance(instance, RealLocalAnnotatorClass)


@patch("image_annotator_lib.api.get_cls_obj_registry", return_value=MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES)
def test_create_web_api_instance_model_not_found_in_toml(mock_get_cls_reg):
    """Test KeyError when Web API model info is missing in available_api_models.toml."""
    model_name_short = "NonExistentWebModel"

    with pytest.raises(KeyError, match=f"Model '{model_name_short}' not found in class registry."):
        api._create_annotator_instance(model_name_short)

    mock_get_cls_reg.assert_called_once()


def test_get_annotator_instance_caches_instance():
    """Test that get_annotator_instance caches the created instance."""
    # Use a realistic local model name
    model_name = "wd-v1-4-convnext-tagger-v2"
    RealLocalAnnotatorClass = MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES[model_name]

    mock_instance = MagicMock(spec=RealLocalAnnotatorClass)
    with patch(
        "image_annotator_lib.api._create_annotator_instance", return_value=mock_instance
    ) as mock_create:
        instance1 = api.get_annotator_instance(model_name)
        assert instance1 is mock_instance
        mock_create.assert_called_once_with(model_name)
        assert model_name in api._MODEL_INSTANCE_REGISTRY
        assert api._MODEL_INSTANCE_REGISTRY[model_name] is mock_instance

        instance2 = api.get_annotator_instance(model_name)
        assert instance2 is mock_instance
        # _create_annotator_instance should only be called once due to caching
        mock_create.assert_called_once()

    # Fixture handles cleanup
