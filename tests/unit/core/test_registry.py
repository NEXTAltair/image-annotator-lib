from unittest.mock import MagicMock, call, patch

import pytest

# Assuming registry.py is in src.image_annotator_lib.core
from image_annotator_lib.core import api_model_discovery, config, registry
from image_annotator_lib.core.base import BaseAnnotator
from image_annotator_lib.core.config import ModelConfigRegistry  # Import for mocking


# Mock annotator classes for testing _find_annotator_class_by_provider
class GoogleApiAnnotator(BaseAnnotator):
    pass


class OpenAIApiAnnotator(BaseAnnotator):
    pass


class AnthropicApiAnnotator(BaseAnnotator):
    pass


class OpenRouterApiAnnotator(BaseAnnotator):
    pass


class SomeOtherAnnotator(BaseAnnotator):
    pass


# Sample available classes for testing provider mapping
MOCK_AVAILABLE_CLASSES = {
    "GoogleApiAnnotator": GoogleApiAnnotator,
    "OpenAIApiAnnotator": OpenAIApiAnnotator,
    "AnthropicApiAnnotator": AnthropicApiAnnotator,
    "OpenRouterApiAnnotator": OpenRouterApiAnnotator,
    "SomeOtherAnnotator": SomeOtherAnnotator,
}

# Sample API model data for testing config update
MOCK_API_MODELS = {
    "google/gemini-pro-1.5": {
        "provider": "google",
        "model_name_short": "Gemini 1.5 Pro",
        # ... other keys
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "model_name_short": "GPT-4o",
        # ... other keys
    },
    "unknown/some-model": {
        "provider": "some-unknown-provider",
        "model_name_short": "Unknown Model",
        # ... other keys
    },
    "anthropic/claude-3-sonnet": {
        # Missing provider to test skipping
        "model_name_short": "Claude 3 Sonnet",
    },
    "invalid_format_model": "this_is_not_a_dict",  # Test invalid format
}


@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._update_config_with_api_models")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")  # Mock logger init
def test_initialize_registry_api_models_file_not_exists_calls_fetch_and_update(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_update_config,
    mock_register,
):
    """Test initialize_registry calls API fetch and config update when file not exists."""
    mock_config_path.exists.return_value = False

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_called_once()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._update_config_with_api_models")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_api_models_file_exists_skips_fetch(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_update_config,
    mock_register,
):
    """Test initialize_registry skips API fetch but calls config update when file exists."""
    mock_config_path.exists.return_value = True

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_not_called()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._update_config_with_api_models")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_continues_if_fetch_api_fails(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_update_config,
    mock_register,
):
    """Test initialize_registry continues process even if API fetch fails."""
    mock_config_path.exists.return_value = False
    mock_fetch_api_models.side_effect = Exception("API Error")

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_called_once()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


# --- Tests for _update_config_with_api_models ---


@patch("image_annotator_lib.core.config.config_registry.add_default_setting")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_success(
    mock_load_api_models,
    mock_gather_classes,
    mock_add_setting,
):
    """Test _update_config_with_api_models successfully calls add_default_setting."""
    mock_load_api_models.return_value = MOCK_API_MODELS
    mock_gather_classes.return_value = MOCK_AVAILABLE_CLASSES

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")

    expected_calls = [
        # Google model -> GoogleApiAnnotator
        call("Gemini 1.5 Pro", "class", "GoogleApiAnnotator"),
        call("Gemini 1.5 Pro", "max_output_tokens", 1800),
        # OpenAI model -> OpenAIApiAnnotator
        call("GPT-4o", "class", "OpenAIApiAnnotator"),
        call("GPT-4o", "max_output_tokens", 1800),
        # Unknown provider -> OpenRouterApiAnnotator (fallback based on corrected logic)
        call("Unknown Model", "class", "OpenRouterApiAnnotator"),
        call("Unknown Model", "max_output_tokens", 1800),
        # Missing provider model is skipped, invalid format is skipped
    ]
    # Use assert_has_calls with any_order=True as dict iteration order isn't guaranteed
    mock_add_setting.assert_has_calls(expected_calls, any_order=True)
    # Check total calls, excluding skipped models
    assert mock_add_setting.call_count == len(expected_calls)


@patch("image_annotator_lib.core.config.config_registry.add_default_setting")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_no_api_data(
    mock_load_api_models,
    mock_gather_classes,
    mock_add_setting,
):
    """Test _update_config_with_api_models when no API models are loaded."""
    mock_load_api_models.return_value = {}

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_not_called()  # Should not gather if no models
    mock_add_setting.assert_not_called()  # Should not add settings


@patch("image_annotator_lib.core.config.config_registry.add_default_setting")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_no_classes(
    mock_load_api_models,
    mock_gather_classes,
    mock_add_setting,
):
    """Test _update_config_with_api_models when no annotator classes are found."""
    mock_load_api_models.return_value = MOCK_API_MODELS
    mock_gather_classes.return_value = {}  # Simulate no classes found

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")

    # Expect calls, but all classes should fallback to OpenRouterApiAnnotator
    expected_calls = [
        call("Gemini 1.5 Pro", "class", "OpenRouterApiAnnotator"),
        call("Gemini 1.5 Pro", "max_output_tokens", 1800),
        call("GPT-4o", "class", "OpenRouterApiAnnotator"),
        call("GPT-4o", "max_output_tokens", 1800),
        call("Unknown Model", "class", "OpenRouterApiAnnotator"),
        call("Unknown Model", "max_output_tokens", 1800),
    ]
    mock_add_setting.assert_has_calls(expected_calls, any_order=True)
    assert mock_add_setting.call_count == len(expected_calls)


# --- Tests for _find_annotator_class_by_provider ---


@pytest.mark.parametrize(
    "provider_name, expected_class_name",
    [
        ("google", "GoogleApiAnnotator"),
        ("GOOGLE", "GoogleApiAnnotator"),  # Case-insensitive
        ("openai", "OpenAIApiAnnotator"),
        ("Anthropic", "AnthropicApiAnnotator"),
        ("some_other", "OpenRouterApiAnnotator"),
        ("unknown", "OpenRouterApiAnnotator"),
        ("router", "OpenRouterApiAnnotator"),
        ("no_match_provider", "OpenRouterApiAnnotator"),
    ],
)
@patch("image_annotator_lib.core.registry.logger")  # Mock logger to suppress warnings
def test_find_annotator_class_by_provider(mock_logger, provider_name, expected_class_name):
    """Test the provider name to class name mapping logic based on corrected spec."""
    result = registry._find_annotator_class_by_provider(provider_name, MOCK_AVAILABLE_CLASSES)
    assert result == expected_class_name


@patch("image_annotator_lib.core.registry.logger")  # Mock logger
def test_find_annotator_class_by_provider_no_available_classes(mock_logger):
    """Test fallback when available_classes is empty."""
    result = registry._find_annotator_class_by_provider("google", {})  # Pass empty dict
    assert result == "OpenRouterApiAnnotator"
    # Check that a warning was logged because the specific provider 'google' was expected
    # but no classes were available to match against.
    mock_logger.warning.assert_called_once()
