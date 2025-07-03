from unittest.mock import call, patch

import pytest

# Assuming registry.py is in src.image_annotator_lib.core
from image_annotator_lib.core import registry
from image_annotator_lib.core.base.annotator import BaseAnnotator


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


@pytest.mark.unit
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
    # Reset singleton state before test
    registry._REGISTRY_INITIALIZED = False
    
    mock_config_path.exists.return_value = False

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_called_once()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


@pytest.mark.unit
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
    # Reset singleton state before test
    registry._REGISTRY_INITIALIZED = False
    
    mock_config_path.exists.return_value = True

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_not_called()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


@pytest.mark.unit
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
    # Reset singleton state before test
    registry._REGISTRY_INITIALIZED = False
    
    mock_config_path.exists.return_value = False
    mock_fetch_api_models.side_effect = Exception("API Error")

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    mock_config_path.exists.assert_called_once()
    mock_fetch_api_models.assert_called_once()
    mock_update_config.assert_called_once()
    mock_register.assert_called_once()


# --- Tests for _update_config_with_api_models ---


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.config_registry")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_success(
    mock_load_api_models,
    mock_gather_classes,
    mock_config_registry,
):
    """Test _update_config_with_api_models successfully calls add_default_setting."""
    mock_load_api_models.return_value = MOCK_API_MODELS
    mock_gather_classes.return_value = MOCK_AVAILABLE_CLASSES

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")

    expected_calls = [
        # PydanticAI統一実装: 全WebAPIモデルはPydanticAIWebAPIAnnotatorを使用
        call("Gemini 1.5 Pro", "class", "PydanticAIWebAPIAnnotator"),
        call("Gemini 1.5 Pro", "api_model_id", "google/gemini-pro-1.5"),
        call("Gemini 1.5 Pro", "max_output_tokens", 1800),
        call("GPT-4o", "class", "PydanticAIWebAPIAnnotator"),
        call("GPT-4o", "api_model_id", "openai/gpt-4o"),
        call("GPT-4o", "max_output_tokens", 1800),
        call("Unknown Model", "class", "PydanticAIWebAPIAnnotator"),
        call("Unknown Model", "api_model_id", "unknown/some-model"),
        call("Unknown Model", "max_output_tokens", 1800),
        # Missing provider model is skipped, invalid format is skipped
    ]
    # Use assert_has_calls with any_order=True as dict iteration order isn't guaranteed
    mock_config_registry.add_default_setting.assert_has_calls(expected_calls, any_order=True)
    # Check total calls, excluding skipped models
    assert mock_config_registry.add_default_setting.call_count == len(expected_calls)


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.config_registry")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_no_api_data(
    mock_load_api_models,
    mock_gather_classes,
    mock_config_registry,
):
    """Test _update_config_with_api_models when no API models are loaded."""
    mock_load_api_models.return_value = {}

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_not_called()  # Should not gather if no models
    mock_config_registry.add_default_setting.assert_not_called()  # Should not add settings


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.config_registry")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_update_config_with_api_models_no_classes(
    mock_load_api_models,
    mock_gather_classes,
    mock_config_registry,
):
    """Test _update_config_with_api_models when no annotator classes are found."""
    mock_load_api_models.return_value = MOCK_API_MODELS
    mock_gather_classes.return_value = {}  # Simulate no classes found

    registry._update_config_with_api_models()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")

    # PydanticAI統一実装: クラスが見つからない場合でもPydanticAIWebAPIAnnotatorを使用
    expected_calls = [
        call("Gemini 1.5 Pro", "class", "PydanticAIWebAPIAnnotator"),
        call("Gemini 1.5 Pro", "api_model_id", "google/gemini-pro-1.5"),
        call("Gemini 1.5 Pro", "max_output_tokens", 1800),
        call("GPT-4o", "class", "PydanticAIWebAPIAnnotator"),
        call("GPT-4o", "api_model_id", "openai/gpt-4o"),
        call("GPT-4o", "max_output_tokens", 1800),
        call("Unknown Model", "class", "PydanticAIWebAPIAnnotator"),
        call("Unknown Model", "api_model_id", "unknown/some-model"),
        call("Unknown Model", "max_output_tokens", 1800),
    ]
    mock_config_registry.add_default_setting.assert_has_calls(expected_calls, any_order=True)
    assert mock_config_registry.add_default_setting.call_count == len(expected_calls)


# --- _find_annotator_class_by_provider tests removed ---
# PydanticAI統一実装では、この関数は常にPydanticAIWebAPIAnnotatorを返すだけの
# 単純な関数になったため、テストは不要になりました。


# --- Test for singleton pattern ---


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._update_config_with_api_models")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_singleton_pattern(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_update_config,
    mock_register,
):
    """Test initialize_registry uses singleton pattern and only initializes once."""
    # Reset singleton state before test
    registry._REGISTRY_INITIALIZED = False
    
    mock_config_path.exists.return_value = True

    # First call should execute all initialization steps
    registry.initialize_registry()
    
    # Verify first call executed all steps
    assert mock_init_logger.call_count == 1
    assert mock_config_path.exists.call_count == 1
    assert mock_fetch_api_models.call_count == 0  # Should skip when file exists
    assert mock_update_config.call_count == 1
    assert mock_register.call_count == 1
    
    # Second call should skip all initialization due to singleton pattern
    registry.initialize_registry()
    
    # Verify second call did NOT execute any additional steps
    assert mock_init_logger.call_count == 2  # init_logger always runs
    assert mock_config_path.exists.call_count == 1  # Should not check again
    assert mock_fetch_api_models.call_count == 0  # Still 0
    assert mock_update_config.call_count == 1  # Still 1
    assert mock_register.call_count == 1  # Still 1
    
    # Third call should also skip
    registry.initialize_registry()
    
    # Verify third call also skipped
    assert mock_init_logger.call_count == 3  # Only this increases
    assert mock_config_path.exists.call_count == 1  # Still 1
    assert mock_fetch_api_models.call_count == 0  # Still 0
    assert mock_update_config.call_count == 1  # Still 1
    assert mock_register.call_count == 1  # Still 1
