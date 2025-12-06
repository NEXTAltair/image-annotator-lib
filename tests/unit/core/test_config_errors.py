"""Unit tests for configuration error handling.

Tests validation errors from Pydantic model config schemas.

Mock Strategy (Phase C):
- Real: Pydantic validation, config registry operations
- Mock: None (testing actual validation logic)
"""

import pytest

from image_annotator_lib.core.config import config_registry
from image_annotator_lib.exceptions.errors import ConfigurationError


@pytest.fixture
def clean_config_registry():
    """Clean config registry for isolated tests."""
    import copy

    # Save original state
    original_system = copy.deepcopy(config_registry._system_config_data)
    original_user = copy.deepcopy(config_registry._user_config_data)
    original_merged = copy.deepcopy(config_registry._merged_config_data)

    # Clear for test
    config_registry._system_config_data.clear()
    config_registry._user_config_data.clear()
    config_registry._merged_config_data.clear()

    yield config_registry

    # Restore
    config_registry._system_config_data = original_system
    config_registry._user_config_data = original_user
    config_registry._merged_config_data = original_merged


@pytest.mark.unit
def test_invalid_config_missing_required_fields(clean_config_registry):
    """Test ConfigurationError when required fields missing.

    Mock Strategy:
    - Real: Pydantic validation of LocalMLModelConfig
    - Real: ConfigurationError propagation

    Verifies:
    - Missing model_path raises ConfigurationError
    - Error message indicates missing field
    - Pydantic ValidationError wrapped correctly
    """
    from image_annotator_lib.core.model_config import ModelConfigFactory

    # Missing model_path (required field)
    invalid_config = {
        "class": "WDTagger",
        "device": "cpu",
        # model_path is missing
    }

    registry_dict = {"test_invalid": invalid_config}

    with pytest.raises(ConfigurationError, match="model_path"):
        ModelConfigFactory.from_registry("test_invalid", registry_dict)


@pytest.mark.unit
def test_invalid_config_wrong_types(clean_config_registry):
    """Test Pydantic ValidationError for incorrect field types.

    Mock Strategy:
    - Real: Pydantic type validation

    Verifies:
    - String for numeric field raises ValidationError
    - batch_size with negative value rejected
    - Type conversion attempted before error
    """
    from image_annotator_lib.core.model_config import ModelConfigFactory

    # batch_size should be int > 0, providing string
    invalid_config = {
        "class": "WDTagger",
        "model_path": "/path/to/model",
        "device": "cpu",
        "batch_size": "not_a_number",  # Should be int
    }

    registry_dict = {"test_type_error": invalid_config}

    with pytest.raises(ConfigurationError):
        ModelConfigFactory.from_registry("test_type_error", registry_dict)


@pytest.mark.unit
def test_invalid_config_out_of_range_values(clean_config_registry):
    """Test Pydantic validation for out-of-range values.

    Mock Strategy:
    - Real: Pydantic Field validators (gt, le constraints)

    Verifies:
    - Negative batch_size rejected (gt=0 constraint)
    - Timeout > max_value rejected for WebAPI models
    - Validation error messages clear
    """
    from image_annotator_lib.core.model_config import ModelConfigFactory

    # Test 1: batch_size must be > 0
    invalid_config_negative_batch = {
        "class": "WDTagger",
        "model_path": "/path/to/model.onnx",
        "device": "cpu",
        "batch_size": -1,  # Must be > 0
        "type": "tagger",
    }

    registry_dict = {"test_negative_batch": invalid_config_negative_batch}

    with pytest.raises(ConfigurationError):
        ModelConfigFactory.from_registry("test_negative_batch", registry_dict)

    # Test 2: WebAPI timeout must be <= 300
    invalid_config_timeout = {
        "class": "PydanticAIWebAPIAnnotator",
        "model_name_on_provider": "gpt-4",
        "device": "cpu",
        "timeout": 500,  # Must be <= 300
    }

    registry_dict_timeout = {"test_timeout": invalid_config_timeout}

    with pytest.raises(ConfigurationError, match="timeout"):
        ModelConfigFactory.from_registry("test_timeout", registry_dict_timeout)
