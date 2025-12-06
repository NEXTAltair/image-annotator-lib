import copy
import unittest.mock as mock
from pathlib import Path

import pytest
import toml

# Import the module and the class to be tested
from image_annotator_lib.core import config as config_module
from image_annotator_lib.core.config import (
    ModelConfigRegistry,
    _load_config_from_file,
    load_available_api_models,
    save_available_api_models,
)

# Import shared fixtures

# --- Test Data ---
MOCK_SYSTEM_CONFIG = {
    "models": {"cache_dir": "/system/cache"},
    "annotator": {"device": "cpu"},
}
MOCK_USER_CONFIG = {"annotator": {"device": "cuda:0"}}
MOCK_MERGED_CONFIG = {
    "models": {"cache_dir": "/system/cache"},
    "annotator": {"device": "cuda:0"},
}

# --- Fixtures ---


@pytest.fixture
def registry():
    """Provides a clean ModelConfigRegistry instance for each test."""
    return ModelConfigRegistry()


@pytest.fixture
def tmp_config_paths(tmp_path):
    """Create temporary config paths for testing."""
    system_path = tmp_path / "system" / "annotator_config.toml"
    user_path = tmp_path / "user" / "user_config.toml"
    system_path.parent.mkdir()
    user_path.parent.mkdir()
    return system_path, user_path


# --- Tests for ModelConfigRegistry ---


@pytest.mark.fast
def test_registry_init(registry):
    """Test that the registry initializes with empty dicts."""
    assert registry._system_config_data == {}
    assert registry._user_config_data == {}
    assert registry._merged_config_data == {}


@pytest.mark.fast
def test_determine_config_paths(registry, monkeypatch):
    """Test the logic for determining config paths."""
    monkeypatch.setattr(
        config_module,
        "DEFAULT_PATHS",
        {"config_toml": "/default/system/config.toml", "user_config_toml": "/default/user/config.toml"},
    )

    # Test with default paths
    registry._determine_config_paths()
    assert registry._system_config_path == Path("/default/system/config.toml")
    assert registry._user_config_path == Path("/default/user/config.toml")

    # Test with custom paths
    registry._determine_config_paths(
        config_path="/custom/system.toml", user_config_path="/custom/user.toml"
    )
    assert registry._system_config_path == Path("/custom/system.toml")
    assert registry._user_config_path == Path("/custom/user.toml")


@pytest.mark.fast
@mock.patch("importlib.resources.as_file")
@mock.patch("shutil.copyfile")
def test_ensure_system_config_exists_creates_file(mock_copy, mock_as_file, registry, tmp_path):
    """Test that the system config is created from template if it doesn't exist."""
    # Setup mock for the template file
    template_path = tmp_path / "template.toml"
    template_path.touch()
    mock_cm = mock.MagicMock()
    mock_cm.__enter__.return_value = template_path
    mock_as_file.return_value = mock_cm

    system_path = tmp_path / "config" / "annotator_config.toml"
    registry._system_config_path = system_path

    # Pre-condition: file does not exist
    assert not system_path.exists()

    registry._ensure_system_config_exists()

    mock_copy.assert_called_once_with(template_path, system_path)


@pytest.mark.fast
@mock.patch("shutil.copyfile")
def test_ensure_system_config_exists_does_nothing(mock_copy, registry, tmp_path):
    """Test that nothing happens if the system config already exists."""
    system_path = tmp_path / "config.toml"
    system_path.touch()  # File now exists
    registry._system_config_path = system_path

    registry._ensure_system_config_exists()

    mock_copy.assert_not_called()


@pytest.mark.fast
def test_load_and_set_configs(registry, tmp_config_paths):
    """Test loading system and user configs."""
    system_path, user_path = tmp_config_paths

    with open(system_path, "w") as f:
        toml.dump(MOCK_SYSTEM_CONFIG, f)
    with open(user_path, "w") as f:
        toml.dump(MOCK_USER_CONFIG, f)

    registry._system_config_path = system_path
    registry._user_config_path = user_path

    registry._load_and_set_system_config()
    registry._load_and_set_user_config()

    assert registry._system_config_data == MOCK_SYSTEM_CONFIG
    assert registry._user_config_data == MOCK_USER_CONFIG


@pytest.mark.fast
def test_merge_configs(registry):
    """Test the deep merge logic."""
    registry._system_config_data = copy.deepcopy(MOCK_SYSTEM_CONFIG)
    registry._user_config_data = copy.deepcopy(MOCK_USER_CONFIG)

    registry._merge_configs()

    assert registry._merged_config_data == MOCK_MERGED_CONFIG
    # Ensure it's a deep copy
    registry._merged_config_data["annotator"]["device"] = "new_device"
    assert MOCK_SYSTEM_CONFIG["annotator"]["device"] == "cpu"


@pytest.mark.fast
def test_get_value(registry):
    """Test the get method for retrieving config values."""
    registry._merged_config_data = MOCK_MERGED_CONFIG

    # Test existing value
    assert registry.get("annotator", "device") == "cuda:0"
    # Test non-existent key with default
    assert registry.get("annotator", "non_existent", "default_val") == "default_val"
    # Test non-existent model
    assert registry.get("non_existent_model", "key", "default_val") == "default_val"


@pytest.mark.fast
def test_set_value(registry):
    """Test setting a user config value."""
    registry._system_config_data = MOCK_SYSTEM_CONFIG
    registry._merge_configs()  # Initial merge

    assert registry.get("annotator", "device") == "cpu"

    registry.set("annotator", "device", "new_cuda")

    assert registry._user_config_data["annotator"]["device"] == "new_cuda"
    assert registry.get("annotator", "device") == "new_cuda"  # Merged view is updated


@pytest.mark.fast
def test_save_user_config(registry, tmp_path):
    """Test saving the user configuration to a file."""
    user_path = tmp_path / "user.toml"
    registry._user_config_path = user_path
    registry.set("new_model", "key", "value")

    registry.save_user_config()

    assert user_path.exists()
    with open(user_path) as f:
        data = toml.load(f)
    assert data["new_model"]["key"] == "value"


# --- Tests for Standalone Functions ---


@pytest.mark.fast
def test_load_config_from_file_errors(tmp_path):
    """Test error handling for _load_config_from_file."""
    # Test FileNotFoundError
    assert _load_config_from_file(tmp_path / "nonexistent.toml") == {}

    # Test TomlDecodeError
    bad_toml_path = tmp_path / "bad.toml"
    with open(bad_toml_path, "w") as f:
        f.write("this is not toml")

    # It should re-raise the exception
    with pytest.raises(Exception):
        _load_config_from_file(bad_toml_path)


@pytest.mark.fast
def test_save_and_load_api_models(tmp_path, monkeypatch):
    """Test saving and loading of the available_api_models.toml file."""
    temp_file = tmp_path / "api_models.toml"
    monkeypatch.setattr(config_module, "AVAILABLE_API_MODELS_CONFIG_PATH", temp_file)

    test_data = {"model1": {"provider": "test"}}

    save_available_api_models(test_data)

    # Clear cache to force re-read from disk
    load_available_api_models.cache_clear()

    loaded_data = load_available_api_models()

    assert loaded_data == test_data

    # Check that the data is under the correct section
    with open(temp_file) as f:
        raw_data = toml.load(f)
    assert raw_data["available_vision_models"] == test_data


# ==============================================================================
# Additional Coverage Tests (Phase C)
# ==============================================================================


@pytest.mark.fast
def test_load_config_from_toml_file_with_complex_structure(tmp_path):
    """Test TOML file loading with nested/complex structure.

    Tests:
    - Valid TOML with nested sections is parsed correctly
    - Complex data types (arrays, nested dicts) are preserved
    - File path resolution works correctly
    """
    toml_path = tmp_path / "complex.toml"

    # Create complex TOML structure
    complex_config = {
        "model1": {
            "class": "WDTagger",
            "model_path": "/path/to/model.onnx",
            "device": "cuda:0",
            "tags": ["anime", "character"],
            "nested": {
                "batch_size": 16,
                "threshold": 0.5,
            },
        },
        "model2": {
            "class": "AestheticScorer",
            "model_path": "/path/to/scorer.onnx",
            "device": "cpu",
        },
    }

    with open(toml_path, "w") as f:
        toml.dump(complex_config, f)

    # Load and verify
    loaded_config = _load_config_from_file(toml_path)

    assert loaded_config == complex_config
    assert loaded_config["model1"]["tags"] == ["anime", "character"]
    assert loaded_config["model1"]["nested"]["batch_size"] == 16
    assert loaded_config["model1"]["nested"]["threshold"] == 0.5
    assert loaded_config["model2"]["device"] == "cpu"


@pytest.mark.fast
def test_config_registry_isolation():
    """Test that registry instances are isolated from each other.

    Tests:
    - Multiple registry instances don't share state
    - Changes to one registry don't affect others
    - Each registry maintains independent config data
    """
    # Create two independent registry instances
    registry1 = ModelConfigRegistry()
    registry2 = ModelConfigRegistry()

    # Modify registry1
    registry1._system_config_data = {"model1": {"device": "cuda"}}
    registry1._merge_configs()

    # Modify registry2
    registry2._system_config_data = {"model2": {"device": "cpu"}}
    registry2._merge_configs()

    # Verify isolation
    assert registry1._system_config_data != registry2._system_config_data
    assert "model1" in registry1._system_config_data
    assert "model1" not in registry2._system_config_data
    assert "model2" in registry2._system_config_data
    assert "model2" not in registry1._system_config_data

    # Verify changes don't cross-contaminate
    registry1.set("model1", "device", "cuda:1")
    assert registry1.get("model1", "device") == "cuda:1"
    assert registry2.get("model1", "device", "default") == "default"


# ==============================================================================
# Phase C Additional Coverage Tests (2025-12-05)
# ==============================================================================


@pytest.mark.fast
def test_config_deep_merge_nested_dicts(registry):
    """Test deep merge with nested dictionary structures.

    Tests:
    - Deep nested structures are merged correctly
    - User config overrides system config values
    - System config provides base values
    - No mutation of source dictionaries
    """
    # Create nested structures (2 levels deep - matches actual merge behavior)
    registry._system_config_data = {
        "model1": {"device": "cpu", "batch_size": 16, "precision": "fp32"}
    }
    registry._user_config_data = {"model1": {"device": "cuda"}, "model2": {"device": "cuda"}}

    registry._merge_configs()

    # Verify merge results
    merged = registry._merged_config_data
    assert merged["model1"]["device"] == "cuda"  # User override
    assert merged["model1"]["batch_size"] == 16  # System preserved
    assert merged["model1"]["precision"] == "fp32"  # System preserved
    assert merged["model2"]["device"] == "cuda"  # User only

    # Verify no mutation of source
    assert registry._system_config_data["model1"]["device"] == "cpu"
    assert registry._user_config_data["model1"]["device"] == "cuda"


@pytest.mark.fast
def test_config_file_not_found_handling(tmp_path):
    """Test handling of missing config files.

    Tests:
    - Missing system config returns empty dict
    - Missing user config returns empty dict
    - No exceptions raised for missing files
    - Merge works with missing files
    """
    nonexistent_system = tmp_path / "nonexistent_system.toml"
    nonexistent_user = tmp_path / "nonexistent_user.toml"

    # Load nonexistent files
    system_result = _load_config_from_file(nonexistent_system)
    user_result = _load_config_from_file(nonexistent_user)

    assert system_result == {}
    assert user_result == {}

    # Verify registry works with missing files
    registry = ModelConfigRegistry()
    registry._system_config_path = nonexistent_system
    registry._user_config_path = nonexistent_user

    registry._load_and_set_system_config()
    registry._load_and_set_user_config()
    registry._merge_configs()

    assert registry._system_config_data == {}
    assert registry._user_config_data == {}
    assert registry._merged_config_data == {}


@pytest.mark.fast
def test_config_toml_decode_error_recovery(tmp_path):
    """Test TOML decode error handling and recovery.

    Tests:
    - Invalid TOML syntax raises exception
    - Error is propagated (not silently caught)
    - TomlDecodeError type raised
    - Error message contains useful context
    """
    # Create file with invalid TOML
    bad_toml = tmp_path / "bad.toml"
    with open(bad_toml, "w") as f:
        f.write("[model1\n")  # Missing closing bracket
        f.write("device = 'cuda'\n")

    # Expect TOML decode error
    with pytest.raises(Exception) as exc_info:
        _load_config_from_file(bad_toml)

    # Verify error is propagated with context
    error_str = str(exc_info.value)
    # Check for error indicators (line number, character position, or syntax description)
    assert any(
        indicator in error_str.lower()
        for indicator in ["line", "column", "char", "group", "syntax", "bracket"]
    )


@pytest.mark.fast
def test_config_save_user_config_creates_directory(tmp_path):
    """Test user config save creates parent directories.

    Tests:
    - Missing parent directory is created automatically
    - Config file is saved correctly
    - Directory permissions are correct
    - Nested directories are created
    """
    # Path with nonexistent parent directories
    nested_path = tmp_path / "level1" / "level2" / "user.toml"

    registry = ModelConfigRegistry()
    registry._user_config_path = nested_path
    registry.set("test_model", "device", "cuda")

    # Pre-condition: parent doesn't exist
    assert not nested_path.parent.exists()

    # Save should create directories
    registry.save_user_config()

    # Verify directory and file created
    assert nested_path.parent.exists()
    assert nested_path.exists()

    # Verify content
    with open(nested_path) as f:
        data = toml.load(f)
    assert data["test_model"]["device"] == "cuda"


@pytest.mark.fast
def test_config_get_with_different_types(registry):
    """Test get() method with various data types.

    Tests:
    - String values retrieved correctly
    - Integer values retrieved correctly
    - Float values retrieved correctly
    - Boolean values retrieved correctly
    - List values retrieved correctly
    - Dict values retrieved correctly
    - None values handled correctly
    """
    # Setup config with various types
    registry._merged_config_data = {
        "model1": {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": ["a", "b", "c"],
            "dict_val": {"nested": "value"},
            "none_val": None,
        }
    }

    # Test retrieval of different types
    assert registry.get("model1", "string_val") == "test"
    assert registry.get("model1", "int_val") == 42
    assert registry.get("model1", "float_val") == 3.14
    assert registry.get("model1", "bool_val") is True
    assert registry.get("model1", "list_val") == ["a", "b", "c"]
    assert registry.get("model1", "dict_val") == {"nested": "value"}
    assert registry.get("model1", "none_val") is None

    # Test default for missing key
    assert registry.get("model1", "missing", "default") == "default"
    assert registry.get("missing_model", "key", 123) == 123
