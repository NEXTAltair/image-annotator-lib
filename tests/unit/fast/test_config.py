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
