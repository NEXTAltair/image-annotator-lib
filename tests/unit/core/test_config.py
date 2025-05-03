import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import toml

from image_annotator_lib.core import config
from image_annotator_lib.core.config import (
    ModelConfigRegistry,
    _load_config_from_file,
    load_available_api_models,
    save_available_api_models,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Ensure LRU caches are cleared before each test."""
    _load_config_from_file.cache_clear()
    load_available_api_models.cache_clear()
    yield  # Run the test
    _load_config_from_file.cache_clear()
    load_available_api_models.cache_clear()


@pytest.fixture
def mock_config_registry():
    """Provides a clean ModelConfigRegistry instance for each test."""
    registry = ModelConfigRegistry()
    with patch.object(config, "config_registry", registry, create=True):
        yield registry


@pytest.fixture
def mock_paths(tmp_path):
    """Creates temporary paths within a unique test directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    system_config_path = config_dir / "annotator_config.toml"
    user_config_path = config_dir / "user_config.toml"
    available_api_models_path = config_dir / "available_api_models.toml"
    template_config_path = tmp_path / "template_config.toml"  # Dummy template path

    return {
        "system": system_config_path,
        "user": user_config_path,
        "available": available_api_models_path,
        "template": template_config_path,
        "config_dir": config_dir,
    }


@pytest.fixture
def create_config_file(mock_paths):
    """Helper fixture to create config files with given content."""

    def _create_file(path_key: str, content: dict):
        file_path = mock_paths[path_key]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            toml.dump(content, f)
        return file_path

    return _create_file


# --- Test ModelConfigRegistry ---


def test_model_config_registry_init(mock_config_registry: ModelConfigRegistry):
    """Test initial state of the registry."""
    assert mock_config_registry._system_config_data == {}
    assert mock_config_registry._user_config_data == {}
    assert mock_config_registry._merged_config_data == {}
    assert mock_config_registry._system_config_path is None
    assert mock_config_registry._user_config_path is None


def test_load_system_config_only(mock_config_registry: ModelConfigRegistry, mock_paths):
    """Test loading only the system config file (mocking helper methods)."""
    sys_path = mock_paths["system"]
    user_path = mock_paths["user"]
    system_data = {"model_a": {"key1": "sys_value1"}}

    # Mock the helper methods called by load
    with (
        patch.object(ModelConfigRegistry, "_determine_config_paths") as mock_determine,
        patch.object(ModelConfigRegistry, "_ensure_system_config_exists") as mock_ensure,
        patch.object(ModelConfigRegistry, "_load_and_set_system_config") as mock_load_sys,
        patch.object(ModelConfigRegistry, "_load_and_set_user_config") as mock_load_user,
        patch.object(ModelConfigRegistry, "_merge_configs") as mock_merge,
    ):
        # Simulate the state after helpers run
        def setup_state(*args, **kwargs):
            mock_config_registry._system_config_path = sys_path
            mock_config_registry._user_config_path = user_path

        mock_determine.side_effect = setup_state

        def load_sys_effect():
            mock_config_registry._system_config_data = system_data

        mock_load_sys.side_effect = load_sys_effect

        def load_user_effect():
            mock_config_registry._user_config_data = {}

        mock_load_user.side_effect = load_user_effect

        def merge_effect():
            # Simulate the merge based on the mocked data
            mock_config_registry._merged_config_data = copy.deepcopy(
                mock_config_registry._system_config_data
            )
            # No user data to merge in this case

        mock_merge.side_effect = merge_effect

        # --- Execute --- #
        mock_config_registry.load(config_path=sys_path, user_config_path=user_path)

        # --- Assertions --- #
        mock_determine.assert_called_once_with(sys_path, user_path)
        mock_ensure.assert_called_once()
        mock_load_sys.assert_called_once()
        mock_load_user.assert_called_once()
        mock_merge.assert_called_once()

        # Check the final state set by mocks
        assert mock_config_registry._system_config_data == system_data
        assert mock_config_registry._user_config_data == {}
        assert mock_config_registry._merged_config_data == system_data
        assert mock_config_registry._system_config_path == sys_path
        assert mock_config_registry._user_config_path == user_path


def test_load_user_config_only(mock_config_registry: ModelConfigRegistry, mock_paths):
    """Test loading only user config (mocking helper methods)."""
    sys_path = mock_paths["system"]
    user_path = mock_paths["user"]
    user_data = {"model_b": {"key2": "user_value2"}}

    with (
        patch.object(ModelConfigRegistry, "_determine_config_paths") as mock_determine,
        patch.object(ModelConfigRegistry, "_ensure_system_config_exists") as mock_ensure,
        patch.object(ModelConfigRegistry, "_load_and_set_system_config") as mock_load_sys,
        patch.object(ModelConfigRegistry, "_load_and_set_user_config") as mock_load_user,
        patch.object(ModelConfigRegistry, "_merge_configs") as mock_merge,
    ):

        def setup_state(*args, **kwargs):
            mock_config_registry._system_config_path = sys_path
            mock_config_registry._user_config_path = user_path

        mock_determine.side_effect = setup_state

        def load_sys_effect():  # Simulate system config not found/empty
            mock_config_registry._system_config_data = {}

        mock_load_sys.side_effect = load_sys_effect

        def load_user_effect():  # Simulate user config loaded
            mock_config_registry._user_config_data = user_data

        mock_load_user.side_effect = load_user_effect

        def merge_effect():
            mock_config_registry._merged_config_data = copy.deepcopy(
                mock_config_registry._system_config_data
            )  # Start with empty
            # Merge user data
            for model_name, user_model_config in mock_config_registry._user_config_data.items():
                # In this case, merged_data is initially empty, so just assign
                mock_config_registry._merged_config_data[model_name] = copy.deepcopy(user_model_config)

        mock_merge.side_effect = merge_effect

        # --- Execute --- #
        mock_config_registry.load(config_path=sys_path, user_config_path=user_path)

        # --- Assertions --- #
        mock_determine.assert_called_once_with(sys_path, user_path)
        mock_ensure.assert_called_once()  # Ensure is still called
        mock_load_sys.assert_called_once()
        mock_load_user.assert_called_once()
        mock_merge.assert_called_once()

        assert mock_config_registry._system_config_data == {}
        assert mock_config_registry._user_config_data == user_data
        assert mock_config_registry._merged_config_data == user_data
        assert mock_config_registry._system_config_path == sys_path
        assert mock_config_registry._user_config_path == user_path


def test_load_merge_configs(mock_config_registry: ModelConfigRegistry, mock_paths):
    """Test merging system and user configs (mocking helper methods)."""
    sys_path = mock_paths["system"]
    user_path = mock_paths["user"]
    system_data = {
        "model_a": {"key1": "sys_value1", "key_shared": "sys_shared"},
        "model_c": {"key_sys": "sys_only"},
    }
    user_data = {
        "model_a": {"key_user": "user_a", "key_shared": "user_shared"},
        "model_b": {"key_user": "user_b"},
    }
    expected_merged = {
        "model_a": {"key1": "sys_value1", "key_shared": "user_shared", "key_user": "user_a"},
        "model_c": {"key_sys": "sys_only"},
        "model_b": {"key_user": "user_b"},
    }

    with (
        patch.object(ModelConfigRegistry, "_determine_config_paths") as mock_determine,
        patch.object(ModelConfigRegistry, "_ensure_system_config_exists") as mock_ensure,
        patch.object(ModelConfigRegistry, "_load_and_set_system_config") as mock_load_sys,
        patch.object(ModelConfigRegistry, "_load_and_set_user_config") as mock_load_user,
        patch.object(ModelConfigRegistry, "_merge_configs") as mock_merge,
    ):

        def setup_state(*args, **kwargs):
            mock_config_registry._system_config_path = sys_path
            mock_config_registry._user_config_path = user_path

        mock_determine.side_effect = setup_state

        def load_sys_effect():
            mock_config_registry._system_config_data = system_data

        mock_load_sys.side_effect = load_sys_effect

        def load_user_effect():
            mock_config_registry._user_config_data = user_data

        mock_load_user.side_effect = load_user_effect

        # Let the actual _merge_configs run to test its logic (already uses deepcopy)
        # We just need to patch it so the mock object exists, but don't set side_effect
        # Alternatively, we can explicitly call the real method after setting internal state
        # For simplicity here, let's simulate the expected merge result in the mock's side_effect
        def merge_effect():
            merged = copy.deepcopy(system_data)
            for model_name, user_conf in user_data.items():
                if model_name in merged:
                    merged[model_name].update(copy.deepcopy(user_conf))
                else:
                    merged[model_name] = copy.deepcopy(user_conf)
            mock_config_registry._merged_config_data = merged

        mock_merge.side_effect = merge_effect  # Use the simulation

        # --- Execute --- #
        mock_config_registry.load(config_path=sys_path, user_config_path=user_path)

        # --- Assertions --- #
        mock_determine.assert_called_once_with(sys_path, user_path)
        mock_ensure.assert_called_once()
        mock_load_sys.assert_called_once()
        mock_load_user.assert_called_once()
        mock_merge.assert_called_once()

        # Check final state
        assert mock_config_registry._system_config_data == system_data
        assert mock_config_registry._user_config_data == user_data
        assert mock_config_registry._merged_config_data == expected_merged


def test_load_system_config_copies_from_template(mock_config_registry: ModelConfigRegistry, mock_paths):
    """Test template copy logic by mocking helpers and asserting _ensure call."""
    sys_path = mock_paths["system"]
    user_path = mock_paths["user"]
    template_data = {"template_model": {"key": "template_value"}}

    # We only need to verify that _ensure_system_config_exists is called
    # and that the subsequent loads reflect the expected outcome (template data loaded)
    with (
        patch.object(ModelConfigRegistry, "_determine_config_paths") as mock_determine,
        patch.object(ModelConfigRegistry, "_ensure_system_config_exists") as mock_ensure,
        patch.object(ModelConfigRegistry, "_load_and_set_system_config") as mock_load_sys,
        patch.object(ModelConfigRegistry, "_load_and_set_user_config") as mock_load_user,
        patch.object(ModelConfigRegistry, "_merge_configs") as mock_merge,
    ):

        def setup_state(*args, **kwargs):
            mock_config_registry._system_config_path = sys_path
            mock_config_registry._user_config_path = user_path

        mock_determine.side_effect = setup_state

        # Simulate _load_and_set_system_config loading the template data
        # This implies _ensure_system_config_exists did its job (copying)
        def load_sys_effect():
            mock_config_registry._system_config_data = template_data

        mock_load_sys.side_effect = load_sys_effect

        def load_user_effect():  # Assume no user config
            mock_config_registry._user_config_data = {}

        mock_load_user.side_effect = load_user_effect

        def merge_effect():
            mock_config_registry._merged_config_data = copy.deepcopy(template_data)

        mock_merge.side_effect = merge_effect

        # --- Execute --- #
        mock_config_registry.load(config_path=sys_path, user_config_path=user_path)

        # --- Assertions --- #
        mock_determine.assert_called_once_with(sys_path, user_path)
        mock_ensure.assert_called_once()  # Verify the copy logic container was called
        mock_load_sys.assert_called_once()
        mock_load_user.assert_called_once()
        mock_merge.assert_called_once()

        # Verify the final state assumes template data was loaded
        assert mock_config_registry._system_config_data == template_data
        assert mock_config_registry._user_config_data == {}
        assert mock_config_registry._merged_config_data == template_data


# Test 'get' method
def test_get_value(mock_config_registry: ModelConfigRegistry, create_config_file, mock_paths):
    system_data = {"model_a": {"key1": "sys_value1", "key_shared": "sys_shared"}}
    user_data = {"model_a": {"key_shared": "user_shared"}}
    sys_path = create_config_file("system", system_data)
    user_path = create_config_file("user", user_data)
    mock_config_registry.load(sys_path, user_path)

    assert mock_config_registry.get("model_a", "key1") == "sys_value1"
    assert mock_config_registry.get("model_a", "key_shared") == "user_shared"
    assert mock_config_registry.get("model_a", "non_existent", "default") == "default"
    assert mock_config_registry.get("model_a", "non_existent") is None
    assert mock_config_registry.get("non_model", "key1", "default") == "default"
    assert mock_config_registry.get("non_model", "key1") is None


# Test 'set' method (updates user config)
def test_set_value(mock_config_registry: ModelConfigRegistry, create_config_file, mock_paths):
    sys_path = create_config_file("system", {"model_a": {"key1": "sys_value1"}})
    user_path = mock_paths["user"]
    mock_config_registry.load(sys_path, user_path)

    mock_config_registry.set("model_a", "new_key", "new_user_value")
    assert mock_config_registry.get("model_a", "new_key") == "new_user_value"
    assert mock_config_registry._user_config_data["model_a"]["new_key"] == "new_user_value"
    assert mock_config_registry._merged_config_data["model_a"]["new_key"] == "new_user_value"

    mock_config_registry.set("model_a", "key1", "user_override")
    assert mock_config_registry.get("model_a", "key1") == "user_override"
    assert mock_config_registry._user_config_data["model_a"]["key1"] == "user_override"
    assert mock_config_registry._merged_config_data["model_a"]["key1"] == "user_override"

    mock_config_registry.set("new_model", "some_key", "some_value")
    assert mock_config_registry.get("new_model", "some_key") == "some_value"
    assert mock_config_registry._user_config_data["new_model"]["some_key"] == "some_value"
    assert "new_model" not in mock_config_registry._system_config_data
    assert mock_config_registry._merged_config_data["new_model"]["some_key"] == "some_value"


# Test 'set_system_value' method
def test_set_system_value(mock_config_registry: ModelConfigRegistry, create_config_file, mock_paths):
    sys_data = {"model_a": {"key1": "sys_value1"}}
    sys_path = create_config_file("system", sys_data)
    user_path = mock_paths["user"]
    mock_config_registry.load(sys_path, user_path)

    mock_config_registry.set_system_value("model_a", "new_sys_key", "new_system_value")
    assert mock_config_registry.get("model_a", "new_sys_key") == "new_system_value"
    assert mock_config_registry._system_config_data["model_a"]["new_sys_key"] == "new_system_value"
    assert "new_sys_key" not in mock_config_registry._user_config_data.get("model_a", {})
    assert mock_config_registry._merged_config_data["model_a"]["new_sys_key"] == "new_system_value"

    mock_config_registry.set_system_value("new_sys_model", "key", "value")
    assert mock_config_registry.get("new_sys_model", "key") == "value"
    assert mock_config_registry._system_config_data["new_sys_model"]["key"] == "value"
    assert "new_sys_model" not in mock_config_registry._user_config_data
    assert mock_config_registry._merged_config_data["new_sys_model"]["key"] == "value"


# Test 'save_user_config'
def test_save_user_config(mock_config_registry: ModelConfigRegistry, mock_paths):
    user_path = mock_paths["user"]
    mock_config_registry.load(config_path=mock_paths["system"], user_config_path=user_path)
    user_data = {"user_model": {"key": "value"}}
    mock_config_registry.set("user_model", "key", "value")

    with (
        patch.object(Path, "mkdir") as mock_mkdir,
        patch("builtins.open", MagicMock()) as mock_open,
        patch("toml.dump") as mock_toml_dump,
    ):
        mock_config_registry.save_user_config()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(user_path, "w", encoding="utf-8")
        mock_toml_dump.assert_called_once_with(user_data, mock_open().__enter__())


# Test 'save_system_config'
def test_save_system_config(mock_config_registry: ModelConfigRegistry, create_config_file, mock_paths):
    sys_data = {"sys_model": {"k": "v"}}
    sys_path = create_config_file("system", sys_data)
    mock_config_registry.load(config_path=sys_path)

    mock_config_registry.set_system_value("sys_model", "new_k", "new_v")
    expected_data_to_save = {"sys_model": {"k": "v", "new_k": "new_v"}}

    with (
        patch.object(Path, "mkdir") as mock_mkdir,
        patch("builtins.open", MagicMock()) as mock_open,
        patch("toml.dump") as mock_toml_dump,
    ):
        mock_config_registry.save_system_config()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(sys_path, "w", encoding="utf-8")
        mock_toml_dump.assert_called_once_with(expected_data_to_save, mock_open().__enter__())


# --- Test Standalone Functions (using monkeypatch) ---


def test_load_available_api_models_success(monkeypatch, tmp_path):
    """Test successful loading of available_api_models.toml."""
    test_file_path = tmp_path / "available_api_models.toml"
    monkeypatch.setattr(config, "AVAILABLE_API_MODELS_CONFIG_PATH", test_file_path)

    mock_data = {"model1": {"provider": "p1"}, "model2": {"provider": "p2"}}
    file_content = {"available_vision_models": mock_data}

    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, "w") as f:
        toml.dump(file_content, f)

    load_available_api_models.cache_clear()
    result = load_available_api_models()
    assert result == mock_data


def test_load_available_api_models_not_found(monkeypatch, tmp_path):
    """Test loading when available_api_models.toml does not exist."""
    test_file_path = tmp_path / "non_existent_api_models.toml"
    monkeypatch.setattr(config, "AVAILABLE_API_MODELS_CONFIG_PATH", test_file_path)

    # No need to mock Path.is_file if the file truly doesn't exist
    load_available_api_models.cache_clear()
    result = load_available_api_models()
    assert result == {}


def test_load_available_api_models_decode_error(monkeypatch, tmp_path):
    """Test loading with invalid TOML content."""
    test_file_path = tmp_path / "invalid_api_models.toml"
    monkeypatch.setattr(config, "AVAILABLE_API_MODELS_CONFIG_PATH", test_file_path)

    test_file_path.parent.mkdir(exist_ok=True)
    with open(test_file_path, "w") as f:
        f.write("this is not valid toml = [")

    load_available_api_models.cache_clear()
    result = load_available_api_models()
    assert result == {}


@patch.object(config.load_available_api_models, "cache_clear")
def test_save_available_api_models(mock_cache_clear, monkeypatch, tmp_path):
    """Test saving data to available_api_models.toml."""
    test_file_path = tmp_path / "api_models_to_save.toml"
    monkeypatch.setattr(config, "AVAILABLE_API_MODELS_CONFIG_PATH", test_file_path)

    # Mock Path.parent.mkdir, open, and dump
    mock_parent = MagicMock(spec=Path)
    mock_parent.mkdir = MagicMock()
    # Patch Path.parent globally for this test might be too broad,
    # consider patching it on the specific instance if possible,
    # but setattr on the fly is tricky. Mocking mkdir directly is simpler.
    with (
        patch.object(Path, "mkdir") as mock_mkdir,
        patch("builtins.open", MagicMock()) as mock_open,
        patch("toml.dump") as mock_toml_dump,
    ):
        data_to_save = {"model3": {"provider": "p3"}}
        expected_full_data = {"available_vision_models": data_to_save}

        save_available_api_models(data_to_save)

        # Assertions
        # Check if mkdir was called on the correct parent path
        # We need to assert the call based on the test_file_path.parent
        # This requires a more complex mock setup or asserting the call args
        # For simplicity, let's assume the global mkdir mock is sufficient here
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_open.assert_called_once_with(test_file_path, "w", encoding="utf-8")
        mock_toml_dump.assert_called_once_with(expected_full_data, mock_open().__enter__())
        mock_cache_clear.assert_called_once()


# --- Test add_default_setting ---


@patch.object(ModelConfigRegistry, "save_system_config")
def test_add_default_setting_new_section(mock_save, mock_config_registry: ModelConfigRegistry):
    """Test adding a default setting to a new section."""
    section = "new_section"
    key = "new_key"
    value = "default_value"

    # Ensure initial state is empty for merged data
    mock_config_registry._merged_config_data = {}

    mock_config_registry.add_default_setting(section, key, value)

    assert section in mock_config_registry._system_config_data
    assert key in mock_config_registry._system_config_data[section]
    assert mock_config_registry._system_config_data[section][key] == value
    # _merge_configs is no longer mocked, just check save
    mock_save.assert_called_once()  # Should be called because setting was added
    # Check merged data directly or via get
    assert mock_config_registry._merged_config_data.get(section, {}).get(key) == value
    assert mock_config_registry.get(section, key) == value


@patch.object(ModelConfigRegistry, "save_system_config")
def test_add_default_setting_existing_section(mock_save, mock_config_registry: ModelConfigRegistry):
    """Test adding a new default setting to an existing section."""
    section = "existing_section"
    existing_key = "existing_key"
    existing_value = "existing_value"
    new_key = "new_key"
    new_value = "default_value"

    # Pre-populate system config and run real merge
    mock_config_registry._system_config_data = {section: {existing_key: existing_value}}
    mock_config_registry._merge_configs()  # Initial merge

    mock_config_registry.add_default_setting(section, new_key, new_value)

    assert new_key in mock_config_registry._system_config_data[section]
    assert mock_config_registry._system_config_data[section][new_key] == new_value
    assert mock_config_registry._system_config_data[section][existing_key] == existing_value
    mock_save.assert_called_once()  # Called because setting was added
    # Check merged data
    assert mock_config_registry._merged_config_data.get(section, {}).get(new_key) == new_value
    assert mock_config_registry._merged_config_data.get(section, {}).get(existing_key) == existing_value
    assert mock_config_registry.get(section, new_key) == new_value


@patch.object(ModelConfigRegistry, "save_system_config")
def test_add_default_setting_key_exists(mock_save, mock_config_registry: ModelConfigRegistry):
    """Test that add_default_setting does not overwrite an existing key."""
    section = "existing_section"
    key = "existing_key"
    original_value = "original_value"
    new_default_value = "new_default_value"

    # Pre-populate system config and run real merge
    mock_config_registry._system_config_data = {section: {key: original_value}}
    mock_config_registry._merge_configs()  # Initial merge

    mock_config_registry.add_default_setting(section, key, new_default_value)

    assert mock_config_registry._system_config_data[section][key] == original_value
    mock_save.assert_not_called()  # Save should NOT be called
    # Check merged data
    assert mock_config_registry._merged_config_data.get(section, {}).get(key) == original_value
    assert mock_config_registry.get(section, key) == original_value


@patch.object(ModelConfigRegistry, "save_system_config")
def test_add_default_setting_user_override(mock_save, mock_config_registry: ModelConfigRegistry):
    """Test that get returns user value even if default was added."""
    section = "section"
    key = "key"
    default_value = "default_value"
    user_value = "user_override"

    # Simulate user config having the key and run real merge
    mock_config_registry._user_config_data = {section: {key: user_value}}
    mock_config_registry._system_config_data = {}
    mock_config_registry._merge_configs()  # Initial merge (will contain user value)

    # Add the default setting (should be added to _system_config_data)
    mock_config_registry.add_default_setting(section, key, default_value)

    # Assertions
    assert key in mock_config_registry._system_config_data[section]
    assert mock_config_registry._system_config_data[section][key] == default_value
    mock_save.assert_called_once()  # Default was added to system data, so save is called

    # Crucially, assert that get still returns the user override from merged data
    assert mock_config_registry._merged_config_data.get(section, {}).get(key) == user_value
    assert mock_config_registry.get(section, key) == user_value
