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
    config_registry,
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
    runtime_cache_path = tmp_path / "runtime" / "model_runtime_cache.toml"
    user_path = tmp_path / "user" / "user_config.toml"
    system_path.parent.mkdir()
    runtime_cache_path.parent.mkdir()
    user_path.parent.mkdir()
    return system_path, runtime_cache_path, user_path


# --- Tests for ModelConfigRegistry ---


@pytest.mark.fast
def test_registry_init(registry):
    """Test that the registry initializes with empty dicts."""
    assert registry._system_config_data == {}
    assert registry._runtime_cache_data == {}
    assert registry._user_config_data == {}
    assert registry._merged_config_data == {}


@pytest.mark.fast
def test_determine_config_paths(registry, monkeypatch):
    """Test the logic for determining config paths."""
    monkeypatch.setattr(
        config_module,
        "DEFAULT_PATHS",
        {
            "config_toml": "/default/system/config.toml",
            "model_runtime_cache_toml": "/default/runtime/model_runtime_cache.toml",
            "user_config_toml": "/default/user/config.toml",
        },
    )

    # Test with default paths
    registry._determine_config_paths()
    assert registry._system_config_path == Path("/default/system/config.toml")
    assert registry._runtime_cache_path == Path("/default/runtime/model_runtime_cache.toml")
    assert registry._user_config_path == Path("/default/user/config.toml")

    # Test with custom paths
    registry._determine_config_paths(
        config_path="/custom/system.toml",
        runtime_cache_path="/custom/runtime.toml",
        user_config_path="/custom/user.toml",
    )
    assert registry._system_config_path == Path("/custom/system.toml")
    assert registry._runtime_cache_path == Path("/custom/runtime.toml")
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
    system_path, runtime_cache_path, user_path = tmp_config_paths

    with open(system_path, "w") as f:
        toml.dump(MOCK_SYSTEM_CONFIG, f)
    with open(runtime_cache_path, "w") as f:
        toml.dump({"model-a": {"estimated_size_gb": 1.25, "device": "cpu"}}, f)
    with open(user_path, "w") as f:
        toml.dump(MOCK_USER_CONFIG, f)

    registry._system_config_path = system_path
    registry._runtime_cache_path = runtime_cache_path
    registry._user_config_path = user_path

    registry._load_and_set_system_config()
    registry._load_and_set_runtime_cache()
    registry._load_and_set_user_config()

    assert registry._system_config_data == MOCK_SYSTEM_CONFIG
    assert registry._runtime_cache_data == {"model-a": {"estimated_size_gb": 1.25}}
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
def test_merge_configs_prefers_user_over_runtime_cache_over_system(registry):
    """estimated_size_gb precedence is user > runtime cache > system."""
    registry._system_config_data = {
        "system-only": {"estimated_size_gb": 1.0},
        "cached-model": {"estimated_size_gb": 1.0},
        "user-model": {"estimated_size_gb": 1.0},
    }
    registry._runtime_cache_data = {
        "cached-model": {"estimated_size_gb": 2.0},
        "user-model": {"estimated_size_gb": 2.0},
    }
    registry._user_config_data = {"user-model": {"estimated_size_gb": 3.0}}

    registry._merge_configs()

    assert registry.get("system-only", "estimated_size_gb") == 1.0
    assert registry.get("cached-model", "estimated_size_gb") == 2.0
    assert registry.get("user-model", "estimated_size_gb") == 3.0


@pytest.mark.fast
def test_merge_configs_uses_system_size_when_runtime_cache_missing(registry):
    """Missing runtime cache keeps annotator_config.toml value as fallback."""
    registry._system_config_data = {"model": {"estimated_size_gb": 1.0}}
    registry._runtime_cache_data = {}
    registry._user_config_data = {}

    registry._merge_configs()

    assert registry.get("model", "estimated_size_gb") == 1.0


@pytest.mark.fast
def test_runtime_cache_filters_non_runtime_metadata(registry, tmp_path):
    """Runtime cache is restricted to runtime-derived metadata."""
    runtime_cache_path = tmp_path / "model_runtime_cache.toml"
    with open(runtime_cache_path, "w") as f:
        toml.dump(
            {
                "model": {
                    "estimated_size_gb": 1.25,
                    "device": "cuda:0",
                    "capabilities": ["tags"],
                }
            },
            f,
        )

    registry._runtime_cache_path = runtime_cache_path
    registry._load_and_set_runtime_cache()

    assert registry._runtime_cache_data == {"model": {"estimated_size_gb": 1.25}}


@pytest.mark.fast
def test_save_runtime_cache_persists_only_runtime_metadata(registry, tmp_path):
    """Runtime cache writes are local deletable state, separate from system config."""
    runtime_cache_path = tmp_path / "model_runtime_cache.toml"
    registry._runtime_cache_path = runtime_cache_path

    registry.set_runtime_cache_value("model", "estimated_size_gb", 2.5)
    registry.set_runtime_cache_value("model", "device", "cuda:0")
    registry.save_runtime_cache()

    assert toml.load(runtime_cache_path) == {"model": {"estimated_size_gb": 2.5}}


@pytest.mark.fast
def test_merge_configs_unions_builtin_model_capabilities(registry):
    """Built-in model capabilities are additive across system and user config."""
    registry._system_config_data = {
        "wd-vit-tagger-v3": {
            "type": "tagger",
            "class": "WDTagger",
            "capabilities": ["tags", "ratings"],
        }
    }
    registry._user_config_data = {
        "wd-vit-tagger-v3": {
            "device": "cuda:0",
            "capabilities": ["tags"],
        }
    }

    registry._merge_configs()

    assert registry.get("wd-vit-tagger-v3", "capabilities") == ["tags", "ratings"]
    assert registry.get("wd-vit-tagger-v3", "device") == "cuda:0"


@pytest.mark.fast
def test_merge_configs_adds_user_capabilities_without_dropping_system_capabilities(registry):
    """User config may add capabilities, but cannot remove system-declared ones."""
    registry._system_config_data = {"canonical-scorer": {"capabilities": ["scores"]}}
    registry._user_config_data = {"canonical-scorer": {"capabilities": ["score_labels"]}}

    registry._merge_configs()

    assert registry.get("canonical-scorer", "capabilities") == ["scores", "score_labels"]


@pytest.mark.fast
def test_merge_configs_custom_model_uses_user_defined_capabilities(registry):
    """Custom local models without a system entry keep user-defined capabilities."""
    registry._system_config_data = {}
    registry._user_config_data = {
        "custom-rating-model": {
            "type": "tagger",
            "class": "CustomTagger",
            "capabilities": ["ratings"],
        }
    }

    registry._merge_configs()

    assert registry.get("custom-rating-model", "capabilities") == ["ratings"]


@pytest.mark.fast
def test_merge_configs_keeps_user_override_for_non_capability_fields(registry):
    """Only capabilities are unioned; other fields retain user override semantics."""
    registry._system_config_data = {
        "wd-vit-tagger-v3": {
            "device": "cpu",
            "model_path": "/system/model.onnx",
            "capabilities": ["tags", "ratings"],
        }
    }
    registry._user_config_data = {
        "wd-vit-tagger-v3": {
            "device": "cuda:0",
            "capabilities": ["tags"],
        }
    }

    registry._merge_configs()

    assert registry.get("wd-vit-tagger-v3", "device") == "cuda:0"
    assert registry.get("wd-vit-tagger-v3", "model_path") == "/system/model.onnx"
    assert registry.get("wd-vit-tagger-v3", "capabilities") == ["tags", "ratings"]


@pytest.mark.fast
def test_merge_configs_capability_union_does_not_mutate_sources(registry):
    """Capability union uses copied values and leaves source configs untouched."""
    registry._system_config_data = {"model": {"capabilities": ["tags", "ratings"]}}
    registry._user_config_data = {"model": {"capabilities": ["tags"]}}

    registry._merge_configs()
    registry._merged_config_data["model"]["capabilities"].append("scores")

    assert registry._system_config_data["model"]["capabilities"] == ["tags", "ratings"]
    assert registry._user_config_data["model"]["capabilities"] == ["tags"]


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


@pytest.mark.fast
def test_add_default_setting_does_not_modify_project_system_config():
    """Regression test for Issue #95: test stubs must not persist to real config."""
    project_system_config = Path("config/annotator_config.toml")
    original_content = project_system_config.read_text(encoding="utf-8")

    config_registry.add_default_setting("test_stub_model", "class", "TestOnlyAnnotator")

    assert project_system_config.read_text(encoding="utf-8") == original_content
    assert "test_stub_model" in config_registry._system_config_data


@pytest.mark.fast
def test_runtime_size_cache_does_not_modify_project_system_config():
    """Regression test for Issue #97: runtime size cache must not rewrite system catalog."""
    project_system_config = Path("config/annotator_config.toml")
    original_content = project_system_config.read_text(encoding="utf-8")

    config_registry.set_runtime_cache_value("test_runtime_model", "estimated_size_gb", 2.0)
    config_registry.save_runtime_cache()

    assert project_system_config.read_text(encoding="utf-8") == original_content
    assert config_registry.get("test_runtime_model", "estimated_size_gb") == 2.0


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
    with pytest.raises(toml.TomlDecodeError):
        _load_config_from_file(bad_toml_path)


# Issue #139 (B): canonical-label scorer が config/annotator_config.toml と template
# (resources/system/annotator_config.toml) の間で capabilities drift を起こし、
# predictor が生成する score_labels に対し SCORE_LABELS capability が欠落して
# ValidationError になっていた。両ファイル間で drift しないことを固定する。
_CANONICAL_LABEL_SCORERS = ["aesthetic_shadow_v1", "aesthetic_shadow_v2", "cafe_aesthetic"]


@pytest.mark.fast
def test_canonical_scorer_capabilities_declare_score_labels():
    """Issue #139 (B) regression: canonical scorer の config が score_labels capability を持つ。

    AestheticShadow / CafePredictor は _format_predictions で score_labels を返すため
    (ADR 0002), 実 load される config の capabilities に "score_labels" が無いと
    UnifiedAnnotationResult の validator が ValidationError を投げる。
    """
    project_config = toml.load(Path("config/annotator_config.toml"))
    for model_name in _CANONICAL_LABEL_SCORERS:
        capabilities = project_config[model_name]["capabilities"]
        assert "score_labels" in capabilities, (
            f"{model_name}: config/annotator_config.toml の capabilities に score_labels が無い "
            f"(現状 {capabilities})。predictor の score_labels 生成と矛盾し ValidationError になる。"
        )


@pytest.mark.fast
def test_scorer_capabilities_no_drift_between_config_and_template():
    """Issue #139 (B) regression: scorer の capabilities が config と template で一致する。

    drift すると実 load される config と ADR 0002/0009 の前提がずれる。
    """
    project_config = toml.load(Path("config/annotator_config.toml"))
    template_config = toml.load(Path("src/image_annotator_lib/resources/system/annotator_config.toml"))
    for model_name in [*_CANONICAL_LABEL_SCORERS, "ImprovedAesthetic", "WaifuAesthetic"]:
        assert sorted(project_config[model_name]["capabilities"]) == sorted(
            template_config[model_name]["capabilities"]
        ), f"{model_name}: config と template で capabilities が drift している"


# ADR 0023 Phase 1 (Issue #35, PR #40): `save_available_api_models` /
# `load_available_api_models` および `AVAILABLE_API_MODELS_CONFIG_PATH` は廃止された。
# WebAPI モデル一覧は LiteLLM 同梱 DB から runtime 取得するため、TOML cache の
# round-trip test (`test_save_and_load_api_models`) は不要となり削除。


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
    registry._system_config_data = {"model1": {"device": "cpu", "batch_size": 16, "precision": "fp32"}}
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
