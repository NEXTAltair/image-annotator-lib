"""Mutable model preferences and runtime state belong to one OS user."""

import importlib.util
from pathlib import Path

import pytest
import toml
from platformdirs import user_cache_path, user_config_path, user_log_path

from image_annotator_lib.core import config as config_module
from image_annotator_lib.core import constants
from image_annotator_lib.core.config import ModelConfigRegistry, _load_config_from_file

pytestmark = pytest.mark.fast


def load_paths(monkeypatch, tmp_path, environment):
    """Load the path module without reinitializing the global model registry."""
    for kind in ("CONFIG", "CACHE", "LOG"):
        monkeypatch.setenv(f"IMAGE_ANNOTATOR_LIB_{kind}_HOME", str(tmp_path / environment / kind.lower()))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("isolated_model_paths", constants.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_paths_are_outside_shared_checkout(monkeypatch):
    for kind in ("CONFIG", "CACHE", "LOG"):
        monkeypatch.delenv(f"IMAGE_ANNOTATOR_LIB_{kind}_HOME", raising=False)
    spec = importlib.util.spec_from_file_location("default_model_paths", constants.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.USER_CONFIG_PATH == user_config_path("image-annotator-lib", appauthor=False) / (
        "user_config.toml"
    )
    assert module.MODEL_RUNTIME_CACHE_PATH == user_cache_path("image-annotator-lib", appauthor=False) / (
        "model_runtime_cache.toml"
    )
    assert module.LOG_DIR == user_log_path("image-annotator-lib", appauthor=False)
    assert module.SYSTEM_CONFIG_PATH == Path.cwd() / "config" / "annotator_config.toml"
    assert module.CACHE_DIR == Path.cwd() / "models"
    assert not module.USER_CONFIG_PATH.is_relative_to(module.PROJECT_ROOT)
    assert not module.MODEL_RUNTIME_CACHE_PATH.is_relative_to(module.PROJECT_ROOT)


def test_each_environment_preserves_other_override_cache_and_shared_catalog(monkeypatch, tmp_path):
    shared = tmp_path / "config"
    shared.mkdir(exist_ok=True)
    system = shared / "annotator_config.toml"
    system.write_text('[local]\nmodel_path = "portable-model"\nestimated_size_gb = 1.0\n')
    # Legacy machine overrides/cache remain untouched and are never loaded.
    legacy_user = shared / "user_config.toml"
    legacy_user.write_text('[local]\nmodel_path = "H:/legacy/foreign-model"\n')
    legacy_cache = shared / "model_runtime_cache.toml"
    legacy_cache.write_text("[local]\nestimated_size_gb = 99.0\n")
    before = {path: path.read_bytes() for path in (system, legacy_user, legacy_cache)}

    saved = {}
    for environment, model in (("windows", "H:/local/model"), ("linux", "/opt/local/model")):
        paths = load_paths(monkeypatch, tmp_path, environment)
        monkeypatch.setattr(config_module, "DEFAULT_PATHS", paths.DEFAULT_PATHS)
        registry = ModelConfigRegistry()
        registry.load()
        assert registry.get("local", "model_path") == "portable-model"
        assert registry.get("local", "estimated_size_gb") == 1.0
        registry.set("local", "model_path", model)
        registry.set_runtime_cache_value("local", "estimated_size_gb", 2.0)
        registry.save_user_config()
        registry.save_runtime_cache()
        saved[environment] = (
            paths,
            paths.USER_CONFIG_PATH.read_bytes(),
            paths.MODEL_RUNTIME_CACHE_PATH.read_bytes(),
        )
        _load_config_from_file.cache_clear()
        registry.load()
        assert registry.get("local", "model_path") == model
        assert registry.get("local", "estimated_size_gb") == 2.0

    for paths, user_bytes, cache_bytes in saved.values():
        assert paths.USER_CONFIG_PATH.read_bytes() == user_bytes
        assert paths.MODEL_RUNTIME_CACHE_PATH.read_bytes() == cache_bytes
    assert {path: path.read_bytes() for path in before} == before


def test_explicit_registry_paths_preserve_portable_mode(tmp_path):
    system, user, runtime = (
        tmp_path / filename for filename in ("system.toml", "user.toml", "runtime.toml")
    )
    system.write_text('[local]\nmodel_path = "default"\n')
    user.write_text('[local]\nmodel_path = "custom"\nestimated_size_gb = 3.0\n')
    runtime.write_text("[local]\nestimated_size_gb = 2.0\n")
    registry = ModelConfigRegistry()
    registry.load(system, user, runtime)
    assert registry.get("local", "model_path") == "custom"
    assert registry.get("local", "estimated_size_gb") == 3.0
    registry.set_runtime_cache_value("local", "estimated_size_gb", 4.0)
    registry.save_runtime_cache()
    assert toml.load(runtime)["local"]["estimated_size_gb"] == 4.0
    assert toml.load(user)["local"]["estimated_size_gb"] == 3.0


def test_install_markers_use_local_runtime_directory_and_keep_custom_cache_compatibility(
    monkeypatch, tmp_path
):
    from image_annotator_lib import model_installer

    paths = load_paths(monkeypatch, tmp_path, "windows")
    monkeypatch.setattr(model_installer, "DEFAULT_PATHS", paths.DEFAULT_PATHS)
    marker = model_installer._marker_path("org/model")
    assert marker.parent.parent == paths.RUNTIME_CACHE_DIR / "install_markers"
    assert marker.name == "org__model.json"
    monkeypatch.setattr(model_installer, "DEFAULT_PATHS", {"cache_dir": tmp_path / "custom-models"})
    assert model_installer._marker_path("org/model") == (
        tmp_path / "custom-models" / "install_markers" / "org__model.json"
    )
