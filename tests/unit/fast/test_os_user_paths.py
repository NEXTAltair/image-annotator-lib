"""Mutable model preferences and runtime state belong to one OS user."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    assert module.MODEL_RUNTIME_CACHE_PATH == (
        user_cache_path("image-annotator-lib", appauthor=False)
        / "projects"
        / module.runtime_state_namespace(module.SYSTEM_CONFIG_PATH, module.CACHE_DIR)
        / "model_runtime_cache.toml"
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


def test_runtime_metadata_is_namespaced_by_explicit_catalog_and_model_cache(monkeypatch, tmp_path):
    paths = load_paths(monkeypatch, tmp_path, "windows")
    catalog_a, catalog_b = (
        tmp_path / "checkout-a" / "catalog.toml",
        tmp_path / "checkout-b" / "catalog.toml",
    )
    for path, model in ((catalog_a, "model-a"), (catalog_b, "model-b")):
        path.parent.mkdir()
        path.write_text(f'[same-model-name]\nmodel_path = "{model}"\nestimated_size_gb = 1.0\n')
    monkeypatch.setattr(config_module, "DEFAULT_PATHS", paths.DEFAULT_PATHS)
    monkeypatch.setattr(config_module, "MODEL_RUNTIME_CACHE_PATH", paths.MODEL_RUNTIME_CACHE_PATH)
    monkeypatch.setattr(
        config_module, "runtime_cache_path_for_catalog", paths.runtime_cache_path_for_catalog
    )
    left, right = ModelConfigRegistry(), ModelConfigRegistry()
    left.load(config_path=catalog_a)
    left.set_runtime_cache_value("same-model-name", "estimated_size_gb", 4.0)
    left.save_runtime_cache()
    right.load(config_path=catalog_b)
    assert right.get("same-model-name", "estimated_size_gb") == 1.0
    assert left._runtime_cache_path != right._runtime_cache_path
    assert paths.runtime_cache_path_for_catalog(catalog_a, tmp_path / "cache-a") != (
        paths.runtime_cache_path_for_catalog(catalog_a, tmp_path / "cache-b")
    )


def test_relative_home_overrides_stay_anchored_after_chdir(monkeypatch, tmp_path):
    for kind in ("CONFIG", "CACHE", "LOG"):
        monkeypatch.setenv(f"IMAGE_ANNOTATOR_LIB_{kind}_HOME", f"local-{kind.lower()}")
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("relative_model_paths", constants.__file__)
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert paths.USER_CONFIG_DIR == tmp_path / "local-config"
    assert paths.RUNTIME_CACHE_DIR == tmp_path / "local-cache"
    assert paths.LOG_DIR == tmp_path / "local-log"


def test_inplace_custom_cache_keeps_existing_portable_install_markers(monkeypatch, tmp_path):
    from image_annotator_lib import model_installer

    custom_cache = tmp_path / "custom-models"
    monkeypatch.setitem(model_installer.DEFAULT_PATHS, "cache_dir", custom_cache)
    assert model_installer._marker_path("org/model") == custom_cache / "install_markers" / "org__model.json"


def test_install_markers_use_the_same_catalog_identity_as_runtime_metadata(monkeypatch, tmp_path):
    from image_annotator_lib import model_installer

    model_cache = tmp_path / "models"
    marker_root = tmp_path / "isolated-markers"
    catalog_a, catalog_b = tmp_path / "catalog-a.toml", tmp_path / "catalog-b.toml"
    monkeypatch.setattr(
        model_installer, "DEFAULT_PATHS", {"cache_dir": model_cache, "install_marker_dir": marker_root}
    )
    monkeypatch.setattr(model_installer, "config_registry", SimpleNamespace(_system_config_path=catalog_a))
    marker_a = model_installer._marker_path("same-model")
    monkeypatch.setattr(model_installer, "config_registry", SimpleNamespace(_system_config_path=catalog_b))
    marker_b = model_installer._marker_path("same-model")
    assert marker_a.parent.name == constants.runtime_state_namespace(catalog_a, model_cache)
    assert marker_b.parent.name == constants.runtime_state_namespace(catalog_b, model_cache)
    assert marker_a != marker_b
