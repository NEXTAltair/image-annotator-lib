"""Readonly initialization is tested outside pytest's registry autoload shortcut."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from image_annotator_lib import CONFIG_READ_ONLY_ENV, ReadOnlyConfigError
from image_annotator_lib.core.config import ModelConfigRegistry


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("entrypoint", ["list_annotator_info", "list_available_annotators"])
def test_cold_public_import_readonly_policy(tmp_path, prepared, entrypoint):
    cwd = tmp_path / "cold 日本語"
    cwd.mkdir()
    config = cwd / "config" / "annotator_config.toml"
    if prepared:
        config.parent.mkdir()
        config.write_text("# Existing model configuration\n")
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    env[CONFIG_READ_ONLY_ENV] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    code = """
import json, socket
from pathlib import Path
from unittest.mock import patch
with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
    from image_annotator_lib import ReadOnlyConfigError, list_annotator_info
    try:
        infos = list_annotator_info()
        result = {'ok': True, 'count': len(infos)}
    except ReadOnlyConfigError as exc:
        result = {'ok': False, 'action': exc.details['action']}
print(json.dumps(result))
"""
    code = code.replace("list_annotator_info", entrypoint)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=90
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["ok"] is prepared, result
    if prepared:
        assert result["count"] > 0
        assert config.read_text() == "# Existing model configuration\n"
    else:
        assert result["action"] == "existing_system_config_required"
        assert not config.parent.exists()


def test_readonly_config_disappears_after_check_without_copy(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("# ready\n")
    registry = ModelConfigRegistry()
    ensure = registry._ensure_system_config_exists

    def remove_after_check():
        ensure()
        config.unlink()

    monkeypatch.setenv(CONFIG_READ_ONLY_ENV, "1")
    monkeypatch.setattr(registry, "_ensure_system_config_exists", remove_after_check)
    copy = Mock(side_effect=AssertionError("copy forbidden"))
    monkeypatch.setattr("image_annotator_lib.core.config.shutil.copyfile", copy)
    with pytest.raises(ReadOnlyConfigError, match="system_config_unreadable"):
        registry.load(config_path=config)
    assert not config.exists()
    copy.assert_not_called()


@pytest.mark.parametrize("method", ["save_system_config", "save_user_config", "save_runtime_cache"])
def test_readonly_blocks_all_configuration_persistence(tmp_path, monkeypatch, method):
    monkeypatch.setenv(CONFIG_READ_ONLY_ENV, "1")
    registry = ModelConfigRegistry()
    registry._system_config_data = {"test": {"class": "Dummy"}}
    registry._user_config_data = {"test": {"prompt": "synthetic"}}
    registry._runtime_cache_data = {"test": {"estimated_size_gb": 1.0}}
    target = tmp_path / "not-created" / "config.toml"
    with pytest.raises(ReadOnlyConfigError, match="write_forbidden"):
        getattr(registry, method)(target)
    assert not target.parent.exists()


@pytest.mark.parametrize("content", ["broken = 1\n", "broken = [1]\n", "[invalid\n"])
def test_readonly_rejects_invalid_required_model_configuration(tmp_path, monkeypatch, content):
    monkeypatch.setenv(CONFIG_READ_ONLY_ENV, "1")
    config = tmp_path / "required.toml"
    config.write_text(content)
    with pytest.raises(ReadOnlyConfigError, match="system_config_unreadable"):
        ModelConfigRegistry().load(config_path=config)
    assert config.read_text() == content


@pytest.mark.parametrize("content", ["model = 1\n", "other = [1]\n", "[invalid\n"])
def test_readonly_ignores_malformed_optional_user_configuration(tmp_path, monkeypatch, content):
    monkeypatch.setenv(CONFIG_READ_ONLY_ENV, "1")
    system = tmp_path / "system.toml"
    system.write_text('[model]\nclass = "Dummy"\n')
    user = tmp_path / "user.toml"
    user.write_text(content)
    registry = ModelConfigRegistry()
    registry.load(config_path=system, user_config_path=user)
    assert registry.get("model", "class") == "Dummy"
    assert user.read_text() == content
