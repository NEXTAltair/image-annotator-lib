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
def test_cold_public_import_readonly_policy(tmp_path, prepared):
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
    try:
        from image_annotator_lib import list_annotator_info
        infos = list_annotator_info()
        result = {'ok': True, 'count': len(infos)}
    except Exception as exc:
        result = {'ok': False, 'error_type': type(exc).__name__}
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=90
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["ok"] is prepared, result
    if prepared:
        assert config.read_text() == "# Existing model configuration\n"
    else:
        assert result["error_type"] == "ReadOnlyConfigError"
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
