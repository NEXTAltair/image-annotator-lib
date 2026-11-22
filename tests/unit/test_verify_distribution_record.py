"""tools/verify_distribution_record.py が RECORD の hash 照合で欠落・上書きを検出すること (Issue #175)。"""

import base64
import hashlib
import importlib.util
from importlib import metadata
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TOOL = Path(__file__).resolve().parents[2] / "tools" / "verify_distribution_record.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_distribution_record", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _install(site: Path, name: str, files: dict[str, bytes]) -> metadata.Distribution:
    """RECORD 付きの最小 distribution を site に配置する (pip が作る形式)。"""
    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    rows = []
    for relative, data in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(f"{relative},{_digest(data)},{len(data)}")
    rows.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metadata.PathDistribution(info)


def test_intact_distribution_has_no_problems(tmp_path: Path) -> None:
    distribution = _install(tmp_path, "fakeort_gpu", {"fakeort/binding.bin": b"gpu", "fakeort/a.py": b"1"})

    assert _load_tool().verify_distribution(distribution) == []


def test_file_overwritten_by_another_distribution_is_a_mismatch(tmp_path: Path) -> None:
    """CPU 版 (別 distribution) が同じパスを上書きした状態を hash 不一致として検出する。"""
    gpu = _install(tmp_path, "fakeort_gpu", {"fakeort/binding.bin": b"gpu", "fakeort/a.py": b"1"})
    (tmp_path / "fakeort" / "binding.bin").write_bytes(b"cpu")

    assert _load_tool().verify_distribution(gpu) == ["mismatch: fakeort/binding.bin"]


def test_file_removed_by_uninstalling_another_distribution_is_missing(tmp_path: Path) -> None:
    """もう片方の uninstall で共有ファイルが消えた状態を欠落として検出する。"""
    gpu = _install(tmp_path, "fakeort_gpu", {"fakeort/binding.bin": b"gpu", "fakeort/a.py": b"1"})
    (tmp_path / "fakeort" / "binding.bin").unlink()

    assert _load_tool().verify_distribution(gpu) == ["missing: fakeort/binding.bin"]
