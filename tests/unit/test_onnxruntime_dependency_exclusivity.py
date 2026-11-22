"""onnxruntime の CPU 版と GPU 版が同じ環境へ同時に入らない依存定義であることの検証 (Issue #175)。

``onnxruntime`` と ``onnxruntime-gpu`` は同じ ``onnxruntime`` モジュールを配置するため、
同時に導入すると後から書かれたファイルが勝ち、CUDA 版を入れても CPU 実行になることがある。
OS ごとに pyproject.toml と uv.lock の environment marker を評価し、片方だけが選ばれることを確かめる。
"""

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CPU_DISTRIBUTION = "onnxruntime"
GPU_DISTRIBUTION = "onnxruntime-gpu"
# Python の sys.platform と platform.system() の組。OS ごとの marker 評価に使う。
PLATFORMS = {
    "windows": {"sys_platform": "win32", "platform_system": "Windows"},
    "linux": {"sys_platform": "linux", "platform_system": "Linux"},
    "macos": {"sys_platform": "darwin", "platform_system": "Darwin"},
}


def _environment(platform: str) -> dict[str, str]:
    return {**default_environment(), **PLATFORMS[platform]}


def _selected(entries: list[tuple[str, str | None]], platform: str) -> list[str]:
    """(distribution 名, marker) の一覧から、その OS で有効な ONNX Runtime の distribution を返す。"""
    environment = _environment(platform)
    return sorted(
        name
        for name, marker in entries
        if name in {CPU_DISTRIBUTION, GPU_DISTRIBUTION}
        and (marker is None or Marker(marker).evaluate(environment))
    )


def _pyproject_requirements() -> list[tuple[str, str | None]]:
    """pyproject.toml の全依存 (本体・optional・dependency-groups) から (名前, marker) を集める。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs: list[str] = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    for group in data.get("dependency-groups", {}).values():
        specs.extend(item for item in group if isinstance(item, str))
    entries = []
    for spec in specs:
        requirement = Requirement(spec)
        marker = str(requirement.marker) if requirement.marker else None
        entries.append((requirement.name, marker))
    return entries


def _lock_dependents() -> list[tuple[str, str | None]]:
    """uv.lock の全 package のうち、ONNX Runtime に依存する宣言から (名前, marker) を集める。"""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    entries = []
    for package in lock["package"]:
        for dependency in package.get("dependencies", []):
            entries.append((dependency["name"], dependency.get("marker")))
    return entries


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_pyproject_selects_exactly_one_onnxruntime_distribution(platform: str) -> None:
    selected = _selected(_pyproject_requirements(), platform)

    assert len(selected) == 1, f"{platform}: {selected}"


def test_pyproject_selects_gpu_distribution_except_on_macos() -> None:
    """Windows / Linux は CUDA 利用のため GPU 版、macOS には GPU 版が無いため CPU 版。"""
    entries = _pyproject_requirements()

    assert _selected(entries, "windows") == [GPU_DISTRIBUTION]
    assert _selected(entries, "linux") == [GPU_DISTRIBUTION]
    assert _selected(entries, "macos") == [CPU_DISTRIBUTION]


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_lockfile_selects_exactly_one_onnxruntime_distribution(platform: str) -> None:
    """uv.lock の依存宣言 (推移的依存を含む全 package) でも片方だけが選ばれる。"""
    selected = _selected(_lock_dependents(), platform)

    assert len(selected) == 1, f"{platform}: {selected}"


def test_lockfile_matches_pyproject_selection() -> None:
    pyproject = _pyproject_requirements()
    lock = _lock_dependents()

    for platform in PLATFORMS:
        assert _selected(lock, platform) == _selected(pyproject, platform), platform


def test_selector_detects_the_pre_fix_definition() -> None:
    """修正前の定義 (CPU 版を無条件に要求) を、検証が重複として検出できること。"""
    pre_fix = [
        (CPU_DISTRIBUTION, None),
        (GPU_DISTRIBUTION, "platform_system != 'Darwin'"),
    ]

    assert _selected(pre_fix, "windows") == [CPU_DISTRIBUTION, GPU_DISTRIBUTION]
    assert _selected(pre_fix, "macos") == [CPU_DISTRIBUTION]
