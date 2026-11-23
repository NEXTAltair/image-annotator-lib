"""合成 ONNX モデル生成 (build_tagger_onnx) の予算チェックのテスト (Issue #193)。

大きな引数を渡すと重みだけで数十 GiB を確保してしまうため、生成前に必要量を見積もり、
予算を超える場合は重みの生成 (tagger_weights) とファイル作成の前に拒否することを確かめる。
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from unit.fixtures import onnx_models
from unit.fixtures.onnx_models import (
    SYNTHETIC_MODEL_BUDGET_ENV,
    SyntheticModelBudgetError,
    build_tagger_onnx,
    estimate_tagger_onnx_bytes,
)

SMALL_BUDGET = 1 * 1024 * 1024


@pytest.fixture
def weight_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """tagger_weights を呼び出しを記録する spy に差し替え、記録先を返す。"""
    calls: list[tuple[int, int]] = []
    real: Callable[[int, int], np.ndarray] = onnx_models.tagger_weights

    def spy(input_elements: int, tag_count: int) -> np.ndarray:
        calls.append((input_elements, tag_count))
        return real(input_elements, tag_count)

    monkeypatch.setattr(onnx_models, "tagger_weights", spy)
    return calls


@pytest.mark.unit
@pytest.mark.fast
def test_oversized_model_is_rejected_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weight_calls: list[tuple[int, int]]
) -> None:
    monkeypatch.delenv(SYNTHETIC_MODEL_BUDGET_ENV, raising=False)
    path = tmp_path / "oversized.onnx"

    with pytest.raises(SyntheticModelBudgetError, match="予算"):
        build_tagger_onnx(
            path, batch_dim="batch", height=64, width=64, tag_count=1000, max_bytes=SMALL_BUDGET
        )
    assert weight_calls == []
    assert not path.exists()

    # 事故時のサイズ (float32 の重みだけで約 22.43 GiB) を既定の予算で呼んでも割当なしで拒否する
    incident_path = tmp_path / "incident.onnx"
    tracemalloc.start()
    try:
        with pytest.raises(SyntheticModelBudgetError) as excinfo:
            build_tagger_onnx(incident_path, batch_dim="batch", height=448, width=448, tag_count=10000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 * 1024 * 1024
    assert weight_calls == []
    assert not incident_path.exists()
    message = str(excinfo.value)
    assert f"{estimate_tagger_onnx_bytes(448, 448, 10000):,}" in message
    assert "max_bytes" in message
    assert SYNTHETIC_MODEL_BUDGET_ENV in message


@pytest.mark.unit
@pytest.mark.fast
def test_model_within_budget_is_built(tmp_path: Path, weight_calls: list[tuple[int, int]]) -> None:
    path = tmp_path / "small.onnx"

    result = build_tagger_onnx(path, batch_dim="batch", max_bytes=SMALL_BUDGET)

    assert result == path
    assert path.exists()
    assert len(path.read_bytes()) > 0
    assert weight_calls == [(3 * 8 * 8, 6)]


@pytest.mark.unit
@pytest.mark.fast
def test_budget_can_be_raised_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 中間サイズ: float32 の重みは約 0.6 MiB、見積もりは約 4.7 MiB
    size = {"height": 32, "width": 32, "tag_count": 50}
    estimated = estimate_tagger_onnx_bytes(32, 32, 50)
    assert SMALL_BUDGET < estimated < 16 * 1024 * 1024

    # 既定の予算 (環境変数で小さくした状態) では拒否する
    monkeypatch.setenv(SYNTHETIC_MODEL_BUDGET_ENV, str(SMALL_BUDGET))
    rejected = tmp_path / "rejected.onnx"
    with pytest.raises(SyntheticModelBudgetError):
        build_tagger_onnx(rejected, batch_dim="batch", **size)
    assert not rejected.exists()

    # max_bytes を明示的に引き上げると、環境変数より優先されて生成できる
    raised = tmp_path / "raised.onnx"
    build_tagger_onnx(raised, batch_dim="batch", max_bytes=16 * 1024 * 1024, **size)
    assert raised.stat().st_size > 0

    # 環境変数で引き上げても生成できる
    monkeypatch.setenv(SYNTHETIC_MODEL_BUDGET_ENV, str(16 * 1024 * 1024))
    by_env = tmp_path / "by_env.onnx"
    build_tagger_onnx(by_env, batch_dim="batch", **size)
    assert by_env.stat().st_size > 0

    # 不正な予算は ValueError (重みは作らない)
    for invalid in ("abc", "0", "-1", ""):
        monkeypatch.setenv(SYNTHETIC_MODEL_BUDGET_ENV, invalid)
        with pytest.raises(ValueError, match=SYNTHETIC_MODEL_BUDGET_ENV):
            build_tagger_onnx(tmp_path / "invalid.onnx", batch_dim="batch")
    with pytest.raises(ValueError, match="max_bytes"):
        build_tagger_onnx(tmp_path / "invalid.onnx", batch_dim="batch", max_bytes=0)
    assert not (tmp_path / "invalid.onnx").exists()


@pytest.mark.unit
@pytest.mark.fast
def test_estimate_matches_incident_size() -> None:
    float32_weight_bytes = 3 * 448 * 448 * 10000 * 4
    assert float32_weight_bytes == 24_084_480_000
    assert estimate_tagger_onnx_bytes(448, 448, 10000) >= float32_weight_bytes
