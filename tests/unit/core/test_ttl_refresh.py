"""Unit tests for TTL-based auto-refresh of available_api_models.toml."""

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

import image_annotator_lib.core.api_model_discovery as _disc_module
from image_annotator_lib.core.api_model_discovery import (
    discover_available_vision_models,
    should_refresh,
    trigger_background_refresh,
)
from image_annotator_lib.core.config import (
    load_api_models_meta,
    load_available_api_models,
    load_last_refresh,
    save_available_api_models,
)


# --- フィクスチャ ---


@pytest.fixture
def toml_path(tmp_path: Path) -> Generator[Path, None, None]:
    """TOML ファイルパスをテスト用一時ディレクトリにリダイレクトする。"""
    p = tmp_path / "available_api_models.toml"
    with (
        patch("image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", p),
        patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", p),
    ):
        load_available_api_models.cache_clear()
        yield p
        load_available_api_models.cache_clear()


@pytest.fixture(autouse=True)
def reset_refresh_lock() -> Generator[None, None, None]:
    """各テストの前後でモジュールレベルの _refresh_lock を新規 Lock に交換する。

    前テストで lock が残るとスレッドが即スキップするため、テスト間分離が必要。
    """
    original_lock = _disc_module._refresh_lock
    _disc_module._refresh_lock = threading.Lock()
    yield
    _disc_module._refresh_lock = original_lock


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# --- load_last_refresh テスト ---


@pytest.mark.unit
def test_load_last_refresh_returns_none_when_file_missing(toml_path: Path) -> None:
    assert load_last_refresh() is None


@pytest.mark.unit
def test_load_last_refresh_returns_none_when_meta_section_missing(toml_path: Path) -> None:
    save_available_api_models({})
    assert load_last_refresh() is None


@pytest.mark.unit
def test_load_last_refresh_returns_datetime_when_present(toml_path: Path, now_utc: datetime) -> None:
    save_available_api_models({}, last_refresh=now_utc)
    result = load_last_refresh()
    assert result is not None
    assert abs((result - now_utc).total_seconds()) < 1


@pytest.mark.unit
def test_load_last_refresh_returns_timezone_aware_datetime(toml_path: Path, now_utc: datetime) -> None:
    save_available_api_models({}, last_refresh=now_utc)
    result = load_last_refresh()
    assert result is not None
    assert result.tzinfo is not None


@pytest.mark.unit
def test_load_last_refresh_normalizes_naive_datetime_to_utc(toml_path: Path) -> None:
    """タイムゾーンなし文字列（手編集等）は UTC として扱い、aware datetime を返す。"""
    import toml

    toml_path.write_text(
        '[meta]\nlast_refresh = "2026-04-20T10:00:00"\n[available_vision_models]\n',
        encoding="utf-8",
    )
    load_available_api_models.cache_clear()
    result = load_last_refresh()
    assert result is not None
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc


# --- save_available_api_models + [meta] テスト ---


@pytest.mark.unit
def test_save_writes_meta_last_refresh(toml_path: Path, now_utc: datetime) -> None:
    data = {"google/gemini-test": {"provider": "Google"}}
    save_available_api_models(data, last_refresh=now_utc)

    meta = load_api_models_meta()
    assert "last_refresh" in meta


@pytest.mark.unit
def test_save_preserves_meta_when_last_refresh_none(toml_path: Path, now_utc: datetime) -> None:
    """last_refresh=None で保存しても既存 meta.last_refresh が保持される。"""
    save_available_api_models({}, last_refresh=now_utc)
    first_refresh = load_last_refresh()

    save_available_api_models({"some/model": {"provider": "Test"}}, last_refresh=None)
    second_refresh = load_last_refresh()

    assert first_refresh is not None
    assert second_refresh is not None
    assert abs((first_refresh - second_refresh).total_seconds()) < 1


@pytest.mark.unit
def test_save_ignores_non_dict_meta_section(toml_path: Path, now_utc: datetime) -> None:
    """TOML の [meta] が非 dict（破損）の場合でも save が正常完了し meta を上書きする。"""
    toml_path.write_text(
        'meta = "broken"\n[available_vision_models]\n',
        encoding="utf-8",
    )
    load_available_api_models.cache_clear()
    save_available_api_models({}, last_refresh=now_utc)
    meta = load_api_models_meta()
    assert "last_refresh" in meta


@pytest.mark.unit
def test_save_preserves_existing_model_data(toml_path: Path, now_utc: datetime) -> None:
    data = {"openai/gpt-4o": {"provider": "OpenAI", "model_name_short": "gpt-4o"}}
    save_available_api_models(data, last_refresh=now_utc)
    load_available_api_models.cache_clear()
    loaded = load_available_api_models()
    assert "openai/gpt-4o" in loaded


# --- should_refresh テスト ---


@pytest.mark.unit
def test_should_refresh_returns_true_when_file_missing(toml_path: Path) -> None:
    assert should_refresh() is True


@pytest.mark.unit
def test_should_refresh_returns_true_when_last_refresh_missing(toml_path: Path) -> None:
    save_available_api_models({})
    assert should_refresh() is True


@pytest.mark.unit
def test_should_refresh_returns_true_when_ttl_exceeded(toml_path: Path) -> None:
    old_refresh = datetime.now(timezone.utc) - timedelta(days=8)
    save_available_api_models({}, last_refresh=old_refresh)
    assert should_refresh(ttl_days=7) is True


@pytest.mark.unit
def test_should_refresh_returns_false_when_within_ttl(toml_path: Path) -> None:
    recent_refresh = datetime.now(timezone.utc) - timedelta(days=1)
    save_available_api_models({}, last_refresh=recent_refresh)
    assert should_refresh(ttl_days=7) is False


@pytest.mark.unit
def test_should_refresh_returns_true_exactly_at_ttl_boundary(toml_path: Path) -> None:
    """TTL ちょうどは「期限切れ」と判定される（> 判定）。"""
    boundary_refresh = datetime.now(timezone.utc) - timedelta(days=7, seconds=1)
    save_available_api_models({}, last_refresh=boundary_refresh)
    assert should_refresh(ttl_days=7) is True


@pytest.mark.unit
def test_should_refresh_respects_env_var(toml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recent_refresh = datetime.now(timezone.utc) - timedelta(days=2)
    save_available_api_models({}, last_refresh=recent_refresh)

    monkeypatch.setenv("IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS", "1")
    assert should_refresh() is True

    monkeypatch.setenv("IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS", "30")
    assert should_refresh() is False


@pytest.mark.unit
def test_should_refresh_uses_default_ttl_on_invalid_env_var(
    toml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recent_refresh = datetime.now(timezone.utc) - timedelta(days=1)
    save_available_api_models({}, last_refresh=recent_refresh)

    monkeypatch.setenv("IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS", "not_a_number")
    # 既定 TTL=7 日なので 1 日前なら False
    assert should_refresh() is False


# --- trigger_background_refresh テスト ---


@pytest.mark.unit
def test_trigger_background_refresh_does_not_block(toml_path: Path) -> None:
    """refresh が完了するまで待たず、即座に返ること。"""
    barrier = threading.Event()

    def slow_fetch() -> None:
        barrier.wait(timeout=5)

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=slow_fetch,
    ):
        start = time.monotonic()
        thread = trigger_background_refresh()
        elapsed = time.monotonic() - start

    barrier.set()
    thread.join(timeout=2)

    assert elapsed < 0.5, f"trigger_background_refresh がブロックしました: {elapsed:.3f}s"


@pytest.mark.unit
def test_trigger_background_refresh_swallows_exceptions(toml_path: Path) -> None:
    """_fetch_and_update_vision_models が例外を送出してもプロセスが継続すること。"""
    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=RuntimeError("network error"),
    ):
        thread = trigger_background_refresh()
        thread.join(timeout=2)

    assert not thread.is_alive()


@pytest.mark.unit
def test_trigger_background_refresh_dedupe(toml_path: Path) -> None:
    """同時に複数回呼んでも _fetch_and_update_vision_models は 1 回しか実行されない。"""
    call_count = 0
    fetch_started = threading.Event()  # t1 の fetch が開始したら set する

    def counting_fetch() -> None:
        nonlocal call_count
        call_count += 1
        fetch_started.set()  # t1 がロックを保持したまま fetch 中であることを通知
        time.sleep(0.1)

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=counting_fetch,
    ):
        t1 = trigger_background_refresh()
        # t1 が _refresh_lock を保持して counting_fetch に入るまで待つ
        fetch_started.wait(timeout=2)
        t2 = trigger_background_refresh()
        t1.join(timeout=2)
        t2.join(timeout=2)

    assert call_count == 1, f"fetch が複数回実行されました: {call_count} 回"


# --- should_refresh: naive datetime 安全性テスト ---


@pytest.mark.unit
def test_should_refresh_returns_true_on_naive_datetime(toml_path: Path) -> None:
    """load_last_refresh が naive datetime を返しても TypeError を起こさず True を返す。"""
    naive_dt = datetime(2026, 1, 1, 0, 0, 0)  # tzinfo なし
    with patch("image_annotator_lib.core.api_model_discovery.load_last_refresh", return_value=naive_dt):
        result = should_refresh()
    assert result is True


# --- discover_available_vision_models: ロック直列化テスト ---


@pytest.mark.unit
def test_discover_force_refresh_acquires_refresh_lock(toml_path: Path) -> None:
    """force_refresh=True 時、_fetch_and_update_vision_models の実行中に _refresh_lock が保持される。"""
    lock_held_during_fetch: list[bool] = []

    def check_lock() -> dict:
        lock_held_during_fetch.append(_disc_module._refresh_lock.locked())
        return {}

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=check_lock,
    ):
        discover_available_vision_models(force_refresh=True)

    assert lock_held_during_fetch == [True], "fetch 中に _refresh_lock が保持されていません"


@pytest.mark.unit
def test_discover_force_refresh_serializes_with_background_refresh(toml_path: Path) -> None:
    """force_refresh とバックグラウンド refresh は同時に _fetch を実行しない。"""
    concurrent_count = 0
    active_count = 0
    fetch_started = threading.Event()

    def serialized_fetch() -> dict:
        nonlocal concurrent_count, active_count
        active_count += 1
        if active_count > 1:
            concurrent_count += 1
        fetch_started.set()
        time.sleep(0.05)
        active_count -= 1
        return {}

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=serialized_fetch,
    ):
        bg_thread = trigger_background_refresh()
        fetch_started.wait(timeout=2)  # バックグラウンドが fetch 中になるまで待つ
        discover_available_vision_models(force_refresh=True)  # ロック待ちして直列実行
        bg_thread.join(timeout=2)

    assert concurrent_count == 0, "fetch が同時実行されました"
