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
