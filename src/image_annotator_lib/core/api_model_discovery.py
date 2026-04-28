"""API から利用可能なモデル情報を動的に取得する機能を提供します。

メインソース: LiteLLM のローカル DB（pip バンドル、無料・無認証）
フォールバック: OpenRouter API（LiteLLM 未収録の free tier モデル等を補完）
"""

import datetime
import json
import os
import threading
from datetime import datetime as dt
from typing import Any

import litellm
import requests

from ..exceptions.errors import (
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    WebApiError,
)
from .config import (
    load_available_api_models,
    load_last_refresh,
    save_available_api_models,
)
from .constants import (
    AVAILABLE_API_MODELS_CONFIG_PATH,
    DEFAULT_API_MODELS_TTL_DAYS,
    ENV_API_MODELS_TTL_DAYS,
)
from .utils import convert_unix_to_iso8601, logger

_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
_REQUEST_TIMEOUT = 10

# 同時 background refresh を防ぐプロセス内ロック
_refresh_lock = threading.Lock()


def _fetch_from_litellm() -> list[dict[str, Any]]:
    """LiteLLM のローカル DB から Vision 対応モデル一覧を取得する。

    provider/model 形式の ID のみ処理し（エイリアス重複を除外）、
    litellm.supports_vision() で Vision 対応を確認する。

    Returns:
        整形済みモデルデータのリスト（'id' キーを含む）
    """
    results = []
    for model_id in litellm.model_cost.keys():
        if "/" not in model_id:
            continue
        try:
            if not litellm.supports_vision(model_id):
                continue
            info = litellm.get_model_info(model_id)
            if info is None:
                continue
            formatted = _format_litellm_model_for_toml(model_id, info)
            if formatted:
                results.append(formatted)
        except Exception:
            continue
    return results


def _format_litellm_model_for_toml(model_id: str, info: dict[str, Any]) -> dict[str, Any] | None:
    """LiteLLM の get_model_info 結果を TOML 保存形式に整形する。

    'id' キーを含むため、_update_toml_with_api_results に直接渡せる。
    provider/model 形式でない model_id は None を返す。
    """
    if "/" not in model_id:
        return None

    provider_raw, model_name_short = model_id.split("/", 1)
    if provider_raw == "openai":
        provider = "OpenAI"
    else:
        provider = provider_raw.capitalize()

    return {
        "id": model_id,
        "provider": provider,
        "model_name_short": model_name_short,
        "display_name": model_id,
        "mode": info.get("mode", "chat"),
        "max_tokens": info.get("max_tokens"),
        "input_cost_per_token": info.get("input_cost_per_token"),
        "output_cost_per_token": info.get("output_cost_per_token"),
        # last_seen, deprecated_on は _update_toml_with_api_results で付与
    }


def _fetch_from_openrouter_fallback() -> list[dict[str, Any]]:
    """OpenRouter API から Vision モデル一覧を取得する（LiteLLM 未収録モデルの補完用）。

    ネットワーク障害時は空リストを返す（フォールバックなので失敗を伝播しない）。

    Returns:
        整形済みモデルデータのリスト（'id' キーを含む）
    """
    try:
        response = requests.get(_OPENROUTER_API_URL, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        api_data = response.json()
        raw_models = api_data.get("data", [])
        filtered = _filter_openrouter_vision_models(raw_models)
        result = []
        for model_data in filtered:
            model_id = model_data.get("id")
            if not model_id or not isinstance(model_id, str):
                continue
            formatted = _format_openrouter_model_for_toml(model_data)
            if formatted:
                formatted["id"] = model_id
                result.append(formatted)
        return result
    except Exception as e:
        logger.warning(f"OpenRouter フォールバック失敗（無視）: {e}")
        return []


def _fetch_and_update_vision_models() -> dict[str, Any]:
    """LiteLLM DB (メイン) + OpenRouter API (フォールバック) でモデル一覧を更新する。

    Returns:
        updated_toml_data: 更新済みの全モデルデータ（TOML 書き込み失敗時も in-memory の正確なデータを返す）
    """
    now = dt.now(datetime.UTC)
    current_time_iso = now.isoformat(timespec="seconds") + "Z"

    # === LiteLLM メイン取得 ===
    litellm_models = _fetch_from_litellm()
    logger.info(f"LiteLLM DB から {len(litellm_models)} 件の Vision モデルを取得")

    # === OpenRouter フォールバック（LiteLLM 未収録モデルを補完）===
    openrouter_models = _fetch_from_openrouter_fallback()
    litellm_ids = {m["id"] for m in litellm_models}
    additional_models = [m for m in openrouter_models if m["id"] not in litellm_ids]
    if additional_models:
        logger.info(f"OpenRouter フォールバックで {len(additional_models)} 件を追加")

    all_models = litellm_models + additional_models

    # === TOML 読み込み → 更新 → 書き込み ===
    existing_toml_data = load_available_api_models()
    updated_toml_data = _update_toml_with_api_results(existing_toml_data, all_models, current_time_iso)
    save_available_api_models(updated_toml_data, last_refresh=now)

    # TOML 書き込み失敗（save が例外を握りつぶす）でも in-memory の正確なデータを返す
    return updated_toml_data


def should_refresh(ttl_days: int | None = None) -> bool:
    """TTL 超過判定。

    Args:
        ttl_days: 有効期間（日数）。None の場合は環境変数 IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS、
                  未設定時は DEFAULT_API_MODELS_TTL_DAYS（7日）を使用。

    Returns:
        True なら refresh が必要。last_refresh が未記録の場合も True を返す。
    """
    if ttl_days is None:
        env_val = os.environ.get(ENV_API_MODELS_TTL_DAYS)
        if env_val is not None:
            try:
                ttl_days = int(env_val)
            except ValueError:
                logger.warning(f"{ENV_API_MODELS_TTL_DAYS}={env_val!r} が整数でないため既定値を使用します。")
                ttl_days = DEFAULT_API_MODELS_TTL_DAYS
        else:
            ttl_days = DEFAULT_API_MODELS_TTL_DAYS

    last = load_last_refresh()
    if last is None:
        return True

    try:
        age_seconds = (dt.now(datetime.UTC) - last).total_seconds()
    except TypeError:
        logger.warning(f"last_refresh の型が不正のため refresh を実行します: {last!r}")
        return True
    return age_seconds > ttl_days * 86400


def _refresh_worker() -> None:
    """バックグラウンドスレッドで実行する refresh タスク。例外はすべて捕捉する。"""
    if not _refresh_lock.acquire(blocking=False):
        logger.debug("API モデル refresh は既に実行中のため、新規起動をスキップします。")
        return
    try:
        logger.info("バックグラウンド API モデル refresh を開始します。")
        _fetch_and_update_vision_models()
        logger.info("バックグラウンド API モデル refresh が完了しました。")
    except Exception as e:
        logger.warning(f"バックグラウンド API モデル refresh に失敗しました（次回 TTL 超過時に再試行）: {e}")
    finally:
        _refresh_lock.release()


def trigger_background_refresh() -> threading.Thread:
    """バックグラウンドで _fetch_and_update_vision_models() をキックする。

    起動元スレッドをブロックしない。既に refresh 実行中なら _refresh_worker 内でスキップされる。

    Returns:
        起動した daemon スレッド（テスト用）。
    """
    thread = threading.Thread(target=_refresh_worker, daemon=True, name="api-model-refresh")
    thread.start()
    return thread


def discover_available_vision_models(force_refresh: bool = False) -> dict[str, Any]:
    """
    利用可能な Vision 対応モデルの一覧を取得する。

    基本的にはローカルの `available_api_models.toml` ファイルから読み込む。
    `force_refresh=True` が指定された場合、またはTOMLファイルが存在しない/空の場合は、
    LiteLLM のローカル DB から最新情報を取得し、TOMLファイルを更新する。
    LiteLLM に未収録のモデルは OpenRouter API でフォールバック補完する。

    Args:
        force_refresh: True の場合、ローカルファイルを無視して強制的に再取得する。

    Returns:
        成功時: {"models": list[str], "toml_data": dict[str, Any]}
            - "models": 全モデル ID のリスト（deprecated 含む）
            - "toml_data": 全モデルのメタデータ辞書（TOML 書き込み失敗時も in-memory の正確なデータ）
        失敗時: {"error": str}
    """
    if not force_refresh:
        existing_toml_data = load_available_api_models()
        if existing_toml_data:
            model_ids = list(existing_toml_data.keys())
            logger.info(
                f"ローカルファイル ({AVAILABLE_API_MODELS_CONFIG_PATH}) から {len(model_ids)} 件のモデル情報を読み込みました。"
            )
            return {"models": model_ids, "toml_data": existing_toml_data}
        else:
            logger.info(
                f"ローカルファイル ({AVAILABLE_API_MODELS_CONFIG_PATH}) が存在しないか空です。LiteLLM DB から取得します。"
            )

    try:
        logger.info("LiteLLM DB から最新のモデル情報を取得・更新します。")
        with _refresh_lock:
            updated_data = _fetch_and_update_vision_models()
        return {"models": list(updated_data.keys()), "toml_data": updated_data}

    except (ApiTimeoutError, ApiRequestError, ApiServerError, WebApiError) as e:
        logger.error(f"モデル取得中にエラーが発生: {e}")
        return {"error": str(e)}
    except Exception as e:
        error_message = f"モデル取得中に予期せぬエラーが発生しました: {e}"
        logger.exception(error_message)
        return {"error": error_message}


# --- OpenRouter フォールバック用ヘルパー関数 ---


def _filter_openrouter_vision_models(raw_models: list[Any]) -> list[dict[str, Any]]:
    """OpenRouter モデルリストから Vision・構造化出力・ツール利用対応のモデルをフィルタリングする。"""
    compatible_models = []
    for model_data in raw_models:
        if not isinstance(model_data, dict):
            continue

        architecture = model_data.get("architecture", {})
        if not isinstance(architecture, dict):
            continue
        input_modalities = architecture.get("input_modalities", [])
        if not isinstance(input_modalities, list) or "image" not in input_modalities:
            continue

        supported_parameters = model_data.get("supported_parameters")
        is_structured_output_supported = False
        is_tool_use_supported = False

        if supported_parameters and isinstance(supported_parameters, list):
            if "structured_outputs" in supported_parameters:
                is_structured_output_supported = True
            if "tools" in supported_parameters:
                is_tool_use_supported = True

        if not (is_structured_output_supported and is_tool_use_supported):
            continue

        compatible_models.append(model_data)

    return compatible_models


def _format_openrouter_model_for_toml(api_model_data: dict[str, Any]) -> dict[str, Any] | None:
    """OpenRouter API から取得したモデルデータを TOML 保存形式に整形する。

    必須キーが欠けている場合は None を返す。
    'id' キーは呼び出し元で付与する。
    """
    required_keys = ["id", "name", "created", "architecture"]
    if not all(key in api_model_data for key in required_keys):
        return None

    architecture = api_model_data["architecture"]
    if not isinstance(architecture, dict) or not all(
        k in architecture for k in ["modality", "input_modalities"]
    ):
        return None

    provider = "Unknown"
    model_name_short = api_model_data["name"]
    display_name = api_model_data["name"]
    model_id = api_model_data["id"]

    name_separator = ": "
    id_separator = "/"

    if name_separator in display_name:
        parts = display_name.split(name_separator, 1)
        provider = parts[0].strip()
        model_name_short = parts[1].strip()
    elif id_separator in model_id:
        parts = model_id.split(id_separator, 1)
        raw_provider = parts[0].strip()
        if raw_provider == "openai":  # pragma: no cover
            provider = "OpenAI"
        else:
            provider = raw_provider.capitalize() if raw_provider else "Unknown"
        model_name_short = parts[1].strip()

    created_timestamp = api_model_data.get("created")
    created_iso = convert_unix_to_iso8601(created_timestamp, model_id_for_log=model_id)

    return {
        "provider": provider,
        "model_name_short": model_name_short,
        "display_name": display_name,
        "created": created_iso,
        "modality": architecture.get("modality"),
        "input_modalities": architecture.get("input_modalities"),
        # last_seen, deprecated_on は呼び出し元 (_update_toml_with_api_results) で追加
    }


def _update_toml_with_api_results(
    existing_toml_data: dict[str, Any],
    api_models_formatted: list[dict[str, Any]],
    current_time_iso: str,
) -> dict[str, Any]:
    """既存の TOML データと取得結果をマージし、last_seen/deprecated_on を更新する。

    Args:
        existing_toml_data: 既存の TOML データ。
        api_models_formatted: 取得・整形されたモデルデータのリスト（'id' キーを含む）。
        current_time_iso: 現在時刻の ISO 8601 文字列。

    Returns:
        更新された TOML データ。
    """
    updated_data = existing_toml_data.copy()
    api_model_ids = {model["id"] for model in api_models_formatted}

    # 取得できたモデルを更新（last_seen 更新、deprecated_on をクリア）
    for model_data_with_id in api_models_formatted:
        model_id = model_data_with_id.get("id")
        if not model_id:
            continue

        final_model_entry = model_data_with_id.copy()
        del final_model_entry["id"]

        final_model_entry["last_seen"] = current_time_iso
        final_model_entry["deprecated_on"] = None

        updated_data[model_id] = final_model_entry

    # 既存 TOML にしか存在しないモデルに deprecated_on を設定
    for model_id, existing_entry in existing_toml_data.items():
        if model_id not in api_model_ids:
            if isinstance(existing_entry, dict) and existing_entry.get("deprecated_on") is None:
                existing_entry["deprecated_on"] = current_time_iso
                updated_data[model_id] = existing_entry

    return updated_data
