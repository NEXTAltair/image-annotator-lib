"""API から利用可能なモデル情報を動的に取得する機能を提供します。"""

import datetime
import json
from datetime import datetime as dt
from datetime import timedelta, timezone  # datetime を dt としてインポート
from pathlib import Path
from typing import Any  # TOML データ用に一時的に使用

import requests
import toml

from ..exceptions import (
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    WebApiError,
)
from .config import (
    load_available_api_models,  # TOML 読み込み関数
    save_available_api_models,  # TOML 書き込み関数
)
from .constants import AVAILABLE_API_MODELS_CONFIG_PATH
from .utils import convert_unix_to_iso8601, logger

# キャッシュ用のデータ構造 -> 削除
# _model_cache: dict[str, list[str] | str] = {}
# _cache_expiry: dt | None = None
# _CACHE_DURATION = timedelta(hours=1)

_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
_REQUEST_TIMEOUT = 10


def _fetch_and_update_vision_models() -> list[str]:
    """API からモデルを取得し、フィルタリング、整形、TOML 更新を行い、モデル ID リストを返す。"""
    now = dt.now(timezone.utc)
    current_time_iso = now.isoformat(timespec="seconds") + "Z"

    # === API 呼び出し ===
    response = requests.get(_OPENROUTER_API_URL, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    # === ここまで ===

    # === レスポンス解析 ===
    try:
        api_data = response.json()
    except json.JSONDecodeError as e:
        raise WebApiError("API応答の JSON パースに失敗しました。", provider_name="OpenRouter") from e

    if not isinstance(api_data, dict) or "data" not in api_data:
        raise WebApiError(
            "API応答の形式が不正です (ルートが辞書でないか 'data' キーが存在しません)。",
            provider_name="OpenRouter",
        )

    raw_model_list = api_data["data"]
    if not isinstance(raw_model_list, list):
        raise WebApiError(
            "API応答の形式が不正です ('data' フィールドがリストではありません)。",
            provider_name="OpenRouter",
        )
    # === ここまで ===

    # === Vision モデルフィルタリング ===
    filtered_models_raw = _filter_vision_models(raw_model_list)
    # === ここまで ===

    # === モデルデータ整形 ===
    formatted_models_dict: dict[str, dict[str, Any]] = {}
    formatted_models_list: list[dict[str, Any]] = []
    for model_data in filtered_models_raw:
        model_id = model_data.get("id")
        if not model_id or not isinstance(model_id, str):
            continue
        formatted = _format_model_data_for_toml(model_data)
        if formatted:
            if model_id not in formatted_models_dict:
                # TOMLに保存する整形済みデータにはidを含めない
                formatted_models_dict[model_id] = formatted
                # _update_toml_with_api_results に渡すリストにはidを含める
                formatted_with_id = formatted.copy()
                formatted_with_id["id"] = model_id
                formatted_models_list.append(formatted_with_id)
    # === ここまで ===

    # === TOML 読み込み ===
    existing_toml_data = load_available_api_models()
    # === ここまで ===

    # === TOML データ更新 (last_seen, deprecated_on) ===
    updated_toml_data = _update_toml_with_api_results(
        existing_toml_data,
        formatted_models_list,
        current_time_iso,
    )
    # === ここまで ===

    # === TOML 書き込み ===
    save_available_api_models(updated_toml_data)
    # === ここまで ===

    # 保存後のデータからモデル ID リストを作成して返す
    updated_model_ids: list[str] = list(updated_toml_data.keys())
    return updated_model_ids


def discover_available_vision_models(force_refresh: bool = False) -> dict[str, list[str] | str]:
    """
    利用可能な Vision 対応モデルの一覧を取得する。

    基本的にはローカルの `available_api_models.toml` ファイルから読み込む。
    `force_refresh=True` が指定された場合、またはTOMLファイルが存在しない/空の場合は、
    OpenRouter API を介して最新情報を取得し、TOMLファイルを更新する。

    Args:
        force_refresh: True の場合、ローカルファイルを無視して強制的に API から再取得する。

    Returns:
        モデル ID のリスト、またはエラーメッセージを含む辞書。
        キーは "models" (成功時) または "error" (失敗時)。
        例:
            成功時: {"models": ["openai/gpt-4o", "google/gemini-pro-vision", ...]}
            失敗時: {"error": "API 接続エラー: <詳細>"}
    """
    # global _cache_expiry  # グローバル変数を参照・更新することを宣言 -> 削除
    # now = dt.now(timezone.utc) # 削除

    # 1. キャッシュ確認 -> TOML読み込みに変更
    if not force_refresh:
        existing_toml_data = load_available_api_models()
        if existing_toml_data:
            # TOMLデータが存在すれば、そこからモデルIDリストを生成して返す
            model_ids = list(existing_toml_data.keys())
            logger.info(
                f"ローカルファイル ({AVAILABLE_API_MODELS_CONFIG_PATH}) から {len(model_ids)} 件のモデル情報を読み込みました。"
            )
            return {"models": model_ids}
        else:
            logger.info(
                f"ローカルファイル ({AVAILABLE_API_MODELS_CONFIG_PATH}) が存在しないか空です。APIから取得します。"
            )
    # force_refresh=True または TOMLデータがない場合は API 取得へ

    try:
        # === API 取得と TOML 更新処理をヘルパー関数で実行 ===
        logger.info("APIから最新のモデル情報を取得・更新します。")
        updated_model_ids = _fetch_and_update_vision_models()
        # === ここまで ===

        # === キャッシュ更新 (成功時) -> 削除 ===
        result_data: dict[str, list[str] | str] = {"models": updated_model_ids}
        # _model_cache.clear() # 削除
        # _model_cache.update(result_data) # 削除
        # _cache_expiry = now + _CACHE_DURATION  # 成功時に有効期限を設定 (モジュールレベル変数を更新) -> 削除
        return result_data
        # --- ここまで成功時の処理 ---

    except (ApiTimeoutError, ApiRequestError, ApiServerError, WebApiError) as e:
        logger.error(f"APIモデル取得中にエラーが発生: {e}")
        return {"error": str(e)}
    except requests.exceptions.Timeout:
        error_message = f"APIリクエストがタイムアウトしました ({_REQUEST_TIMEOUT}秒)。"
        logger.error(error_message)
        return {"error": error_message}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_details = e.response.text
        if 400 <= status_code < 500:
            error_message = f"APIリクエストエラー (ステータスコード: {status_code}): {error_details}"
        elif 500 <= status_code < 600:
            error_message = f"APIサーバーエラー (ステータスコード: {status_code}): {error_details}"
        else:
            error_message = f"HTTPエラー (ステータスコード: {status_code}): {error_details}"
        logger.error(error_message)
        return {"error": error_message}
    except requests.exceptions.RequestException as e:
        error_message = f"APIへの接続またはリクエスト中にエラーが発生しました: {e}"
        logger.error(error_message)
        return {"error": error_message}
    except Exception as e:
        error_message = f"モデル取得中に予期せぬエラーが発生しました: {e}"
        logger.exception(error_message)
        return {"error": error_message}


# --- ヘルパー関数など (必要に応じて追加) ---


def _filter_vision_models(raw_models: list[Any]) -> list[dict[str, Any]]:
    """モデルリストから Vision (画像入力) 対応モデルのみをフィルタリングする。"""
    vision_models = []
    for model_data in raw_models:
        if not isinstance(model_data, dict):
            continue  # 辞書でないデータはスキップ

        architecture = model_data.get("architecture", {})
        if not isinstance(architecture, dict):
            continue  # architecture が辞書でない場合はスキップ

        input_modalities = architecture.get("input_modalities", [])
        if not isinstance(input_modalities, list):
            continue  # input_modalities がリストでない場合はスキップ

        if "image" in input_modalities:
            vision_models.append(model_data)

    return vision_models


def _format_model_data_for_toml(api_model_data: dict[str, Any]) -> dict[str, Any] | None:
    """API から取得したモデルデータを TOML 保存形式に整形する。

    必須キーが欠けている場合は None を返す。
    """
    required_keys = ["id", "name", "created", "architecture"]
    if not all(key in api_model_data for key in required_keys):
        # 必須キーが不足しているデータはスキップ (ログ出力を検討)
        return None

    architecture = api_model_data["architecture"]
    if not isinstance(architecture, dict) or not all(
        k in architecture for k in ["modality", "input_modalities"]
    ):
        # architecture 構造が不正なデータはスキップ
        return None

    # --- プロバイダー/モデル名分割 (ハイブリッド) ---
    provider = "Unknown"
    model_name_short = api_model_data["name"]  # デフォルト
    display_name = api_model_data["name"]
    model_id = api_model_data["id"]

    name_separator = ": "
    id_separator = "/"

    # 1. name を ": " で分割
    if name_separator in display_name:
        parts = display_name.split(name_separator, 1)
        provider = parts[0].strip()
        model_name_short = parts[1].strip()
    # 2. name が分割できず、id に "/" があれば id で分割
    elif id_separator in model_id:
        parts = model_id.split(id_separator, 1)
        # プロバイダー名を整形 (例: openai -> OpenAI, google -> Google)
        raw_provider = parts[0].strip()
        if raw_provider == "openai": #pragma: no cover
            provider = "OpenAI"   # openai の場合のみ特別扱い
        else:
            provider = raw_provider.capitalize() if raw_provider else "Unknown"
        model_name_short = parts[1].strip()
    # 3. フォールバック ( provider="Unknown", model_name_short=display_name は初期値)

    # --- タイムスタンプ変換 (utils を使用) ---
    created_timestamp = api_model_data.get("created")  # .get() を使用
    # utils の関数を呼び出し
    created_iso = convert_unix_to_iso8601(created_timestamp, model_id_for_log=model_id)

    # --- 整形されたデータを構築 ---
    formatted = {
        # キーはモデル ID なので不要 -> _update_toml_with_api_results に渡す前に含める
        "provider": provider,
        "model_name_short": model_name_short,
        "display_name": display_name,
        "created": created_iso,
        "modality": architecture.get("modality"),
        "input_modalities": architecture.get("input_modalities"),
        # last_seen, deprecated_on は呼び出し元 (_update_toml_with_api_results) で追加
    }
    return formatted


def _update_toml_with_api_results(
    existing_toml_data: dict[str, Any],
    api_models_formatted: list[dict[str, Any]],
    current_time_iso: str,
) -> dict[str, Any]:
    """既存の TOML データと API 結果をマージし、last_seen/deprecated_on を更新する。

    Args:
        existing_toml_data: 既存の TOML データ (available_vision_models セクション)。
        api_models_formatted: API から取得し、整形されたモデルデータのリスト。
                               各要素は _format_model_data_for_toml の戻り値と 'id' を含む。
                               リスト内のモデル ID は重複しない前提。
        current_time_iso: 現在時刻の ISO 8601 文字列。

    Returns:
        更新された TOML データ (available_vision_models セクション)。
    """
    updated_data = existing_toml_data.copy()  # 元の辞書を変更しないようにコピー
    api_model_ids = {model["id"] for model in api_models_formatted}  # 高速なルックアップ用セット

    # 1. API から取得できたモデルを更新 (last_seen 更新、deprecated_on 削除)
    for model_data_with_id in api_models_formatted:
        model_id = model_data_with_id.get("id")
        if not model_id:
            continue

        # TOMLに保存するデータから id を削除
        final_model_entry = model_data_with_id.copy()
        del final_model_entry["id"]  # idキーを削除

        final_model_entry["last_seen"] = current_time_iso
        final_model_entry["deprecated_on"] = None  # または del final_model_entry["deprecated_on"] if exists
        # 存在しないキーは削除を試みる代わりに None を設定する方が安全

        updated_data[model_id] = final_model_entry

    # 2. 既存の TOML データにしか存在しないモデルに deprecated_on を設定
    for model_id, existing_entry in existing_toml_data.items():
        if model_id not in api_model_ids:
            # API 結果になく、かつ deprecated_on がまだ設定されていない場合
            if isinstance(existing_entry, dict) and existing_entry.get("deprecated_on") is None:
                existing_entry["deprecated_on"] = current_time_iso
                # last_seen は更新しない
                updated_data[model_id] = existing_entry  # 更新されたエントリを反映
            # 既に deprecated_on が設定されている場合は何もしない

    return updated_data
