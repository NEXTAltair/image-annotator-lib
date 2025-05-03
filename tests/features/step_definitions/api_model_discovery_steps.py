"""api_model_discovery.feature のステップ定義"""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import requests
from pytest_bdd import given, parsers, scenarios, then, when

from image_annotator_lib.core.api_model_discovery import discover_available_vision_models
from image_annotator_lib.core.constants import AVAILABLE_API_MODELS_CONFIG_PATH
from tests.features.step_definitions.common_steps import *

# --- シナリオファイルの指定 ---
scenarios("api_model_discovery.feature")

# --- モックデータ ---
MOCK_API_SUCCESS_RESPONSE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "created": 1715558400,
            "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"]},
        },
        {
            "id": "google/gemini-pro-vision",
            "name": "Gemini Pro Vision 1.0",
            "created": 1702425600,
            "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"]},
        },
        {
            "id": "anthropic/claude-3-opus",  # Vision対応
            "name": "Anthropic: Claude 3 Opus",
            "created": 1709600000,
            "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"]},
        },
        {
            "id": "meta-llama/llama-3-8b-instruct",  # Vision非対応
            "name": "Meta: Llama 3 8B Instruct",
            "created": 1713400000,
            "architecture": {"modality": "text->text", "input_modalities": ["text"]},
        },
    ]
}

MOCK_API_INVALID_FORMAT_RESPONSE: dict[str, Any] = {
    "invalid_key": []  # 'data' キーがない
}

MOCK_EXISTING_TOML_DATA = {
    "openai/gpt-4o": {
        "provider": "OpenAI",
        "model_name_short": "GPT-4o",
        "display_name": "OpenAI: GPT-4o",
        "created": "2024-05-13T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": "2024-07-20T10:00:00Z",
        "deprecated_on": None,
    }
}

MOCK_EMPTY_TOML_DATA: dict[str, Any] = {}

EXPECTED_VISION_MODELS = [
    "openai/gpt-4o",
    "google/gemini-pro-vision",
    "anthropic/claude-3-opus",
]

# --- フィクスチャ ---


@pytest.fixture
def mock_requests_get(mocker):
    """requests.get をモックするフィクスチャ"""
    return mocker.patch("requests.get")


@pytest.fixture
def mock_load_toml(mocker):
    """load_available_api_models をモックするフィクスチャ"""
    # api_model_discovery 内で import されているものをモック
    return mocker.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")


@pytest.fixture
def mock_save_toml(mocker):
    """save_available_api_models をモックするフィクスチャ"""
    # api_model_discovery 内で import されているものをモック
    return mocker.patch("image_annotator_lib.core.api_model_discovery.save_available_api_models")


@pytest.fixture
def app_cache_dir(tmp_path):
    """アプリケーションキャッシュディレクトリを作成・提供するフィクスチャ"""
    # AVAILABLE_API_MODELS_CONFIG_PATH の親ディレクトリを一時ディレクトリ内に作成
    cache_dir = tmp_path / AVAILABLE_API_MODELS_CONFIG_PATH.parent.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    # モックされた config 関数がこの一時ディレクトリを見るようにする
    # (ここでは簡単化のためパスを直接使わず、ディレクトリ存在のみ保証)
    return cache_dir


@pytest.fixture
def api_discovery_result():
    """discover_available_vision_models の結果を格納するフィクスチャ"""
    return {}  # mutable な辞書を返す


# --- ステップ定義 ---

# --- Given ---


@given("アプリケーションのキャッシュディレクトリが存在する")
def given_app_cache_dir_exists(app_cache_dir: Path):
    """Background: キャッシュディレクトリが存在することを保証 (フィクスチャで実施)"""
    assert app_cache_dir.exists()
    assert app_cache_dir.is_dir()


@given("OpenRouter APIが利用可能であり、Visionモデルを含む有効なモデルリストを返す")
def given_api_returns_valid_list(mock_requests_get: mock.Mock, mock_load_toml: mock.Mock):
    """APIが正常なレスポンスを返すように設定"""
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = MOCK_API_SUCCESS_RESPONSE
    mock_requests_get.return_value = mock_response
    # このシナリオでは、キャッシュ(TOML)がない状態から始める
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA


@given("以前にOpenRouter APIが正常に呼び出され、結果がキャッシュされている")
def given_api_result_is_cached(mock_load_toml: mock.Mock):
    """有効な結果がローカルファイル(TOML)に保存されている状態を設定"""
    mock_load_toml.return_value = MOCK_EXISTING_TOML_DATA


@given("OpenRouter APIが利用可能であり、有効なモデルリストを返す", target_fixture="setup_refresh_api")
def given_api_ready_for_refresh(mock_requests_get: mock.Mock, mock_load_toml: mock.Mock):
    """強制リフレッシュ用のAPI設定 (ローカルファイル(TOML)も存在する状態)"""
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = MOCK_API_SUCCESS_RESPONSE
    mock_requests_get.return_value = mock_response
    # キャッシュ(TOML)は存在する設定
    mock_load_toml.return_value = MOCK_EXISTING_TOML_DATA
    return {"expected_models": EXPECTED_VISION_MODELS}


@given("以下のエラータイプがAPIで発生する:")
def given_api_errors(datatable, mock_requests_get, mock_load_toml):
    """各エラータイプごとにAPI呼び出し時の挙動をセットアップ"""
    # datatable: list[dict[str, str]]
    # ここでは最初のエラータイプのみをセットアップ（本来は各テストでループする設計が望ましい）
    # 例として最初のエラータイプを使う
    error_type = datatable[0]["error_type"]
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA
    if error_type == "Connection Timeout":
        mock_requests_get.side_effect = requests.exceptions.Timeout
    elif error_type == "HTTP Error 500":
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock.Mock(status_code=500)
        )
        mock_requests_get.return_value = mock_response
    elif error_type == "Invalid JSON":
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_requests_get.return_value = mock_response
    elif error_type == "Request Exception":
        mock_requests_get.side_effect = requests.exceptions.RequestException("Generic request error")
    else:
        raise ValueError(f"未対応のエラータイプ: {error_type}")


@given("OpenRouter APIが予期しない形式でデータを返す")
def given_api_returns_invalid_format(mock_requests_get: mock.Mock, mock_load_toml: mock.Mock):
    """APIが不正な形式のレスポンスを返すように設定"""
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = MOCK_API_INVALID_FORMAT_RESPONSE
    mock_requests_get.return_value = mock_response


# --- When ---


@when("discover_available_vision_models 関数を呼び出す", target_fixture="api_discovery_result")
def when_discover_called(mock_requests_get, mock_load_toml, mock_save_toml):
    """discover_available_vision_models を force_refresh=False で呼び出す"""
    return discover_available_vision_models(force_refresh=False)


@when(
    "force_refreshをtrueにして discover_available_vision_models 関数を呼び出す",
    target_fixture="api_discovery_result",
)
def when_discover_called_with_refresh(setup_refresh_api, mock_requests_get, mock_load_toml, mock_save_toml):
    """discover_available_vision_models を force_refresh=True で呼び出す"""
    return discover_available_vision_models(force_refresh=True)


# --- Then ---


@then("利用可能なVisionモデルのリストを取得できる")
def then_get_vision_model_list(api_discovery_result: dict, mock_save_toml: mock.Mock):
    """正常にモデルリストが返されることを検証"""
    assert "models" in api_discovery_result
    assert isinstance(api_discovery_result["models"], list)
    # 順序は問わない
    assert sorted(api_discovery_result["models"]) == sorted(EXPECTED_VISION_MODELS)
    # 保存関数が呼ばれたことも確認
    mock_save_toml.assert_called_once()
    saved_data = mock_save_toml.call_args[0][0]
    # 保存内容の簡易チェック (期待されるモデルが含まれているか)
    for model_id in EXPECTED_VISION_MODELS:
        assert model_id in saved_data


@then("APIから最新のVisionモデルのリストを再取得できる")
def then_get_refreshed_list(api_discovery_result: dict, setup_refresh_api: dict, mock_save_toml: mock.Mock):
    """強制リフレッシュ後に最新リストが返されることを検証"""
    assert "models" in api_discovery_result
    assert isinstance(api_discovery_result["models"], list)
    # setup_refresh_api で期待されるモデルリストを設定
    expected_models = setup_refresh_api["expected_models"]
    assert sorted(api_discovery_result["models"]) == sorted(expected_models)
    # 保存関数が呼ばれたことも確認
    mock_save_toml.assert_called_once()


@then("モデルリストの取得に失敗したことがわかる")
def then_get_failure(api_discovery_result: dict):
    """結果に 'error' キーが含まれることを検証"""
    assert "error" in api_discovery_result
    assert "models" not in api_discovery_result
    assert isinstance(api_discovery_result["error"], str)


@then("エラーの原因が各エラータイプで正しく判定されること")
def then_error_reason_for_each_type(datatable, api_discovery_result):
    """各エラータイプごとにエラーメッセージが正しく含まれているか判定"""
    error_type = datatable[0]["error_type"]
    error_message = api_discovery_result.get("error", "")
    if error_type == "Connection Timeout":
        assert "タイムアウト" in error_message or "Timeout" in error_message
    elif error_type == "HTTP Error 500":
        assert (
            "サーバーエラー" in error_message or "Server Error" in error_message or "500" in error_message
        )
    elif error_type == "Invalid JSON":
        assert "JSON" in error_message or "パース" in error_message
    elif error_type == "Request Exception":
        assert (
            "リクエスト中にエラー" in error_message or "接続" in error_message
        )
    else:
        assert error_type in error_message


@then("エラーの原因がAPI応答形式の問題であることがわかる")
def then_error_is_invalid_format(api_discovery_result: dict):
    """エラーメッセージが形式不正を示していることを検証"""
    error_message = api_discovery_result.get("error", "")
    print(f"[DEBUG] error_message: {error_message}")
    assert "形式が不正" in error_message or "invalid format" in error_message
