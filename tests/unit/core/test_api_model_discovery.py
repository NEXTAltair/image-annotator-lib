"""api_model_discovery モジュールのユニットテスト。"""

from typing import Any

import pytest
import json

# テスト対象のモジュールからヘルパー関数をインポート
# 注意: プライベート関数 (_始まり) のテストは通常推奨されないが、
#       今回は複雑なロジックを含むため、直接テストする。
from image_annotator_lib.core.api_model_discovery import (
    _format_model_data_for_toml,
    _update_toml_with_api_results,
)

# --- Test Data --- #

VALID_API_DATA_NAME_SPLIT = {
    "id": "openai/gpt-4o",
    "name": "OpenAI: GPT-4o",
    "created": 1715558400,  # 2024-05-13T00:00:00Z
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

VALID_API_DATA_ID_SPLIT = {
    "id": "google/gemini-pro-vision",
    "name": "Gemini Pro Vision 1.0",  # Name に : がない
    "created": 1702425600,  # 2023-12-13T00:00:00Z
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

VALID_API_DATA_FALLBACK = {
    "id": "some-random-model",  # ID に / がない
    "name": "Some Random Model",  # Name に : がない
    "created": 1715558400,
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

VALID_API_DATA_OPENAI_CASE = {
    "id": "openai/gpt-4o-mini",
    "name": "GPT-4o-mini",  # Name に : がない
    "created": 1721260800,  # 2024-07-18T00:00:00Z
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

INVALID_API_DATA_MISSING_KEY = {
    "id": "test/missing-key",
    # "name": "Missing Name",
    "created": 1715558400,
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

INVALID_API_DATA_INVALID_ARCH = {
    "id": "test/invalid-arch",
    "name": "Invalid Arch",
    "created": 1715558400,
    "architecture": {
        # "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
}

# --- Tests for _format_model_data_for_toml --- #


def test_format_model_data_name_split():
    """Name によるプロバイダー/モデル名分割が正しく行われるかテスト。"""
    result = _format_model_data_for_toml(VALID_API_DATA_NAME_SPLIT)
    assert result is not None
    assert result["provider"] == "OpenAI"
    assert result["model_name_short"] == "GPT-4o"
    assert result["display_name"] == "OpenAI: GPT-4o"
    assert result["created"] == "2024-05-13T00:00:00Z"
    assert result["modality"] == "text+image->text"
    assert result["input_modalities"] == ["text", "image"]
    assert "last_seen" not in result
    assert "deprecated_on" not in result


def test_format_model_data_id_split():
    """ID によるプロバイダー/モデル名分割と大文字化が正しく行われるかテスト。"""
    result = _format_model_data_for_toml(VALID_API_DATA_ID_SPLIT)
    assert result is not None
    assert result["provider"] == "Google"  # Capitalized
    assert result["model_name_short"] == "gemini-pro-vision"
    assert result["display_name"] == "Gemini Pro Vision 1.0"
    assert result["created"] == "2023-12-13T00:00:00Z"


def test_format_model_data_fallback():
    """Name/ID で分割できない場合のフォールバック処理をテスト。"""
    result = _format_model_data_for_toml(VALID_API_DATA_FALLBACK)
    assert result is not None
    assert result["provider"] == "Unknown"
    assert result["model_name_short"] == "Some Random Model"
    assert result["display_name"] == "Some Random Model"


def test_format_model_data_openai_case():
    """openai プロバイダー名が特別扱いされるかテスト。"""
    result = _format_model_data_for_toml(VALID_API_DATA_OPENAI_CASE)
    assert result is not None
    assert result["provider"] == "OpenAI"  # Not just "Openai"
    assert result["model_name_short"] == "gpt-4o-mini"
    assert result["display_name"] == "GPT-4o-mini"
    assert result["created"] == "2024-07-18T00:00:00Z"


def test_format_model_data_missing_key():
    """必須キーが欠けている場合に None を返すかテスト。"""
    result = _format_model_data_for_toml(INVALID_API_DATA_MISSING_KEY)
    assert result is None


def test_format_model_data_invalid_arch():
    """architecture 構造が不正な場合に None を返すかテスト。"""
    result = _format_model_data_for_toml(INVALID_API_DATA_INVALID_ARCH)
    assert result is None


# convert_unix_to_iso8601 は utils でテスト済みと仮定し、ここでは統合テストのみ

# --- Tests for _update_toml_with_api_results --- #

# テストで使用する現在時刻
CURRENT_TIME_ISO = "2024-05-20T12:00:00Z"
PREVIOUS_TIME_ISO = "2024-05-19T12:00:00Z"

# テストデータ: 整形済み API モデルリスト (ID を含む必要がある)
FORMATTED_API_MODELS_LIST = [
    {
        "id": "model/a",
        "provider": "ProviderA",
        "model_name_short": "Model A",
        "display_name": "Model A Display",
        "created": "2024-01-01T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
    {
        "id": "model/b",
        "provider": "ProviderB",
        "model_name_short": "Model B",
        "display_name": "Model B Display",
        "created": "2024-01-02T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
    # model/d は API 結果に存在するが、既存 TOML には存在しない (新規追加)
    {
        "id": "model/d",
        "provider": "ProviderD",
        "model_name_short": "Model D",
        "display_name": "Model D Display",
        "created": "2024-01-04T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
    # model/e は既存 TOML で deprecated だったが、API 結果に再登場
    {
        "id": "model/e",
        "provider": "ProviderE",
        "model_name_short": "Model E",
        "display_name": "Model E Display",
        "created": "2024-01-05T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
    },
]

# テストデータ: 既存の TOML データ
EXISTING_TOML_DATA = {
    "model/a": {
        "provider": "ProviderA",
        "model_name_short": "Model A Old",  # 他のフィールドは更新される想定
        "display_name": "Model A Old Display",
        "created": "2024-01-01T00:00:00Z",
        "modality": "text",
        "input_modalities": ["text"],
        "last_seen": PREVIOUS_TIME_ISO,
        "deprecated_on": None,
    },
    # model/b は API 結果にも存在する
    "model/b": {
        "provider": "ProviderB",
        "model_name_short": "Model B",
        "display_name": "Model B Display",
        "created": "2024-01-02T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": PREVIOUS_TIME_ISO,
        "deprecated_on": None,
    },
    # model/c は API 結果に存在しない (deprecated になるはず)
    "model/c": {
        "provider": "ProviderC",
        "model_name_short": "Model C",
        "display_name": "Model C Display",
        "created": "2024-01-03T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": PREVIOUS_TIME_ISO,
        "deprecated_on": None,
    },
    # model/e は deprecated だったが API 結果に再登場 (deprecated_on が消えるはず)
    "model/e": {
        "provider": "ProviderE",
        "model_name_short": "Model E Old",
        "display_name": "Model E Old Display",
        "created": "2024-01-05T00:00:00Z",
        "modality": "text",
        "input_modalities": ["text"],
        "last_seen": "2024-05-18T12:00:00Z",  # 過去の last_seen
        "deprecated_on": PREVIOUS_TIME_ISO,  # 既に deprecated
    },
    # model/f は deprecated で API にも現れない (そのままのはず)
    "model/f": {
        "provider": "ProviderF",
        "model_name_short": "Model F",
        "display_name": "Model F Display",
        "created": "2024-01-06T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": "2024-05-18T12:00:00Z",
        "deprecated_on": PREVIOUS_TIME_ISO,
    },
}

# 期待される結果データ
EXPECTED_UPDATED_DATA = {
    "model/a": {
        "provider": "ProviderA",
        "model_name_short": "Model A",
        "display_name": "Model A Display",
        "created": "2024-01-01T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": CURRENT_TIME_ISO,  # 更新されている
        "deprecated_on": None,  # None のまま
    },
    "model/b": {
        "provider": "ProviderB",
        "model_name_short": "Model B",
        "display_name": "Model B Display",
        "created": "2024-01-02T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": CURRENT_TIME_ISO,  # 更新されている
        "deprecated_on": None,
    },
    "model/c": {
        "provider": "ProviderC",
        "model_name_short": "Model C",
        "display_name": "Model C Display",
        "created": "2024-01-03T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": PREVIOUS_TIME_ISO,  # 更新されない
        "deprecated_on": CURRENT_TIME_ISO,  # 設定されている
    },
    "model/d": {
        "provider": "ProviderD",
        "model_name_short": "Model D",
        "display_name": "Model D Display",
        "created": "2024-01-04T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": CURRENT_TIME_ISO,  # 設定されている
        "deprecated_on": None,  # 設定されている
    },
    "model/e": {
        "provider": "ProviderE",
        "model_name_short": "Model E",
        "display_name": "Model E Display",
        "created": "2024-01-05T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": CURRENT_TIME_ISO,  # 更新されている
        "deprecated_on": None,  # None に戻っている
    },
    "model/f": {
        "provider": "ProviderF",
        "model_name_short": "Model F",
        "display_name": "Model F Display",
        "created": "2024-01-06T00:00:00Z",
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "last_seen": "2024-05-18T12:00:00Z",  # 更新されない
        "deprecated_on": PREVIOUS_TIME_ISO,  # 維持される
    },
}


def test_update_toml_with_api_results():
    """既存 TOML と API 結果のマージ、last_seen/deprecated_on 更新をテスト。"""
    # 関数を呼び出し
    actual_updated_data = _update_toml_with_api_results(
        EXISTING_TOML_DATA, FORMATTED_API_MODELS_LIST, CURRENT_TIME_ISO
    )

    # 結果を比較 (キーの順序は問わない)
    assert actual_updated_data == EXPECTED_UPDATED_DATA


# --- Tests for _filter_vision_models --- #

# テストデータ
RAW_MODELS_FOR_FILTER_TEST = [
    # 有効な Vision モデル
    {
        "id": "vision/model1",
        "name": "Vision Model 1",
        "architecture": {"input_modalities": ["text", "image"]},
    },
    # 無効なデータ (辞書ではない)
    "not_a_dict",
    # 無効なデータ (architecture がない)
    {"id": "invalid/no_arch", "name": "No Architecture"},
    # 無効なデータ (architecture が辞書ではない)
    {
        "id": "invalid/arch_not_dict",
        "name": "Architecture Not Dict",
        "architecture": "not_a_dict",
    },
    # 無効なデータ (input_modalities がない)
    {
        "id": "invalid/no_modalities",
        "name": "No Input Modalities",
        "architecture": {},  # modality もないが、input_modalities のチェックが先
    },
    # 無効なデータ (input_modalities がリストではない)
    {
        "id": "invalid/modalities_not_list",
        "name": "Modalities Not List",
        "architecture": {"input_modalities": "not_a_list"},
    },
    # Vision モデルではない (image が含まれない)
    {
        "id": "text/model2",
        "name": "Text Model 2",
        "architecture": {"input_modalities": ["text", "audio"]},
    },
    # 有効な Vision モデル (2つ目)
    {
        "id": "vision/model3",
        "name": "Vision Model 3",
        "architecture": {"input_modalities": ["image"]},
    },
]


def test_filter_vision_models():
    """_filter_vision_models が正しく Vision モデルのみを抽出するかテスト。"""
    # image_annotator_lib.core.api_model_discovery から _filter_vision_models をインポート
    from image_annotator_lib.core.api_model_discovery import _filter_vision_models

    filtered = _filter_vision_models(RAW_MODELS_FOR_FILTER_TEST)

    # 結果の検証
    assert len(filtered) == 2  # 有効な Vision モデルは2つだけ
    assert filtered[0]["id"] == "vision/model1"
    assert filtered[1]["id"] == "vision/model3"


# --- Tests for discover_available_vision_models --- #

import unittest.mock as mock

import requests

# テスト対象のメイン関数
from image_annotator_lib.core.api_model_discovery import discover_available_vision_models
from image_annotator_lib.exceptions.errors import (
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    WebApiError,
)

# モック用のデータ
MOCK_API_SUCCESS_RESPONSE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "created": 1715558400,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
            },
        },
        {
            "id": "google/gemini-pro-vision",
            "name": "Gemini Pro Vision 1.0",
            "created": 1702425600,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
            },
        },
        {
            "id": "anthropic/claude-3-opus",  # Vision対応
            "name": "Anthropic: Claude 3 Opus",
            "created": 1709600000,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
            },
        },
        {
            "id": "meta-llama/llama-3-8b-instruct",  # Vision非対応
            "name": "Meta: Llama 3 8B Instruct",
            "created": 1713400000,
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
            },
        },
    ]
}

MOCK_EMPTY_TOML_DATA: dict[str, Any] = {}
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


@mock.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")
def test_discover_from_existing_toml(mock_load_toml):
    """既存のTOMLファイルからモデルリストを正常に読み込むケース。"""
    mock_load_toml.return_value = MOCK_EXISTING_TOML_DATA

    result = discover_available_vision_models(force_refresh=False)

    assert "models" in result
    assert isinstance(result["models"], list)
    assert list(MOCK_EXISTING_TOML_DATA.keys()) == result["models"]
    mock_load_toml.assert_called_once()


@mock.patch("image_annotator_lib.core.api_model_discovery.save_available_api_models")
@mock.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")
@mock.patch("requests.get")
def test_discover_force_refresh_success(mock_requests_get, mock_load_toml, mock_save_toml):
    """force_refresh=True でAPIから正常に取得･更新するケース。"""
    # モックの設定
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = MOCK_API_SUCCESS_RESPONSE
    mock_requests_get.return_value = mock_response
    mock_load_toml.return_value = MOCK_EXISTING_TOML_DATA  # 既存データがあってもAPIを呼ぶ

    result = discover_available_vision_models(force_refresh=True)

    # 検証
    assert "models" in result
    expected_vision_models = [
        "openai/gpt-4o",
        "google/gemini-pro-vision",
        "anthropic/claude-3-opus",
    ]
    # 順序は問わないが、内容が一致するか
    assert sorted(result["models"]) == sorted(expected_vision_models)

    mock_requests_get.assert_called_once()
    mock_load_toml.assert_called_once()  # _fetch_and_update 内で呼ばれる
    mock_save_toml.assert_called_once()
    # mock_save_toml に渡されたデータの内容も検証可能
    saved_data = mock_save_toml.call_args[0][0]
    assert "openai/gpt-4o" in saved_data
    assert "google/gemini-pro-vision" in saved_data
    assert "anthropic/claude-3-opus" in saved_data
    assert "meta-llama/llama-3-8b-instruct" not in saved_data  # Vision非対応は除外
    assert saved_data["openai/gpt-4o"]["last_seen"] is not None


@mock.patch("image_annotator_lib.core.api_model_discovery.save_available_api_models")
@mock.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")
@mock.patch("requests.get")
def test_discover_initial_load_success(mock_requests_get, mock_load_toml, mock_save_toml):
    """初回起動時 (TOML空) にAPIから正常に取得･更新するケース。"""
    # モックの設定
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = MOCK_API_SUCCESS_RESPONSE
    mock_requests_get.return_value = mock_response
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA  # TOMLは空

    result = discover_available_vision_models(force_refresh=False)

    # 検証 (force_refresh_success と同様)
    assert "models" in result
    expected_vision_models = [
        "openai/gpt-4o",
        "google/gemini-pro-vision",
        "anthropic/claude-3-opus",
    ]
    assert sorted(result["models"]) == sorted(expected_vision_models)

    mock_requests_get.assert_called_once()
    mock_load_toml.assert_called()  # if not force_refresh と _fetch_and_update 内で呼ばれる
    assert mock_load_toml.call_count >= 1  # 少なくとも1回は呼ばれる
    mock_save_toml.assert_called_once()


# APIエラーハンドリングのテストケース
@pytest.mark.parametrize(
    "exception_to_raise, expected_error_type",
    [
        (requests.exceptions.Timeout, ApiTimeoutError),
        (
            requests.exceptions.HTTPError(response=mock.Mock(status_code=500, text="Server Error")),
            ApiServerError,
        ),
        (
            requests.exceptions.HTTPError(response=mock.Mock(status_code=400, text="Bad Request")),
            ApiRequestError,
        ),
        (
            requests.exceptions.RequestException("Connection failed"),
            WebApiError,
        ),  # RequestException -> WebApiError
        (json.JSONDecodeError("Invalid JSON", "", 0), WebApiError),  # JSONDecodeError -> WebApiError
        (Exception("Some generic error"), Exception),  # Generic Exception
    ],
)
@mock.patch("image_annotator_lib.core.api_model_discovery.save_available_api_models")
@mock.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")
@mock.patch("requests.get")
def test_discover_api_errors(
    mock_requests_get, mock_load_toml, mock_save_toml, exception_to_raise, expected_error_type
):
    """API呼び出し時に各種エラーが発生した場合のテスト。"""
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA  # API呼び出しをトリガー

    # requests.get が指定された例外を発生させるように設定
    # HTTPError の場合は raise_for_status で発生させるシナリオも考慮
    if isinstance(exception_to_raise, requests.exceptions.HTTPError):
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = exception_to_raise
        mock_requests_get.return_value = mock_response
    elif isinstance(exception_to_raise, json.JSONDecodeError):
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = exception_to_raise
        mock_requests_get.return_value = mock_response
    else:
        mock_requests_get.side_effect = exception_to_raise

    result = discover_available_vision_models(
        force_refresh=True
    )  # エラー時はforce_refresh=TrueでもFalseでも同じはず

    assert "error" in result
    assert isinstance(result["error"], str)
    # ここでは具体的なエラーメッセージ文字列までは検証しない (変わりうるため)
    # 例外の型が期待通りにラップされているか、などの確認は可能だが、
    # 今回は戻り値の辞書の形式のみ確認する

    mock_save_toml.assert_not_called()  # エラー時は保存しない


@mock.patch("image_annotator_lib.core.api_model_discovery.save_available_api_models")
@mock.patch("image_annotator_lib.core.api_model_discovery.load_available_api_models")
@mock.patch("requests.get")
def test_discover_invalid_response_format(mock_requests_get, mock_load_toml, mock_save_toml):
    """API応答のフォーマットが不正な場合のテスト (例: dataキー欠損)。"""
    mock_load_toml.return_value = MOCK_EMPTY_TOML_DATA

    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"not_data": []}  # 不正な形式
    mock_requests_get.return_value = mock_response

    result = discover_available_vision_models(force_refresh=True)

    assert "error" in result
    assert isinstance(result["error"], str)
    assert "形式が不正" in result["error"]  # エラーメッセージに形式不正が含まれることを確認

    mock_save_toml.assert_not_called()
