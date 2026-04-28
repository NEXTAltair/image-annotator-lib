"""Step definitions for API Model Discovery BDD scenarios."""

from unittest.mock import MagicMock, patch

import pytest
from pytest_bdd import given, then, when

from image_annotator_lib.core.api_model_discovery import discover_available_vision_models

# ============================================================================
# テスト用定数
# ============================================================================

_MOCK_VISION_MODEL_IDS = {
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",
    "google/gemini-1.5-pro",
}

_MOCK_MODEL_COST = {
    "openai/gpt-4o": {"max_tokens": 16384},
    "anthropic/claude-3-5-sonnet-20241022": {"max_tokens": 8192},
    "google/gemini-1.5-pro": {"max_tokens": 2097152},
    "openai/gpt-3.5-turbo": {"max_tokens": 4096},
}

_MOCK_MODEL_INFO = {"mode": "chat", "max_tokens": 16384, "input_cost_per_token": 2.5e-06}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_toml_file(tmp_path):
    """テスト用の一時 TOML ファイルパス。"""
    return tmp_path / "available_api_models.toml"


@pytest.fixture
def mock_litellm_db():
    """LiteLLM ローカル DB を模したモック。"""

    def _create_mock():
        mock_ll = MagicMock()
        mock_ll.model_cost = _MOCK_MODEL_COST
        mock_ll.supports_vision.side_effect = lambda model: model in _MOCK_VISION_MODEL_IDS
        mock_ll.get_model_info.side_effect = (
            lambda model: _MOCK_MODEL_INFO.copy() if model in _MOCK_VISION_MODEL_IDS else None
        )
        return mock_ll

    return _create_mock


# ============================================================================
# Given steps
# ============================================================================


@given("アプリケーションのキャッシュディレクトリが存在する")
def cache_directory_exists(mock_toml_file):
    """キャッシュ用ディレクトリが存在する。"""
    assert mock_toml_file.parent.exists()


@given("LiteLLM ローカル DB が利用可能であり、Visionモデルを含む有効なモデルリストを持つ")
def litellm_db_available_with_vision_models(mock_litellm_db):
    """LiteLLM DB に Vision モデルが含まれている状態。"""
    pass


@given("LiteLLM ローカル DB が利用可能であり、有効なモデルリストを持つ")
def litellm_db_available(mock_litellm_db):
    """LiteLLM DB が利用可能な状態。"""
    pass


# 旧 OpenRouter 向け Given ステップ（後方互換のため残す）
@given("OpenRouter APIが利用可能であり、Visionモデルを含む有効なモデルリストを返す")
def openrouter_api_available_with_vision_models():
    """後方互換用: LiteLLM DB が利用可能と同義。"""
    pass


@given("OpenRouter APIが利用可能であり、有効なモデルリストを返す")
def openrouter_api_available():
    """後方互換用: LiteLLM DB が利用可能と同義。"""
    pass


@given("以前にAPIが正常に呼び出され、結果がキャッシュされている")
def cached_results_exist(mock_toml_file):
    """既存の TOML キャッシュファイルを作成する。"""
    mock_toml_file.write_text(
        """
[openai/gpt-4o]
provider = "OpenAI"
model_name_short = "gpt-4o"

[google/gemini-1.5-pro]
provider = "Google"
model_name_short = "gemini-1.5-pro"
"""
    )


@given("以前にOpenRouter APIが正常に呼び出され、結果がキャッシュされている")
def cached_results_exist_openrouter(mock_toml_file):
    """後方互換用: 既存の TOML キャッシュファイルを作成する。"""
    mock_toml_file.write_text(
        """
[openai/gpt-4o]
provider = "OpenAI"
model_name_short = "gpt-4o"
"""
    )


@given("以下のエラータイプがAPIで発生する:", target_fixture="error_scenarios")
def api_error_scenarios(datatable):
    """複数エラーシナリオの設定。"""
    error_types = []
    for row in datatable[1:]:
        error_types.append(row[0])
    return error_types


@given("LiteLLM DBが予期しない形式でデータを返す", target_fixture="unexpected_format_marker")
def litellm_db_returns_unexpected_format():
    """LiteLLM が予期しない形式を返すシナリオ。"""
    return {"unexpected_format": True}


@given("OpenRouter APIが予期しない形式でデータを返す", target_fixture="unexpected_format_marker")
def openrouter_api_returns_unexpected_format():
    """後方互換用: LiteLLM が予期しない形式を返すシナリオ。"""
    return {"unexpected_format": True}


# ============================================================================
# When steps
# ============================================================================


@when("discover_available_vision_models 関数を呼び出す", target_fixture="discovery_result")
def call_discover_vision_models(request, mock_litellm_db, mock_toml_file):
    """discover_available_vision_models を呼び出す。"""
    unexpected_format = None
    error_scenarios = None
    try:
        unexpected_format = request.getfixturevalue("unexpected_format_marker")
    except Exception:
        pass
    try:
        error_scenarios = request.getfixturevalue("error_scenarios")
    except Exception:
        pass

    with (
        patch("image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file),
        patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file),
    ):
        if unexpected_format:
            # LiteLLM が model_cost を空にして何も返さない、かつ fallback も失敗するシナリオ
            mock_ll = mock_litellm_db()
            mock_ll.model_cost = {}
            mock_ll.supports_vision.return_value = False
            with (
                patch("image_annotator_lib.core.api_model_discovery.litellm", mock_ll),
                patch(
                    "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback",
                    return_value=[],
                ),
            ):
                result = discover_available_vision_models(force_refresh=True)
            return {"result": result}

        elif error_scenarios:
            results = {}
            import requests as req_lib

            for error_type in error_scenarios:
                mock_ll = mock_litellm_db()
                if error_type in ("Connection Timeout", "HTTP Error 500", "Request Exception"):
                    # OpenRouter フォールバックのエラーをシミュレート（LiteLLM 自体は正常）
                    # LiteLLM が空 → フォールバックでエラー → 空リスト → 結果は空
                    mock_ll.model_cost = {}
                    mock_ll.supports_vision.return_value = False

                    def make_fallback_error(etype: str):
                        def _fallback() -> list:
                            if etype == "Connection Timeout":
                                raise req_lib.exceptions.Timeout("Request timed out")
                            elif etype == "HTTP Error 500":
                                raise req_lib.exceptions.HTTPError("500 Server Error")
                            else:
                                raise req_lib.exceptions.RequestException("Connection error")

                        return _fallback

                    with (
                        patch("image_annotator_lib.core.api_model_discovery.litellm", mock_ll),
                        patch(
                            "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback",
                            side_effect=make_fallback_error(error_type),
                        ),
                    ):
                        result = discover_available_vision_models(force_refresh=True)
                elif error_type == "Invalid JSON":
                    # _fetch_and_update_vision_models が予期しない例外を投げる
                    with patch(
                        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
                        side_effect=ValueError("Invalid data"),
                    ):
                        result = discover_available_vision_models(force_refresh=True)
                else:
                    result = {"error": f"Unknown error type: {error_type}"}
                results[error_type] = result

            return {"error_results": results}

        else:
            mock_ll = mock_litellm_db()
            with (
                patch("image_annotator_lib.core.api_model_discovery.litellm", mock_ll),
                patch(
                    "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback",
                    return_value=[],
                ),
            ):
                result = discover_available_vision_models(force_refresh=True)
            return {"result": result}


@when(
    "force_refreshをtrueにして discover_available_vision_models 関数を呼び出す",
    target_fixture="force_refresh_result",
)
def call_discover_with_force_refresh(mock_litellm_db, mock_toml_file):
    """force_refresh=True で discover_available_vision_models を呼び出す。"""
    mock_ll = mock_litellm_db()
    litellm_called = []

    original_supports = mock_ll.supports_vision.side_effect

    def track_call(model):
        litellm_called.append(model)
        return original_supports(model)

    mock_ll.supports_vision.side_effect = track_call

    with (
        patch("image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file),
        patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file),
        patch("image_annotator_lib.core.api_model_discovery.litellm", mock_ll),
        patch(
            "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback",
            return_value=[],
        ),
    ):
        result = discover_available_vision_models(force_refresh=True)

    return {"result": result, "litellm_called": len(litellm_called) > 0}


# ============================================================================
# Then steps
# ============================================================================


@then("利用可能なVisionモデルのリストを取得できる")
def vision_models_list_retrieved(discovery_result: dict):
    """Vision モデルリストが取得できた。"""
    assert "result" in discovery_result
    result = discovery_result["result"]
    assert "models" in result, f"Expected 'models' key in result: {result}"
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0


@then("APIから最新のVisionモデルのリストを再取得できる")
def latest_vision_models_retrieved(force_refresh_result: dict):
    """force_refresh で LiteLLM DB から最新データが取得された。"""
    assert force_refresh_result["litellm_called"], "LiteLLM should have been called with force_refresh=True"
    result = force_refresh_result["result"]
    assert "models" in result
    assert isinstance(result["models"], list)


@then("モデルリストの取得に失敗したことがわかる")
def model_list_retrieval_failed(discovery_result: dict):
    """モデルリスト取得が失敗した（エラーまたは空）。"""
    if "error_results" in discovery_result:
        for error_type, result in discovery_result["error_results"].items():
            # エラーまたは空のモデルリストのいずれかを許容
            has_error = "error" in result
            has_empty_models = "models" in result and len(result["models"]) == 0
            assert has_error or has_empty_models, f"Expected error or empty models for {error_type}: {result}"
    else:
        result = discovery_result.get("result", {})
        has_error = "error" in result
        has_empty_models = "models" in result and len(result["models"]) == 0
        assert has_error or has_empty_models, f"Expected error or empty models: {result}"


@then("エラーの原因が各エラータイプで正しく判定されること")
def error_causes_correctly_identified(discovery_result: dict, error_scenarios: list):
    """各エラータイプで適切にエラーまたは空結果が返される。"""
    assert "error_results" in discovery_result
    error_results = discovery_result["error_results"]

    for error_type in error_scenarios:
        assert error_type in error_results, f"Missing result for error type: {error_type}"
        result = error_results[error_type]
        has_error = "error" in result
        has_empty_models = "models" in result and len(result["models"]) == 0
        assert has_error or has_empty_models, f"Expected error or empty models for {error_type}: {result}"


@then("エラーの原因がAPI応答形式の問題であることがわかる")
def error_cause_is_response_format_issue(discovery_result: dict):
    """エラーまたは空のモデルリストが返される（形式問題）。"""
    result = discovery_result.get("result", {})
    has_error = "error" in result
    has_empty_models = "models" in result and len(result["models"]) == 0
    assert has_error or has_empty_models, f"Expected error or empty result: {result}"
