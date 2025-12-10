"""Step definitions for API Model Discovery BDD scenarios."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from pytest_bdd import given, parsers, then, when

from image_annotator_lib.core.api_model_discovery import discover_available_vision_models

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_openrouter_response():
    """Create a valid OpenRouter API response structure."""

    def _create_response(model_count: int = 3) -> dict:
        models = []
        for i in range(model_count):
            models.append(
                {
                    "id": f"test/model-{i}",
                    "name": f"Test Model {i}",
                    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                    "context_length": 128000,
                }
            )
        return {"data": models}

    return _create_response


@pytest.fixture
def mock_cache_directory(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_toml_file(tmp_path):
    """Create a temporary TOML file for caching."""
    toml_file = tmp_path / "available_api_models.toml"
    return toml_file


# ============================================================================
# Given steps
# ============================================================================


@given("アプリケーションのキャッシュディレクトリが存在する")
def cache_directory_exists(mock_cache_directory):
    """Verify cache directory exists."""
    assert mock_cache_directory.exists()
    assert mock_cache_directory.is_dir()


@given("OpenRouter APIが利用可能であり、Visionモデルを含む有効なモデルリストを返す")
def openrouter_api_available_with_vision_models(mock_openrouter_response):
    """Mock OpenRouter API to return valid Vision models."""
    # This will be used in the when step with patch
    pass


@given("OpenRouter APIが利用可能であり、有効なモデルリストを返す")
def openrouter_api_available(mock_openrouter_response):
    """Mock OpenRouter API to return valid models."""
    pass


@given("以前にOpenRouter APIが正常に呼び出され、結果がキャッシュされている")
def cached_results_exist(mock_toml_file, mock_openrouter_response):
    """Create cached TOML file with previous results."""
    # Mock the TOML file path
    with patch("image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file):
        # Write a simple TOML structure
        mock_toml_file.write_text(
            """
[cached-model-1]
name = "Cached Model 1"
vision = true

[cached-model-2]
name = "Cached Model 2"
vision = true
"""
        )


@given("以下のエラータイプがAPIで発生する:", target_fixture="error_scenarios")
def api_error_scenarios(datatable):
    """Configure API to raise different error types."""
    error_types = []
    # datatable is a list of lists, skip header row
    for row in datatable[1:]:  # Skip header
        error_types.append(row[0])  # Get first column value
    return error_types


@given("OpenRouter APIが予期しない形式でデータを返す", target_fixture="unexpected_format_marker")
def openrouter_api_returns_unexpected_format():
    """Mock OpenRouter API to return unexpected format."""
    return {"unexpected_format": True}


# ============================================================================
# When steps
# ============================================================================


@when("discover_available_vision_models 関数を呼び出す", target_fixture="discovery_result")
def call_discover_vision_models(request, mock_openrouter_response, mock_toml_file):
    """Call discover_available_vision_models function."""
    # Check if this is an error scenario
    error_scenarios = None
    unexpected_format = None
    try:
        error_scenarios = request.getfixturevalue("error_scenarios")
    except Exception:
        pass
    try:
        unexpected_format = request.getfixturevalue("unexpected_format_marker")
    except Exception:
        pass

    # Patch the TOML file path
    with patch(
        "image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file
    ), patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file):
        if unexpected_format:
            # Test unexpected API response format
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                # Return invalid format (missing "data" key)
                mock_response.json.return_value = {"models": ["invalid-structure"]}
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                result = discover_available_vision_models(force_refresh=True)
                return {"result": result}
        elif error_scenarios:
            # Test each error type
            results = {}
            for error_type in error_scenarios:
                with patch("requests.get") as mock_get:
                    # Configure mock based on error type
                    if error_type == "Connection Timeout":
                        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
                    elif error_type == "HTTP Error 500":
                        mock_response = MagicMock()
                        mock_response.status_code = 500
                        mock_response.text = "Internal Server Error"
                        mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)
                    elif error_type == "Invalid JSON":
                        mock_response = MagicMock()
                        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
                        mock_response.raise_for_status.return_value = None
                        mock_get.return_value = mock_response
                    elif error_type == "Request Exception":
                        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

                    result = discover_available_vision_models(force_refresh=True)
                    results[error_type] = result

            return {"error_results": results}
        else:
            # Normal scenario - mock successful API response
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_openrouter_response(3)
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                result = discover_available_vision_models(force_refresh=True)
                return {"result": result}


@when(
    "force_refreshをtrueにして discover_available_vision_models 関数を呼び出す",
    target_fixture="force_refresh_result",
)
def call_discover_with_force_refresh(mock_openrouter_response, mock_toml_file):
    """Call discover_available_vision_models with force_refresh=True."""
    with patch(
        "image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file
    ), patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", mock_toml_file), patch(
        "requests.get"
    ) as mock_get:
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_openrouter_response(5)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = discover_available_vision_models(force_refresh=True)
        return {"result": result, "api_called": mock_get.called}


# ============================================================================
# Then steps
# ============================================================================


@then("利用可能なVisionモデルのリストを取得できる")
def vision_models_list_retrieved(discovery_result: dict):
    """Verify Vision models list was retrieved."""
    assert "result" in discovery_result
    result = discovery_result["result"]
    assert "models" in result, f"Expected 'models' key in result: {result}"
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0


@then("APIから最新のVisionモデルのリストを再取得できる")
def latest_vision_models_retrieved(force_refresh_result: dict):
    """Verify latest Vision models were retrieved from API."""
    assert force_refresh_result["api_called"], "API should have been called with force_refresh=True"
    result = force_refresh_result["result"]
    assert "models" in result
    assert isinstance(result["models"], list)


@then("モデルリストの取得に失敗したことがわかる")
def model_list_retrieval_failed(discovery_result: dict):
    """Verify model list retrieval failed."""
    if "error_results" in discovery_result:
        # Multiple error scenarios
        for error_type, result in discovery_result["error_results"].items():
            assert "error" in result, f"Expected 'error' key for {error_type}: {result}"
    else:
        # Single error scenario
        result = discovery_result.get("result", {})
        assert "error" in result, f"Expected 'error' key in result: {result}"


@then("エラーの原因が各エラータイプで正しく判定されること")
def error_causes_correctly_identified(discovery_result: dict, error_scenarios: list):
    """Verify error causes are correctly identified."""
    assert "error_results" in discovery_result
    error_results = discovery_result["error_results"]

    for error_type in error_scenarios:
        assert error_type in error_results, f"Missing result for error type: {error_type}"
        result = error_results[error_type]
        assert "error" in result, f"Expected error for {error_type}"
        error_message = result["error"]

        # Verify error message contains relevant information
        if error_type == "Connection Timeout":
            assert "タイムアウト" in error_message or "timeout" in error_message.lower()
        elif error_type == "HTTP Error 500":
            assert "500" in error_message or "サーバーエラー" in error_message
        elif error_type == "Invalid JSON":
            assert "JSON" in error_message or "parse" in error_message.lower()
        elif error_type == "Request Exception":
            assert "接続" in error_message or "Connection" in error_message or "リクエスト" in error_message


@then("エラーの原因がAPI応答形式の問題であることがわかる")
def error_cause_is_response_format_issue(discovery_result: dict):
    """Verify error is due to API response format issue."""
    result = discovery_result.get("result", {})
    assert "error" in result
    error_message = result["error"]
    # Error message should indicate format/structure issue
    assert any(
        keyword in error_message for keyword in ["形式", "format", "構造", "structure", "不正", "invalid"]
    ), f"Error message should indicate format issue: {error_message}"
