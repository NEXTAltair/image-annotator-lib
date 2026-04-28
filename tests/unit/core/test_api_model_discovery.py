"""Unit tests for api_model_discovery - LiteLLM-driven discovery."""

from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.api_model_discovery import (
    _fetch_from_litellm,
    _fetch_from_openrouter_fallback,
    _format_litellm_model_for_toml,
    _update_toml_with_api_results,
    discover_available_vision_models,
)

# --- テスト用定数 ---

MOCK_LITELLM_MODEL_COST = {
    "openai/gpt-4o": {"max_tokens": 16384, "input_cost_per_token": 2.5e-06},
    "openai/gpt-3.5-turbo": {"max_tokens": 4096},
    "anthropic/claude-3-5-sonnet-20241022": {"max_tokens": 8192},
    "google/gemini-1.5-pro": {"max_tokens": 2097152},
    "gpt-4o": {"max_tokens": 16384},  # prefix なしエイリアス → スキップされる
}

MOCK_VISION_MODEL_IDS = {"openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "google/gemini-1.5-pro"}

MOCK_MODEL_INFO = {
    "mode": "chat",
    "max_tokens": 16384,
    "input_cost_per_token": 2.5e-06,
    "output_cost_per_token": 1.0e-05,
}


# --- フィクスチャ ---


@pytest.fixture
def mock_litellm():
    """litellm モジュールをモック。model_cost, supports_vision, get_model_info を制御する。"""
    with patch("image_annotator_lib.core.api_model_discovery.litellm") as mock_ll:
        mock_ll.model_cost = MOCK_LITELLM_MODEL_COST
        mock_ll.supports_vision.side_effect = lambda model: model in MOCK_VISION_MODEL_IDS
        mock_ll.get_model_info.side_effect = (
            lambda model: MOCK_MODEL_INFO.copy() if model in MOCK_VISION_MODEL_IDS else None
        )
        yield mock_ll


@pytest.fixture
def mock_toml_paths(tmp_path):
    """TOML ファイルパスをテスト用一時ディレクトリにリダイレクトする。"""
    toml_path = tmp_path / "available_api_models.toml"
    with (
        patch("image_annotator_lib.core.api_model_discovery.AVAILABLE_API_MODELS_CONFIG_PATH", toml_path),
        patch("image_annotator_lib.core.config.AVAILABLE_API_MODELS_CONFIG_PATH", toml_path),
    ):
        yield toml_path


# ==============================================================================
# _fetch_from_litellm() のテスト
# ==============================================================================


@pytest.mark.unit
def test_fetch_from_litellm_returns_only_vision_models(mock_litellm):
    """Vision 対応モデルのみ返し、非対応・prefix なし ID は除外される。"""
    results = _fetch_from_litellm()
    ids = {m["id"] for m in results}

    assert "openai/gpt-4o" in ids
    assert "anthropic/claude-3-5-sonnet-20241022" in ids
    assert "google/gemini-1.5-pro" in ids
    assert "openai/gpt-3.5-turbo" not in ids  # Vision 非対応
    assert "gpt-4o" not in ids  # prefix なし → スキップ


@pytest.mark.unit
def test_fetch_from_litellm_result_contains_id(mock_litellm):
    """返されたエントリに 'id' キーが含まれる（_update_toml_with_api_results 用）。"""
    results = _fetch_from_litellm()
    for entry in results:
        assert "id" in entry


@pytest.mark.unit
def test_fetch_from_litellm_skips_on_exception(mock_litellm):
    """supports_vision が例外を投げても他のモデルの処理は継続する。"""
    def supports_vision_with_error(model: str) -> bool:
        if model == "anthropic/claude-3-5-sonnet-20241022":
            raise RuntimeError("API error")
        return model in MOCK_VISION_MODEL_IDS

    mock_litellm.supports_vision.side_effect = supports_vision_with_error

    results = _fetch_from_litellm()
    ids = {m["id"] for m in results}

    assert "openai/gpt-4o" in ids
    assert "anthropic/claude-3-5-sonnet-20241022" not in ids  # 例外でスキップ
    assert "google/gemini-1.5-pro" in ids


# ==============================================================================
# _format_litellm_model_for_toml() のテスト
# ==============================================================================


@pytest.mark.unit
def test_format_litellm_model_required_keys():
    """整形結果が必要なキーをすべて持つ。"""
    info = MOCK_MODEL_INFO.copy()
    result = _format_litellm_model_for_toml("openai/gpt-4o", info)

    assert result is not None
    for key in ("id", "provider", "model_name_short", "display_name", "mode", "max_tokens"):
        assert key in result, f"Key '{key}' missing from result"


@pytest.mark.unit
def test_format_litellm_model_openai_provider_capitalization():
    """openai プロバイダーは 'OpenAI' に変換される。"""
    result = _format_litellm_model_for_toml("openai/gpt-4o", MOCK_MODEL_INFO.copy())

    assert result is not None
    assert result["provider"] == "OpenAI"
    assert result["model_name_short"] == "gpt-4o"


@pytest.mark.unit
def test_format_litellm_model_other_provider_capitalize():
    """openai 以外のプロバイダーは capitalize される。"""
    result = _format_litellm_model_for_toml("anthropic/claude-3-5-sonnet", MOCK_MODEL_INFO.copy())

    assert result is not None
    assert result["provider"] == "Anthropic"
    assert result["model_name_short"] == "claude-3-5-sonnet"


@pytest.mark.unit
def test_format_litellm_model_rejects_no_prefix():
    """provider prefix のない model_id は None を返す。"""
    result = _format_litellm_model_for_toml("gpt-4o", MOCK_MODEL_INFO.copy())

    assert result is None


@pytest.mark.unit
def test_format_litellm_model_cost_fields():
    """コスト情報が正しくマッピングされる。"""
    info = {"mode": "chat", "max_tokens": 8192, "input_cost_per_token": 3e-06, "output_cost_per_token": 1.5e-05}
    result = _format_litellm_model_for_toml("google/gemini-1.5-pro", info)

    assert result is not None
    assert result["max_tokens"] == 8192
    assert result["input_cost_per_token"] == 3e-06
    assert result["output_cost_per_token"] == 1.5e-05


# ==============================================================================
# _fetch_from_openrouter_fallback() のテスト
# ==============================================================================


@pytest.mark.unit
def test_fetch_from_openrouter_fallback_network_error_returns_empty():
    """ネットワーク障害時は空リストを返し、例外を伝播しない。"""
    import requests

    with patch("image_annotator_lib.core.api_model_discovery.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        result = _fetch_from_openrouter_fallback()

    assert result == []


@pytest.mark.unit
def test_fetch_from_openrouter_fallback_timeout_returns_empty():
    """タイムアウト時は空リストを返す。"""
    import requests

    with patch("image_annotator_lib.core.api_model_discovery.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = _fetch_from_openrouter_fallback()

    assert result == []


@pytest.mark.unit
def test_fetch_from_openrouter_fallback_returns_vision_models():
    """正常時は Vision 対応モデルのリストを返す。"""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {
                "id": "openrouter/free-model",
                "name": "OpenRouter: Free Model",
                "created": 1700000000,
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["structured_outputs", "tools"],
            }
        ]
    }

    with patch("image_annotator_lib.core.api_model_discovery.requests.get", return_value=mock_response):
        result = _fetch_from_openrouter_fallback()

    assert len(result) == 1
    assert result[0]["id"] == "openrouter/free-model"


# ==============================================================================
# discover_available_vision_models() のテスト
# ==============================================================================


@pytest.mark.unit
def test_discover_uses_cache_when_toml_exists(mock_litellm, mock_toml_paths):
    """force_refresh=False かつ TOML が存在する場合、LiteLLM は呼ばれない。"""
    mock_toml_paths.write_text('[openai/gpt-4o]\nprovider="OpenAI"\n')

    result = discover_available_vision_models(force_refresh=False)

    assert "models" in result
    assert "openai/gpt-4o" in result["models"]
    mock_litellm.supports_vision.assert_not_called()


@pytest.mark.unit
def test_discover_calls_litellm_when_toml_empty(mock_litellm, mock_toml_paths):
    """force_refresh=False かつ TOML が空の場合、LiteLLM から取得する。

    load_available_api_models は lru_cache を持つため、直接モックして空 dict を返す。
    """
    with (
        patch("image_annotator_lib.core.api_model_discovery.load_available_api_models", return_value={}),
        patch("image_annotator_lib.core.api_model_discovery.save_available_api_models"),
        patch("image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback", return_value=[]),
    ):
        result = discover_available_vision_models(force_refresh=False)

    assert "models" in result
    mock_litellm.supports_vision.assert_called()


@pytest.mark.unit
def test_discover_force_refresh_calls_litellm(mock_litellm, mock_toml_paths):
    """force_refresh=True で LiteLLM から取得し、TOML を更新する。"""
    mock_toml_paths.write_text('[openai/gpt-4o]\nprovider="OpenAI"\n')

    with patch("image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback", return_value=[]):
        result = discover_available_vision_models(force_refresh=True)

    assert "models" in result
    assert "openai/gpt-4o" in result["models"]
    mock_litellm.supports_vision.assert_called()
    assert mock_toml_paths.exists()


@pytest.mark.unit
def test_discover_returns_error_on_unexpected_exception(mock_litellm, mock_toml_paths):
    """予期しない例外発生時は {'error': ...} を返す。"""
    mock_litellm.model_cost = {}  # 空にして _fetch_from_litellm が []を返すようにする
    mock_litellm.supports_vision.side_effect = None

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        side_effect=RuntimeError("Unexpected"),
    ):
        result = discover_available_vision_models(force_refresh=True)

    assert "error" in result


# ==============================================================================
# _update_toml_with_api_results() のテスト（既存動作の確認）
# ==============================================================================


@pytest.mark.unit
def test_update_toml_sets_last_seen_on_new_model():
    """新規モデルに last_seen が設定される。"""
    existing: dict = {}
    api_models = [{"id": "openai/gpt-4o", "provider": "OpenAI", "mode": "chat"}]

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert "openai/gpt-4o" in result
    assert result["openai/gpt-4o"]["last_seen"] == "2026-04-27T00:00:00Z"
    assert result["openai/gpt-4o"]["deprecated_on"] is None


@pytest.mark.unit
def test_update_toml_sets_deprecated_on_for_missing_model():
    """API 結果に存在しない既存モデルに deprecated_on が設定される。"""
    existing = {
        "openai/old-model": {"provider": "OpenAI", "deprecated_on": None, "last_seen": "2025-01-01T00:00:00Z"}
    }
    api_models: list = []  # API からは何も返らない

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert result["openai/old-model"]["deprecated_on"] == "2026-04-27T00:00:00Z"


@pytest.mark.unit
def test_update_toml_does_not_overwrite_existing_deprecated_on():
    """既に deprecated_on が設定されているモデルは上書きしない。"""
    existing_date = "2025-06-01T00:00:00Z"
    existing = {
        "openai/very-old-model": {"provider": "OpenAI", "deprecated_on": existing_date}
    }
    api_models: list = []

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert result["openai/very-old-model"]["deprecated_on"] == existing_date
