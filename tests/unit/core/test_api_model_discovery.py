"""Unit tests for api_model_discovery - LiteLLM-driven discovery."""

from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.api_model_discovery import (
    _fetch_from_litellm,
    _fetch_from_openrouter_fallback,
    _format_litellm_model_for_toml,
    is_allowed_provider,
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
    "groq/llama-3.1-70b": {"max_tokens": 8192},  # 許可外プロバイダー → スキップされる
    "cohere/command-r": {"max_tokens": 4096},  # 許可外プロバイダー → スキップされる
}

MOCK_VISION_MODEL_IDS = {"openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "google/gemini-1.5-pro"}

MOCK_MODEL_INFO = {
    "mode": "chat",
    "max_tokens": 16384,
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "supports_vision": True,
    "supports_response_schema": True,
    "supports_function_calling": True,
    "supports_tool_choice": True,
    "supports_parallel_function_calling": True,
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
        mock_ll.get_model_info.side_effect = lambda model: (
            MOCK_MODEL_INFO.copy() if model in MOCK_VISION_MODEL_IDS else None
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
# is_allowed_provider() のテスト
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id",
    [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "gemini/gemini-1.5-pro",
        "vertex_ai/gemini-1.5-pro",
        "google/gemini-1.5-pro",
        "openrouter/anthropic/claude-3-opus",
    ],
)
def test_is_allowed_provider_accepts_three_majors_and_openrouter(model_id):
    """OpenAI / Anthropic / Google (gemini/vertex_ai/google) / OpenRouter は許可される。"""
    assert is_allowed_provider(model_id) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id",
    [
        "groq/llama-3.1-70b",
        "cohere/command-r",
        "mistral/mistral-large",
        "azure/gpt-4o",  # Azure は許可外 (OpenAI 直接ではない)
        "xai/grok-2",
        "gpt-4o",  # prefix なし
        "",
    ],
)
def test_is_allowed_provider_rejects_others(model_id):
    """三大プロバイダーと OpenRouter 以外は除外される。"""
    assert is_allowed_provider(model_id) is False


# ==============================================================================
# _fetch_from_litellm() のテスト
# ==============================================================================


@pytest.mark.unit
def test_fetch_from_litellm_returns_only_vision_models(mock_litellm):
    """Vision + structured output 対応モデルのみ返し、非対応・prefix なし ID は除外される。"""
    results = _fetch_from_litellm()
    ids = {m["id"] for m in results}

    assert "openai/gpt-4o" in ids
    assert "anthropic/claude-3-5-sonnet-20241022" in ids
    assert "google/gemini-1.5-pro" in ids
    assert "openai/gpt-3.5-turbo" not in ids  # Vision 非対応
    assert "gpt-4o" not in ids  # prefix なし → スキップ
    assert "groq/llama-3.1-70b" not in ids  # 許可外プロバイダー → スキップ
    assert "cohere/command-r" not in ids  # 許可外プロバイダー → スキップ


@pytest.mark.unit
def test_fetch_from_litellm_skips_disallowed_providers_before_litellm_calls(mock_litellm):
    """許可外プロバイダーは supports_vision / get_model_info を呼ぶ前に除外される。"""
    _ = _fetch_from_litellm()

    called_models = {call.args[0] for call in mock_litellm.supports_vision.call_args_list}
    assert "groq/llama-3.1-70b" not in called_models
    assert "cohere/command-r" not in called_models
    assert "openai/gpt-4o" in called_models


@pytest.mark.unit
def test_fetch_from_litellm_excludes_vision_model_without_response_schema(mock_litellm):
    """Vision 対応でも structured output 非対応なら除外される。"""

    def get_model_info(model: str):
        info = MOCK_MODEL_INFO.copy()
        if model == "google/gemini-1.5-pro":
            info["supports_response_schema"] = False
        return info if model in MOCK_VISION_MODEL_IDS else None

    mock_litellm.get_model_info.side_effect = get_model_info

    results = _fetch_from_litellm()
    ids = {m["id"] for m in results}

    assert "openai/gpt-4o" in ids
    assert "google/gemini-1.5-pro" not in ids


@pytest.mark.unit
def test_fetch_from_litellm_excludes_non_chat_modes(mock_litellm):
    """画像生成など AnnotationSchema 実行対象外 mode は除外される。"""

    def get_model_info(model: str):
        info = MOCK_MODEL_INFO.copy()
        if model == "google/gemini-1.5-pro":
            info["mode"] = "image_generation"
        return info if model in MOCK_VISION_MODEL_IDS else None

    mock_litellm.get_model_info.side_effect = get_model_info

    results = _fetch_from_litellm()
    ids = {m["id"] for m in results}

    assert "openai/gpt-4o" in ids
    assert "google/gemini-1.5-pro" not in ids


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
    for key in (
        "id",
        "provider",
        "model_name_short",
        "display_name",
        "mode",
        "max_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "supports_vision",
        "supports_response_schema",
        "supports_function_calling",
        "supports_tool_choice",
    ):
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
    info = {
        "mode": "chat",
        "max_tokens": 8192,
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
    }
    result = _format_litellm_model_for_toml("google/gemini-1.5-pro", info)

    assert result is not None
    assert result["max_tokens"] == 8192
    assert result["input_cost_per_token"] == 3e-06
    assert result["output_cost_per_token"] == 1.5e-05


@pytest.mark.unit
def test_format_litellm_model_capability_fields():
    """LiteLLM capability 情報が TOML 保存形式に引き継がれる。"""
    result = _format_litellm_model_for_toml("openai/gpt-4o", MOCK_MODEL_INFO.copy())

    assert result is not None
    assert result["supports_vision"] is True
    assert result["supports_response_schema"] is True
    assert result["supports_function_calling"] is True
    assert result["supports_tool_choice"] is True
    assert result["supports_parallel_function_calling"] is True
    assert result["max_input_tokens"] == 128000
    assert result["max_output_tokens"] == 16384


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
    assert "toml_data" in result
    assert "openai/gpt-4o" in result["models"]
    assert "openai/gpt-4o" in result["toml_data"]
    mock_litellm.supports_vision.assert_not_called()


@pytest.mark.unit
def test_discover_filters_disallowed_providers_from_cache(mock_litellm, mock_toml_paths):
    """既存 TOML に許可外プロバイダーのエントリが残っていても結果から除外される。"""
    mock_toml_paths.write_text(
        '[openai/gpt-4o]\nprovider="OpenAI"\n'
        '[groq/llama-3.1-70b]\nprovider="Groq"\n'
        '[cohere/command-r]\nprovider="Cohere"\n'
    )

    result = discover_available_vision_models(force_refresh=False)

    assert "openai/gpt-4o" in result["models"]
    assert "groq/llama-3.1-70b" not in result["models"]
    assert "cohere/command-r" not in result["models"]
    assert "groq/llama-3.1-70b" not in result["toml_data"]
    assert "cohere/command-r" not in result["toml_data"]


@pytest.mark.unit
def test_discover_calls_litellm_when_toml_empty(mock_litellm, mock_toml_paths):
    """force_refresh=False かつ TOML が空の場合、LiteLLM から取得する。

    load_available_api_models は lru_cache を持つため、直接モックして空 dict を返す。
    """
    with (
        patch("image_annotator_lib.core.api_model_discovery.load_available_api_models", return_value={}),
        patch("image_annotator_lib.core.api_model_discovery.save_available_api_models"),
        patch(
            "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback", return_value=[]
        ),
    ):
        result = discover_available_vision_models(force_refresh=False)

    assert "models" in result
    assert "toml_data" in result
    mock_litellm.supports_vision.assert_called()


@pytest.mark.unit
def test_discover_force_refresh_calls_litellm(mock_litellm, mock_toml_paths):
    """force_refresh=True で LiteLLM から取得し、TOML を更新する。"""
    mock_toml_paths.write_text('[openai/gpt-4o]\nprovider="OpenAI"\n')

    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_from_openrouter_fallback", return_value=[]
    ):
        result = discover_available_vision_models(force_refresh=True)

    assert "models" in result
    assert "toml_data" in result
    assert "openai/gpt-4o" in result["models"]
    assert "openai/gpt-4o" in result["toml_data"]
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
        "openai/old-model": {
            "provider": "OpenAI",
            "deprecated_on": None,
            "last_seen": "2025-01-01T00:00:00Z",
        }
    }
    api_models: list = []  # API からは何も返らない

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert result["openai/old-model"]["deprecated_on"] == "2026-04-27T00:00:00Z"


@pytest.mark.unit
def test_update_toml_does_not_overwrite_existing_deprecated_on():
    """既に deprecated_on が設定されているモデルは上書きしない。"""
    existing_date = "2025-06-01T00:00:00Z"
    existing = {"openai/very-old-model": {"provider": "OpenAI", "deprecated_on": existing_date}}
    api_models: list = []

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert result["openai/very-old-model"]["deprecated_on"] == existing_date


@pytest.mark.unit
def test_update_toml_drops_disallowed_providers_from_existing():
    """既存 TOML に許可外プロバイダーが残っていたら結果から物理削除される。"""
    existing = {
        "openai/gpt-4o": {"provider": "OpenAI", "deprecated_on": None},
        "xai/grok-2": {"provider": "Xai", "deprecated_on": None},
        "vercel/v0-1.5-md": {"provider": "Vercel", "deprecated_on": None},
        "zai-org/glm-4.5v": {"provider": "Zai-org", "deprecated_on": None},
    }
    api_models: list = []

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert "openai/gpt-4o" in result
    assert "xai/grok-2" not in result
    assert "vercel/v0-1.5-md" not in result
    assert "zai-org/glm-4.5v" not in result


@pytest.mark.unit
def test_update_toml_drops_disallowed_providers_from_api_results():
    """API 結果に許可外プロバイダーが紛れていても TOML には書き込まれない。"""
    existing: dict = {}
    api_models = [
        {"id": "openai/gpt-4o", "provider": "OpenAI", "mode": "chat"},
        {"id": "xai/grok-2", "provider": "Xai", "mode": "chat"},
    ]

    result = _update_toml_with_api_results(existing, api_models, "2026-04-27T00:00:00Z")

    assert "openai/gpt-4o" in result
    assert "xai/grok-2" not in result


@pytest.mark.unit
def test_discover_force_refresh_filters_disallowed_from_fetch_result(mock_litellm, mock_toml_paths):
    """force_refresh パスでも戻り値から許可外プロバイダーが除外される。

    _fetch_and_update_vision_models が許可外を返してきても、最終的な return 時点でフィルタする
    防御的二段フィルタを検証する。
    """
    mock_toml_paths.write_text("")

    fake_updated = {
        "openai/gpt-4o": {"provider": "OpenAI"},
        "xai/grok-2": {"provider": "Xai"},
    }
    with patch(
        "image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models",
        return_value=fake_updated,
    ):
        result = discover_available_vision_models(force_refresh=True)

    assert "openai/gpt-4o" in result["models"]
    assert "xai/grok-2" not in result["models"]
    assert "xai/grok-2" not in result["toml_data"]
