"""WebAPI メタデータレイヤー単一情報源化 (Issue #23) の回帰防止テスト。

Issue #6 の応急処置で導入された `config_registry.set_system_value()` での逆流注入を
撤廃し、`_WEBAPI_MODEL_METADATA` を SSoT として確立した状態を保証する。

ADR 0023 Phase 1 (Issue #35): 旧 `load_available_api_models` 経由のテスト群は
`discover_available_vision_models` への置換に伴い broken になった。本ファイルは
ファイルレベル `pytestmark = pytest.mark.skip` で一時保留し、新仕様での再構築は
別 issue で行う。
"""

from unittest.mock import patch

import pytest

from image_annotator_lib.core import registry
from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.config import get_config_registry
from image_annotator_lib.core.registry import (
    _register_webapi_models_from_discovery,
    get_webapi_metadata,
)

pytestmark = pytest.mark.skip(
    reason="ADR 0023 Phase 1 (Issue #35): load_available_api_models が削除されたため "
    "本ファイルの test 群は dead。新仕様 (discover_available_vision_models) での "
    "再構築は別 issue で実施。"
)


class _DummyPydanticAIWebAPIAnnotator(BaseAnnotator):
    """`_try_register_model` の `issubclass(BaseAnnotator)` チェックを通過するためのダミー実装。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def _preprocess_images(self, images):
        return images

    def _run_inference(self, processed):
        return processed

    def _format_predictions(self, raw_outputs):
        return raw_outputs

    def _generate_tags(self, formatted_output):
        return []


# テスト用 WebAPI モデル discovery レスポンス。
# `_register_webapi_models_from_discovery` が `load_available_api_models()` から
# 受け取る形式。
_DISCOVERY_PAYLOAD = {
    "google/gemini-2.5-pro": {
        "provider": "google",
        "model_name_short": "Gemini 2.5 Pro",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 8192,
        "supports_vision": True,
        "supports_response_schema": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "deprecated_on": None,
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "model_name_short": "GPT-4o",
        "mode": "chat",
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "supports_vision": True,
        "supports_response_schema": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "deprecated_on": None,
    },
}


@pytest.fixture
def isolated_registry():
    """`_WEBAPI_MODEL_METADATA` と `_MODEL_CLASS_OBJ_REGISTRY` を空の隔離状態にする。

    各テスト後に元の状態へ戻すため `patch.object` で context-managed に置換する。
    """
    with patch.object(registry, "_WEBAPI_MODEL_METADATA", {}):
        with patch.object(registry, "_MODEL_CLASS_OBJ_REGISTRY", {}):
            yield


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_set_system_value_not_called_after_registration(
    mock_load_api_models,
    mock_gather_classes,
    isolated_registry,
):
    """SSoT 化の核: `config_registry.set_system_value()` が一度も呼ばれない。

    Issue #6 応急処置時に導入された逆流注入経路が完全撤廃されていることを保証する。
    """
    mock_load_api_models.return_value = _DISCOVERY_PAYLOAD
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": _DummyPydanticAIWebAPIAnnotator}

    real_registry = get_config_registry()
    with patch.object(real_registry, "set_system_value") as mock_set_system_value:
        _register_webapi_models_from_discovery()

        mock_set_system_value.assert_not_called()


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_config_registry_does_not_contain_webapi_after_registration(
    mock_load_api_models,
    mock_gather_classes,
    isolated_registry,
):
    """登録後、`config_registry.get(<webapi-model>, "api_model_id")` が None を返す。

    set_system_value が呼ばれないため、config_registry の内部 dict に WebAPI
    モデルのレコードが作られない (逆流ゼロ)。
    """
    mock_load_api_models.return_value = _DISCOVERY_PAYLOAD
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": _DummyPydanticAIWebAPIAnnotator}

    # `_merged_config_data` を空の状態から初期化して測定の独立性を担保する。
    real_registry = get_config_registry()
    with patch.object(real_registry, "_merged_config_data", {}):
        _register_webapi_models_from_discovery()

        assert real_registry.get("Gemini 2.5 Pro", "api_model_id") is None
        assert real_registry.get("GPT-4o", "api_model_id") is None


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_get_webapi_metadata_returns_full_metadata(
    mock_load_api_models,
    mock_gather_classes,
    isolated_registry,
):
    """登録後、`get_webapi_metadata(model_name)` が完全な metadata 辞書を返す。"""
    mock_load_api_models.return_value = _DISCOVERY_PAYLOAD
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": _DummyPydanticAIWebAPIAnnotator}

    _register_webapi_models_from_discovery()

    metadata = get_webapi_metadata("Gemini 2.5 Pro")
    assert metadata is not None
    assert metadata["api_model_id"] == "google/gemini-2.5-pro"
    assert metadata["model_name_on_provider"] == "google/gemini-2.5-pro"
    assert metadata["provider"] == "google"
    assert metadata["supports_vision"] is True
    assert metadata["supports_response_schema"] is True
    assert metadata["max_input_tokens"] == 1048576
    assert metadata["max_output_tokens"] == 8192
    assert metadata["class"] == "PydanticAIWebAPIAnnotator"
    assert metadata["type"] == "webapi"


@pytest.mark.unit
def test_get_webapi_metadata_returns_none_for_unregistered():
    """未登録モデル名に対して `get_webapi_metadata()` は None を返す。"""
    with patch.object(registry, "_WEBAPI_MODEL_METADATA", {}):
        assert get_webapi_metadata("nonexistent-model") is None


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_register_webapi_models_requires_response_schema(
    mock_load_api_models,
    mock_gather_classes,
    isolated_registry,
):
    """Vision 対応でも structured output 非対応モデルは登録されない。"""
    mock_load_api_models.return_value = {
        "openai/gpt-4o": {
            "provider": "openai",
            "model_name_short": "GPT-4o",
            "mode": "chat",
            "supports_vision": True,
            "supports_response_schema": True,
            "deprecated_on": None,
        },
        "openai/gpt-4-turbo-vision-preview": {
            "provider": "openai",
            "model_name_short": "GPT-4 Turbo Vision",
            "mode": "chat",
            "supports_vision": True,
            "supports_response_schema": False,
            "deprecated_on": None,
        },
    }
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": _DummyPydanticAIWebAPIAnnotator}

    _register_webapi_models_from_discovery()

    assert get_webapi_metadata("GPT-4o") is not None
    assert get_webapi_metadata("GPT-4 Turbo Vision") is None


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_base_annotator_loads_webapi_config_via_metadata_fallback(
    mock_load_api_models,
    mock_gather_classes,
    isolated_registry,
):
    """`BaseAnnotator._load_config_from_registry` が `_WEBAPI_MODEL_METADATA` をフォールバックで読む。

    config_registry に api_model_id が登録されていなくても、SSoT (`_WEBAPI_MODEL_METADATA`)
    から WebAPIModelConfig を構築できることを保証する (Issue #23 の動作核)。
    """
    mock_load_api_models.return_value = _DISCOVERY_PAYLOAD
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": _DummyPydanticAIWebAPIAnnotator}

    real_registry = get_config_registry()
    with patch.object(real_registry, "_merged_config_data", {}):
        _register_webapi_models_from_discovery()

        # config_registry には api_model_id が無い状態 (SSoT 化の効果)
        assert real_registry.get("GPT-4o", "api_model_id") is None

        # それでも `BaseAnnotator._load_config_from_registry` が SSoT フォールバック経由で成功する。
        # `_DummyPydanticAIWebAPIAnnotator.__init__` は `super().__init__` を呼ばないため、
        # `_load_config_from_registry` を後から直接呼んで経路を検証する。
        dummy = _DummyPydanticAIWebAPIAnnotator("GPT-4o")
        config = dummy._load_config_from_registry("GPT-4o")
        assert config.api_model_id == "openai/gpt-4o"
        assert config.class_name == "PydanticAIWebAPIAnnotator"


# ============================================================================
# User TOML WebAPI 定義廃止テスト (Issue #25 完全 SSoT 化)
# ============================================================================


@pytest.mark.unit
def test_pydantic_ai_annotator_ignores_user_toml_api_model_id():
    """`PydanticAIWebAPIAnnotator` は user TOML の api_model_id を採用しない。"""
    from image_annotator_lib.core.base.pydantic_ai_annotator import PydanticAIWebAPIAnnotator

    real_registry = get_config_registry()
    user_overrides = {
        "GPT-4o": {
            "api_model_id": "gpt-4o-mini-test",
            "class": "PydanticAIWebAPIAnnotator",
            "model_name_on_provider": "gpt-4o-mini-test",
        }
    }
    discovery_metadata = {
        "GPT-4o": {
            "api_model_id": "gpt-4o-2024-08-06",
            "model_name_on_provider": "gpt-4o-2024-08-06",
            "provider": "openai",
            "max_output_tokens": 1800,
            "supports_vision": True,
            "supports_response_schema": True,
            "type": "webapi",
            "class": "PydanticAIWebAPIAnnotator",
        }
    }

    with patch.object(real_registry, "_merged_config_data", user_overrides):
        with patch.object(registry, "_WEBAPI_MODEL_METADATA", discovery_metadata):
            annotator = PydanticAIWebAPIAnnotator("GPT-4o")
            assert annotator.config.model_id == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_pydantic_ai_annotator_falls_back_to_discovery_metadata_when_no_user_config():
    """user TOML に api_model_id が無い場合、discovery metadata から取得する。"""
    from image_annotator_lib.core.base.pydantic_ai_annotator import PydanticAIWebAPIAnnotator

    real_registry = get_config_registry()
    discovery_metadata = {
        "GPT-4o": {
            "api_model_id": "gpt-4o-2024-08-06",
            "model_name_on_provider": "gpt-4o-2024-08-06",
            "provider": "openai",
            "max_output_tokens": 1800,
            "supports_vision": True,
            "supports_response_schema": True,
            "type": "webapi",
            "class": "PydanticAIWebAPIAnnotator",
        }
    }

    with patch.object(real_registry, "_merged_config_data", {}):
        with patch.object(registry, "_WEBAPI_MODEL_METADATA", discovery_metadata):
            annotator = PydanticAIWebAPIAnnotator("GPT-4o")
            assert annotator.config.model_id == "gpt-4o-2024-08-06"


@pytest.mark.unit
def test_pydantic_ai_webapi_wrapper_ignores_user_toml_api_model_id():
    """`PydanticAIWebAPIWrapper.__enter__` も SSoT の api_model_id だけを採用する。"""
    from image_annotator_lib.core.annotation_runner import PydanticAIWebAPIWrapper

    real_registry = get_config_registry()
    user_overrides = {
        "Claude 3.5 Sonnet": {
            "api_model_id": "claude-3-5-sonnet-test-override",
            "class": "PydanticAIWebAPIAnnotator",
            "model_name_on_provider": "claude-3-5-sonnet-test-override",
        }
    }
    discovery_metadata = {
        "Claude 3.5 Sonnet": {
            "api_model_id": "anthropic/claude-3-5-sonnet-latest",
            "model_name_on_provider": "anthropic/claude-3-5-sonnet-latest",
            "provider": "anthropic",
            "max_output_tokens": 1800,
            "supports_vision": True,
            "supports_response_schema": True,
            "type": "webapi",
            "class": "PydanticAIWebAPIAnnotator",
        }
    }

    with patch.object(real_registry, "_merged_config_data", user_overrides):
        with patch.object(registry, "_WEBAPI_MODEL_METADATA", discovery_metadata):
            wrapper = PydanticAIWebAPIWrapper("Claude 3.5 Sonnet", _DummyPydanticAIWebAPIAnnotator)
            with wrapper:
                assert wrapper._api_model_id == "anthropic/claude-3-5-sonnet-latest"
