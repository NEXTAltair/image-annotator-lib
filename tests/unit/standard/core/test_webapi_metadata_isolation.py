"""WebAPI メタデータレイヤー単一情報源化 (Issue #23) の回帰防止テスト。

Issue #6 の応急処置で導入された `config_registry.set_system_value()` での逆流注入を
撤廃し、`_WEBAPI_MODEL_METADATA` を SSoT として確立した状態を保証する。

以下を verify する:
1. `_register_webapi_models_from_discovery` 実行後、登録された WebAPI モデルが
   `config_registry` に **入っていない** こと (逆流ゼロの保証)
2. `get_webapi_metadata()` getter が完全な metadata 辞書を返すこと
3. 未登録モデルに対しては `get_webapi_metadata()` が None を返すこと
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
        "deprecated_on": None,
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "model_name_short": "GPT-4o",
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
