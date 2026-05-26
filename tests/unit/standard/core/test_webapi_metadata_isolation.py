"""WebAPI メタデータレイヤー単一情報源化 (Issue #23) の回帰防止テスト。

Issue #6 の応急処置で導入された `config_registry.set_system_value()` での逆流注入を
撤廃し、`_WEBAPI_MODEL_METADATA` を SSoT として確立した状態を保証する。

Issue #269 で現仕様 (ADR 0023 Phase 1 / #45 / Phase 1.x) に合わせ再構築した:

- 旧 `load_available_api_models` モックは `discover_available_vision_models` モックへ
  置換 (`_register_webapi_models_from_discovery` は内部で discovery を呼ぶ)。
- WebAPI モデル登録の主条件は `supports_response_schema` → `supports_vision` +
  `supports_function_calling` (#45)。
- 削除済みの `PydanticAIWebAPIAnnotator` / `PydanticAIWebAPIWrapper` を参照していた
  test は、SSoT 優先 (`_WEBAPI_MODEL_METADATA` が user TOML を override する) の検証へ
  読み替えた。registry 登録クラスは `WebApiAnnotator` に統合済み。
"""

from unittest.mock import patch

import pytest

from image_annotator_lib.core import registry
from image_annotator_lib.core.config import get_config_registry
from image_annotator_lib.core.registry import (
    _register_webapi_models_from_discovery,
    get_webapi_metadata,
)
from image_annotator_lib.webapi.api_model_discovery import _is_litellm_model_annotation_compatible

# `_register_webapi_models_from_discovery` は関数内で
# `from ..webapi.api_model_discovery import discover_available_vision_models` する。
# patch は定義元モジュールを対象にする。
_DISCOVERY_PATCH_TARGET = "image_annotator_lib.webapi.api_model_discovery.discover_available_vision_models"


def _model_info(model_short: str, provider: str) -> dict:
    """`discover_available_vision_models()` の metadata エントリ相当を生成する。

    `_format_litellm_metadata` の出力形式 (Issue #51 以降、`model_name_short` は
    `provider/model` 形式の完全 ID) に合わせる。
    """
    return {
        "provider": provider,
        "model_name_short": model_short,
        "display_name": model_short,
        "mode": "chat",
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "max_tokens": 16384,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_parallel_function_calling": False,
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "deprecation_date": None,
    }


def _discovery_result(*model_shorts: tuple[str, str]) -> dict:
    """`discover_available_vision_models()` の戻り値形式を生成する。"""
    metadata = {short: _model_info(short, provider) for short, provider in model_shorts}
    return {"models": list(metadata.keys()), "metadata": metadata}


# テスト用 WebAPI モデル discovery レスポンス。
_DISCOVERY_RESULT = _discovery_result(
    ("google/gemini-2.5-pro", "Google"),
    ("openai/gpt-4o", "OpenAI"),
)


@pytest.fixture
def isolated_registry():
    """`_WEBAPI_MODEL_METADATA` と `_MODEL_CLASS_OBJ_REGISTRY` を空の隔離状態にする。

    各テスト後に元の状態へ戻すため `patch.object` で context-managed に置換する。
    """
    with patch.object(registry, "_WEBAPI_MODEL_METADATA", {}):
        with patch.object(registry, "_MODEL_CLASS_OBJ_REGISTRY", {}):
            yield


@pytest.mark.unit
def test_set_system_value_not_called_after_registration(isolated_registry):
    """SSoT 化の核: `config_registry.set_system_value()` が一度も呼ばれない。

    Issue #6 応急処置時に導入された逆流注入経路が完全撤廃されていることを保証する。
    """
    real_registry = get_config_registry()
    with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
        with patch.object(real_registry, "set_system_value") as mock_set_system_value:
            _register_webapi_models_from_discovery()

            mock_set_system_value.assert_not_called()


@pytest.mark.unit
def test_config_registry_does_not_contain_webapi_after_registration(isolated_registry):
    """登録後、`config_registry.get(<webapi-model>, ...)` が None を返す。

    set_system_value が呼ばれないため、config_registry の内部 dict に WebAPI
    モデルのレコードが作られない (逆流ゼロ)。
    """
    real_registry = get_config_registry()
    with patch.object(real_registry, "_merged_config_data", {}):
        with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
            _register_webapi_models_from_discovery()

            assert real_registry.get("openai/gpt-4o", "litellm_model_id") is None
            assert real_registry.get("google/gemini-2.5-pro", "litellm_model_id") is None


@pytest.mark.unit
def test_get_webapi_metadata_returns_full_metadata(isolated_registry):
    """登録後、`get_webapi_metadata(model_name)` が完全な metadata 辞書を返す。"""
    with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
        _register_webapi_models_from_discovery()

    metadata = get_webapi_metadata("google/gemini-2.5-pro")
    assert metadata is not None
    assert metadata["litellm_model_id"] == "google/gemini-2.5-pro"
    assert metadata["model_name_on_provider"] == "google/gemini-2.5-pro"
    assert metadata["provider"] == "google"
    assert metadata["supports_vision"] is True
    assert metadata["supports_function_calling"] is True
    assert metadata["type"] == "webapi"
    assert metadata["class"] == "WebApiAnnotator"
    # Issue #82: discovery metadata は rating を含む全 capability を明示する。
    assert metadata["capabilities"] == ["tags", "captions", "scores", "ratings"]


@pytest.mark.unit
def test_discovered_webapi_model_advertises_rating_capability(isolated_registry):
    """Issue #82: discovery 登録モデルは `get_model_capabilities` で RATINGS を含む。

    discovery metadata が `capabilities` を明示しないと fallback で RATINGS が
    欠落し、rating prompt / 正規化経路が `annotate()` から到達不能になる
    (PR #85 Codex review P1)。
    """
    from image_annotator_lib.core.types import TaskCapability
    from image_annotator_lib.core.utils import get_model_capabilities

    with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
        _register_webapi_models_from_discovery()

    caps = get_model_capabilities("google/gemini-2.5-pro")
    assert caps == {
        TaskCapability.TAGS,
        TaskCapability.CAPTIONS,
        TaskCapability.SCORES,
        TaskCapability.RATINGS,
    }


@pytest.mark.unit
def test_get_webapi_metadata_returns_none_for_unregistered():
    """未登録モデル名に対して `get_webapi_metadata()` は None を返す。"""
    with patch.object(registry, "_WEBAPI_MODEL_METADATA", {}):
        assert get_webapi_metadata("nonexistent-model") is None


@pytest.mark.unit
def test_annotation_compatibility_requires_vision_and_function_calling():
    """WebAPI モデル登録の主条件は `supports_vision` + `supports_function_calling` (#45)。

    旧 test (`..._requires_response_schema`) は `supports_response_schema` を判定条件に
    していたが、ADR 0023 Phase 1 (#45) で structured output は PydanticAI default Tool
    Output で得る方針に変わり、`supports_function_calling` が主条件に統一された。
    判定は `discover_available_vision_models` 内の `_is_litellm_model_annotation_compatible`
    が担う。
    """
    compatible = {"mode": "chat", "supports_vision": True, "supports_function_calling": True}
    assert _is_litellm_model_annotation_compatible(compatible) is True

    # vision 対応でも function_calling 非対応なら除外される
    no_function_calling = {"mode": "chat", "supports_vision": True, "supports_function_calling": False}
    assert _is_litellm_model_annotation_compatible(no_function_calling) is False

    # function_calling 対応でも vision 非対応なら除外される
    no_vision = {"mode": "chat", "supports_vision": False, "supports_function_calling": True}
    assert _is_litellm_model_annotation_compatible(no_vision) is False

    # response_schema は判定に使われない (対応有無で結果が変わらない)
    assert (
        _is_litellm_model_annotation_compatible({**compatible, "supports_response_schema": False}) is True
    )


@pytest.mark.unit
def test_webapi_config_resolvable_only_via_metadata_ssot(isolated_registry):
    """WebAPI モデルの設定は `_WEBAPI_MODEL_METADATA` (SSoT) 経由でのみ解決される。

    旧 `PydanticAIWebAPIAnnotator` test (`...falls_back_to_discovery_metadata...`) の
    意図を読み替え。`BaseAnnotator._load_config_from_registry` は `get_webapi_metadata`
    を最優先で参照する。登録後、WebAPI モデルは `get_webapi_metadata` で解決でき、かつ
    `config_registry` 側には一切レコードが作られない (SSoT 一元化 + 逆流ゼロ)。
    """
    real_registry = get_config_registry()
    with patch.object(real_registry, "_merged_config_data", {}):
        with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
            _register_webapi_models_from_discovery()

        # SSoT 経由では解決できる
        metadata = get_webapi_metadata("openai/gpt-4o")
        assert metadata is not None
        assert metadata["model_name_on_provider"] == "openai/gpt-4o"
        # config_registry 側には WebAPI モデルのレコードが無い
        assert real_registry.get_all_config().get("openai/gpt-4o") is None


@pytest.mark.unit
def test_webapi_metadata_ssot_unaffected_by_conflicting_user_toml(isolated_registry):
    """同名 user TOML エントリは `_WEBAPI_MODEL_METADATA` を汚染しない。

    旧 `PydanticAIWebAPIAnnotator` test (`...ignores_user_toml_api_model_id`) の意図を
    読み替え。`_register_webapi_models_from_discovery` は discovery 結果のみから
    `_WEBAPI_MODEL_METADATA` を構築し、config_registry / user TOML を入力に使わない。
    矛盾する同名 user TOML があっても SSoT は discovery 由来の値を保持する。
    """
    real_registry = get_config_registry()
    user_overrides = {
        "openai/gpt-4o": {
            "model_name_on_provider": "user-toml-wrong-id",
            "class": "WebApiAnnotator",
        }
    }
    with patch.object(real_registry, "_merged_config_data", user_overrides):
        with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
            _register_webapi_models_from_discovery()

        metadata = get_webapi_metadata("openai/gpt-4o")
        assert metadata is not None
        # user TOML の "user-toml-wrong-id" ではなく discovery SSoT の値
        assert metadata["litellm_model_id"] == "openai/gpt-4o"
        assert metadata["model_name_on_provider"] == "openai/gpt-4o"


@pytest.mark.unit
def test_register_webapi_models_keys_metadata_by_model_name_short(isolated_registry):
    """`_WEBAPI_MODEL_METADATA` は `model_name_short` をキーに `litellm_model_id` を保持する。

    旧 `PydanticAIWebAPIAnnotator` test (`...falls_back_to_discovery_metadata...`) の意図を
    読み替え。discovery が SSoT であることを、キー = `model_name_short` (完全 ID) /
    値の `litellm_model_id` が discovery 由来であることで検証する。
    """
    with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
        _register_webapi_models_from_discovery()

    # discovery の全モデルが model_name_short をキーに登録される
    assert set(registry._WEBAPI_MODEL_METADATA.keys()) == {
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
    }
    for model_short in ("openai/gpt-4o", "google/gemini-2.5-pro"):
        metadata = get_webapi_metadata(model_short)
        assert metadata is not None
        assert metadata["litellm_model_id"] == model_short


@pytest.mark.unit
def test_registered_webapi_model_resolves_to_webapi_annotator(isolated_registry):
    """登録済み WebAPI モデルは `WebApiAnnotator` クラスへ解決される。

    旧 `PydanticAIWebAPIWrapper` test の意図を読み替え。ADR 0023 Phase 1 で WebAPI
    系の入口は `WebApiAnnotator` 1 種に統合された。`_register_webapi_models_from_discovery`
    がモデルクラス registry に `WebApiAnnotator` を登録し、metadata の `class` も
    `WebApiAnnotator` であることを検証する。
    """
    from image_annotator_lib.webapi.annotator import WebApiAnnotator

    with patch(_DISCOVERY_PATCH_TARGET, return_value=_DISCOVERY_RESULT):
        _register_webapi_models_from_discovery()

    for model_short in ("openai/gpt-4o", "google/gemini-2.5-pro"):
        assert registry._MODEL_CLASS_OBJ_REGISTRY.get(model_short) is WebApiAnnotator
        metadata = get_webapi_metadata(model_short)
        assert metadata is not None
        assert metadata["class"] == "WebApiAnnotator"
