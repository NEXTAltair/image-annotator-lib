"""image_annotator_lib.core.registry の純粋関数 unit test。

ADR 0023 Phase 1 (Issue #35) で削除された旧 API (`load_available_api_models` /
`_fetch_and_update_vision_models` / `should_refresh` / `trigger_background_refresh` /
`AVAILABLE_API_MODELS_CONFIG_PATH` / 旧 `PydanticAIWebAPIAnnotator` 等) を mock した
test は本ファイルから削除された。`_register_webapi_models_from_discovery` /
`initialize_registry` の test は新仕様 (LiteLLM 同梱 DB / `WebApiAnnotator`) に追従する
形で別途書き直す予定 (Phase 1.5 issue 切り出し対象)。

本ファイルでは `_is_obsolete_annotator_class` / `_resolve_model_class` /
`_try_register_model` の純粋関数 test を保持する。
"""

import pytest

from image_annotator_lib.core import registry
from image_annotator_lib.core.base.annotator import BaseAnnotator

# ==============================================================================
# Test _is_obsolete_annotator_class
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_true_for_old_api_annotators():
    """古いプロバイダー固有クラスは obsolete 判定される。"""
    assert registry._is_obsolete_annotator_class("OpenAIApiAnnotator") is True
    assert registry._is_obsolete_annotator_class("GoogleApiAnnotator") is True
    assert registry._is_obsolete_annotator_class("AnthropicApiAnnotator") is True
    # Chat/Response 系も obsolete
    assert registry._is_obsolete_annotator_class("OpenAIApiChatAnnotator") is True
    assert registry._is_obsolete_annotator_class("OpenAIApiResponseAnnotator") is True


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_true_for_legacy_pydantic_ai():
    """ADR 0023 Phase 1 (Issue #35): PydanticAIWebAPIAnnotator は WebApiAnnotator に統合され obsolete 化。"""
    assert registry._is_obsolete_annotator_class("PydanticAIWebAPIAnnotator") is True


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_false_for_webapi_annotator():
    """WebApiAnnotator は新規統一クラスで obsolete ではない。"""
    assert registry._is_obsolete_annotator_class("WebApiAnnotator") is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_false_for_non_api():
    """非 API クラスは obsolete ではない。"""
    assert registry._is_obsolete_annotator_class("LocalMLAnnotator") is False
    assert registry._is_obsolete_annotator_class("SomeOtherClass") is False


# ==============================================================================
# Test _resolve_model_class
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_webapi_annotator_from_user_toml_is_rejected():
    """ADR 0023 Phase 1 (Codex P1, PR #40): user TOML 経由の `class = "WebApiAnnotator"`
    指定は registry に登録されない (broken path 防止)。

    WebApiAnnotator の registry 登録は LiteLLM 同梱 DB 由来の
    `_register_webapi_models_from_discovery()` が排他的に行うため、user TOML から
    指定された場合は warning + skip する。
    """
    available = {"SomeLocal": type("SomeLocal", (), {})}

    result = registry._resolve_model_class("WebApiAnnotator", "test-model", available, "annotator")
    assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_obsolete_returns_none():
    """古いプロバイダー固有クラス・PydanticAIWebAPIAnnotator が指定された場合、None が返される。"""
    available = {"OpenAIApiAnnotator": type("OpenAIApiAnnotator", (), {})}

    result = registry._resolve_model_class("OpenAIApiAnnotator", "test-model", available, "annotator")
    assert result is None

    result = registry._resolve_model_class(
        "PydanticAIWebAPIAnnotator", "test-model", available, "annotator"
    )
    assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_local_model():
    """ローカルモデルクラスは正しく解決される。"""
    local_cls = type("LocalMLAnnotator", (), {})
    available = {"LocalMLAnnotator": local_cls}

    result = registry._resolve_model_class("LocalMLAnnotator", "test-model", available, "annotator")
    assert result is local_cls


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_not_found():
    """存在しないクラスが指定された場合、None が返される。"""
    result = registry._resolve_model_class("NonExistentClass", "test-model", {}, "annotator")
    assert result is None


# ==============================================================================
# Test _try_register_model
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_success():
    """base_class のサブクラスが正常に登録される。"""
    reg: dict = {}
    model_cls = type("TestModel", (BaseAnnotator,), {})

    result = registry._try_register_model(reg, "test-model", model_cls, BaseAnnotator)
    assert result is True
    assert "test-model" in reg


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_with_predict_method():
    """predict メソッドを持つクラスは base_class のサブクラスでなくても登録される。"""
    reg: dict = {}
    model_cls = type("PredictModel", (), {"predict": lambda self: None})

    result = registry._try_register_model(reg, "test-model", model_cls, BaseAnnotator)
    assert result is True
    assert "test-model" in reg


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_reject_incompatible():
    """base_class のサブクラスでも predict も持たないクラスは拒否される。"""
    reg: dict = {}
    model_cls = type("IncompatibleModel", (), {})

    result = registry._try_register_model(reg, "test-model", model_cls, BaseAnnotator)
    assert result is False
    assert "test-model" not in reg


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_overwrite_warning():
    """既存エントリへの上書き登録が成功する。"""
    old_cls = type("OldModel", (BaseAnnotator,), {})
    new_cls = type("NewModel", (BaseAnnotator,), {})
    reg: dict = {"test-model": old_cls}

    result = registry._try_register_model(reg, "test-model", new_cls, BaseAnnotator)
    assert result is True
    assert reg["test-model"] is new_cls
