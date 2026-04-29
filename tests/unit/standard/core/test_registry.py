from unittest.mock import MagicMock, call, patch

import pytest

# Assuming registry.py is in src.image_annotator_lib.core
from image_annotator_lib.core import registry
from image_annotator_lib.core.base.annotator import BaseAnnotator


# Mock annotator classes for testing _find_annotator_class_by_provider
class GoogleApiAnnotator(BaseAnnotator):
    pass


class OpenAIApiAnnotator(BaseAnnotator):
    pass


class AnthropicApiAnnotator(BaseAnnotator):
    pass


class OpenRouterApiAnnotator(BaseAnnotator):
    pass


class SomeOtherAnnotator(BaseAnnotator):
    pass


# Sample available classes for testing provider mapping
MOCK_AVAILABLE_CLASSES = {
    "GoogleApiAnnotator": GoogleApiAnnotator,
    "OpenAIApiAnnotator": OpenAIApiAnnotator,
    "AnthropicApiAnnotator": AnthropicApiAnnotator,
    "OpenRouterApiAnnotator": OpenRouterApiAnnotator,
    "SomeOtherAnnotator": SomeOtherAnnotator,
}

# Sample API model data for testing config update
MOCK_API_MODELS = {
    "google/gemini-pro-1.5": {
        "provider": "google",
        "model_name_short": "Gemini 1.5 Pro",
        # ... other keys
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "model_name_short": "GPT-4o",
        # ... other keys
    },
    "unknown/some-model": {
        "provider": "some-unknown-provider",
        "model_name_short": "Unknown Model",
        # ... other keys
    },
    "anthropic/claude-3-sonnet": {
        # Missing provider to test skipping
        "model_name_short": "Claude 3 Sonnet",
    },
    "invalid_format_model": "this_is_not_a_dict",  # Test invalid format
}


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.os.getenv")
@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._register_webapi_models_from_discovery")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")  # Mock logger init
def test_initialize_registry_api_models_file_not_exists_calls_fetch_and_update(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_register_webapi,
    mock_register,
    mock_getenv,
):
    """Test initialize_registry calls API fetch when file not exists; WebAPI registration skipped."""
    registry._REGISTRY_INITIALIZED = False

    mock_getenv.return_value = "false"
    mock_config_path.exists.return_value = False

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    # exists() is called twice: once in the elif check, once before WebAPI registration
    assert mock_config_path.exists.call_count == 2
    mock_fetch_api_models.assert_called_once()
    mock_register_webapi.assert_not_called()  # file absent → no registration
    mock_register.assert_called_once()


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.os.getenv", return_value="false")
@patch("image_annotator_lib.core.api_model_discovery.should_refresh", return_value=False)
@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._register_webapi_models_from_discovery")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_api_models_file_exists_skips_fetch(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_register_webapi,
    mock_register,
    _mock_should_refresh,
    _mock_getenv,
):
    """Test initialize_registry skips API fetch but calls WebAPI registration when file exists."""
    registry._REGISTRY_INITIALIZED = False

    mock_config_path.exists.return_value = True

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    assert mock_config_path.exists.call_count == 2
    mock_fetch_api_models.assert_not_called()
    mock_register_webapi.assert_called_once()
    mock_register.assert_called_once()


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.os.getenv")
@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._register_webapi_models_from_discovery")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_continues_if_fetch_api_fails(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_register_webapi,
    mock_register,
    mock_getenv,
):
    """Test initialize_registry continues process even if API fetch fails."""
    registry._REGISTRY_INITIALIZED = False

    mock_getenv.return_value = "false"
    mock_config_path.exists.return_value = False
    mock_fetch_api_models.side_effect = Exception("API Error")

    registry.initialize_registry()

    mock_init_logger.assert_called_once()
    assert mock_config_path.exists.call_count == 2
    mock_fetch_api_models.assert_called_once()
    mock_register_webapi.assert_not_called()  # file absent → no registration
    mock_register.assert_called_once()


# --- Tests for _register_webapi_models_from_discovery ---

# MOCK_API_MODELS にアクティブモデルと廃止モデルを含む
MOCK_API_MODELS_WITH_DEPRECATED = {
    "google/gemini-pro-1.5": {
        "provider": "google",
        "model_name_short": "Gemini 1.5 Pro",
        "deprecated_on": None,
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "model_name_short": "GPT-4o",
        "deprecated_on": None,
    },
    "openai/old-model": {
        "provider": "openai",
        "model_name_short": "Old Model",
        "deprecated_on": "2024-01-01T00:00:00Z",  # 廃止済み → スキップ
    },
    "anthropic/claude-3-sonnet": {
        # model_name_short missing → スキップ
        "provider": "anthropic",
        "deprecated_on": None,
    },
    "invalid_format_model": "this_is_not_a_dict",  # 不正形式 → スキップ
}


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._try_register_model")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_register_webapi_models_from_discovery_success(
    mock_load_api_models,
    mock_gather_classes,
    mock_try_register,
):
    """アクティブなモデルのみが直接レジストリに登録される。"""
    pydantic_cls = MagicMock()
    mock_load_api_models.return_value = MOCK_API_MODELS_WITH_DEPRECATED
    mock_gather_classes.return_value = {"PydanticAIWebAPIAnnotator": pydantic_cls}
    mock_try_register.return_value = True

    registry._WEBAPI_MODEL_METADATA.clear()
    registry._register_webapi_models_from_discovery()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")

    # 廃止済み・不正形式・必須キー欠落のモデルを除いた 2 件のみ登録
    assert mock_try_register.call_count == 2
    registered_names = [c.args[1] for c in mock_try_register.call_args_list]
    assert "Gemini 1.5 Pro" in registered_names
    assert "GPT-4o" in registered_names
    assert "Old Model" not in registered_names  # deprecated_on → スキップ

    # メタデータが保存されている
    assert "Gemini 1.5 Pro" in registry._WEBAPI_MODEL_METADATA
    assert registry._WEBAPI_MODEL_METADATA["Gemini 1.5 Pro"]["api_model_id"] == "google/gemini-pro-1.5"
    registry._WEBAPI_MODEL_METADATA.clear()


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_register_webapi_models_from_discovery_no_api_data(
    mock_load_api_models,
    mock_gather_classes,
):
    """利用可能なモデル情報がない場合、クラス収集をスキップする。"""
    mock_load_api_models.return_value = {}

    registry._register_webapi_models_from_discovery()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_not_called()


@pytest.mark.unit
@patch("image_annotator_lib.core.registry._try_register_model")
@patch("image_annotator_lib.core.registry._gather_available_classes")
@patch("image_annotator_lib.core.registry.load_available_api_models")
def test_register_webapi_models_from_discovery_no_pydantic_class(
    mock_load_api_models,
    mock_gather_classes,
    mock_try_register,
):
    """PydanticAIWebAPIAnnotator が見つからない場合、登録を中断する。"""
    mock_load_api_models.return_value = MOCK_API_MODELS
    mock_gather_classes.return_value = {}  # PydanticAIWebAPIAnnotator 不在

    registry._register_webapi_models_from_discovery()

    mock_load_api_models.assert_called_once()
    mock_gather_classes.assert_called_once_with("model_class")
    mock_try_register.assert_not_called()  # 登録処理に到達しない


# --- Test for singleton pattern ---


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.os.getenv", return_value="false")
@patch("image_annotator_lib.core.api_model_discovery.should_refresh", return_value=False)
@patch("image_annotator_lib.core.registry.register_annotators")
@patch("image_annotator_lib.core.registry._register_webapi_models_from_discovery")
@patch("image_annotator_lib.core.api_model_discovery._fetch_and_update_vision_models")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
@patch("image_annotator_lib.core.utils.init_logger")
def test_initialize_registry_singleton_pattern(
    mock_init_logger,
    mock_config_path,
    mock_fetch_api_models,
    mock_register_webapi,
    mock_register,
    _mock_should_refresh,
    _mock_getenv,
):
    """Test initialize_registry uses singleton pattern and only initializes once."""
    registry._REGISTRY_INITIALIZED = False

    mock_config_path.exists.return_value = True

    registry.initialize_registry()

    assert mock_init_logger.call_count == 1
    assert mock_config_path.exists.call_count == 2
    assert mock_fetch_api_models.call_count == 0
    assert mock_register_webapi.call_count == 1
    assert mock_register.call_count == 1

    # Second call should skip all initialization due to singleton pattern
    registry.initialize_registry()

    assert mock_init_logger.call_count == 2  # init_logger always runs
    assert mock_config_path.exists.call_count == 2  # unchanged
    assert mock_fetch_api_models.call_count == 0
    assert mock_register_webapi.call_count == 1  # unchanged
    assert mock_register.call_count == 1  # unchanged

    registry.initialize_registry()

    assert mock_init_logger.call_count == 3
    assert mock_config_path.exists.call_count == 2  # unchanged
    assert mock_fetch_api_models.call_count == 0
    assert mock_register_webapi.call_count == 1  # unchanged
    assert mock_register.call_count == 1  # unchanged


# ==============================================================================
# Test _is_obsolete_annotator_class
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_true_for_old_api_annotators():
    """古いプロバイダー固有クラスはobsolete判定される。"""
    assert registry._is_obsolete_annotator_class("OpenAIApiAnnotator") is True
    assert registry._is_obsolete_annotator_class("GoogleApiAnnotator") is True
    assert registry._is_obsolete_annotator_class("AnthropicApiAnnotator") is True
    # Chat/Response系もobsolete
    assert registry._is_obsolete_annotator_class("OpenAIApiChatAnnotator") is True
    assert registry._is_obsolete_annotator_class("OpenAIApiResponseAnnotator") is True


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_false_for_pydantic_ai():
    """PydanticAIWebAPIAnnotatorはobsoleteではない。"""
    assert registry._is_obsolete_annotator_class("PydanticAIWebAPIAnnotator") is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_obsolete_annotator_class_false_for_non_api():
    """非APIクラスはobsoleteではない。"""
    assert registry._is_obsolete_annotator_class("LocalMLAnnotator") is False
    assert registry._is_obsolete_annotator_class("SomeOtherClass") is False


# ==============================================================================
# Test _resolve_model_class
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_pydantic_ai():
    """PydanticAIWebAPIAnnotatorが指定された場合、統一実装が返される。"""
    mock_pydantic_class = type("PydanticAIWebAPIAnnotator", (), {})
    available = {"SomeLocal": type("SomeLocal", (), {})}

    result = registry._resolve_model_class(
        "PydanticAIWebAPIAnnotator", "test-model", available, mock_pydantic_class, "annotator"
    )
    assert result is mock_pydantic_class


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_obsolete_returns_none():
    """古いプロバイダー固有クラスが指定された場合、Noneが返される。"""
    available = {"OpenAIApiAnnotator": type("OpenAIApiAnnotator", (), {})}

    result = registry._resolve_model_class(
        "OpenAIApiAnnotator", "test-model", available, None, "annotator"
    )
    assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_local_model():
    """ローカルモデルクラスは正しく解決される。"""
    local_cls = type("LocalMLAnnotator", (), {})
    available = {"LocalMLAnnotator": local_cls}

    result = registry._resolve_model_class(
        "LocalMLAnnotator", "test-model", available, None, "annotator"
    )
    assert result is local_cls


@pytest.mark.unit
@pytest.mark.fast
def test_resolve_model_class_not_found():
    """存在しないクラスが指定された場合、Noneが返される。"""
    result = registry._resolve_model_class(
        "NonExistentClass", "test-model", {}, None, "annotator"
    )
    assert result is None


# ==============================================================================
# Test _try_register_model
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_success():
    """base_classのサブクラスが正常に登録される。"""
    reg: dict = {}
    model_cls = type("TestModel", (BaseAnnotator,), {})

    result = registry._try_register_model(reg, "test-model", model_cls, BaseAnnotator)
    assert result is True
    assert "test-model" in reg


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_with_predict_method():
    """predictメソッドを持つクラスはbase_classのサブクラスでなくても登録される。"""
    reg: dict = {}
    model_cls = type("PredictModel", (), {"predict": lambda self: None})

    result = registry._try_register_model(reg, "test-model", model_cls, BaseAnnotator)
    assert result is True
    assert "test-model" in reg


@pytest.mark.unit
@pytest.mark.fast
def test_try_register_model_reject_incompatible():
    """base_classのサブクラスでもpredictも持たないクラスは拒否される。"""
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


# ==============================================================================
# Test _discover_and_update_api_models: background refresh ordering
# ==============================================================================


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.os.getenv", return_value="false")
@patch("image_annotator_lib.core.api_model_discovery.trigger_background_refresh")
@patch("image_annotator_lib.core.api_model_discovery.should_refresh", return_value=True)
@patch("image_annotator_lib.core.registry._register_webapi_models_from_discovery")
@patch("image_annotator_lib.core.registry.AVAILABLE_API_MODELS_CONFIG_PATH")
def test_background_refresh_starts_after_register_webapi(
    mock_config_path,
    mock_register_webapi,
    _mock_should_refresh,
    mock_trigger,
    _mock_getenv,
) -> None:
    """TTL 超過時、trigger_background_refresh は _register_webapi_models_from_discovery の後に呼ばれる。"""
    call_order: list[str] = []
    mock_config_path.exists.return_value = True
    mock_register_webapi.side_effect = lambda: call_order.append("register_webapi")
    mock_trigger.side_effect = lambda: call_order.append("trigger_refresh")

    registry._discover_and_update_api_models(skip_api_discovery=False)

    assert call_order == ["register_webapi", "trigger_refresh"], (
        f"呼び出し順が期待と異なります: {call_order}"
    )
