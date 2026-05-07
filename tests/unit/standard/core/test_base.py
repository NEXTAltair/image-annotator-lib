"""
src/image_annotator_lib/core/base.py のユニットテスト。

ADR 0023 Phase 1 (Issue #35): WebApiBaseAnnotator は廃止された。WebAPI の test は
`tests/unit/core/test_webapi_annotator.py` を参照。本ファイルでは `BaseAnnotator` と
ローカル ML 系 base class (`TransformersBaseAnnotator` 等) のみを扱う。
"""

from typing import Self
from unittest.mock import Mock, patch

import pytest
from PIL import Image

# テスト対象のインポート
from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.base.transformers import TransformersBaseAnnotator
from image_annotator_lib.core.config import ModelConfigRegistry
from image_annotator_lib.exceptions.errors import (
    OutOfMemoryError,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def setup_test_base_annotator_config():
    """Setup test model configuration for BaseAnnotator tests."""
    from image_annotator_lib.core.config import config_registry

    # Use unique model name to avoid conflicts
    test_model_name = "test_base_annotator_model"

    # Cleanup first to ensure no leftover settings
    try:
        merged_data = getattr(config_registry, "_merged_config_data", {})
        merged_data.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass

    # Set up LocalMLModelConfig-compatible configuration (with model_path)
    config = {
        "model_path": "/test/path/model",
        "device": "cpu",
        "class": "ConcreteAnnotator",
    }
    for key, value in config.items():
        config_registry.add_default_setting(test_model_name, key, value)

    yield

    # Cleanup after test
    try:
        merged_data = getattr(config_registry, "_merged_config_data", {})
        merged_data.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass


@pytest.fixture(autouse=True, scope="class")
def setup_test_transformers_config():
    """Setup test model configuration for TransformersBaseAnnotator tests."""
    from image_annotator_lib.core.config import config_registry

    # Use unique model name specific to transformers tests
    test_model_name = "test_transformers_base_model"

    # Comprehensive cleanup first to ensure no leftover settings
    try:
        # Clean from all config stores
        merged_data = getattr(config_registry, "_merged_config_data", {})
        merged_data.pop(test_model_name, None)
        system_data = getattr(config_registry, "_system_config_data", {})
        system_data.pop(test_model_name, None)
        user_data = getattr(config_registry, "_user_config_data", {})
        user_data.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass

    # Set up LocalMLModelConfig-compatible configuration (base fields only)
    # Note: max_length and processor_path are intentionally NOT included here
    # because they would be rejected by Pydantic validation (extra='forbid')
    # TransformersBaseAnnotator reads them directly via config_registry.get() with defaults
    config = {
        "model_path": "/test/path/transformers_model",
        "device": "cpu",
        "class": "TransformersBaseAnnotator",
    }
    for key, value in config.items():
        config_registry.add_default_setting(test_model_name, key, value)

    yield

    # Comprehensive cleanup after test
    try:
        merged_data = getattr(config_registry, "_merged_config_data", {})
        merged_data.pop(test_model_name, None)
        system_data = getattr(config_registry, "_system_config_data", {})
        system_data.pop(test_model_name, None)
        user_data = getattr(config_registry, "_user_config_data", {})
        user_data.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass


@pytest.fixture
def mock_config_registry_fixture():
    """Provides a mock for the config_registry with a default test_model config."""
    registry = ModelConfigRegistry()
    # Pre-populate with a default configuration for 'test_model'
    registry.set_system_value("test_model", "device", "cpu")
    registry.set_system_value("test_model", "chunk_size", 8)
    registry.set_system_value("test_model", "model_path", "/test/path")
    registry.set_system_value("test_model", "max_length", 75)
    registry.set_system_value("test_model", "processor_path", "/processor/path")
    registry.set_system_value("test_model", "prompt_template", "Test prompt")
    registry.set_system_value("test_model", "timeout", 30)
    registry.set_system_value("test_model", "retry_count", 5)
    registry.set_system_value("test_model", "retry_delay", 2.0)
    registry.set_system_value("test_model", "min_request_interval", 0.5)
    registry.set_system_value("test_model", "max_output_tokens", 2000)
    return registry


# --- Concrete implementations for testing abstract classes ---


class ConcreteAnnotator(BaseAnnotator):
    def __init__(self, model_name: str = "test_base_annotator_model"):
        super().__init__(model_name)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        pass

    def _preprocess_images(self, i):
        return i

    def _run_inference(self, p):
        return [{"inference": "result"} for _ in p]

    def _format_predictions(self, r):
        return r

    def _generate_tags(self, f):
        return ["test_tag"]


# ADR 0023 Phase 1 (Issue #35): ConcreteWebApiAnnotator は WebApiBaseAnnotator 継承で
# あったため削除された。WebAPI の test は tests/unit/core/test_webapi_annotator.py 参照。


# --- Tests ---


class TestBaseAnnotator:
    @pytest.mark.unit
    def test_init_success(self):
        """正常な初期化のテスト。"""
        annotator = ConcreteAnnotator()
        assert annotator.model_name == "test_base_annotator_model"
        # ADR 0023 Phase 1 (Issue #35): BaseAnnotator.__init__ から device 判定が分離された
        # ため、サブクラスが device を設定しない限り sentinel ("") が残る。本
        # ConcreteAnnotator は ML 系 base class を経由しないため device は "" のまま。
        assert annotator.device == ""

    @pytest.mark.unit
    def test_init_no_config_error(self):
        """設定が見つからない場合のエラーテスト。"""
        with pytest.raises(ValueError, match="Model 'non_existent_model' not found in config_registry"):
            ConcreteAnnotator("non_existent_model")

    @pytest.mark.unit
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_handles_out_of_memory(self, mock_logger):
        """Predict handles OutOfMemoryError gracefully."""

        class OOMAnnotator(ConcreteAnnotator):
            def _run_inference(self, p):
                raise OutOfMemoryError("OOM test")

        annotator = OOMAnnotator()
        results = annotator.predict([Mock(spec=Image.Image)], ["hash"])
        assert "メモリ不足エラー" in results[0].error
        mock_logger.error.assert_called()

    @pytest.mark.unit
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_handles_general_exception(self, mock_logger):
        """Predict handles general exceptions gracefully."""

        class ExcAnnotator(ConcreteAnnotator):
            def _run_inference(self, p):
                raise ValueError("General test error")

        annotator = ExcAnnotator()
        results = annotator.predict([Mock(spec=Image.Image)], ["hash"])
        assert "予期せぬエラー" in results[0].error
        mock_logger.exception.assert_called()


class TestTransformersBaseAnnotator:
    @pytest.mark.unit
    def test_init(self):
        """初期化のテスト（デフォルト値確認）。"""
        annotator = TransformersBaseAnnotator("test_transformers_base_model")
        # max_length and processor_path use default values when not in config
        assert annotator.max_length == 75  # default from config_registry.get(..., 75)
        assert annotator.processor_path is None  # default from config_registry.get(..., None)

    # ADR 0023 Phase 1: TransformersBaseAnnotator._generate_tags メソッドは廃止された
    # (BaseAnnotator の `_format_predictions` で UnifiedAnnotationResult を直接構築するため)。
    # test_generate_tags_logic はスコープ外として削除。


# TestWebApiBaseAnnotator は ADR 0023 Phase 1 (Issue #35) で削除された。
# WebAPI annotator の test は tests/unit/core/test_webapi_annotator.py 参照。
