"""
src/image_annotator_lib/core/base.py のユニットテスト。
"""

from typing import Any, Self
from unittest.mock import Mock, patch

import pytest
from PIL import Image

# テスト対象のインポート
from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.base.transformers import TransformersBaseAnnotator
from image_annotator_lib.core.base.webapi import WebApiBaseAnnotator
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


@pytest.fixture(autouse=True, scope="class")
def setup_test_webapi_config():
    """Setup test model configuration for WebApiBaseAnnotator tests in this file."""
    from image_annotator_lib.core.config import config_registry

    # Use unique model name for webapi tests in test_base.py
    test_model_name = "test_webapi_base_model"

    # Cleanup first to ensure no leftover settings
    try:
        merged_data = getattr(config_registry, "_merged_config_data", {})
        merged_data.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass

    # Set up WebAPIModelConfig-compatible configuration (no model_path)
    config = {
        "device": "cpu",
        "class": "ConcreteWebApiAnnotator",
        "api_model_id": "test-api-model-id",
        "model_name_on_provider": "test-provider-model",
        "prompt_template": "Test prompt",
        "timeout": 30,
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


class ConcreteWebApiAnnotator(WebApiBaseAnnotator):
    def __init__(self, model_name: str = "test_webapi_base_model"):
        super().__init__(model_name)

    def _run_inference(self, processed_images: list[str]) -> list[dict[str, Any]]:
        # Dummy implementation for testing
        return [{"tags": ["web_tag"]}] * len(processed_images)


# --- Tests ---


class TestBaseAnnotator:
    @pytest.mark.unit
    def test_init_success(self):
        """正常な初期化のテスト。"""
        annotator = ConcreteAnnotator()
        assert annotator.model_name == "test_base_annotator_model"
        assert annotator.device == "cpu"
        # chunk_size is not a property of BaseAnnotator anymore

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

    @pytest.mark.unit
    def test_generate_tags_logic(self):
        """_generate_tags handles string and non-string inputs."""
        annotator = TransformersBaseAnnotator("test_transformers_base_model")
        assert annotator._generate_tags("tag1, tag2") == ["tag1, tag2"]
        assert annotator._generate_tags(["tag1", "tag2"]) == []
        assert annotator._generate_tags(123) == []


class TestWebApiBaseAnnotator:
    @pytest.mark.unit
    def test_init(self):
        """初期化のテスト。"""
        annotator = ConcreteWebApiAnnotator()
        assert annotator.prompt_template == "Test prompt"
        assert annotator.timeout == 30

    @pytest.mark.unit
    @patch("base64.b64encode", return_value=b"encoded_data")
    def test_preprocess_images(self, mock_b64):
        """_preprocess_images correctly encodes images."""
        annotator = ConcreteWebApiAnnotator()
        mock_image = Mock(spec=Image.Image)
        results = annotator._preprocess_images([mock_image])
        assert results == ["encoded_data"]
        mock_image.save.assert_called_once()

    @pytest.mark.unit
    def test_parse_common_json_response(self):
        """Test parsing of common JSON responses."""
        annotator = ConcreteWebApiAnnotator()
        # Test with dict
        result = annotator._parse_common_json_response({"tags": ["a"], "captions": ["b"], "score": 0.5})
        assert result["annotation"]["tags"] == ["a"]
        # Test with JSON string
        result = annotator._parse_common_json_response('{"tags": ["b"], "captions": ["c"], "score": 0.6}')
        assert result["annotation"]["tags"] == ["b"]
        # Test with invalid JSON
        result = annotator._parse_common_json_response("not json")
        assert "JSON解析エラー" in result["error"]

    @pytest.mark.unit
    def test_extract_tags_from_text(self):
        """Test tag extraction from various text formats."""
        annotator = ConcreteWebApiAnnotator()
        assert annotator._extract_tags_from_text('{"tags": ["a"]}') == ["a"]
        assert annotator._extract_tags_from_text("tags: a, b, c") == ["a", "b", "c"]
        assert annotator._extract_tags_from_text("a, b, c") == ["a", "b", "c"]
