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
    ConfigurationError,
    OutOfMemoryError,
)

# --- Fixtures ---


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
    def _run_inference(self, processed_images: list[str]) -> list[dict[str, Any]]:
        # Dummy implementation for testing
        return [{"tags": ["web_tag"]}] * len(processed_images)


# --- Tests ---


@patch("image_annotator_lib.core.base.annotator.config_registry")
class TestBaseAnnotator:
    @pytest.mark.unit
    def test_init_success(self, mock_config_registry):
        """正常な初期化のテスト。"""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
            ("test_model", "chunk_size"): 8,
        }.get((name, key), default)
        annotator = ConcreteAnnotator("test_model")
        assert annotator.model_name == "test_model"
        assert annotator.device == "cpu"
        # chunk_size is not a property of BaseAnnotator anymore

    @pytest.mark.unit
    def test_init_no_config_error(self, mock_config_registry):
        """設定が見つからない場合のエラーテスト。"""
        mock_config_registry.get.side_effect = ConfigurationError(
            "'non_existent_model' の設定が config_registry に見つかりません"
        )
        with pytest.raises(
            ConfigurationError, match="'non_existent_model' の設定が config_registry に見つかりません"
        ):
            ConcreteAnnotator("non_existent_model")

    @pytest.mark.unit
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_handles_out_of_memory(self, mock_logger, mock_config_registry):
        """Predict handles OutOfMemoryError gracefully."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
        }.get((name, key), default)

        class OOMAnnotator(ConcreteAnnotator):
            def _run_inference(self, p):
                raise OutOfMemoryError("OOM test")

        annotator = OOMAnnotator("test_model")
        results = annotator.predict([Mock(spec=Image.Image)], ["hash"])
        assert "メモリ不足エラー" in results[0]["error"]
        mock_logger.error.assert_called()

    @pytest.mark.unit
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_handles_general_exception(self, mock_logger, mock_config_registry):
        """Predict handles general exceptions gracefully."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
        }.get((name, key), default)

        class ExcAnnotator(ConcreteAnnotator):
            def _run_inference(self, p):
                raise ValueError("General test error")

        annotator = ExcAnnotator("test_model")
        results = annotator.predict([Mock(spec=Image.Image)], ["hash"])
        assert "予期せぬエラー" in results[0]["error"]
        mock_logger.exception.assert_called()


@patch("image_annotator_lib.core.base.transformers.config_registry")
class TestTransformersBaseAnnotator:
    @pytest.mark.unit
    def test_init(self, mock_config_registry):
        """初期化のテスト。"""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
            ("test_model", "max_length"): 75,
            ("test_model", "processor_path"): "/processor/path",
        }.get((name, key), default)
        annotator = TransformersBaseAnnotator("test_model")
        assert annotator.max_length == 75
        assert annotator.processor_path == "/processor/path"

    @pytest.mark.unit
    def test_generate_tags_logic(self, mock_config_registry):
        """_generate_tags handles string and non-string inputs."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
            ("test_model", "max_length"): 75,
            ("test_model", "processor_path"): "/processor/path",
        }.get((name, key), default)
        annotator = TransformersBaseAnnotator("test_model")
        assert annotator._generate_tags("tag1, tag2") == ["tag1, tag2"]
        assert annotator._generate_tags(["tag1", "tag2"]) == []
        assert annotator._generate_tags(123) == []


@patch("image_annotator_lib.core.base.webapi.config_registry")
class TestWebApiBaseAnnotator:
    @pytest.mark.unit
    def test_init(self, mock_config_registry):
        """初期化のテスト。"""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
            ("test_model", "prompt_template"): "Test prompt",
            ("test_model", "timeout"): 30,
        }.get((name, key), default)
        annotator = ConcreteWebApiAnnotator("test_model")
        assert annotator.prompt_template == "Test prompt"
        assert annotator.timeout == 30

    @pytest.mark.unit
    @patch("base64.b64encode", return_value=b"encoded_data")
    def test_preprocess_images(self, mock_b64, mock_config_registry):
        """_preprocess_images correctly encodes images."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
        }.get((name, key), default)
        annotator = ConcreteWebApiAnnotator("test_model")
        mock_image = Mock(spec=Image.Image)
        results = annotator._preprocess_images([mock_image])
        assert results == ["encoded_data"]
        mock_image.save.assert_called_once()

    @pytest.mark.unit
    def test_parse_common_json_response(self, mock_config_registry):
        """Test parsing of common JSON responses."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
        }.get((name, key), default)
        annotator = ConcreteWebApiAnnotator("test_model")
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
    def test_extract_tags_from_text(self, mock_config_registry):
        """Test tag extraction from various text formats."""
        mock_config_registry.get.side_effect = lambda name, key, default=None: {
            ("test_model", "model_path"): "/test/path",
            ("test_model", "device"): "cpu",
        }.get((name, key), default)
        annotator = ConcreteWebApiAnnotator("test_model")
        assert annotator._extract_tags_from_text('{"tags": ["a"]}') == ["a"]
        assert annotator._extract_tags_from_text("tags: a, b, c") == ["a", "b", "c"]
        assert annotator._extract_tags_from_text("a, b, c") == ["a", "b", "c"]
