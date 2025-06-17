"""src/image_annotator_lib/core/base.py のユニットテスト。

このモジュールは、画像アノテーションライブラリの基底クラスと型定義をテストします。
主要なテスト対象：
- BaseAnnotator 抽象基底クラス
- フレームワーク別基底クラス（TransformersBaseAnnotator, TensorflowBaseAnnotator, etc.）
- 型定義（AnnotationResult, TagConfidence, etc.）
- 共通メソッドの動作
"""

from pathlib import Path
from typing import Any, Self
from unittest.mock import Mock, patch

import pytest
from PIL import Image

# テスト対象のインポート
from image_annotator_lib.core.base import (
    AnnotationResult,
    BaseAnnotator,
    ONNXComponents,
    TagConfidence,
    TensorFlowComponents,
    TransformersBaseAnnotator,
    TransformersComponents,
    WebApiBaseAnnotator,
)
from image_annotator_lib.exceptions.errors import (
    ConfigurationError,
    OutOfMemoryError,
)


class TestAnnotationResult:
    """AnnotationResult 型定義のテスト。"""

    def test_annotation_result_structure(self):
        """AnnotationResult の構造が正しいことを確認。"""
        result: AnnotationResult = {
            "phash": "test_hash",
            "tags": ["tag1", "tag2"],
            "formatted_output": {"test": "data"},
            "error": None,
        }

        assert result["phash"] == "test_hash"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["formatted_output"] == {"test": "data"}
        assert result["error"] is None

    def test_annotation_result_optional_fields(self):
        """AnnotationResult のオプションフィールドが正しく動作することを確認。"""
        # 最小限の構造
        result: AnnotationResult = {
            "tags": ["tag1"],
        }

        assert result["tags"] == ["tag1"]
        # オプションフィールドは存在しなくても良い
        assert "phash" not in result
        assert "formatted_output" not in result
        assert "error" not in result

    def test_annotation_result_with_error(self):
        """エラーを含む AnnotationResult の構造を確認。"""
        result: AnnotationResult = {
            "phash": None,
            "tags": [],
            "formatted_output": None,
            "error": "テストエラー",
        }

        assert result["phash"] is None
        assert result["tags"] == []
        assert result["formatted_output"] is None
        assert result["error"] == "テストエラー"


class TestTagConfidence:
    """TagConfidence 型定義のテスト。"""

    def test_tag_confidence_structure(self):
        """TagConfidence の構造が正しいことを確認。"""
        tag_conf: TagConfidence = {
            "confidence": 0.85,
            "source": "test_model",
        }

        assert tag_conf["confidence"] == 0.85
        assert tag_conf["source"] == "test_model"


class TestComponentTypes:
    """各種 Components 型定義のテスト。"""

    def test_transformers_components(self):
        """TransformersComponents の構造を確認。"""
        mock_model = Mock()
        mock_processor = Mock()

        components: TransformersComponents = {
            "model": mock_model,
            "processor": mock_processor,
        }

        assert components["model"] is mock_model
        assert components["processor"] is mock_processor

    def test_onnx_components(self):
        """ONNXComponents の構造を確認。"""
        mock_session = Mock()
        test_path = Path("/test/path.csv")

        components: ONNXComponents = {
            "session": mock_session,
            "csv_path": test_path,
        }

        assert components["session"] is mock_session
        assert components["csv_path"] == test_path

    def test_tensorflow_components(self):
        """TensorFlowComponents の構造を確認。"""
        mock_model = Mock()
        test_path = Path("/test/model")

        components: TensorFlowComponents = {
            "model_dir": test_path,
            "model": mock_model,
        }

        assert components["model_dir"] == test_path
        assert components["model"] is mock_model


class ConcreteAnnotator(BaseAnnotator):
    """テスト用の具象 BaseAnnotator クラス。"""

    def __enter__(self) -> Self:
        self.components = {"test": "component"}
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.components = None

    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        return [{"processed": True} for _ in images]

    def _run_inference(self, processed: Any) -> Any:
        return [{"inference": "result"} for _ in processed]

    def _format_predictions(self, raw_outputs: Any) -> list[Any]:
        return [{"formatted": True} for _ in raw_outputs]

    def _generate_tags(self, formatted_output: Any) -> list[str]:
        return ["test_tag"]


class TestBaseAnnotator:
    """BaseAnnotator 抽象基底クラスのテスト。"""

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init_success(self, mock_utils, mock_config_registry):
        """正常な初期化のテスト。"""
        # モックの設定
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 8,
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = ConcreteAnnotator("test_model")

        assert annotator.model_name == "test_model"
        assert annotator.device == "cuda"
        assert annotator.chunk_size == 8
        assert annotator.model_path == "/test/path"
        assert annotator.components is None

    @patch('image_annotator_lib.core.base.config_registry')
    def test_init_no_config_error(self, mock_config_registry):
        """設定が見つからない場合のエラーテスト。"""
        mock_config_registry.get_all_config.return_value = {}

        with pytest.raises(ConfigurationError, match="モデル 'test_model' の設定が config_registry に見つかりません"):
            ConcreteAnnotator("test_model")

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init_invalid_device_config(self, mock_utils, mock_config_registry):
        """無効なデバイス設定のテスト。"""
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): 123,  # 無効な型
            ("test_model", "chunk_size"): 8,
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = ConcreteAnnotator("test_model")

        # デフォルト値が使用されることを確認
        mock_utils.determine_effective_device.assert_called_with("cuda", "test_model")

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init_invalid_chunk_size_config(self, mock_utils, mock_config_registry):
        """無効なチャンクサイズ設定のテスト。"""
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): "invalid",  # 無効な型
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = ConcreteAnnotator("test_model")

        # デフォルト値が使用されることを確認
        assert annotator.chunk_size == 8

    def test_generate_result(self):
        """_generate_result メソッドのテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): "/test/path",
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = ConcreteAnnotator("test_model")

            # 文字列タグのテスト
            result = annotator._generate_result(
                phash="test_hash",
                tags="single_tag",
                formatted_output={"test": "output"},
                error=None
            )

            expected = {
                "phash": "test_hash",
                "tags": ["single_tag"],
                "formatted_output": {"test": "output"},
                "error": None,
            }
            assert result == expected

            # リストタグのテスト
            result = annotator._generate_result(
                phash="test_hash",
                tags=["tag1", "tag2"],
                formatted_output={"test": "output"},
                error="test_error"
            )

            expected = {
                "phash": "test_hash",
                "tags": ["tag1", "tag2"],
                "formatted_output": {"test": "output"},
                "error": "test_error",
            }
            assert result == expected

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    @patch('image_annotator_lib.core.base.torch.no_grad')
    def test_predict_success(self, mock_no_grad, mock_utils, mock_config_registry):
        """predict メソッドの正常動作テスト。"""
        # モックの設定
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 2,  # 小さなチャンクサイズでテスト
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        annotator = ConcreteAnnotator("test_model")

        # テスト用の画像とハッシュリスト
        test_images = [Mock(spec=Image.Image) for _ in range(3)]
        test_phash_list = ["hash1", "hash2", "hash3"]

        results = annotator.predict(test_images, test_phash_list)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["phash"] == f"hash{i+1}"
            assert result["tags"] == ["test_tag"]
            assert result["formatted_output"] == {"formatted": True}
            assert result["error"] is None

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    @patch('image_annotator_lib.core.base.torch.no_grad')
    def test_predict_with_memory_error(self, mock_no_grad, mock_utils, mock_config_registry):
        """predict メソッドでメモリエラーが発生した場合のテスト。"""
        # モックの設定
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 2,
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        class MemoryErrorAnnotator(ConcreteAnnotator):
            def _preprocess_images(self, images: list[Image.Image]) -> Any:
                raise OutOfMemoryError("テストメモリエラー")

        annotator = MemoryErrorAnnotator("test_model")

        test_images = [Mock(spec=Image.Image) for _ in range(2)]
        test_phash_list = ["hash1", "hash2"]

        results = annotator.predict(test_images, test_phash_list)

        assert len(results) == 2
        for result in results:
            assert result["tags"] == []
            assert result["formatted_output"] is None
            assert result["error"] == "メモリ不足エラー"

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    @patch('image_annotator_lib.core.base.torch.no_grad')
    def test_predict_with_tag_generation_error(self, mock_no_grad, mock_utils, mock_config_registry):
        """predict メソッドでタグ生成エラーが発生した場合のテスト。"""
        # モックの設定
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 2,
            ("test_model", "model_path"): "/test/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        class TagErrorAnnotator(ConcreteAnnotator):
            def _generate_tags(self, formatted_output: Any) -> list[str]:
                raise ValueError("タグ生成テストエラー")

        annotator = TagErrorAnnotator("test_model")

        test_images = [Mock(spec=Image.Image)]
        test_phash_list = ["hash1"]

        results = annotator.predict(test_images, test_phash_list)

        assert len(results) == 1
        result = results[0]
        assert result["tags"] == []
        assert result["formatted_output"] == {"formatted": True}
        assert "タグ生成エラー: タグ生成テストエラー" in result["error"]


class TestTransformersBaseAnnotator:
    """TransformersBaseAnnotator のテスト。"""

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init(self, mock_utils, mock_config_registry):
        """初期化のテスト。"""
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 8,
            ("test_model", "model_path"): "/test/path",
            ("test_model", "max_length"): 100,
            ("test_model", "processor_path"): "/processor/path",
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = TransformersBaseAnnotator("test_model")

        assert annotator.model_name == "test_model"
        assert annotator.max_length == 100
        assert annotator.processor_path == "/processor/path"

    def test_generate_tags_with_string(self):
        """_generate_tags メソッドで文字列入力のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): "/test/path",
                ("test_model", "max_length"): 75,
                ("test_model", "processor_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = TransformersBaseAnnotator("test_model")

            # 文字列入力のテスト
            result = annotator._generate_tags("test caption")
            assert result == ["test caption"]

            # 非文字列入力のテスト（ログ出力の確認は困難なので、戻り値のみ確認）
            result = annotator._generate_tags(123)
            assert result == []

            # None入力のテスト
            result = annotator._generate_tags(None)
            assert result == []


class TestWebApiBaseAnnotator:
    """WebApiBaseAnnotator のテスト。"""

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init(self, mock_utils, mock_config_registry):
        """初期化のテスト。"""
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 8,
            ("test_model", "model_path"): None,  # WebAPIではNone
            ("test_model", "prompt_template"): "Test prompt",
            ("test_model", "timeout"): 30,
            ("test_model", "retry_count"): 5,
            ("test_model", "retry_delay"): 2.0,
            ("test_model", "min_request_interval"): 0.5,
            ("test_model", "max_output_tokens"): 2000,
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = WebApiBaseAnnotator("test_model")

        assert annotator.model_name == "test_model"
        assert annotator.prompt_template == "Test prompt"
        assert annotator.timeout == 30
        assert annotator.retry_count == 5
        assert annotator.retry_delay == 2.0
        assert annotator.min_request_interval == 0.5
        assert annotator.max_output_tokens == 2000

    @patch('image_annotator_lib.core.base.config_registry')
    @patch('image_annotator_lib.core.base.utils')
    def test_init_with_invalid_values(self, mock_utils, mock_config_registry):
        """無効な設定値での初期化テスト。"""
        mock_config_registry.get_all_config.return_value = {"test_model": {}}
        mock_config_registry.get.side_effect = lambda model, key, default=None: {
            ("test_model", "device"): "cuda",
            ("test_model", "chunk_size"): 8,
            ("test_model", "model_path"): None,
            ("test_model", "timeout"): "invalid",  # 無効な値
            ("test_model", "retry_count"): "invalid",  # 無効な値
            ("test_model", "retry_delay"): "invalid",  # 無効な値
            ("test_model", "min_request_interval"): "invalid",  # 無効な値
            ("test_model", "max_output_tokens"): "invalid",  # 無効な値
        }.get((model, key), default)
        mock_utils.determine_effective_device.return_value = "cuda"

        annotator = WebApiBaseAnnotator("test_model")

        # デフォルト値が使用されることを確認
        assert annotator.timeout == 60
        assert annotator.retry_count == 3
        assert annotator.retry_delay == 1.0
        assert annotator.min_request_interval == 1.0
        assert annotator.max_output_tokens is None

    def test_preprocess_images(self):
        """_preprocess_images メソッドのテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # モック画像を作成
            mock_image = Mock(spec=Image.Image)
            mock_image.save = Mock()

            with patch('base64.b64encode') as mock_b64encode:
                mock_b64encode.return_value.decode.return_value = "encoded_image_data"

                result = annotator._preprocess_images([mock_image])

                assert result == ["encoded_image_data"]
                mock_image.save.assert_called_once()

    def test_parse_common_json_response_with_dict(self):
        """_parse_common_json_response メソッドで辞書入力のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils, \
             patch('image_annotator_lib.core.base.AnnotationSchema') as mock_schema:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            # AnnotationSchemaのモック設定
            mock_validated = Mock()
            mock_validated.model_dump.return_value = {"tags": ["test"], "caption": "test caption"}
            mock_schema.model_validate.return_value = mock_validated

            annotator = WebApiBaseAnnotator("test_model")

            test_dict = {"tags": ["test"], "caption": "test caption"}
            result = annotator._parse_common_json_response(test_dict)

            assert result["annotation"] == {"tags": ["test"], "caption": "test caption"}
            assert result["error"] is None

    def test_parse_common_json_response_with_json_string(self):
        """_parse_common_json_response メソッドでJSON文字列入力のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # 正常なJSON文字列
            json_string = '{"tags": ["test1", "test2"], "caption": "test caption"}'
            result = annotator._parse_common_json_response(json_string)

            assert result["annotation"] == {"tags": ["test1", "test2"], "caption": "test caption"}
            assert result["error"] is None

            # Annotationキーを含むJSON
            json_string = '{"Annotation": {"tags": ["test1"], "score": 0.8}}'
            result = annotator._parse_common_json_response(json_string)

            assert result["annotation"] == {"tags": ["test1"], "score": 0.8}
            assert result["error"] is None

    def test_parse_common_json_response_with_invalid_json(self):
        """_parse_common_json_response メソッドで無効なJSON入力のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # 無効なJSON文字列
            invalid_json = '{"invalid": json}'
            result = annotator._parse_common_json_response(invalid_json)

            assert result["annotation"] is None
            assert "JSON解析エラー" in result["error"]

    def test_extract_tags_from_text_json_format(self):
        """_extract_tags_from_text メソッドでJSON形式のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # JSON形式のタグリスト
            json_text = '{"tags": ["tag1", "tag2", "tag3"]}'
            result = annotator._extract_tags_from_text(json_text)
            assert result == ["tag1", "tag2", "tag3"]

            # ネストしたJSON形式
            json_text = '{"Annotation": {"tags": ["nested1", "nested2"]}}'
            result = annotator._extract_tags_from_text(json_text)
            assert result == ["nested1", "nested2"]

            # JSON配列形式
            json_text = '["array1", "array2", "array3"]'
            result = annotator._extract_tags_from_text(json_text)
            assert result == ["array1", "array2", "array3"]

    def test_extract_tags_from_text_comma_separated(self):
        """_extract_tags_from_text メソッドでカンマ区切り形式のテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # カンマ区切り形式
            text = "tag1, tag2, tag3"
            result = annotator._extract_tags_from_text(text)
            assert result == ["tag1", "tag2", "tag3"]

            # プレフィックス付き
            text = "tags: tag1, tag2, tag3"
            result = annotator._extract_tags_from_text(text)
            assert result == ["tag1", "tag2", "tag3"]

    def test_generate_tags_with_formatted_output(self):
        """_generate_tags メソッドのテスト。"""
        with patch('image_annotator_lib.core.base.config_registry') as mock_config_registry, \
             patch('image_annotator_lib.core.base.utils') as mock_utils:

            mock_config_registry.get_all_config.return_value = {"test_model": {}}
            mock_config_registry.get.side_effect = lambda model, key, default=None: {
                ("test_model", "device"): "cuda",
                ("test_model", "chunk_size"): 8,
                ("test_model", "model_path"): None,
            }.get((model, key), default)
            mock_utils.determine_effective_device.return_value = "cuda"

            annotator = WebApiBaseAnnotator("test_model")

            # 正常なフォーマット済み出力
            formatted_output = {
                "annotation": {"tags": ["tag1", "tag2"]},
                "error": None
            }
            result = annotator._generate_tags(formatted_output)
            assert result == ["tag1", "tag2"]

            # エラーを含む出力
            formatted_output = {
                "annotation": None,
                "error": "テストエラー"
            }
            result = annotator._generate_tags(formatted_output)
            assert result == []

            # annotationがNoneの場合
            formatted_output = {
                "annotation": None,
                "error": None
            }
            result = annotator._generate_tags(formatted_output)
            assert result == []


if __name__ == "__main__":
    pytest.main([__file__])
