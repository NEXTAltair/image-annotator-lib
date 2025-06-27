"""BaseAnnotator クラスのテスト"""

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.exceptions.errors import OutOfMemoryError


class MockAnnotator(BaseAnnotator):
    """テスト用の BaseAnnotator 実装"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True

    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        return [f"processed_{i}" for i in range(len(images))]

    def _run_inference(self, processed: Any) -> Any:
        return [f"inference_{item}" for item in processed]

    def _format_predictions(self, raw_outputs: Any) -> Any:
        return [f"formatted_{item}" for item in raw_outputs]

    def _generate_tags(self, formatted_output: Any) -> list[str]:
        return [f"tag1_{formatted_output}", f"tag2_{formatted_output}"]


class TestBaseAnnotator:
    """BaseAnnotator クラスのテスト"""

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.config_registry")
    def test_init_with_default_device(self, mock_config):
        """初期化テスト - デフォルトデバイス"""

        def mock_get(model: str, key: str, default: str = "cpu") -> str:
            if model == "test_model" and key == "model_path":
                return "/path/to/model"
            elif model == "test_model" and key == "device":
                return default
            return default

        mock_config.get.side_effect = mock_get

        annotator = MockAnnotator("test_model")

        assert annotator.model_name == "test_model"
        assert annotator.model_path == "/path/to/model"
        assert annotator.device == "cpu"
        assert annotator.components is None

        mock_config.get.assert_any_call("test_model", "model_path")
        mock_config.get.assert_any_call("test_model", "device", "cpu")

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.config_registry")
    def test_init_with_custom_device(self, mock_config):
        """初期化テスト - カスタムデバイス"""

        def mock_get(model: str, key: str, default: str = "cpu") -> str:
            if model == "test_model" and key == "model_path":
                return "/path/to/model"
            elif model == "test_model" and key == "device":
                return "cuda"
            return default

        mock_config.get.side_effect = mock_get

        annotator = MockAnnotator("test_model")

        assert annotator.device == "cuda"

    @pytest.mark.standard
    def test_context_manager(self):
        """コンテキストマネージャーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")

            assert not annotator.entered
            assert not annotator.exited

            with annotator:
                assert annotator.entered
                assert not annotator.exited

            assert annotator.entered
            assert annotator.exited

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.imagehash")
    def test_calculate_phash_success(self, mock_imagehash):
        """知覚ハッシュ計算成功テスト"""
        mock_hash = Mock()
        mock_hash.__str__ = Mock(return_value="abc123")
        mock_imagehash.phash.return_value = mock_hash

        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            image = Mock(spec=Image.Image)

            result = annotator._calculate_phash(image)

            assert result == "abc123"
            mock_imagehash.phash.assert_called_once_with(image)

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.imagehash")
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_calculate_phash_failure(self, mock_logger, mock_imagehash):
        """知覚ハッシュ計算失敗テスト"""
        mock_imagehash.phash.side_effect = Exception("Hash calculation failed")

        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            image = Mock(spec=Image.Image)

            result = annotator._calculate_phash(image)

            assert result is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_empty_images(self, mock_logger):
        """空の画像リストでの予測テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")

            result = annotator.predict([])

            assert result == []
            mock_logger.warning.assert_called_once_with(
                "空の画像リストが渡されました。アノテーションをスキップします。"
            )

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_success_single_image(self, mock_logger):
        """単一画像での予測成功テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            image = cast(Image.Image, Mock(spec=Image.Image))

            with patch.object(annotator, "_calculate_phash", return_value="test_hash"):
                result = annotator.predict([image])

            assert len(result) == 1
            assert result[0].get("phash") == "test_hash"
            assert result[0].get("tags") == [
                "tag1_formatted_inference_processed_0",
                "tag2_formatted_inference_processed_0",
            ]
            assert result[0].get("formatted_output") == "formatted_inference_processed_0"
            assert result[0].get("error") is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_success_multiple_images(self, mock_logger):
        """複数画像での予測成功テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(3)]

            with patch.object(annotator, "_calculate_phash", side_effect=["hash1", "hash2", "hash3"]):
                result = annotator.predict(images)

            assert len(result) == 3
            for i, res in enumerate(result):
                assert res.get("phash") == f"hash{i + 1}"
                assert res.get("tags") == [
                    f"tag1_formatted_inference_processed_{i}",
                    f"tag2_formatted_inference_processed_{i}",
                ]
                assert res.get("error") is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_with_provided_phash(self, mock_logger):
        """事前計算されたハッシュでの予測テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]
            phash_list = ["provided_hash1", "provided_hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 2
            assert result[0].get("phash") == "provided_hash1"
            assert result[1].get("phash") == "provided_hash2"

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_tag_generation_error(self, mock_logger):
        """タグ生成エラーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            annotator._generate_tags = Mock(side_effect=Exception("Tag generation failed"))

            image = cast(Image.Image, Mock(spec=Image.Image))
            phash_list = ["test_hash"]

            result = annotator.predict([image], phash_list)

            assert len(result) == 1
            assert result[0].get("phash") == "test_hash"
            assert result[0].get("tags") == []
            assert result[0].get("formatted_output") is None
            error_msg = result[0].get("error")
            assert error_msg is not None and "タグ生成エラー" in error_msg
            mock_logger.exception.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_out_of_memory_error(self, mock_logger):
        """メモリ不足エラーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=OutOfMemoryError("Out of memory"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]
            phash_list = ["hash1", "hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 2
            for i, res in enumerate(result):
                assert res.get("phash") == f"hash{i + 1}"
                assert res.get("tags") == []
                assert res.get("formatted_output") is None
                assert res.get("error") == "メモリ不足エラー"

            mock_logger.error.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_unexpected_error(self, mock_logger):
        """予期せぬエラーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=RuntimeError("Unexpected error"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]

            result = annotator.predict(images)

            assert len(result) == 2
            for res in result:
                assert res.get("phash") is None
                assert res.get("tags") == []
                assert res.get("formatted_output") is None
                error_msg = res.get("error")
                assert error_msg is not None and "予期せぬエラー" in error_msg

            mock_logger.exception.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_single_formatted_output(self, mock_logger):
        """単一の整形出力での予測テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            # _format_predictions が単一の値を返すようにモック
            annotator._format_predictions = Mock(return_value="single_formatted_output")

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]

            with patch.object(annotator, "_calculate_phash", side_effect=["hash1", "hash2"]):
                result = annotator.predict(images)

            assert len(result) == 2
            for res in result:
                assert res.get("formatted_output") == "single_formatted_output"
                assert res.get("tags") == ["tag1_single_formatted_output", "tag2_single_formatted_output"]

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_timing_logs(self, mock_logger):
        """処理時間ログのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            image = cast(Image.Image, Mock(spec=Image.Image))

            with patch.object(annotator, "_calculate_phash", return_value="test_hash"):
                annotator.predict([image])

            # デバッグログが3回呼ばれることを確認(前処理、推論、整形の時間)
            debug_calls = [call for call in mock_logger.debug.call_args_list if "時間:" in str(call)]
            assert len(debug_calls) == 3

    @pytest.mark.standard
    def test_abstract_methods_not_implemented(self):
        """抽象メソッドが実装されていない場合のテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            # BaseAnnotator を直接インスタンス化しようとするとエラーになることを確認
            with pytest.raises(TypeError):
                BaseAnnotator("test_model")  # type: ignore

    @pytest.mark.standard
    def test_abstract_methods_raise_not_implemented(self):
        """抽象メソッドが NotImplementedError を発生させることのテスト"""

        # 部分的に実装されたクラスを作成
        class PartialAnnotator(BaseAnnotator):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def _preprocess_images(self, images: list[Image.Image]) -> Any:
                raise NotImplementedError()

            def _run_inference(self, processed: Any) -> Any:
                raise NotImplementedError()

            def _format_predictions(self, raw_outputs: Any) -> Any:
                raise NotImplementedError()

            def _generate_tags(self, formatted_output: Any) -> list[str]:
                raise NotImplementedError()

        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = PartialAnnotator("test_model")

            with pytest.raises(NotImplementedError):
                annotator._preprocess_images([])

            with pytest.raises(NotImplementedError):
                annotator._run_inference([])

            with pytest.raises(NotImplementedError):
                annotator._format_predictions([])

            with pytest.raises(NotImplementedError):
                annotator._generate_tags([])

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_phash_list_shorter_than_images(self, mock_logger):
        """ハッシュリストが画像リストより短い場合のテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(3)]
            phash_list = ["hash1", "hash2"]  # 画像より少ない

            with patch.object(annotator, "_calculate_phash", return_value="calculated_hash"):
                result = annotator.predict(images, phash_list)

            assert len(result) == 3
            assert result[0].get("phash") == "hash1"
            assert result[1].get("phash") == "hash2"
            assert result[2].get("phash") == "calculated_hash"  # 計算されたハッシュ

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_error_with_phash_list(self, mock_logger):
        """エラー時にハッシュリストが正しく使用されることのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry"):
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=RuntimeError("Test error"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]
            phash_list = ["error_hash1", "error_hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 2
            assert result[0].get("phash") == "error_hash1"
            assert result[1].get("phash") == "error_hash2"
