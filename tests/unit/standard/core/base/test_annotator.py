"""BaseAnnotator クラスのテスト"""

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult
from image_annotator_lib.exceptions.errors import OutOfMemoryError


@pytest.fixture(autouse=True)
def setup_test_model_config():
    """Setup test model configuration for all tests."""
    from image_annotator_lib.core.config import config_registry

    config = {
        "model_path": "/path/to/model",
        "device": "cpu",
        "class": "MockAnnotator",
        "capabilities": ["tags"],
    }
    for key, value in config.items():
        config_registry.add_default_setting("test_model", key, value)

    # Store original get_all_config for potential restoration
    original_get_all_config = config_registry.get_all_config

    yield

    # Cleanup
    try:
        config_registry._config.pop("test_model", None)
    except (AttributeError, KeyError):
        pass


def setup_mock_config_registry(mock_config, model_config: dict[str, Any]):
    """Helper to setup mock config_registry with get_all_config."""
    mock_config.get_all_config.return_value = {"test_model": model_config}


class MockAnnotator(BaseAnnotator):
    """テスト用の BaseAnnotator 実装。

    Note:
        ADR 0023 Phase 1 で `_format_predictions` は `UnifiedAnnotationResult` の
        list を返す契約に統一された (`_build_results` が型違反を `TypeError` で弾く)。
        `_generate_tags` は `BaseAnnotator` の pipeline から外れた (onnx / tensorflow の
        tagger サブクラス専用ヘルパーに降格) ため本 Mock では実装しない。
        本 Mock と `test_predict_*` 系 test は Issue #269 で現契約に合わせ再構築した。
    """

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

    def _format_predictions(self, raw_outputs: Any) -> list[UnifiedAnnotationResult]:
        """推論結果を `UnifiedAnnotationResult` の list へ整形する (現契約)。"""
        return [
            UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities={TaskCapability.TAGS},
                tags=[f"tag1_{item}", f"tag2_{item}"],
            )
            for item in raw_outputs
        ]


class TestBaseAnnotator:
    """BaseAnnotator クラスのテスト"""

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.config_registry")
    def test_init_with_default_device(self, mock_config):
        """初期化テスト - デフォルトデバイス"""
        # Mock get_all_config to return full model config dict
        mock_config.get_all_config.return_value = {
            "test_model": {
                "model_path": "/path/to/model",
                "device": "cpu",
                "class": "MockAnnotator",
            }
        }

        annotator = MockAnnotator("test_model")

        assert annotator.model_name == "test_model"
        assert annotator.model_path == "/path/to/model"
        # Issue #35: device 判定はサブクラスへ移譲。MockAnnotator は BaseAnnotator 直系のため
        # device sentinel "" が残る。
        assert annotator.device == ""
        assert annotator._config.device == "cpu"
        assert annotator.components is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.config_registry")
    def test_init_with_custom_device(self, mock_config, mock_cuda_available):
        """初期化テスト - カスタムデバイス"""
        # Mock get_all_config to return full model config dict with custom device
        mock_config.get_all_config.return_value = {
            "test_model": {
                "model_path": "/path/to/model",
                "device": "cuda",
                "class": "MockAnnotator",
            }
        }

        annotator = MockAnnotator("test_model")

        # Issue #35: device 判定はサブクラスへ移譲。MockAnnotator は BaseAnnotator 直系のため
        # device sentinel "" が残る。
        assert annotator.device == ""
        assert annotator._config.device == "cuda"

    @pytest.mark.standard
    def test_context_manager(self):
        """コンテキストマネージャーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
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

        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
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

        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            image = Mock(spec=Image.Image)

            result = annotator._calculate_phash(image)

            assert result is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_empty_images(self, mock_logger):
        """空の画像リストでの予測テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")

            result = annotator.predict([])

            assert result == []
            mock_logger.warning.assert_called_once_with(
                "空の画像リストが渡されました。アノテーションをスキップします。"
            )

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_success_single_image(self, mock_logger):
        """単一画像での予測成功テスト。

        `predict()` は `_format_predictions` が返す `UnifiedAnnotationResult` を
        そのまま結果として返す (ADR 0023 Phase 1 契約)。
        """
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            image = cast(Image.Image, Mock(spec=Image.Image))

            result = annotator.predict([image])

            assert len(result) == 1
            assert isinstance(result[0], UnifiedAnnotationResult)
            assert result[0].tags == [
                "tag1_inference_processed_0",
                "tag2_inference_processed_0",
            ]
            assert result[0].error is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_success_multiple_images(self, mock_logger):
        """複数画像での予測成功テスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(3)]

            result = annotator.predict(images)

            assert len(result) == 3
            for i, res in enumerate(result):
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.tags == [
                    f"tag1_inference_processed_{i}",
                    f"tag2_inference_processed_{i}",
                ]
                assert res.error is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_with_provided_phash(self, mock_logger):
        """事前計算されたハッシュを渡しても結果は正常に返る (phash は api.py で処理)。"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]
            phash_list = ["provided_hash1", "provided_hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 2
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.error is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_format_predictions_error(self, mock_logger):
        """`_format_predictions` が例外を投げた場合のエラーハンドリングテスト。

        Note:
            旧 test (`test_predict_tag_generation_error`) は `_generate_tags` の例外を
            検証していたが、ADR 0023 Phase 1 で `_generate_tags` は `BaseAnnotator` の
            pipeline から外れた。整形段階 (`_format_predictions`) の例外を検証する
            test として再定義した (Issue #269)。preprocess 段階の例外を検証する
            `test_predict_unexpected_error` とは例外発生段階で差別化される。
        """
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            annotator._format_predictions = Mock(side_effect=RuntimeError("Format failed"))

            image = cast(Image.Image, Mock(spec=Image.Image))

            result = annotator.predict([image])

            assert len(result) == 1
            assert isinstance(result[0], UnifiedAnnotationResult)
            error_msg = result[0].error
            assert error_msg is not None and "予期せぬエラー" in error_msg
            mock_logger.exception.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_out_of_memory_error(self, mock_logger):
        """メモリ不足エラーのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=OutOfMemoryError("Out of memory"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]

            result = annotator.predict(images)

            assert len(result) == 2
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.error == "メモリ不足エラー"

            mock_logger.error.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_unexpected_error(self, mock_logger):
        """予期せぬエラーのテスト (preprocess 段階で RuntimeError)。"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=RuntimeError("Unexpected error"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]

            result = annotator.predict(images)

            assert len(result) == 2
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                error_msg = res.error
                assert error_msg is not None and "予期せぬエラー" in error_msg

            mock_logger.exception.assert_called_once()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_single_formatted_output(self, mock_logger):
        """`_format_predictions` が単一値を返すと全画像へブロードキャストされる。

        `_execute_pipeline` は `_format_predictions` の戻り値が list でない場合
        `[output] * len(images)` でブロードキャストする。単一の
        `UnifiedAnnotationResult` を返すケースを検証する。
        """
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            single_result = UnifiedAnnotationResult(
                model_name="test_model",
                capabilities={TaskCapability.TAGS},
                tags=["broadcast_tag"],
            )
            # _format_predictions が単一の UnifiedAnnotationResult を返すようにモック
            annotator._format_predictions = Mock(return_value=single_result)

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]

            result = annotator.predict(images)

            assert len(result) == 2
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.tags == ["broadcast_tag"]
            # 同一オブジェクトがブロードキャストされる (_execute_pipeline)
            assert result[0] is result[1]

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_timing_logs(self, mock_logger):
        """処理時間ログのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            image = cast(Image.Image, Mock(spec=Image.Image))

            annotator.predict([image])

            # デバッグログが3回呼ばれることを確認(前処理、推論、整形の時間)
            debug_calls = [call for call in mock_logger.debug.call_args_list if "時間:" in str(call)]
            assert len(debug_calls) == 3

    @pytest.mark.standard
    def test_abstract_methods_not_implemented(self):
        """抽象メソッドが実装されていない場合のテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
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

        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "PartialAnnotator"}
            )
            annotator = PartialAnnotator("test_model")

            with pytest.raises(NotImplementedError):
                annotator._preprocess_images([])

            with pytest.raises(NotImplementedError):
                annotator._run_inference([])

            with pytest.raises(NotImplementedError):
                annotator._format_predictions([])

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_phash_list_shorter_than_images(self, mock_logger):
        """phash_listを渡しても結果は正常に返却されることのテスト (phashはapi.pyで処理)"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(3)]
            phash_list = ["hash1", "hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 3
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.error is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.annotator.logger")
    def test_predict_error_with_phash_list(self, mock_logger):
        """エラー時にphash_listを渡してもエラー結果が返ることのテスト"""
        with patch("image_annotator_lib.core.base.annotator.config_registry") as mock_config:
            setup_mock_config_registry(
                mock_config, {"model_path": "/path/to/model", "device": "cpu", "class": "MockAnnotator"}
            )
            annotator = MockAnnotator("test_model")
            annotator._preprocess_images = Mock(side_effect=RuntimeError("Test error"))

            images = [cast(Image.Image, Mock(spec=Image.Image)) for _ in range(2)]
            phash_list = ["error_hash1", "error_hash2"]

            result = annotator.predict(images, phash_list)

            assert len(result) == 2
            for res in result:
                assert isinstance(res, UnifiedAnnotationResult)
                assert res.error is not None
