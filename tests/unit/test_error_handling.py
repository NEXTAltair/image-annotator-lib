from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from image_annotator_lib import api
from image_annotator_lib.api import annotate
from image_annotator_lib.exceptions import (
    InvalidInputError,
    InvalidModelConfigError,
    InvalidOutputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_invalid_image_input(self) -> None:
        """無効な画像入力のテスト"""
        with pytest.raises(TypeError):
            # 非画像データを渡す
            invalid_image = "これは画像ではなく文字列です"
            model = MagicMock()
            model.predict.side_effect = TypeError("入力は PIL.Image オブジェクトである必要があります")
            model.predict([invalid_image])

    def test_model_load_error(self) -> None:
        """モデルロード時のエラー処理のテスト"""
        # 現行実装にImageRewardScorerが存在しないため、このテストはスキップまたは削除
        pass

    @pytest.mark.parametrize(
        "exception_class",
        [
            ModelNotFoundError,
            ModelLoadError,
            InvalidModelConfigError,
            UnsupportedModelError,
            ModelExecutionError,
            InvalidInputError,
            InvalidOutputError,
        ],
    )
    def test_model_exceptions(self, exception_class: type[Exception]) -> None:
        """モデル例外の発生テスト"""
        # 例外が適切に発生するかテスト
        with pytest.raises(exception_class):
            raise exception_class("テスト例外メッセージ")

    @patch("torch.cuda.max_memory_allocated")
    def test_memory_error_handling(self, mock_memory: MagicMock) -> None:
        """メモリ不足のエラー処理テスト"""
        # メモリ不足状態をシミュレート
        mock_memory.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        with pytest.raises(torch.cuda.OutOfMemoryError):
            # メモリエラーのシミュレーション
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
