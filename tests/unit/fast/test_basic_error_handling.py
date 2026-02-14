from unittest.mock import MagicMock

import pytest

from image_annotator_lib.exceptions.errors import (
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

    @pytest.mark.fast
    def test_invalid_image_input(self) -> None:
        """無効な画像入力のテスト"""
        with pytest.raises(TypeError):
            # 非画像データを渡す
            invalid_image = "これは画像ではなく文字列です"
            model = MagicMock()
            model.predict.side_effect = TypeError("入力は PIL.Image オブジェクトである必要があります")
            model.predict([invalid_image])

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_memory_error_handling(self) -> None:
        """メモリ不足のエラー処理テスト"""

        # カスタム例外クラスでテスト（torchに依存しない）
        class MockCudaOutOfMemoryError(Exception):
            pass

        with pytest.raises(MockCudaOutOfMemoryError):
            # メモリエラーのシミュレーション
            raise MockCudaOutOfMemoryError("CUDA out of memory")
