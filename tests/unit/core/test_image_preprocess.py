"""ADR 0023 Phase 1: webapi/image_preprocess.py の unit test."""

from __future__ import annotations

from PIL import Image
from pydantic_ai.messages import BinaryContent

from image_annotator_lib.webapi.image_preprocess import preprocess_images_to_binary


class TestPreprocessImagesToBinary:
    """PIL Image → BinaryContent 変換を確認する。"""

    def test_single_image(self) -> None:
        image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        result = preprocess_images_to_binary([image])

        assert len(result) == 1
        assert isinstance(result[0], BinaryContent)
        assert result[0].media_type == "image/webp"
        assert isinstance(result[0].data, bytes)
        assert len(result[0].data) > 0

    def test_multiple_images(self) -> None:
        images = [
            Image.new("RGB", (10, 10), color=(255, 0, 0)),
            Image.new("RGB", (20, 20), color=(0, 255, 0)),
            Image.new("RGB", (30, 30), color=(0, 0, 255)),
        ]
        result = preprocess_images_to_binary(images)

        assert len(result) == 3
        for binary in result:
            assert isinstance(binary, BinaryContent)
            assert binary.media_type == "image/webp"

    def test_empty_list(self) -> None:
        assert preprocess_images_to_binary([]) == []

    def test_preserves_order(self) -> None:
        red = Image.new("RGB", (10, 10), color=(255, 0, 0))
        green = Image.new("RGB", (20, 20), color=(0, 255, 0))
        result = preprocess_images_to_binary([red, green])

        # WebP 出力では赤画像と緑画像のバイナリは異なるはず
        assert result[0].data != result[1].data
