"""PIL Image → PydanticAI `BinaryContent` 変換 (ADR 0023 Phase 1)。

旧 `pydantic_ai_factory.preprocess_images_to_binary()` を責務単位に切り出したモジュール。
WEBP に統一して送信サイズを抑える方針は維持する。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image
from pydantic_ai.messages import BinaryContent


def preprocess_images_to_binary(images: list[Image.Image]) -> list[BinaryContent]:
    """PIL Image のリストを PydanticAI `BinaryContent` のリストに変換する。

    Args:
        images: PIL Image オブジェクトのリスト。

    Returns:
        Agent に渡せる `BinaryContent` のリスト。WEBP 形式・media_type=`image/webp`。
    """
    binary_contents: list[BinaryContent] = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        binary_contents.append(BinaryContent(data=buffered.getvalue(), media_type="image/webp"))
    return binary_contents


__all__ = ["preprocess_images_to_binary"]
