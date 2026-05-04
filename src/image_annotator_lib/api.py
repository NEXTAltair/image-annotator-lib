"""ライブラリの外部 API 関数 (公開層)。

複数モデルでのアノテーション実行ロジックは core/annotation_runner.py に切り出されている。
本モジュールは公開 API のシグネチャ維持と委譲のみを担う薄い層として設計されている。
"""

from PIL import Image

from .core.annotation_runner import run_annotation
from .core.registry import list_available_annotators as _registry_list_annotators
from .core.types import PHashAnnotationResults

__all__ = ["PHashAnnotationResults", "annotate", "list_available_annotators"]


def annotate(
    images_list: list[Image.Image],
    model_name_list: list[str],
    phash_list: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> PHashAnnotationResults:
    """複数の画像を指定された複数のモデルで評価(アノテーション)する。

    Args:
        images_list: 評価対象の PIL Image オブジェクトのリスト。
        model_name_list: 使用するモデル名のリスト。
        phash_list: 各画像に対応するpHashのリスト。
        api_keys: WebAPIモデル用のAPIキー辞書 (オプション)。

    Returns:
        結果を格納した PHashAnnotationResults 辞書。
    """
    return run_annotation(
        images=images_list,
        model_names=model_name_list,
        phash_list=phash_list,
        api_keys=api_keys,
    )


def list_available_annotators() -> list[str]:
    """利用可能なアノテーターモデル名のリストを返す。

    Returns:
        利用可能なモデル名のリスト
    """
    return _registry_list_annotators()
