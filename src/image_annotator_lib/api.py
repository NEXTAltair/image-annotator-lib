"""ライブラリの外部 API 関数 (公開層)。

複数モデルでのアノテーション実行ロジックは core/annotation_runner.py に切り出されている。
本モジュールは公開 API のシグネチャ維持と委譲のみを担う薄い層として設計されている。
"""

from PIL import Image

from .core.annotation_runner import run_annotation
from .core.registry import list_available_annotators as _registry_list_annotators
from .core.simplified_agent_factory import get_agent_factory
from .core.types import AnnotatorInfo, PHashAnnotationResults
from .core.utils import logger

__all__ = ["PHashAnnotationResults", "annotate", "list_annotator_info", "list_available_annotators"]


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


def list_annotator_info() -> list[AnnotatorInfo]:
    """登録済み全アノテーターのメタデータを返す。

    レジストリ経由のモデル + PydanticAI 直接モデル (``provider/model_id`` 形式)
    を統合した完全リストを返す。重複は登録済みレジストリ側を優先して除外する。

    Returns:
        AnnotatorInfo のリスト (name 昇順でソート済み)。

    See Also:
        list_available_annotators: モデル名のみのリスト。
    """
    from .core.config import config_registry
    from .core.registry import (
        _MODEL_CLASS_OBJ_REGISTRY,
        _REGISTRY_INITIALIZED,
        _build_annotator_info_for_direct_model,
        _build_annotator_info_for_registry_model,
        get_webapi_metadata,
        initialize_registry,
    )

    if not _REGISTRY_INITIALIZED:
        initialize_registry()

    infos: list[AnnotatorInfo] = []
    seen_names: set[str] = set()

    # 1) レジストリ登録済みモデル
    try:
        all_config = config_registry.get_all_config()
    except Exception as e:
        logger.error(f"設定取得に失敗: {e}", exc_info=True)
        all_config = {}

    for model_name, model_class in _MODEL_CLASS_OBJ_REGISTRY.items():
        # ローカル ML モデルは all_config (config_registry) から、
        # WebAPI モデルは _WEBAPI_MODEL_METADATA (SSoT) から取得する。
        # WebAPI モデルは Issue #23 以降 config_registry に登録されないため、
        # `or get_webapi_metadata(...)` のフォールバックで WebAPI 経路に入る。
        model_config = all_config.get(model_name) or get_webapi_metadata(model_name) or {}
        try:
            infos.append(_build_annotator_info_for_registry_model(model_name, model_class, model_config))
            seen_names.add(model_name)
        except Exception as e:
            logger.error(f"モデル '{model_name}' の AnnotatorInfo 構築失敗: {e}", exc_info=True)

    # 2) PydanticAI 直接モデル (レジストリと重複するものは除外)
    try:
        agent_factory = get_agent_factory()
        for model_id in agent_factory.get_available_models():
            if model_id in seen_names:
                continue
            infos.append(_build_annotator_info_for_direct_model(model_id))
            seen_names.add(model_id)
    except Exception as e:
        logger.error(f"PydanticAI 直接モデルの取得失敗: {e}", exc_info=True)

    infos.sort(key=lambda info: info.name)
    logger.info(f"AnnotatorInfo リスト生成完了: {len(infos)} 件")
    return infos
