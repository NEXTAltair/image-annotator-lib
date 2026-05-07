"""ライブラリの外部 API 関数 (公開層)。

複数モデルでのアノテーション実行ロジックは core/annotation_runner.py に切り出されている。
本モジュールは公開 API のシグネチャ維持と委譲のみを担う薄い層として設計されている。
"""

from typing import Any

from PIL import Image

from .core.annotation_runner import run_annotation
from .core.api_model_discovery import get_available_models as _discover_available_models
from .core.registry import list_available_annotators as _registry_list_annotators
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
    from .core.webapi_annotator import WebApiAnnotator

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
        # WebAPI モデル / ローカル ML モデルの排他分岐 (Issue #26 Codex P2 #6 根本対応):
        #   - 判定基準: **model_class が WebApiAnnotator (またはサブクラス) か**
        #     (ADR 0023 Phase 1 / Issue #35 で class 名文字列照合から issubclass に移行)
        #   - WebAPI モデル: `_WEBAPI_MODEL_METADATA` (SSoT) のみ参照
        #   - ローカル ML モデル: `config_registry` (user TOML) のみ参照
        # 旧来の `or` フォールバック方式 (PR #22) では discovery 経由の `api_model_id`
        # がローカル ML モデルに混入し `_requires_api_key` が誤分類する Codex P2 #6 が
        # 発生していたが、model_class ベースの排他分岐により混入経路が消滅する。
        # 注意 (PR #27 Codex P1): merge 処理 (`{**a, **b}`) は b が dict でない場合
        # `TypeError` を投げるため try ブロック **内** で実施する。malformed user TOML
        # entry (truthy で dict でない値) で listing 全体が abort することを防ぐ。
        try:
            is_webapi_class = issubclass(model_class, WebApiAnnotator)
            raw_user_overrides = all_config.get(model_name)
            user_overrides: dict[str, Any] = (
                raw_user_overrides if isinstance(raw_user_overrides, dict) else {}
            )
            if raw_user_overrides is not None and not isinstance(raw_user_overrides, dict):
                logger.warning(
                    f"モデル '{model_name}' の config_registry エントリが dict ではありません "
                    f"(取得型: {type(raw_user_overrides).__name__})。空 dict として扱います。"
                )
            if is_webapi_class:
                # WebAPI モデル: metadata は SSoT のみ。user TOML は実行時 override 用で、
                # api_model_id/provider/capability 等のモデル定義には使わない。
                raw_webapi_metadata = get_webapi_metadata(model_name)
                webapi_metadata: dict[str, Any] = (
                    raw_webapi_metadata if isinstance(raw_webapi_metadata, dict) else {}
                )
                model_config = webapi_metadata
            else:
                # ローカル ML モデル: user TOML のみ (WebAPI metadata は混入させない)
                model_config = user_overrides
            infos.append(_build_annotator_info_for_registry_model(model_name, model_class, model_config))
            seen_names.add(model_name)
        except Exception as e:
            logger.error(f"モデル '{model_name}' の AnnotatorInfo 構築失敗: {e}", exc_info=True)

    # 2) LiteLLM 直接モデル (レジストリと重複するものは除外)
    # ADR 0023 Phase 1: SimplifiedAgentFactory は廃止され、LiteLLM 同梱 DB を runtime SSoT とする。
    try:
        for model_id in _discover_available_models():
            if model_id in seen_names:
                continue
            infos.append(_build_annotator_info_for_direct_model(model_id))
            seen_names.add(model_id)
    except Exception as e:
        logger.error(f"LiteLLM 直接モデルの取得失敗: {e}", exc_info=True)

    infos.sort(key=lambda info: info.name)
    logger.info(f"AnnotatorInfo リスト生成完了: {len(infos)} 件")
    return infos
