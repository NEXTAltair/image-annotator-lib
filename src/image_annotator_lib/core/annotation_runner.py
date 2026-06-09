"""アノテーション実行ループの内部実装 (ADR 0023 Phase 1)。

このモジュールは `image_annotator_lib.api.annotate()` の内部実装詳細を提供する。
利用者は `image_annotator_lib.api.annotate()` を呼び出すこと。

責務:
- Annotator インスタンス管理 (キャッシュ + ファクトリ)
- 単一/複数モデルの実行
- pHash 単位での結果集約
- モデル実行失敗時の error result 記録

ADR 0023 / Issue #45: WebAPI モデルは LiteLLM 同梱 DB から起動時に discovery され、
`_register_webapi_models_from_discovery()` 経由で registry に自動登録される。
本モジュールは **registry 解決のみ** を行い、registry に存在しない `provider/model`
形式の direct LiteLLM ID dispatch 経路は廃止された (capability check / model
allowlist の責務を discovery / registry SSoT に集約)。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from typing import Any

from PIL import Image

from ..webapi.annotator import WebApiAnnotator
from .base.annotator import BaseAnnotator
from .registry import (
    find_model_class_case_insensitive,
    get_cls_obj_registry,
    get_webapi_metadata,
    initialize_registry,
)
from .types import PHashAnnotationResults, UnifiedAnnotationResult
from .utils import calculate_phash, get_model_capabilities, logger

_MODEL_INSTANCE_REGISTRY: dict[str, Any] = {}


def _is_webapi_annotator_class(annotator_class: type) -> bool:
    """registry 登録の WebAPI annotator クラスか判定する。

    ADR 0023 Phase 1 (Issue #35): WebAPI 系は `WebApiAnnotator` 1 種に統一されたため、
    `issubclass(annotator_class, WebApiAnnotator)` で判定する。サブクラス継承での
    test stub も自然に True になる。
    """
    return issubclass(annotator_class, WebApiAnnotator)


def _create_annotator_instance(model_name: str, api_keys: dict[str, str] | None = None) -> BaseAnnotator:
    """モデル名から annotator インスタンスを生成する (registry のみ参照)。

    ADR 0023 / Issue #45: 探索は registry のみ。`_register_webapi_models_from_discovery()`
    が起動時に LiteLLM 同梱 DB から WebAPI モデルを自動登録するため、`provider/model`
    形式の LiteLLM ID も registry 経由で解決される (例: `openai/gpt-4o` は
    `model_name_short = "openai/gpt-4o"` のキーで登録)。registry に存在しない
    任意の `provider/model` を `WebApiAnnotator` に直接渡す経路は廃止された
    (capability check が抜けて非 vision モデルが API 課金前に弾けない問題を回避)。

    Args:
        model_name: registry 登録名 (起動時 discovery で自動登録された LiteLLM ID
            slash 形式を含む)。
        api_keys: WebAPI モデル用の provider -> API key dict (オプション)。

    Returns:
        `BaseAnnotator` サブクラスのインスタンス。

    Raises:
        KeyError: registry に該当しない場合。
    """
    logger.debug(f"Creating annotator instance for model: '{model_name}'")

    model_result = find_model_class_case_insensitive(model_name)
    if model_result is not None:
        actual_model_name, Annotator_class = model_result

        # registry 登録 WebAPI モデル: metadata の litellm_model_id / mode で WebApiAnnotator を構築。
        # litellm_model_id と mode は同一 metadata から取得するため、二重 lookup を避けて
        # `get_webapi_metadata` を一度だけ呼ぶ。
        if _is_webapi_annotator_class(Annotator_class):
            metadata = get_webapi_metadata(actual_model_name) or {}
            litellm_model_id = metadata.get("litellm_model_id")
            if not litellm_model_id:
                raise KeyError(
                    f"Model '{actual_model_name}' is registered as WebAPI but has no "
                    f"`litellm_model_id` in `_WEBAPI_MODEL_METADATA`"
                )
            mode = metadata.get("mode", "chat")
            logger.debug(
                f"WebApiAnnotator (registry WebAPI): model={actual_model_name}, "
                f"litellm_model_id={litellm_model_id}, mode={mode}"
            )
            return WebApiAnnotator(
                litellm_model_id=litellm_model_id,
                api_keys=api_keys,
                model_name=actual_model_name,
                capabilities=get_model_capabilities(actual_model_name),
                mode=mode,
            )

        # registry 登録 ローカル ML モデル: 直接インスタンス化
        instance = Annotator_class(model_name=actual_model_name)
        logger.debug(
            f"モデル '{model_name}' -> '{actual_model_name}' を直接インスタンス化 "
            f"(クラス: {Annotator_class.__name__})"
        )
        return instance

    # registry 未登録 → KeyError
    registry = get_cls_obj_registry()
    available_models = list(registry.keys())
    error_details = {
        "requested_model": model_name,
        "registry_models_count": len(available_models),
        "registry_sample": available_models[:5],
    }
    logger.error(f"Model resolution failed: {error_details}")
    raise KeyError(
        f"Model '{model_name}' not found in registry. "
        f"WebAPI モデルは起動時 discovery で registry に自動登録されます "
        f"(ADR 0023 / Issue #45: registry 未登録の `provider/model` 直接呼び出しは廃止)。"
    )


def get_annotator_instance(model_name: str, api_keys: dict[str, str] | None = None) -> Any:
    """モデル名からアノテータインスタンスを取得する (キャッシュあり)。

    モデルがすでにロードされている場合はキャッシュから返す。
    まだロードされていない場合は、新たにインスタンスを作成してキャッシュに保存する。

    Args:
        model_name: モデルの名前 (registry 登録名または `provider/model` 形式)。
        api_keys: WebAPI モデル用の provider -> API key dict (オプション)。

    Returns:
        アノテータインスタンス。
    """
    # API キーが実際に指定されている場合のみキャッシュをバイパスする。
    # api_keys={} (空 dict) は「未指定」と同義であり、ローカル ML モデルが毎回
    # 新インスタンスを生成して 2 回目に ModelLoadError になるバグを防ぐ (Issue #146)。
    if api_keys:
        logger.debug(f"APIキー指定のためモデル '{model_name}' の新しいインスタンスを作成")
        return _create_annotator_instance(model_name, api_keys=api_keys)

    if model_name in _MODEL_INSTANCE_REGISTRY:
        logger.debug(f"モデル '{model_name}' はキャッシュから取得されました")
        return _MODEL_INSTANCE_REGISTRY[model_name]

    instance = _create_annotator_instance(model_name)
    _MODEL_INSTANCE_REGISTRY[model_name] = instance
    return instance


def _annotate_model(
    Annotator: BaseAnnotator, images: list[Image.Image], phash_list: list[str]
) -> list[UnifiedAnnotationResult]:
    """1モデル分のアノテーション処理を実施する。

    Args:
        Annotator: アノテータインスタンス。
        images: 処理対象の画像リスト。
        phash_list: 各画像に対応する pHash のリスト。

    Returns:
        統一バリデーションスキーマでの結果リスト。
    """
    with Annotator:
        results: list[UnifiedAnnotationResult] = Annotator.predict(images, phash_list)
    return results


def _process_model_results(
    model_name: str,
    annotation_results: list[UnifiedAnnotationResult],
    results_by_phash: PHashAnnotationResults,
    phash_map: dict[int, str],
) -> None:
    """モデルの結果を pHash ベースの構造に変換する。"""
    for i, result in enumerate(annotation_results):
        phash_key = phash_map.get(i)
        if phash_key is None:
            logger.warning(f"pHash取得失敗: index={i}, model={model_name}")
            continue
        if phash_key not in results_by_phash:
            results_by_phash[phash_key] = {}
        results_by_phash[phash_key][model_name] = result
        logger.debug(f"モデル '{model_name}' の結果を pHash '{phash_key[:8]}...' に格納しました")


def _prepare_phash_map(
    images_list: list[Image.Image],
    phash_list: list[str] | None,
) -> tuple[list[str], dict[int, str]]:
    """画像リストに対する pHash マップを準備する。"""
    phash_map: dict[int, str] = {}

    if phash_list is None:
        phash_list = []
        logger.debug("画像のpHash計算を開始...")
        for i, image in enumerate(images_list):
            phash = calculate_phash(image)
            phash_list.append(phash)
            phash_map[i] = phash
        logger.debug(f"pHash計算完了: {len(phash_list)}個")
    else:
        logger.debug("提供されたpHashリストを使用します。")
        for i, phash in enumerate(phash_list):
            if i < len(images_list):
                phash_map[i] = phash
            else:
                logger.warning(
                    f"pHashリストの要素数が画像リストの要素数を超えています。"
                    f"インデックス {i} 以降のpHashは無視されます。"
                )
                break

    return phash_list, phash_map


def _execute_model_annotation(
    model_name: str,
    images_list: list[Image.Image],
    phash_list: list[str],
    phash_map: dict[int, str],
    results_by_phash: PHashAnnotationResults,
    api_keys: dict[str, str] | None,
) -> None:
    """単一モデルでのアノテーションを実行し、結果を `results_by_phash` に格納する。"""
    try:
        annotator = get_annotator_instance(model_name, api_keys=api_keys)
        annotation_results = _annotate_model(annotator, images_list, phash_list)
        logger.debug(f"モデル '{model_name}' の評価完了。結果件数: {len(annotation_results)}")

        _process_model_results(model_name, annotation_results, results_by_phash, phash_map)

        if len(annotation_results) != len(images_list):
            logger.error(
                f"モデル '{model_name}' の結果リスト長 ({len(annotation_results)}) が"
                f"画像数 ({len(images_list)}) と一致しません。"
            )
            from .utils import get_model_capabilities

            capabilities = get_model_capabilities(model_name)
            error_result = UnifiedAnnotationResult(
                model_name=model_name, capabilities=capabilities, error="処理結果が不足しています"
            )
            for i in range(len(annotation_results), len(images_list)):
                phash_key = phash_map.get(i) or f"unknown_image_{i}"
                if phash_key not in results_by_phash:
                    results_by_phash[phash_key] = {}
                results_by_phash[phash_key][model_name] = error_result

    except Exception as e:
        logger.error(
            f"Model processing fatal error: model={model_name}, "
            f"error_type={type(e).__name__}, message={e!s}, "
            f"images_count={len(images_list)}",
            exc_info=True,
        )
        _record_model_error(model_name, e, phash_list, phash_map, results_by_phash)


def _record_model_error(
    model_name: str,
    error: Exception,
    phash_list: list[str],
    phash_map: dict[int, str],
    results_by_phash: PHashAnnotationResults,
) -> None:
    """モデル処理の致命的エラーを全画像に記録する。"""
    error_message = f"{type(error).__name__}: {error}"
    from .utils import get_model_capabilities

    capabilities = get_model_capabilities(model_name)
    if not capabilities:
        from .types import TaskCapability

        capabilities = {TaskCapability.TAGS}
        logger.warning(f"モデル '{model_name}' のcapabilitiesが未設定のため、デフォルト値 {{TAGS}} を使用")
    error_results = [
        UnifiedAnnotationResult(model_name=model_name, capabilities=capabilities, error=error_message)
        for _ in phash_list
    ]
    _process_model_results(model_name, error_results, results_by_phash, phash_map)


def run_annotation(
    images: list[Image.Image],
    model_names: list[str],
    phash_list: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> PHashAnnotationResults:
    """複数モデルでの一括アノテーション実行 (内部実装)。

    `image_annotator_lib.api.annotate()` の内部実装。利用者は `api.annotate()` を使うこと。

    Args:
        images: 評価対象の PIL Image オブジェクトのリスト。
        model_names: 使用するモデル名のリスト (指定順に実行)。
        phash_list: 各画像に対応する pHash のリスト (None の場合は計算)。
        api_keys: WebAPI モデル用の provider -> API key 辞書 (オプション)。

    Returns:
        pHash をキーとし、その値がモデル名をキーとする `UnifiedAnnotationResult` の辞書。
    """
    registry = get_cls_obj_registry()
    if not registry:
        logger.info("レジストリが空のため初期化を実行します...")
        initialize_registry()
        registry = get_cls_obj_registry()
        logger.info(f"レジストリ初期化完了。登録済みモデル数: {len(registry)}")

    logger.info(f"{len(images)} 枚の画像を {len(model_names)} 個のモデルで評価開始...")
    logger.debug(f"利用可能なモデル: {list(registry.keys())[:10]}...")

    phash_list_final, phash_map = _prepare_phash_map(images, phash_list)
    results_by_phash: PHashAnnotationResults = PHashAnnotationResults()

    for model_name in model_names:
        logger.debug(f"モデル '{model_name}' の評価を開始...")
        _execute_model_annotation(
            model_name, images, phash_list_final, phash_map, results_by_phash, api_keys
        )

    logger.info(f"全モデル ({len(model_names)}個) の評価完了。画像キー数: {len(results_by_phash)}")
    return results_by_phash
