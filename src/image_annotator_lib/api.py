"""ライブラリの外部 API 関数 (既存テスト互換性重視)。

主に Gradio インターフェースや既存の統合テストからの利用を想定しています。
"""

from typing import Any, TypedDict

from PIL import Image

from .core.base import AnnotationResult, BaseAnnotator
from .core.registry import get_cls_obj_registry
from .core.utils import calculate_phash, logger

_MODEL_INSTANCE_REGISTRY: dict[str, Any] = {}


class ModelResultDict(TypedDict, total=False):
    """モデルの評価結果を表す型定義。

    Attributes:
        tags: アノテーション結果の主要な文字列リスト。
        formatted_output: 整形済み出力。
        error: 処理中に発生したエラーメッセージ。エラーがない場合は None。
    """

    tags: list[str] | None
    formatted_output: Any | None
    error: str | None


class PHashAnnotationResults(dict[str, dict[str, ModelResultDict]]):
    """画像のpHashをキーとする評価結果辞書。

    Attributes:
        [phash]: 画像のpHashをキーとする辞書。
                 各キーの値は、モデル名をキーとする辞書。
                 各モデル名の値は、そのモデルの評価結果。
    """

    pass


# REFACTOR: インスタンス管理の改善
# - 現状: _MODEL_INSTANCE_REGISTRYがAPIレイヤーに配置
# - 課題:
#   1. レジストリ機能との分離が不自然
#   2. インスタンス管理の責務がAPIレイヤーにある
# - 当面の方針:
#   - 既存の互換性と安定性を優先
#   - 大規模な改修は次期メジャーバージョンで検討
def _create_annotator_instance(model_name: str) -> BaseAnnotator:
    """
    モデル名に対応するクラスを取得し、インスタンスを生成します。
    Web API モデルもローカルモデルも、model_name（論理名）をそのまま渡して初期化します。

    Args:
        model_name (str): モデルの名前 (Web APIの場合は model_name_short)。

    Returns:
        BaseAnnotator: アノテーターのインスタンス。

    Raises:
        KeyError: 指定された model_name がレジストリに存在しない場合。
    """
    registry = get_cls_obj_registry()
    if model_name not in registry:
        logger.error(f"要求されたモデル名 '{model_name}' はクラスレジストリに見つかりません。")
        raise KeyError(f"Model '{model_name}' not found in class registry.")

    Annotator_class = registry[model_name]
    instance = Annotator_class(model_name=model_name)
    logger.debug(
        f"モデル '{model_name}' の新しいインスタンスを作成しました (クラス: {Annotator_class.__name__})"
    )
    return instance


def get_annotator_instance(model_name: str) -> Any:
    """モデル名からスコアラーインスタンスを取得する

    モデルがすでにロードされている場合はキャッシュから返す。
    まだロードされていない場合は、新たにインスタンスを作成してキャッシュに保存する。

    Args:
        model_name: モデルの名前(models.tomlで定義されたキー)

    Returns:
        スコアラーインスタンス

    Raises:
        ValueError: 指定されたモデル名が設定に存在しない場合
    """
    if model_name in _MODEL_INSTANCE_REGISTRY:
        logger.debug(f"モデル '{model_name}' はキャッシュから取得されました")
        return _MODEL_INSTANCE_REGISTRY[model_name]

    instance = _create_annotator_instance(model_name)
    _MODEL_INSTANCE_REGISTRY[model_name] = instance
    return instance


def _annotate_model(
    Annotator: BaseAnnotator, images: list[Image.Image], phash_list: list[str]
) -> list[AnnotationResult]:
    """1モデル分のアノテーション処理を実施します。
    ･モデルのロード / 復元、予測、キャッシュ &リリースを実行

    Args:
        Annotator: アノテータインスタンス
        images: 処理対象の画像リスト
        phash_list: 各画像に対応するpHashのリスト

    Returns:
        list[AnnotationResult]: アノテーション結果のリスト
    """
    with Annotator:
        results: list[AnnotationResult] = Annotator.predict(images, phash_list)
    return results


def _process_model_results(
    model_name: str,
    annotation_results: list[AnnotationResult],
    results_by_phash: PHashAnnotationResults,
) -> None:
    """モデルの結果を pHash ベースの構造に変換します。

    Args:
        model_name: モデルの名前
        annotation_results: モデルの予測結果リスト
        results_by_phash: pHash をキーとする結果辞書(更新対象)
    """
    for result in annotation_results:
        # pHash が None の場合は代替キーを使用
        phash_key = result.get("phash") or f"unknown_image_{len(results_by_phash)}"

        # 結果辞書の初期化
        if phash_key not in results_by_phash:
            results_by_phash[phash_key] = {}

        # モデルごとの結果を格納
        results_by_phash[phash_key][model_name] = {
            "tags": result.get("tags", []),
            "formatted_output": result.get("formatted_output"),
            "error": result.get("error"),
        }

        logger.debug(f"モデル '{model_name}' の結果を pHash '{phash_key}' に格納しました")


def _handle_error(
    e: Exception,
    model_name: str,
    image_hash: str,
    results_dict: dict[str, dict[str, Any]],
    idx: int,
    total_models: int,
) -> None:
    """エラーを処理し、結果辞書に記録する。"""
    error_type_name = type(e).__name__ # 例外のクラス名を取得
    error_message = f"{error_type_name}: {e!s} (モデル: {model_name})" # メッセージに含める
    logger.error(
        f"モデル '{model_name}' (画像 {idx + 1}/{total_models}) でエラーが発生しました: {e!s}"
    )
    if image_hash not in results_dict:
        results_dict[image_hash] = {}
    results_dict[image_hash][model_name] = {
        "error": error_message,
        "formatted_output": None,
        "tags": None, # エラー時はNoneまたは空リストを返す
        # 必要に応じて他のフィールドもエラー時のデフォルト値を設定
    }


def annotate(
    images_list: list[Image.Image], model_name_list: list[str], phash_list: list[str] | None = None
) -> PHashAnnotationResults:
    """複数の画像を指定された複数のモデルで評価(アノテーション)します。

    各画像のpHashをキーとして、モデルごとの評価結果を整理して返します。
    これにより、各画像に対する複数モデルの結果を簡単に比較できます。

    Args:
        images_list: 評価対象の PIL Image オブジェクトのリスト。
        model_name_list: 使用するモデル名のリスト。
        phash_list: 各画像に対応するpHashのリスト。

    Returns:
        結果を格納した辞書。最上位のキーは画像のpHash (または代替キー 'unknown_image_{index}')、
        次のレベルのキーはモデル名。値は {"tags": [...], "formatted_output": ..., "error": ...} 形式の辞書。
        例: {
            "phash1": {
                "model1": {"tags": ["tag1"], "formatted_output": ..., "error": None},
                "model2": {"tags": ["score_tag"], "formatted_output": ..., "error": None}
            }
        }
        エラーが発生した場合、対応するモデル名のエントリにはエラー情報が含まれます。
    """
    logger.info(f"{len(images_list)} 枚の画像を {len(model_name_list)} 個のモデルで評価開始...")

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
                # phash_list が画像リストより長い場合は、警告を出してループを抜けるなどの処理も検討可能
                logger.warning(
                    f"pHashリストの要素数が画像リストの要素数を超えています。インデックス {i} 以降のpHashは無視されます。"
                )
                break

    # 結果格納用 (pHash ベース)
    # {phash: {model_name: {"tags": [...], "formatted_output": ..., "error": ...}}}
    results_by_phash: PHashAnnotationResults = PHashAnnotationResults()

    # 各モデルで評価
    for model_name in model_name_list:
        logger.debug(f"モデル '{model_name}' の評価を開始...")

        try:
            annotator = get_annotator_instance(model_name)
            annotation_results = _annotate_model(annotator, images_list, phash_list)
            logger.debug(f"モデル '{model_name}' の評価完了。結果件数: {len(annotation_results)}")

            # 結果を pHash ベースの構造に処理
            _process_model_results(model_name, annotation_results, results_by_phash)

            # 結果リストの長さが画像数と一致しない場合のエラーハンドリング
            # (predict が必ず画像数と同じ長さのリストを返す想定だが念のため)
            if len(annotation_results) != len(images_list):
                logger.error(
                    f"モデル '{model_name}' の結果リスト長 ({len(annotation_results)}) が画像数 ({len(images_list)}) と一致しません。"
                )
                error_entry: ModelResultDict = {
                    "tags": None,
                    "formatted_output": None,
                    "error": "処理結果が不足しています",
                }
                for i in range(len(annotation_results), len(images_list)):
                    phash_key = phash_map.get(i) or f"unknown_image_{i}"
                    if phash_key not in results_by_phash:
                        results_by_phash[phash_key] = {}
                    results_by_phash[phash_key][model_name] = error_entry.copy()

        except Exception as e:
            # エラーハンドリング (エラー結果を results_by_phash に設定)
            _handle_error(e, model_name, "", results_by_phash, 0, len(model_name_list))

    logger.info(f"全モデル ({len(model_name_list)}個) の評価完了。画像キー数: {len(results_by_phash)}")
    return results_by_phash
