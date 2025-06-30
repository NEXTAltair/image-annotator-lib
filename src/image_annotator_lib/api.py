"""ライブラリの外部 API 関数 (既存テスト互換性重視)。

主に Gradio インターフェースや既存の統合テストからの利用を想定しています。
"""

from typing import Any, TypedDict

from PIL import Image

from .core.base.annotator import BaseAnnotator
from .core.provider_manager import ProviderManager
from .core.registry import get_cls_obj_registry
from .core.types import AnnotationResult
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
    PydanticAI WebAPIモデルの場合はProvider-level管理を使用します。

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

    # PydanticAI WebAPIアノテーターの場合はProvider-level管理を使用
    if _is_pydantic_ai_webapi_annotator(Annotator_class):
        # Provider-levelラッパーを返す
        return PydanticAIWebAPIWrapper(model_name, Annotator_class)
    else:
        # 従来通りのインスタンス作成
        instance = Annotator_class(model_name=model_name)
        logger.debug(
            f"モデル '{model_name}' の新しいインスタンスを作成しました (クラス: {Annotator_class.__name__})"
        )
        return instance


def _is_pydantic_ai_webapi_annotator(annotator_class) -> bool:
    """PydanticAI WebAPIアノテーターかどうかを判定"""
    # PydanticAIAnnotatorMixinを継承しているかで判定
    from .core.pydantic_ai_factory import PydanticAIAnnotatorMixin

    return issubclass(annotator_class, PydanticAIAnnotatorMixin)


class PydanticAIWebAPIWrapper(BaseAnnotator):
    """PydanticAI WebAPIアノテーター用のProvider-levelラッパー"""

    def __init__(self, model_name: str, annotator_class):
        super().__init__(model_name)
        self.annotator_class = annotator_class
        self._api_model_id = None

    def __enter__(self):
        # Configuration読み込みでapi_model_idを取得
        from .core.config import config_registry

        self._api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Provider-levelで管理されるため何もしない
        pass

    def predict(self, images: list[Image.Image], phash_list: list[str]) -> list[AnnotationResult]:
        """Provider-level実行でのpredict実装"""
        if not self._api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        try:
            # Provider Managerを通して実行
            raw_outputs = ProviderManager.run_inference_with_model(
                model_name=self.model_name, images_list=images, api_model_id=self._api_model_id
            )
        except Exception as e:
            # ProviderManagerからの例外をキャッチし、全画像にエラー結果を返す
            logger.error(f"ProviderManagerでの推論中にエラーが発生: {e}", exc_info=True)
            error_message = f"Failed to run inference: {e}"
            return [
                AnnotationResult(
                    phash=phash_list[i] if i < len(phash_list) else None,
                    tags=[],
                    formatted_output=None,
                    error=error_message,
                )
                for i in range(len(images))
            ]

        # 結果をAnnotationResult形式に変換
        results = []
        for i, raw_output in enumerate(raw_outputs):
            phash = phash_list[i] if i < len(phash_list) else None

            # raw_outputが文字列の場合(エラーメッセージ)、辞書に変換
            if isinstance(raw_output, str):
                raw_output = {"error": raw_output}

            if raw_output.get("error"):
                result = AnnotationResult(
                    phash=phash, tags=[], formatted_output=None, error=raw_output["error"]
                )
            else:
                annotation = raw_output.get("response")
                if annotation:
                    if hasattr(annotation, "tags"):
                        tags = annotation.tags
                    elif isinstance(annotation, dict) and "tags" in annotation:
                        tags = annotation["tags"]
                    else:
                        tags = []

                    result = AnnotationResult(
                        phash=phash, tags=tags, formatted_output=annotation, error=None
                    )
                else:
                    result = AnnotationResult(
                        phash=phash, tags=[], formatted_output=None, error="No response from model"
                    )

            results.append(result)

        return results

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[dict]:
        """Provider Managerを通して推論実行"""
        if not self._api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        return ProviderManager.run_inference_with_model(
            model_name=self.model_name, images_list=processed, api_model_id=self._api_model_id
        )

    def _format_predictions(self, raw_outputs: list[dict]) -> list[dict]:
        """Provider-levelでは整形済みのため変更不要"""
        return raw_outputs

    def _generate_tags(self, formatted_output: list[dict]) -> list[str]:
        """整形済み出力からタグリストを生成"""
        all_tags = []
        for output in formatted_output:
            if output.get("error"):
                continue
            annotation = output.get("response")
            if annotation:
                if hasattr(annotation, "tags"):
                    all_tags.extend(annotation.tags)
                elif isinstance(annotation, dict) and "tags" in annotation:
                    all_tags.extend(annotation["tags"])
        return all_tags


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
    error_type_name = type(e).__name__  # 例外のクラス名を取得
    error_message = f"{error_type_name}: {e!s} (モデル: {model_name})"  # メッセージに含める
    logger.error(f"モデル '{model_name}' (画像 {idx + 1}/{total_models}) でエラーが発生しました: {e!s}")
    if image_hash not in results_dict:
        results_dict[image_hash] = {}
    results_dict[image_hash][model_name] = {
        "error": error_message,
        "formatted_output": None,
        "tags": None,  # エラー時はNoneまたは空リストを返す
        # 必要に応じて他のフィールドもエラー時のデフォルト値を設定
    }


def list_available_annotators() -> list[str]:
    """利用可能なアノテーターモデル名のリストを返す

    Returns:
        利用可能なモデル名のリスト
    """
    from .core.registry import list_available_annotators as _list_available_annotators

    return _list_available_annotators()


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
            # エラーハンドリング: このモデルでの処理は失敗とみなし、全画像にエラーを記録
            logger.error(f"モデル '{model_name}' の処理中に致命的なエラー: {e}", exc_info=True)
            error_message = f"{type(e).__name__}: {e}"
            # predictが例外を投げた場合に備え、全画像にエラー結果を作成
            error_results = [
                AnnotationResult(phash=phash, tags=[], formatted_output=None, error=error_message)
                for phash in phash_list
            ]
            _process_model_results(model_name, error_results, results_by_phash)

    logger.info(f"全モデル ({len(model_name_list)}個) の評価完了。画像キー数: {len(results_by_phash)}")
    return results_by_phash
