"""ライブラリの外部 API 関数 (既存テスト互換性重視)。

主に Gradio インターフェースや既存の統合テストからの利用を想定しています。
"""

from typing import Any

from PIL import Image

from .core.base.annotator import BaseAnnotator
from .core.provider_manager import ProviderManager
from .core.registry import find_model_class_case_insensitive, get_cls_obj_registry
from .core.simplified_agent_factory import get_agent_factory
from .core.types import UnifiedAnnotationResult
from .core.utils import calculate_phash, logger

_MODEL_INSTANCE_REGISTRY: dict[str, Any] = {}


class PHashAnnotationResults(dict[str, dict[str, UnifiedAnnotationResult]]):
    """統一バリデーションスキーマ用の画像pHashをキーとする評価結果辞書。

    Attributes:
        [phash]: 画像のpHashをキーとする辞書。
                 各キーの値は、モデル名をキーとする辞書。
                 各モデル名の値は、型安全なUnifiedAnnotationResult。
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
def _create_annotator_instance(model_name: str, api_keys: dict[str, str] | None = None) -> BaseAnnotator:
    """
    モデル名に対応するクラスを取得し、インスタンスを生成します。
    PydanticAI WebAPIモデルの場合は簡素化されたAgent factoryを使用します。

    Args:
        model_name (str): モデルの名前または model_id。
        api_keys: WebAPIモデル用のAPIキー辞書 (オプション)

    Returns:
        BaseAnnotator: アノテーターのインスタンス。

    Raises:
        KeyError: 指定された model_name がレジストリに存在しない場合。
    """
    logger.debug(f"Creating annotator instance for model: '{model_name}'")

    # Check if it's a direct model_id (e.g., "google/gemini-2.5-pro-preview-03-25")
    agent_factory = get_agent_factory()
    if agent_factory.is_model_available(model_name):
        logger.debug(f"Using simplified Agent factory for model: {model_name}")
        # Create a simplified wrapper for PydanticAI agents
        from .core.simplified_agent_wrapper import SimplifiedAgentWrapper

        return SimplifiedAgentWrapper(model_name, api_keys=api_keys)

    # Fallback to traditional registry-based approach
    model_result = find_model_class_case_insensitive(model_name)
    if model_result is None:
        registry = get_cls_obj_registry()
        available_models = list(registry.keys())
        available_direct_models = agent_factory.get_available_models()

        error_details = {
            "requested_model": model_name,
            "registry_models_count": len(available_models),
            "direct_models_count": len(available_direct_models),
            "registry_sample": available_models[:5],
            "direct_models_sample": available_direct_models[:5],
        }
        logger.error(f"Model resolution failed: {error_details}")
        logger.error(f"要求されたモデル名 '{model_name}' が見つかりません。")
        logger.error(f"レジストリモデル例: {available_models[:3]}")
        logger.error(f"直接利用可能モデル例: {available_direct_models[:3]}")
        raise KeyError(f"Model '{model_name}' not found in registry or available models.")

    actual_model_name, Annotator_class = model_result

    # 実際のモデル名を使用(大文字・小文字が正規化されている場合)
    effective_model_name = actual_model_name

    # PydanticAI WebAPIアノテーターの場合はProvider-level管理を使用
    if _is_pydantic_ai_webapi_annotator(Annotator_class):
        # Provider-levelラッパーを返す(正規化されたモデル名を使用)
        return PydanticAIWebAPIWrapper(effective_model_name, Annotator_class, api_keys=api_keys)
    else:
        # 従来通りのインスタンス作成(正規化されたモデル名を使用)
        # TODO: レガシーアノテーターでもAPIキー対応が必要な場合は、ここでapi_keysを渡す
        instance = Annotator_class(model_name=effective_model_name)
        logger.debug(
            f"モデル '{model_name}' -> '{effective_model_name}' の新しいインスタンスを作成しました (クラス: {Annotator_class.__name__})"
        )
        return instance


def _is_pydantic_ai_webapi_annotator(annotator_class) -> bool:
    """PydanticAI WebAPIアノテーターかどうかを判定"""
    # PydanticAIWebAPIAnnotatorクラス名で判定
    if annotator_class.__name__ == "PydanticAIWebAPIAnnotator":
        return True

    # 従来のWebAPIクラス名で判定（レジストリで統一実装に置換されるため実質的にPydanticAI）
    webapi_class_names = {
        "AnthropicApiAnnotator",
        "GoogleApiAnnotator",
        "OpenAIApiAnnotator",
        "OpenRouterApiAnnotator",
    }

    return annotator_class.__name__ in webapi_class_names


class PydanticAIWebAPIWrapper(BaseAnnotator):
    """PydanticAI WebAPIアノテーター用のProvider-levelラッパー"""

    def __init__(self, model_name: str, annotator_class, api_keys: dict[str, str] | None = None):
        super().__init__(model_name)
        self.annotator_class = annotator_class
        self.api_keys = api_keys
        self._api_model_id = None

    def __enter__(self):
        # Configuration読み込みでapi_model_idを取得
        from .core.config import config_registry

        self._api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Provider-levelで管理されるため何もしない
        pass

    def predict(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[UnifiedAnnotationResult]:
        """Provider-level実行でのpredict実装"""
        if not self._api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        # phash_listが提供されていない場合は計算
        if phash_list is None:
            import imagehash

            phash_list = [str(imagehash.phash(img)) for img in images]

        try:
            # Provider Managerを通して実行（辞書形式の戻り値）
            results_by_phash = ProviderManager.run_inference_with_model(
                model_name=self.model_name,
                images_list=images,
                api_model_id=self._api_model_id,
                api_keys=self.api_keys,
            )
        except Exception as e:
            # ProviderManagerからの例外をキャッチし、全画像にエラー結果を返す
            logger.error(f"ProviderManagerでの推論中にエラーが発生: {e}", exc_info=True)
            error_message = f"Failed to run inference: {e}"
            from .core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name, capabilities=capabilities, error=error_message
                )
                for i in range(len(images))
            ]

        # 辞書形式の結果をリスト形式に変換
        results = []
        for i, _image in enumerate(images):
            phash = phash_list[i] if i < len(phash_list) else None

            # phashに対応する結果を検索
            annotation_result = None
            for result_phash, result in results_by_phash.items():
                if result_phash == phash:
                    annotation_result = result
                    break

            if annotation_result:
                # ProviderManagerから返された結果をそのまま使用（新スキーマ）
                results.append(annotation_result)
            else:
                # 対応する結果が見つからない場合
                logger.warning(f"画像 {i} (phash: {phash}) の結果が見つかりません")
                from .core.utils import get_model_capabilities

                capabilities = get_model_capabilities(self.model_name)
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error="No result found for image",
                    )
                )

        return results

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[dict[str, Any]]:
        """Provider Managerを通して推論実行"""
        if not self._api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        return ProviderManager.run_inference_with_model(
            model_name=self.model_name,
            images_list=processed,
            api_model_id=self._api_model_id,
            api_keys=self.api_keys,
        )

    def _format_predictions(self, raw_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-levelでは整形済みのため変更不要"""
        return raw_outputs

    def _generate_tags(self, formatted_output: list[dict[str, Any]]) -> list[str]:
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


def get_annotator_instance(model_name: str, api_keys: dict[str, str] | None = None) -> Any:
    """モデル名からスコアラーインスタンスを取得する

    モデルがすでにロードされている場合はキャッシュから返す。
    まだロードされていない場合は、新たにインスタンスを作成してキャッシュに保存する。

    Args:
        model_name: モデルの名前(models.tomlで定義されたキー)
        api_keys: WebAPIモデル用のAPIキー辞書 (オプション)

    Returns:
        スコアラーインスタンス

    Raises:
        ValueError: 指定されたモデル名が設定に存在しない場合
    """
    # APIキー指定時はキャッシュを使用しない (設定が動的に変わるため)
    if api_keys is not None:
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
    """1モデル分のアノテーション処理を実施します。
    ･モデルのロード / 復元、予測、キャッシュ &リリースを実行

    Args:
        Annotator: アノテータインスタンス
        images: 処理対象の画像リスト
        phash_list: 各画像に対応するpHashのリスト

    Returns:
        list[UnifiedAnnotationResult]: 統一バリデーションスキーマでの結果リスト
    """
    with Annotator:
        results: list[UnifiedAnnotationResult] = Annotator.predict(images, phash_list)
    return results


def _process_model_results(
    model_name: str,
    annotation_results: list[UnifiedAnnotationResult],
    results_by_phash: PHashAnnotationResults,
) -> None:
    """モデルの結果を pHash ベースの構造に変換します（統一バリデーションスキーマ対応）。

    Args:
        model_name: モデルの名前
        annotation_results: 統一スキーマでのモデル予測結果リスト
        results_by_phash: pHash をキーとする結果辞書(更新対象)
    """
    for i, result in enumerate(annotation_results):
        # pHashは結果に含まれていないため、インデックスベースで生成
        phash_key = f"image_{i}"  # 簡素化したキー（実際のpHashは呼び出し元で管理）

        # 結果辞書の初期化
        if phash_key not in results_by_phash:
            results_by_phash[phash_key] = {}

        # 統一スキーマの結果をそのまま格納
        results_by_phash[phash_key][model_name] = result

        logger.debug(f"モデル '{model_name}' の結果を pHash '{phash_key}' に格納しました")


def _handle_error(
    e: Exception,
    model_name: str,
    image_hash: str,
    results_dict: PHashAnnotationResults,
    idx: int,
    total_models: int,
) -> None:
    """エラーを処理し、結果辞書に記録する（統一バリデーションスキーマ対応）。"""

    error_type_name = type(e).__name__  # 例外のクラス名を取得
    error_message = f"{error_type_name}: {e!s} (モデル: {model_name})"  # メッセージに含める
    logger.error(f"モデル '{model_name}' (画像 {idx + 1}/{total_models}) でエラーが発生しました: {e!s}")

    if image_hash not in results_dict:
        results_dict[image_hash] = {}

    # エラー結果を統一スキーマで作成
    from .core.utils import get_model_capabilities

    capabilities = get_model_capabilities(model_name)
    results_dict[image_hash][model_name] = UnifiedAnnotationResult(
        model_name=model_name, capabilities=capabilities, error=error_message
    )


def list_available_annotators() -> list[str]:
    """利用可能なアノテーターモデル名のリストを返す

    Returns:
        利用可能なモデル名のリスト
    """
    from .core.registry import list_available_annotators as _list_available_annotators

    return _list_available_annotators()


def annotate(
    images_list: list[Image.Image],
    model_name_list: list[str],
    phash_list: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> PHashAnnotationResults:
    """複数の画像を指定された複数のモデルで評価(アノテーション)します。

    各画像のpHashをキーとして、モデルごとの評価結果を整理して返します。
    これにより、各画像に対する複数モデルの結果を簡単に比較できます。

    Args:
        images_list: 評価対象の PIL Image オブジェクトのリスト。
        model_name_list: 使用するモデル名のリスト。
        phash_list: 各画像に対応するpHashのリスト。
        api_keys: WebAPIモデル用のAPIキー辞書 (オプション)。
                 例: {"openai": "sk-...", "anthropic": "sk-ant-..."}
                 指定された場合、設定ファイルより優先されます。

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
    # レジストリが初期化されていることを確認
    from .core.registry import get_cls_obj_registry, initialize_registry

    registry = get_cls_obj_registry()
    if not registry:
        logger.info("レジストリが空のため初期化を実行します...")
        initialize_registry()
        registry = get_cls_obj_registry()
        logger.info(f"レジストリ初期化完了。登録済みモデル数: {len(registry)}")

    logger.info(f"{len(images_list)} 枚の画像を {len(model_name_list)} 個のモデルで評価開始...")
    logger.debug(f"利用可能なモデル: {list(registry.keys())[:10]}...")

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
            annotator = get_annotator_instance(model_name, api_keys=api_keys)
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
                from .core.utils import get_model_capabilities

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
            # エラーハンドリング: このモデルでの処理は失敗とみなし、全画像にエラーを記録
            {
                "model_name": model_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "images_count": len(images_list),
                "phase": "model_processing",
            }
            logger.error(
                f"Model processing fatal error: model={model_name}, error_type={type(e).__name__}, message={e!s}, images_count={len(images_list)}",
                exc_info=True,
            )
            logger.error(f"モデル '{model_name}' の処理中に致命的なエラー: {e}", exc_info=True)
            error_message = f"{type(e).__name__}: {e}"
            # predictが例外を投げた場合に備え、全画像にエラー結果を作成
            from .core.utils import get_model_capabilities

            capabilities = get_model_capabilities(model_name)
            if not capabilities:
                # エラー結果用のデフォルトcapability（タグ生成を想定）
                from .core.types import TaskCapability

                capabilities = {TaskCapability.TAGS}
                logger.warning(
                    f"モデル '{model_name}' のcapabilitiesが未設定のため、デフォルト値 {{TAGS}} を使用"
                )
            error_results = [
                UnifiedAnnotationResult(
                    model_name=model_name, capabilities=capabilities, error=error_message
                )
                for _ in phash_list
            ]
            _process_model_results(model_name, error_results, results_by_phash)

    logger.info(f"全モデル ({len(model_name_list)}個) の評価完了。画像キー数: {len(results_by_phash)}")
    return results_by_phash
