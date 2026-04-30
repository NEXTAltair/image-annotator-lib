"""基底アノテータークラス - すべてのアノテーターの共通基底クラス"""

import time
from abc import ABC, abstractmethod
from typing import Any, Self

import imagehash
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import OutOfMemoryError
from ..config import config_registry
from ..model_config import BaseModelConfig, ModelConfigFactory
from ..types import LoaderComponents, UnifiedAnnotationResult
from ..utils import logger


class BaseAnnotator(ABC):
    """すべてのアノテーターの基底クラス。

    このクラスは、画像アノテーションの共通インターフェースを定義し、
    各フレームワーク固有の実装に必要な抽象メソッドを提供します。
    """

    def __init__(self, model_name: str, config: BaseModelConfig | None = None):
        """BaseAnnotator を初期化します。

        Args:
            model_name: モデルの名前。設定ファイルでの識別子として使用されます。
            config: Config Object (Phase 1B DI)。Noneの場合、config_registryから読み込み。

        Note:
            Phase 1B: Dependency Injection導入
            - config引数経由でConfig Objectを注入可能
            - config=Noneの場合、後方互換のためconfig_registryから読み込み
        """
        self.model_name = model_name
        # Config Object注入 or 後方互換フォールバック
        self._config = config if config is not None else self._load_config_from_registry(model_name)

        # model_pathはLocalMLModelConfig専用(WebAPIModelConfigにはない)
        self.model_path = getattr(self._config, "model_path", None)
        self.device = self._validate_device(self._config.device)
        self.components: LoaderComponents | None = None

    def _validate_device(self, requested_device: str) -> str:
        """要求されたデバイスを検証し、CUDA利用不可の場合はCPUにフォールバック。

        このメソッドは、アノテーターのデバイス設定と実際のデバイス機能の
        一貫性を保証します。ModelLoad.Loaderと同じ検証ロジックを使用して、
        デバイスの不一致問題を防止します。

        Args:
            requested_device: 設定ファイルからのデバイス文字列 ("cuda", "cpu" など)

        Returns:
            検証されたデバイス文字列（CUDA利用不可の場合は "cpu"）

        Note:
            一貫した検証ロジックのため determine_effective_device() を使用。
            パフォーマンス影響: アノテーター初期化あたり <1ms。
        """
        from ..utils import determine_effective_device

        return determine_effective_device(requested_device, self.model_name)

    @abstractmethod
    def __enter__(self) -> Self:
        """コンテキストマネージャーの開始処理。モデルのロードを行います。"""
        raise NotImplementedError("サブクラスは __enter__ を実装する必要があります。")

    @abstractmethod
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャーの終了処理。リソースの解放を行います。"""
        raise NotImplementedError("サブクラスは __exit__ を実装する必要があります。")

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        """画像リストを前処理します。

        Args:
            images: 前処理する PIL Image のリスト。

        Returns:
            前処理済みのデータ。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _preprocess_images を実装する必要があります。")

    @abstractmethod
    def _run_inference(self, processed: Any) -> Any:
        """前処理済みデータで推論を実行します。

        Args:
            processed: _preprocess_images の出力。

        Returns:
            推論結果の生データ。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _run_inference を実装する必要があります。")

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> Any:
        """推論結果を整形します。

        Args:
            raw_outputs: _run_inference の出力。

        Returns:
            整形済みの予測結果。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _format_predictions を実装する必要があります。")

    def predict(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[UnifiedAnnotationResult]:
        """画像リストに対してアノテーションを実行します。

        Args:
            images: アノテーションする PIL Image のリスト。
            phash_list: 事前計算された知覚ハッシュのリスト（オプション）。
                api.pyでpHashマッピングに使用。結果オブジェクトには含まれない。

        Returns:
            各画像のUnifiedAnnotationResult結果のリスト。
        """
        if not images:
            logger.warning("空の画像リストが渡されました。アノテーションをスキップします。")
            return []
        try:
            formatted_outputs = self._execute_pipeline(images)
            return self._build_results(images, formatted_outputs)
        except OutOfMemoryError as mem_e:
            logger.error(f"メモリ不足エラー: {mem_e}")
            return self._create_error_results(images, "メモリ不足エラー")
        except Exception as e:
            logger.exception(f"予期せぬエラー: {e}")
            return self._create_error_results(images, f"予期せぬエラー: {e}")

    def _execute_pipeline(self, images: list[Image.Image]) -> list:
        """前処理→推論→整形のパイプラインを実行する。

        Args:
            images: アノテーション対象の画像リスト。

        Returns:
            整形済みの出力リスト（画像数と同じ長さ）。
        """
        start_time = time.time()
        processed = self._preprocess_images(images)
        logger.debug(f"前処理時間: {time.time() - start_time:.3f}秒")

        start_time = time.time()
        raw_outputs = self._run_inference(processed)
        logger.debug(f"推論時間: {time.time() - start_time:.3f}秒")

        start_time = time.time()
        formatted_outputs = self._format_predictions(raw_outputs)
        logger.debug(f"整形時間: {time.time() - start_time:.3f}秒")

        if not isinstance(formatted_outputs, list):
            formatted_outputs = [formatted_outputs] * len(images)
        return formatted_outputs

    def _build_results(
        self,
        images: list[Image.Image],
        formatted_outputs: list,
    ) -> list[UnifiedAnnotationResult]:
        """各画像の推論出力からアノテーション結果を構築する。

        全アノテーターは `_format_predictions` で `UnifiedAnnotationResult` を返すことが必須。
        旧形式 (dict / list[str] 等) を返した場合は TypeError を送出する。

        Args:
            images: 元画像リスト。
            formatted_outputs: 整形済み推論出力。

        Returns:
            UnifiedAnnotationResult結果のリスト。

        Raises:
            TypeError: `formatted_output` が UnifiedAnnotationResult でない場合。
        """
        results: list[UnifiedAnnotationResult] = []
        for i, (_image, formatted_output) in enumerate(zip(images, formatted_outputs, strict=True)):
            if not isinstance(formatted_output, UnifiedAnnotationResult):
                raise TypeError(
                    f"画像 {i}: _format_predictions は UnifiedAnnotationResult を返す必要があります "
                    f"(モデル: {self.model_name}, 取得型: {type(formatted_output).__name__})"
                )
            results.append(formatted_output)
        return results

    def _create_error_result(self, error_message: str) -> UnifiedAnnotationResult:
        """エラー発生時のアノテーション結果を生成する。

        Args:
            error_message: エラーメッセージ。

        Returns:
            エラー情報を含むUnifiedAnnotationResult。
        """
        from ..utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        return UnifiedAnnotationResult(
            model_name=self.model_name,
            capabilities=capabilities,
            error=error_message,
        )

    def _create_error_results(
        self,
        images: list[Image.Image],
        error_message: str,
    ) -> list[UnifiedAnnotationResult]:
        """全画像に対してエラー結果を一括生成する。

        Args:
            images: 元画像リスト。
            error_message: エラーメッセージ。

        Returns:
            エラー結果のリスト。
        """
        return [self._create_error_result(error_message) for _ in images]

    def _load_config_from_registry(self, model_name: str) -> BaseModelConfig:
        """config_registryからConfig Objectを生成します (後方互換性用)。

        Args:
            model_name: モデル名

        Returns:
            BaseModelConfig: 生成されたConfig Object

        Raises:
            ConfigurationError: 設定の読み込みまたは変換に失敗した場合

        Note:
            Phase 1B: config=None時の後方互換フォールバック
            既存コード (config引数なし) が引き続き動作するよう、
            config_registryから設定を読み込んでConfig Objectに変換します。
        """
        logger.debug(f"後方互換: config_registryから '{model_name}' の設定を読み込み")
        registry_dict = config_registry.get_all_config().get(model_name)
        if not registry_dict:
            # model_nameが存在しない場合のエラー処理
            logger.error(f"モデル '{model_name}' の設定がconfig_registryに存在しません")
            raise ValueError(f"Model '{model_name}' not found in config_registry")
        return ModelConfigFactory.from_registry(model_name, registry_dict)

    def _calculate_phash(self, image: Image.Image) -> str | None:
        """画像の知覚ハッシュを計算します。

        Args:
            image: ハッシュを計算する PIL Image。

        Returns:
            知覚ハッシュの文字列表現。計算に失敗した場合は None。
        """
        try:
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            logger.warning(f"知覚ハッシュの計算に失敗: {e}")
            return None
