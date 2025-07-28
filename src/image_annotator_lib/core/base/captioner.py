"""キャプション生成モデル用の基底クラス。"""

from abc import abstractmethod
from typing import Any, Self, cast

from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import OutOfMemoryError
from ..model_factory import ModelLoad
from ..types import (
    CaptionerAnnotationResult,
    TaskCapability,
    TransformersComponents,
    UnifiedAnnotationResult,
)
from ..utils import logger
from .annotator import BaseAnnotator


class CaptionerBaseAnnotator(BaseAnnotator):
    """キャプション生成モデル用の基底クラス（BLIP、GIT等）"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # キャプション生成固有の設定
        self.max_length: int = 50  # 生成キャプションの最大長
        self.num_beams: int = 5  # ビームサーチのビーム数
        # components の型ヒントを具体的に指定
        self.components: TransformersComponents | None = None

    def __enter__(self) -> Self:
        """
        Transformers キャプション生成モデルコンポーネントをロードします。
        """
        try:
            if self.model_path is None:
                raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")
            logger.info(f"Loading caption generation model: '{self.model_path}'")
            self.components = ModelLoad.load_transformers_components(
                self.model_name, self.model_path, self.device
            )
            self._configure_generation_params()

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"キャプション生成モデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """キャプション生成モデルのリソースを解放します。"""
        logger.debug(f"Exiting context for caption model '{self.model_name}' (exception: {exc_type})")
        if self.components:
            released_components = ModelLoad.release_model_components(
                self.model_name, cast(dict[str, Any], self.components)
            )
            self.components = cast(TransformersComponents, released_components)
        if exc_type:
            logger.error(
                f"キャプション生成モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}"
            )

    def _configure_generation_params(self) -> None:
        """設定ファイルから生成パラメータを読み込み"""
        from ..config import config_registry

        self.max_length = config_registry.get(self.model_name, "max_length", 50)
        self.num_beams = config_registry.get(self.model_name, "num_beams", 5)

        logger.debug(
            f"キャプション生成パラメータ - max_length: {self.max_length}, num_beams: {self.num_beams}"
        )

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        """画像をキャプション生成モデル用に前処理（サブクラスで実装）"""
        raise NotImplementedError("キャプションサブクラスは _preprocess_images を実装する必要があります。")

    @abstractmethod
    def _run_inference(self, processed: Any) -> Any:
        """キャプション生成モデルで推論実行（サブクラスで実装）"""
        raise NotImplementedError("キャプションサブクラスは _run_inference を実装する必要があります。")

    def _format_predictions(self, raw_outputs: Any) -> list[UnifiedAnnotationResult]:
        """キャプション生成結果を統一UnifiedAnnotationResultにフォーマット"""
        from ..utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        results = []

        try:
            # raw_outputsの形式は実装によって異なるため、_extract_captions で統一
            captions = self._extract_captions(raw_outputs)

            for caption_list in captions:
                if isinstance(caption_list, str):
                    caption_list = [caption_list]

                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        tags=None,  # キャプション生成はタグなし
                        captions=caption_list if TaskCapability.CAPTIONS in capabilities else None,
                        scores=None,  # キャプション生成は直接的なスコアなし
                        framework="transformers",
                        raw_output={
                            "generation_params": {
                                "max_length": self.max_length,
                                "num_beams": self.num_beams,
                            },
                            "base_model": self.model_path or "unknown",
                            "raw_output": raw_outputs,
                        },
                    )
                )

        except Exception as e:
            logger.exception(f"キャプション結果のフォーマット中にエラー: {e}")
            # エラーの場合も統一したスキーマで返す
            results.append(
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"キャプション生成エラー: {e}",
                    framework="transformers",
                )
            )

        return results

    @abstractmethod
    def _extract_captions(self, raw_outputs: Any) -> list[list[str]]:
        """生出力からキャプションリストを抽出（サブクラスで実装）"""
        raise NotImplementedError("キャプションサブクラスは _extract_captions を実装する必要があります。")

    def _generate_tags(self, formatted_output: CaptionerAnnotationResult) -> list[str]:
        """新バリデーションスキーマからタグを生成（キャプションはタグではなくキャプションがメイン）"""
        if isinstance(formatted_output, CaptionerAnnotationResult):
            # キャプション生成の場合、主要データはキャプションなのでタグ生成は基本的に空
            # ただし、キャプションをタグとして使用したい場合はサブクラスでオーバーライド可能
            return []
        return []
