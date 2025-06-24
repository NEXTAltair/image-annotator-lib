"""Hugging Face Pipeline を使用するモデル用の基底クラス。"""

from typing import Any, cast

from PIL import Image

# --- ローカルインポート ---
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import TransformersPipelineComponents
from ..utils import logger
from .annotator import BaseAnnotator


class PipelineBaseAnnotator(BaseAnnotator):
    """Hugging Face Pipeline を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.batch_size = config_registry.get(self.model_name, "batch_size", 8)
        self.task = config_registry.get(self.model_name, "task", "image-classification")
        # components の型ヒントを具体的に指定
        self.components: TransformersPipelineComponents | None = None

    def __enter__(self) -> "PipelineBaseAnnotator":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        if self.task is None:
            raise ValueError(f"モデル '{self.model_name}' の task が設定されていません。")
        if self.model_path is None:
            raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")
        if self.batch_size is None:
            raise ValueError(f"モデル '{self.model_name}' の batch_size が設定されていません。")

        loaded_components = ModelLoad.load_transformers_pipeline_components(
            self.task,
            self.model_name,
            self.model_path,
            self.device,
            self.batch_size,
        )
        if loaded_components:
            self.components = loaded_components

        # 型の問題を回避するため、一時的にcast使用
        restored_components = ModelLoad.restore_model_to_cuda(
            self.model_name, self.device, cast(dict[str, Any], self.components)
        )
        self.components = cast(TransformersPipelineComponents, restored_components)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Pipeline モデルをキャッシュします。"""
        logger.debug(f"Exiting context for Pipeline model '{self.model_name}' (exception: {exc_type})")
        cached_components = ModelLoad.cache_to_main_memory(
            self.model_name, cast(dict[str, Any], self.components)
        )
        self.components = cast(TransformersPipelineComponents, cached_components)

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Pipeline は PIL Image を直接受け付けるため、前処理は不要。"""
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[list[dict[str, Any]]]:
        """Pipeline を使用して推論を実行します。"""
        try:
            if not self.components or "pipeline" not in self.components:
                raise RuntimeError("Pipeline がロードされていません。")
            raw_outputs = self.components["pipeline"](processed)
            return raw_outputs
        except Exception as e:
            logger.exception(f"Pipeline 推論中にエラーが発生: {e}")
            raise

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> Any:
        """
        Pipeline の生出力は人間が読めるので不要
        """
        return raw_outputs
