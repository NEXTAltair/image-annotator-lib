"""Hugging Face Pipeline を使用するモデル用の基底クラス。"""

from typing import Any, cast

from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import ModelLoadError
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import TransformersPipelineComponents, UnifiedAnnotationResult
from ..utils import logger
from .annotator import BaseAnnotator


class PipelineBaseAnnotator(BaseAnnotator):
    """Hugging Face Pipeline を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # device 判定はローカル ML 系 base class の責務 (Issue #35 で BaseAnnotator から移譲)
        from ..utils import determine_effective_device

        self.device = determine_effective_device(self._config.device, self.model_name)
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

        # ロード前にキャッシュ状態を確認する (Issue #146)。
        # load_components() は「キャッシュ済み」と「ロード失敗」の両方で None を返すため、
        # 呼び出し前の状態で2ケースを区別する。
        had_cached_state = ModelLoad._get_model_state(self.model_name) is not None

        loaded_components = ModelLoad.load_transformers_pipeline_components(
            self.task,
            self.model_name,
            self.model_path,
            self.device,
            self.batch_size,
        )

        # None = 既キャッシュ済み (self.components は前回 __exit__ で設定済み)。
        # 空コンポーネント = ロード失敗。
        if loaded_components is not None:
            self.components = loaded_components
        elif not self.components:
            if had_cached_state:
                # モデル状態が「キャッシュ済み」(None 返却) だが、このインスタンスはコンポーネントを
                # 保持していない。api_keys={} 指定時の毎回インスタンス再生成などで発生 (Issue #146)。
                # 状態をリセットして強制再ロードする。
                logger.warning(
                    f"モデル '{self.model_name}' は状態「キャッシュ済み」だが、"
                    "このインスタンスはコンポーネントを保持していない。状態をリセットして再ロード。"
                )
                ModelLoad._release_model_state(self.model_name)
                loaded_components = ModelLoad.load_transformers_pipeline_components(
                    self.task, self.model_name, self.model_path, self.device, self.batch_size
                )
                if loaded_components is not None:
                    self.components = loaded_components
                else:
                    error_msg = f"Failed to load pipeline components for model '{self.model_name}'."
                    logger.error(error_msg, exc_info=True)
                    raise ModelLoadError(error_msg, model_path=self.model_path)
            else:
                error_msg = f"Failed to load pipeline components for model '{self.model_name}'."
                logger.error(error_msg, exc_info=True)
                raise ModelLoadError(error_msg, model_path=self.model_path)

        # Restoration Attempt (失敗しても継続可能)
        restored_components = ModelLoad.restore_model_to_cuda(
            self.model_name, cast(dict[str, Any], self.components), self.device
        )

        # Restoration Failure Handling (警告のみ、CPU で継続)
        if restored_components is not None:
            self.components = cast(TransformersPipelineComponents, restored_components)
        else:
            # restore_model_to_cuda() は既に CPU フォールバック済み（None 返却）
            # self.components は CPU 版を維持したまま継続
            logger.warning(
                f"Model '{self.model_name}' will run on CPU. "
                f"CUDA restoration failed but CPU fallback is already complete."
            )
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

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> list[UnifiedAnnotationResult]:
        """
        Pipeline の生出力は人間が読めるので不要
        """
        # Concrete subclasses override this method completely
        return []
