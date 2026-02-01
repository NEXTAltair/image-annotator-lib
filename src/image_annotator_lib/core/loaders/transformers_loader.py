"""Hugging Face Transformers モデルローダー。

AutoModelForVision2Seq と Pipeline の2つのローダーを提供する。

Dependencies:
    - transformers: Hugging Face Transformers ライブラリ (遅延import)
    - torch: PyTorch (遅延import)
"""

from __future__ import annotations

import gc
from typing import Any, cast, override

from ..types import TransformersComponents, TransformersPipelineComponents
from ..utils import logger
from .loader_base import LoaderBase

if __name__ != "__main__":
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from transformers.pipelines.base import Pipeline


class TransformersLoader(LoaderBase):
    """Hugging Face Transformers モデル (AutoModelForVision2Seq) のローダー。"""

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """CPU 上でモデルを一時ロードしてサイズを計算する。"""
        import torch.nn
        from transformers.models.auto.modeling_auto import AutoModelForVision2Seq

        logger.debug(f"一時ロードによる Transformer サイズ計算開始: {model_path}")
        calculated_size_mb = 0.0
        temp_model = None
        try:
            temp_model = AutoModelForVision2Seq.from_pretrained(model_path).to("cpu")
            if isinstance(temp_model, torch.nn.Module):
                calculated_size_mb = LoaderBase._calculate_transformer_size_mb(temp_model)
        except Exception as e:
            logger.warning(f"一時ロード計算中にエラー ({self.model_name}): {e}", exc_info=False)
        finally:
            if temp_model:
                del temp_model
                gc.collect()
        logger.debug(f"一時ロード Transformer サイズ計算完了: {calculated_size_mb:.2f} MB")
        return calculated_size_mb

    @override
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> TransformersComponents:
        """モデルとプロセッサをロードする。"""
        from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
        from transformers.models.auto.processing_auto import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
        return {"model": model, "processor": processor}


class TransformersPipelineLoader(LoaderBase):
    """Hugging Face Transformers Pipeline のローダー。"""

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """CPU 上でパイプラインを一時作成してサイズを計算する。

        kwargs に 'task' が必要。
        """
        import torch.nn
        from transformers.pipelines import pipeline

        task = cast(str, kwargs.get("task"))
        if not task:
            return 0.0

        logger.debug(f"一時ロードによる Pipeline サイズ計算開始: task={task}, path={model_path}")
        calculated_size_mb = 0.0
        temp_pipeline = None
        try:
            temp_pipeline = pipeline(task, model=model_path, device="cpu", batch_size=1)
            if hasattr(temp_pipeline, "model") and isinstance(temp_pipeline.model, torch.nn.Module):
                calculated_size_mb = LoaderBase._calculate_transformer_size_mb(temp_pipeline.model)
        except Exception as e:
            logger.warning(f"一時ロード計算中にエラー ({self.model_name}): {e}", exc_info=False)
        finally:
            if temp_pipeline:
                del temp_pipeline
                gc.collect()
        logger.debug(f"一時ロード Pipeline サイズ計算完了: {calculated_size_mb:.2f} MB")
        return calculated_size_mb

    @override
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> TransformersPipelineComponents:
        """パイプラインオブジェクトをロードする。

        kwargs に 'task' と 'batch_size' が必要。
        """
        from transformers.pipelines import pipeline

        task = cast(str, kwargs.get("task"))
        batch_size = cast(int, kwargs.get("batch_size"))
        if not task or not batch_size:
            raise ValueError("Pipeline loader requires 'task' and 'batch_size' kwargs.")

        pipeline_obj: Pipeline = pipeline(task, model=model_path, device=self.device, batch_size=batch_size)
        return {"pipeline": pipeline_obj}
