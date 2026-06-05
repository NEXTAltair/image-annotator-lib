"""Hugging Face Transformers モデルローダー。

AutoModelForImageTextToText と Pipeline の2つのローダーを提供する。

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
    """Hugging Face Transformers モデル (AutoModelForImageTextToText) のローダー。"""

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """CPU 上でモデルを一時ロードしてサイズを計算する。"""
        import torch.nn
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        logger.debug(f"一時ロードによる Transformer サイズ計算開始: {model_path}")
        calculated_size_mb = 0.0
        temp_model = None
        try:
            temp_model = AutoModelForImageTextToText.from_pretrained(model_path).to("cpu")
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
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
        from transformers.models.auto.processing_auto import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(model_path).to(self.device)
        return {"model": model, "processor": processor}


def _resolve_pipeline_image_processor(model_path: str) -> Any | None:
    """Pipeline 用の image processor を解決する。

    transformers 5.x は feature extractor 群 (`*FeatureExtractor`) を削除した。古い repo の
    `preprocessor_config.json` が legacy な `image_processor_type` (例: ``"ViTFeatureExtractor"``)
    を持つ場合、``AutoImageProcessor`` の後方互換正規化は ``image_processor_type`` が None の時
    しか発火せず、クラス解決に失敗する (Issue #139)。その場合は model の config (``model_type``)
    から対応する ImageProcessor クラスを引いて明示構築する。

    Args:
        model_path: HuggingFace repo 名 / ローカルパス。

    Returns:
        構築できた image processor。解決不能なら None (pipeline の auto 解決に委ねる)。
    """
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(model_path)
    except (ValueError, OSError) as e:
        logger.debug(f"AutoImageProcessor 解決失敗、model_type 経由で再試行: {model_path} ({e})")

    try:
        from transformers import AutoConfig
        from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING

        config = AutoConfig.from_pretrained(model_path)
        # IMAGE_PROCESSOR_MAPPING の値型は transformers のバージョンで揺れる
        # (5.x: backend 別 dict {"torchvision": ViTImageProcessor, "pil": ...} /
        #  旧: (slow, fast) の tuple / 単一クラス)。型 stub と実体が一致しないため Any で受ける。
        processor_entry: Any = IMAGE_PROCESSOR_MAPPING[type(config)]
        if isinstance(processor_entry, dict):
            candidates = [processor_entry.get("torchvision"), *processor_entry.values()]
        elif isinstance(processor_entry, (tuple, list)):
            candidates = list(processor_entry)
        else:
            candidates = [processor_entry]
        processor_cls = next((cls for cls in candidates if cls is not None), None)
        if processor_cls is None:
            return None
        return processor_cls.from_pretrained(model_path)
    except (ValueError, OSError, KeyError) as e:
        logger.warning(f"image processor の明示構築に失敗: {model_path} ({e})")
        return None


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
            image_processor = _resolve_pipeline_image_processor(model_path)
            temp_pipeline = pipeline(
                task, model=model_path, image_processor=image_processor, device="cpu", batch_size=1
            )
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

        image_processor = _resolve_pipeline_image_processor(model_path)
        pipeline_obj: Pipeline = pipeline(
            task,
            model=model_path,
            image_processor=image_processor,
            device=self.device,
            batch_size=batch_size,
        )
        return {"pipeline": pipeline_obj}
