"""CLIP モデルをベースとする Scorer 用の基底クラス。"""

from abc import abstractmethod
from typing import Any, Self, cast

import torch
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import ModelLoadError, OutOfMemoryError
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import CLIPComponents
from ..utils import logger
from .annotator import BaseAnnotator


class ClipBaseAnnotator(BaseAnnotator):
    """CLIP モデルをベースとする Scorer 用の基底クラス。"""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        # base_model は必須設定でデフォルト値なし
        self.base_model = config_registry.get(self.model_name, "base_model")  # 型チェック後に代入
        logger.debug(
            f"ClipBaseAnnotator '{model_name}' initialized. Base CLIP: {self.base_model}, Head: {self.model_path}"
        )
        # components の型ヒントを具体的に指定
        self.components: CLIPComponents | None = None

    def __enter__(self) -> Self:
        """CLIP モデルと分類器ヘッドをロードします。"""
        logger.debug(f"Entering context for CLIP Scorer '{self.model_name}'")
        try:
            if self.base_model is None:
                raise ValueError(f"モデル '{self.model_name}' の base_model が設定されていません。")
            if self.model_path is None:
                raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")
            loaded_components = ModelLoad.load_clip_components(
                model_name=self.model_name,
                base_model=self.base_model,
                model_path=self.model_path,
                device=self.device,
                activation_type=config_registry.get(self.model_name, "activation_type"),
                final_activation_type=config_registry.get(self.model_name, "final_activation_type"),
            )
            if loaded_components:
                self.components = loaded_components
            logger.info(f"CLIP Scorer '{self.model_name}' の準備完了。")

        except (ModelLoadError, OutOfMemoryError, FileNotFoundError, ValueError) as e:
            logger.error(f"CLIP Scorer '{self.model_name}' のロード/復元中にエラー: {e}")
            self.components = None
            raise
        except Exception as e:
            logger.exception(f"CLIP Scorer '{self.model_name}' のロード/復元中に予期せぬエラー: {e}")
            self.components = None
            raise ModelLoadError(f"予期せぬロードエラー: {e}") from e
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """CLIP Scorer モデルをキャッシュします。"""
        logger.debug(f"Exiting context for CLIP Scorer model '{self.model_name}' (exception: {exc_type})")
        try:
            if self.components:
                cached_components = ModelLoad.cache_to_main_memory(self.model_name, cast(dict[str, Any], self.components))
                self.components = cast(CLIPComponents, cached_components)
            else:
                ModelLoad.release_model(self.model_name)
        except Exception:
            ModelLoad.release_model(self.model_name)

    def _preprocess_images(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """画像を CLIP プロセッサで前処理します。"""
        if not self.components or "processor" not in self.components or self.components["processor"] is None:
            raise RuntimeError("CLIP プロセッサがロードされていません。")
        processor = self.components["processor"]
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True)
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.exception(f"CLIP 画像の前処理中にエラー: {e}")
            raise ValueError(f"CLIP 画像の前処理失敗: {e}") from e

    def _run_inference(self, processed: dict[str, torch.Tensor]) -> torch.Tensor:
        """CLIP モデルで画像特徴量を抽出し、分類器ヘッドでスコアを計算します。"""
        if not self.components or "clip_model" not in self.components or self.components["clip_model"] is None:
            raise RuntimeError("CLIP ベースモデルがロードされていません。")
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("分類器ヘッド (model) がロードされていません。")

        clip_model = self.components["clip_model"]
        classifier_head = self.components["model"]

        try:
            with torch.no_grad():
                image_features = clip_model.get_image_features(**processed)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                raw_scores = classifier_head(image_features)
                return raw_scores.squeeze(-1)
        except torch.cuda.OutOfMemoryError as e:
            error_message = f"CUDA OOM: CLIP Scorer '{self.model_name}' 推論中"
            logger.error(error_message)
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            logger.exception(f"CLIP Scorer '{self.model_name}' 推論中にエラー: {e}")
            raise RuntimeError(f"CLIP Scorer 推論エラー: {e}") from e

    def _format_predictions(self, raw_outputs: torch.Tensor) -> list[float]:
        """生のスコアテンソルを float のリストに変換します。"""
        try:
            scores = raw_outputs.cpu().numpy().tolist()
            return [float(s) for s in scores]
        except Exception as e:
            logger.exception(f"スコアテンソルのフォーマット中にエラー: {e}")
            try:
                batch_size = raw_outputs.shape[0]
                return [0.0] * batch_size
            except Exception:
                return []

    @abstractmethod
    def _get_score_tag(self, score: float) -> str:
        """スコア値に基づいてスコアタグ文字列を生成します (サブクラスで実装)。"""
        raise NotImplementedError("サブクラスは _get_score_tag を実装する必要があります。")

    def _generate_tags(self, formatted_output: float) -> list[str]:
        """スコア値からスコアタグを生成します。"""
        return [self._get_score_tag(formatted_output)]
