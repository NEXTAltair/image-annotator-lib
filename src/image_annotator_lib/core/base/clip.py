"""CLIP モデルをベースとする Scorer 用の基底クラス。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from PIL import Image

# Issue #59: module-level `import torch` は CUDA driver 不在 + triton 在り環境で
# SIGSEGV を引き起こす。型ヒントは TYPE_CHECKING 内、runtime 利用箇所は関数内 import に分離する。
if TYPE_CHECKING:
    import torch

# --- ローカルインポート ---
from ...exceptions.errors import ModelLoadError, OutOfMemoryError
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import CLIPComponents, ScoreScale, TaskCapability, UnifiedAnnotationResult
from ..utils import logger
from .annotator import BaseAnnotator


class ClipBaseAnnotator(BaseAnnotator):
    """CLIP モデルをベースとする Scorer 用の基底クラス。"""

    # ADR 0009: subclass が `scores` の各 key の値域を宣言する。
    # 基底クラスの default は空 dict (subclass で override する)。
    SCORE_SCALE: ClassVar[dict[str, ScoreScale]] = {}

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        # device 判定はローカル ML 系 base class の責務 (Issue #35 で BaseAnnotator から移譲)
        from ..utils import determine_effective_device

        self.device = determine_effective_device(self._config.device, self.model_name)
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
                cached_components = ModelLoad.cache_to_main_memory(
                    self.model_name, cast(dict[str, Any], self.components)
                )
                self.components = cast(CLIPComponents, cached_components)
            else:
                ModelLoad.release_model(self.model_name)
        except Exception:
            ModelLoad.release_model(self.model_name)

    def _preprocess_images(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """画像を CLIP プロセッサで前処理します。"""
        if (
            not self.components
            or "processor" not in self.components
            or self.components["processor"] is None
        ):
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
        import torch

        if (
            not self.components
            or "clip_model" not in self.components
            or self.components["clip_model"] is None
        ):
            raise RuntimeError("CLIP ベースモデルがロードされていません。")
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("分類器ヘッド (model) がロードされていません。")

        clip_model = self.components["clip_model"]
        classifier_head = self.components["model"]

        try:
            with torch.no_grad():
                # transformers 5.x: CLIPModel.get_image_features は tensor ではなく
                # BaseModelOutputWithPooling を返す。射影済み image embeds は .pooler_output に入る
                # (旧 4.x の戻り tensor 相当)。古い tensor 戻り値にも備えて getattr で吸収する。
                features_output = clip_model.get_image_features(**processed)
                image_features = getattr(features_output, "pooler_output", features_output)
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

    def _format_predictions(self, raw_outputs: torch.Tensor) -> list[UnifiedAnnotationResult]:
        """CLIP生出力を統一UnifiedAnnotationResultにフォーマット"""
        from ..utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)

        try:
            scores = raw_outputs.cpu().numpy().tolist()
            score_list = [float(s) for s in scores]

            has_scores = TaskCapability.SCORES in capabilities
            # ADR 0009: scores を返す場合のみ値域メタデータも添える (scores と整合)
            score_scales = self.SCORE_SCALE if has_scores and self.SCORE_SCALE else None

            results = []
            for score in score_list:
                results.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        tags=None,  # CLIPスコアラーはタグ生成なし
                        captions=None,  # CLIPスコアラーはキャプション生成なし
                        scores={"aesthetic": score} if has_scores else None,
                        score_scales=score_scales,
                        framework="pytorch",
                        raw_output={
                            "tensor_shape": list(raw_outputs.shape),
                            "raw_score": score,
                            "base_model": self.base_model or "unknown",
                        },
                    )
                )
            return results

        except Exception as e:
            logger.exception(f"スコアテンソルのフォーマット中にエラー: {e}")
            try:
                batch_size = raw_outputs.shape[0]
                # エラーの場合でもUnifiedAnnotationResultを返す
                return [
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=f"スコアテンソルのフォーマットエラー: {e}",
                        framework="pytorch",
                    )
                    for _ in range(batch_size)
                ]
            except Exception:
                return [
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=capabilities,
                        error=f"スコアテンソルのフォーマットエラー: {e}",
                        framework="pytorch",
                    )
                ]
