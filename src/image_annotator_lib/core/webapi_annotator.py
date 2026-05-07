"""WebAPI 推論用の汎用 BaseAnnotator サブクラス (ADR 0023 Phase 1)。

旧 `SimplifiedAgentWrapper` と `PydanticAIWebAPIWrapper` を統合した唯一の
WebAPI 入口。direct model registration (`google/gemini-...` 等) と registry 登録済
WebAPI モデルの双方を本クラスで処理する。

Agent / Provider / Model はキャッシュせず推論呼び出しごとに新規作成する。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from typing import Any, Self

from PIL import Image

from .base.annotator import BaseAnnotator
from .provider_manager import ProviderManager
from .types import AnnotationResult, TaskCapability, UnifiedAnnotationResult
from .utils import calculate_phash, logger


class WebApiAnnotator(BaseAnnotator):
    """LiteLLM ID ベースで WebAPI 推論を実行する汎用アノテーター。

    `BaseAnnotator` を継承するが、`config_registry` への依存を避けるために
    `__init__` で親の初期化を呼ばず、必要最小限の属性のみを直接設定する。
    """

    ADVERTISED_CAPABILITIES: frozenset[TaskCapability] = frozenset(
        {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
    )
    """Phase 1 では `AnnotationSchema` (tags / captions / score) の 3 種を全申告する。"""

    def __init__(
        self,
        litellm_model_id: str,
        api_keys: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> None:
        """`WebApiAnnotator` を初期化する。

        Args:
            litellm_model_id: `openai/gpt-4o` のような LiteLLM 形式 ID。
            api_keys: provider 名 (`openai` / `anthropic` / `google` / `openrouter`)
                をキーとする API key dict。
            model_name: registry 登録済モデル名 (registry 経由の場合に渡す)。
                省略時は `litellm_model_id` をモデル名として扱う (direct model registration)。
        """
        # ADR 0023 Phase 1 / Issue #35:
        # - direct LiteLLM ID 経路 (`google/gemini-...` 等) では config_registry に entry が
        #   無いため、`BaseAnnotator.__init__` (内部で `_load_config_from_registry` を
        #   呼び得る) を踏まずに必要最小限の attribute のみを設定する。
        # - device 判定はローカル ML 系 base class (Transformers / ONNX / TF / CLIP /
        #   Pipeline) の責務として分離されており (Issue #35)、本クラスは "api" 固定。
        # - Agent / Provider / Model はキャッシュしない (ADR 0023)。
        self.model_name = model_name or litellm_model_id
        self.litellm_model_id = litellm_model_id
        self.api_keys = api_keys
        self._config = None
        self.model_path = None
        self.device = "api"
        self.components = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        # ADR 0023: Agent / Provider / Model はキャッシュしないため cleanup 不要。
        return None

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        # 実際の WEBP 変換は ProviderManager 内 (preprocess_images_to_binary) で行う。
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[AnnotationResult]:
        """`ProviderManager` 経由で WebAPI 推論を実行する。

        画像順序を保つため、`ProviderManager` の dict 戻り値を入力画像の順序で
        list 化してから返す。
        """
        results_dict = ProviderManager.run_inference_with_model(
            model_name=self.model_name,
            images_list=processed,
            litellm_model_id=self.litellm_model_id,
            api_keys=self.api_keys,
        )
        ordered: list[AnnotationResult] = []
        for index, image in enumerate(processed):
            phash = calculate_phash(image) or f"unknown_image_{index}"
            result = results_dict.get(phash)
            if result is None:
                logger.warning(
                    f"画像 {index} (phash={phash}) の結果が ProviderManager から返却されませんでした"
                )
                result = AnnotationResult(
                    phash=phash,
                    tags=[],
                    formatted_output=None,
                    error="No result returned for image",
                )
            ordered.append(result)
        return ordered

    def _format_predictions(self, raw_outputs: list[AnnotationResult]) -> list[UnifiedAnnotationResult]:
        formatted: list[UnifiedAnnotationResult] = []
        for ann in raw_outputs:
            error = ann.get("error")
            if error:
                formatted.append(
                    UnifiedAnnotationResult(
                        model_name=self.model_name,
                        capabilities=set(self.ADVERTISED_CAPABILITIES),
                        error=error,
                        framework="pydantic_ai",
                    )
                )
                continue

            formatted_output = ann.get("formatted_output") or {}
            raw_tags = formatted_output.get("tags") if isinstance(formatted_output, dict) else None
            raw_captions = formatted_output.get("captions") if isinstance(formatted_output, dict) else None
            raw_score = formatted_output.get("score") if isinstance(formatted_output, dict) else None

            tags_value: list[str] | None = list(raw_tags) if raw_tags else None
            captions_value: list[str] | None = list(raw_captions) if raw_captions else None
            scores_value: dict[str, float] | None = (
                {"overall": float(raw_score)} if raw_score is not None else None
            )

            formatted.append(
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=set(self.ADVERTISED_CAPABILITIES),
                    tags=tags_value,
                    captions=captions_value,
                    scores=scores_value,
                    framework="pydantic_ai",
                    raw_output={"litellm_model_id": self.litellm_model_id},
                )
            )
        return formatted


__all__ = ["WebApiAnnotator"]
