"""WebAPI 推論用の汎用 BaseAnnotator サブクラス (ADR 0023 Phase 1)。

旧 `SimplifiedAgentWrapper` と `PydanticAIWebAPIWrapper` を統合した唯一の
WebAPI 入口。registry 登録済 WebAPI モデル経由でインスタンス化される
(Issue #45: direct LiteLLM ID dispatch 経路は廃止)。

Agent / Provider / Model はキャッシュせず推論呼び出しごとに新規作成する。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from typing import Any, Self

from PIL import Image

from .base.annotator import BaseAnnotator
from .provider_manager import ProviderManager
from .types import AnnotationResult, RatingPrediction, TaskCapability, UnifiedAnnotationResult
from .utils import calculate_phash, logger


class WebApiAnnotator(BaseAnnotator):
    """LiteLLM ID ベースで WebAPI 推論を実行する汎用アノテーター。

    `BaseAnnotator` を継承するが、`config_registry` への依存を避けるために
    `__init__` で親の初期化を呼ばず、必要最小限の属性のみを直接設定する。
    """

    ADVERTISED_CAPABILITIES: frozenset[TaskCapability] = frozenset(
        {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
    )
    """Default WebAPI capabilities when no task-specific override is provided."""

    def __init__(
        self,
        litellm_model_id: str,
        api_keys: dict[str, str] | None = None,
        model_name: str | None = None,
        capabilities: set[TaskCapability] | frozenset[TaskCapability] | list[str] | None = None,
    ) -> None:
        """`WebApiAnnotator` を初期化する。

        Args:
            litellm_model_id: `openai/gpt-4o` のような LiteLLM 形式 ID。
            api_keys: provider 名 (`openai` / `anthropic` / `google` / `openrouter`)
                をキーとする API key dict。
            model_name: registry 登録済モデル名 (registry 経由の通常呼び出しで渡される)。
                省略時は `litellm_model_id` をモデル名として扱う (テスト stub 等の特殊用途)。
            capabilities: 明示タスク能力。`TaskCapability.RATINGS` が含まれる場合だけ
                rating 出力を `UnifiedAnnotationResult.ratings` として公開する。
        """
        # ADR 0023 Phase 1 / Issue #35 / Issue #45:
        # - WebAPI モデルは registry 経由でのみインスタンス化される (Issue #45 で
        #   direct dispatch 経路廃止)。本 __init__ も registry の litellm_model_id を
        #   そのまま受け取って動作する設計。
        # - `BaseAnnotator.__init__` は config_registry 依存のため踏まず、必要最小限の
        #   attribute のみを直接設定する。
        # - device 判定はローカル ML 系 base class (Transformers / ONNX / TF / CLIP /
        #   Pipeline) の責務として分離されており (Issue #35)、本クラスは "api" 固定。
        # - Agent / Provider / Model はキャッシュしない (ADR 0023)。
        self.model_name = model_name or litellm_model_id
        self.litellm_model_id = litellm_model_id
        self.api_keys = api_keys
        self.capabilities = self._normalize_capabilities(capabilities)
        self._config = None
        self.model_path = None
        self.device = "api"
        self.components = None

    @classmethod
    def _normalize_capabilities(
        cls, capabilities: set[TaskCapability] | frozenset[TaskCapability] | list[str] | None
    ) -> frozenset[TaskCapability]:
        if capabilities is None:
            return cls.ADVERTISED_CAPABILITIES

        normalized: set[TaskCapability] = set()
        for capability in capabilities:
            normalized.add(capability if isinstance(capability, TaskCapability) else TaskCapability(capability))
        return frozenset(normalized) if normalized else cls.ADVERTISED_CAPABILITIES

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
            capabilities=self.capabilities,
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
                        capabilities=set(self.capabilities),
                        error=error,
                        framework="pydantic_ai",
                    )
                )
                continue

            formatted_output = ann.get("formatted_output") or {}
            raw_tags = formatted_output.get("tags") if isinstance(formatted_output, dict) else None
            raw_captions = formatted_output.get("captions") if isinstance(formatted_output, dict) else None
            raw_score = formatted_output.get("score") if isinstance(formatted_output, dict) else None
            raw_ratings = formatted_output.get("ratings") if isinstance(formatted_output, dict) else None

            tags_value: list[str] | None = list(raw_tags) if raw_tags else None
            captions_value: list[str] | None = list(raw_captions) if raw_captions else None
            scores_value: dict[str, float] | None = (
                {"overall": float(raw_score)} if raw_score is not None else None
            )
            ratings_value = (
                self._format_ratings(raw_ratings)
                if TaskCapability.RATINGS in self.capabilities and raw_ratings
                else None
            )

            formatted.append(
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=set(self.capabilities),
                    tags=tags_value if TaskCapability.TAGS in self.capabilities else None,
                    captions=captions_value if TaskCapability.CAPTIONS in self.capabilities else None,
                    scores=scores_value if TaskCapability.SCORES in self.capabilities else None,
                    ratings=ratings_value,
                    framework="pydantic_ai",
                    raw_output={
                        "litellm_model_id": self.litellm_model_id,
                        "formatted_output": formatted_output,
                    },
                )
            )
        return formatted

    @staticmethod
    def _format_ratings(raw_ratings: Any) -> list[RatingPrediction] | None:
        if not isinstance(raw_ratings, list):
            return None

        ratings: list[RatingPrediction] = []
        for item in raw_ratings:
            if isinstance(item, RatingPrediction):
                ratings.append(item)
            elif isinstance(item, dict):
                ratings.append(RatingPrediction.model_validate(item))
        return ratings or None


__all__ = ["WebApiAnnotator"]
