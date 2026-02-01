from typing import Any, Self, override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin
from ...core.types import UnifiedAnnotationResult
from ...core.utils import logger


class OpenAIApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Provider-level PydanticAI OpenAI アノテーター実装

    Provider-level アーキテクチャによる効率的なリソース共有で
    OpenAI API と構造化出力を統合する。
    """

    _PROVIDER_NAME = "openai"
    _PROVIDER_DISPLAY = "OpenAI"

    def __init__(self, model_name: str):
        """OpenAI アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
        """
        WebApiBaseAnnotator.__init__(self, model_name)
        PydanticAIAnnotatorMixin.__init__(self, model_name)

    @override
    def __enter__(self) -> Self:
        """コンテキストマネージャーエントリ - Provider-level Agent準備"""
        logger.info(f"Provider-level OpenAI アノテーター '{self.model_name}' のコンテキストに入ります...")

        try:
            self._setup_agent()
            logger.info(f"Provider-level OpenAI Agent 準備完了 (model: {self.api_model_id})")
        except Exception as e:
            logger.error(f"Provider-level OpenAI Agent 準備エラー: {e}")
            raise

        return self

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャー終了 - Provider-levelで管理されるため何もしない"""
        logger.debug("Provider-level OpenAI Agent コンテキスト終了")

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    @override
    def _run_inference(self, processed: list[Image.Image]) -> list[UnifiedAnnotationResult]:
        """Provider Managerを通して推論実行（UnifiedAnnotationResult対応）"""
        if not self.api_model_id:
            raise ValueError(f"Model {self.model_name} has no api_model_id configured")

        # Provider-level実行に委譲
        return self.run_with_model(processed, self.api_model_id)
