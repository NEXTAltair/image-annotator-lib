from typing import Any, Self, override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin
from ...core.types import UnifiedAnnotationResult
from ...core.utils import logger


class GoogleApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Provider-level PydanticAI Google Gemini アノテーター実装

    Provider-level アーキテクチャによる効率的なリソース共有で
    Google Gemini API と構造化出力を統合する。

    Note:
        Phase 1B: Config Object統合
    """

    _PROVIDER_NAME = "google"
    _PROVIDER_DISPLAY = "Google"

    def __init__(self, model_name: str, config=None):
        """Google アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
            config: WebAPIModelConfig (Phase 1B DI)。Noneの場合、後方互換フォールバック。
        """
        WebApiBaseAnnotator.__init__(self, model_name, config=config)
        PydanticAIAnnotatorMixin.__init__(self, model_name, config=config)
        # 設定を初期化時に読み込む
        self._load_configuration()

    @override
    def __enter__(self) -> Self:
        """コンテキストマネージャーエントリ - Provider-level Agent準備"""
        logger.info(f"Provider-level Google アノテーター '{self.model_name}' のコンテキストに入ります...")

        try:
            self._setup_agent()
            logger.info(f"Provider-level Google Agent 準備完了 (model: {self.api_model_id})")
        except Exception as e:
            logger.error(f"Provider-level Google Agent 準備エラー: {e}")
            raise

        return self

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャー終了 - Provider-levelで管理されるため何もしない"""
        logger.debug("Provider-level Google Agent コンテキスト終了")

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Provider-levelでは前処理は不要"""
        return images

    @override
    def _run_inference(self, processed: list[Image.Image]) -> list[UnifiedAnnotationResult]:
        """Provider Managerを通して推論実行（UnifiedAnnotationResult対応）"""
        if not self.api_model_id:
            from ...core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"Model {self.model_name} has no api_model_id configured",
                    provider_name=self._PROVIDER_NAME,
                    framework="api",
                )
            ] * len(processed)

        # Provider-level実行に委譲
        return self.run_with_model(processed, self.api_model_id)

    @override
    def _format_predictions(
        self, raw_outputs: list[UnifiedAnnotationResult]
    ) -> list[UnifiedAnnotationResult]:
        """Provider-levelでは整形済みのため変更不要（UnifiedAnnotationResult対応）"""
        return raw_outputs

    @override
    def _generate_tags(self, formatted_output: UnifiedAnnotationResult) -> list[str]:
        """整形済み出力からタグリストを生成（UnifiedAnnotationResult対応）"""
        if formatted_output.error:
            return []

        if formatted_output.tags:
            return formatted_output.tags

        return []
