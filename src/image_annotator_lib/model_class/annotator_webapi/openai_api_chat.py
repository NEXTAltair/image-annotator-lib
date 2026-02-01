import base64
from io import BytesIO
from typing import override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin, PydanticAIProviderFactory
from ...core.types import UnifiedAnnotationResult
from ...core.utils import logger


class OpenRouterApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """OpenRouter API を使用して画像に注釈を付けるクラス (Provider-level PydanticAI版)

    Note:
        Phase 1B: Config Object統合
    """

    _PROVIDER_NAME = "openrouter"
    _PROVIDER_DISPLAY = "OpenRouter"

    def __init__(self, model_name: str, config=None):
        """初期化

        Args:
            model_name: モデル名
            config: WebAPIModelConfig (Phase 1B DI)。Noneの場合、後方互換フォールバック。
        """
        WebApiBaseAnnotator.__init__(self, model_name, config=config)
        PydanticAIAnnotatorMixin.__init__(self, model_name, config=config)

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        # OpenRouter専用の設定でAgentを作成
        self._load_configuration()

        # OpenRouter固有の設定データ
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
            "referer": config_registry.get(self.model_name, "referer"),
            "app_name": config_registry.get(self.model_name, "app_name"),
        }

        # OpenRouter固有のAgentを取得 (キャッシュ付き)
        self.agent = PydanticAIProviderFactory.get_cached_agent(
            model_name=self.model_name,
            api_model_id=f"openrouter:{self.api_model_id}",  # openrouter prefix
            api_key=self.api_key.get_secret_value(),
            config_data=config_data,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # Provider Factory で管理されるため、ここでは何もしない
        pass

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[UnifiedAnnotationResult]:
        """PydanticAI Agent を使用して推論を実行する (デフォルトモデル、UnifiedAnnotationResult対応)"""
        # PIL.Image リストに変換
        try:
            pil_images = []
            for item in processed_images:
                if isinstance(item, str):
                    # Base64 文字列の場合
                    image_data = base64.b64decode(item)
                    pil_image = Image.open(BytesIO(image_data))
                    pil_images.append(pil_image)
                elif isinstance(item, bytes):
                    # バイナリデータの場合
                    pil_image = Image.open(BytesIO(item))
                    pil_images.append(pil_image)
                else:
                    # すでに PIL.Image の場合
                    pil_images.append(item)
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            from ...core.utils import get_model_capabilities

            capabilities = get_model_capabilities(self.model_name)
            return [
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    error=f"画像前処理エラー: {e}",
                    provider_name=self._PROVIDER_NAME,
                    framework="api",
                )
            ] * len(processed_images)

        # デフォルトモデルで実行
        return self.run_with_model(pil_images, self.api_model_id)
