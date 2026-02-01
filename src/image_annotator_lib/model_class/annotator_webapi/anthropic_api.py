import base64
from io import BytesIO
from typing import override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin, _is_test_environment
from ...core.types import UnifiedAnnotationResult
from ...core.utils import logger
from ...exceptions.errors import WebApiError


class AnthropicApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Anthropic Claude API を使用するアノテーター (Provider-level PydanticAI版)

    Note:
        Phase 1B: Config Object統合
        - WebAPIModelConfig経由でAPI関連設定を注入
        - 後方互換のためconfig=Noneをサポート
    """

    _PROVIDER_NAME = "anthropic"
    _PROVIDER_DISPLAY = "Anthropic"

    def __init__(self, model_name: str, config=None):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
            config: WebAPIModelConfig (Phase 1B DI)。Noneの場合、後方互換フォールバック。
        """
        WebApiBaseAnnotator.__init__(self, model_name, config=config)
        PydanticAIAnnotatorMixin.__init__(self, model_name, config=config)
        # 設定を初期化時に読み込む
        self._load_configuration()
        # テスト環境ではAPIキー検証をスキップ
        if not _is_test_environment() and (not self.api_key or not self.api_key.get_secret_value()):
            raise WebApiError("APIキーが設定されていません。", provider_name="Anthropic")

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._setup_agent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # Agent は Provider Factory で管理されるため、ここでは何もしない
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
