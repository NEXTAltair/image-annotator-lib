from typing import override
import asyncio
import base64
from io import BytesIO

from PIL import Image
from pydantic import SecretStr
from pydantic_ai.messages import BinaryContent

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput
from ...core.utils import logger
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin
from ...exceptions.errors import (
    ModelNotFoundError,
    WebApiError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiTimeoutError,
    ApiServerError,
)


class AnthropicApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Anthropic Claude API を使用するアノテーター (Provider-level PydanticAI版)"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        WebApiBaseAnnotator.__init__(self, model_name)
        PydanticAIAnnotatorMixin.__init__(self, model_name)

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._setup_agent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # Agent は Provider Factory で管理されるため、ここでは何もしない
        pass

    def run_with_model(self, images: list[Image.Image], model_id: str) -> list[RawOutput]:
        """指定されたモデルIDで推論を実行する"""
        if not self.agent:
            raise WebApiError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。",
                provider_name="Anthropic",
            )
        
        binary_contents = self._preprocess_images_to_binary(images)
        
        results: list[RawOutput] = []
        for binary_content in binary_contents:
            try:
                self._wait_for_rate_limit()
                
                # 指定されたモデルIDで推論実行
                annotation = self._run_inference_with_model(binary_content, model_id)
                results.append({"response": annotation, "error": None})
                
            except Exception as e:
                error_message = self._handle_api_error(e)
                results.append({"response": None, "error": error_message})
        
        return results

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[RawOutput]:
        """PydanticAI Agent を使用して推論を実行する (デフォルトモデル)"""
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
            return [{"response": None, "error": f"画像前処理エラー: {e}"}] * len(processed_images)
        
        # デフォルトモデルで実行
        return self.run_with_model(pil_images, self.api_model_id)

    def _handle_api_error(self, error: Exception) -> str:
        """API エラーを適切な例外に変換"""
        error_str = str(error)
        
        # 404エラーの場合はModelNotFoundErrorでラップ
        if "404" in error_str or "not_found_error" in error_str:
            import re
            m = re.search(r"model: ([\w\.\-\:]+)", error_str)
            model_name = m.group(1) if m else "不明"
            custom_error = ModelNotFoundError(model_name)
            logger.error(f"Anthropic API モデル未検出: {custom_error}")
            raise custom_error
        
        # その他のエラーパターン
        if "authentication" in error_str.lower():
            raise ApiAuthenticationError(f"Anthropic API 認証エラー: {error_str}")
        elif "rate limit" in error_str.lower():
            raise ApiRateLimitError(f"Anthropic API レート制限: {error_str}")
        elif "timeout" in error_str.lower():
            raise ApiTimeoutError(f"Anthropic API タイムアウト: {error_str}")
        elif "500" in error_str or "server error" in error_str.lower():
            raise ApiServerError(f"Anthropic API サーバーエラー: {error_str}")
        
        # 一般エラー
        logger.error(f"Anthropic API エラー: {error}")
        return f"Anthropic API Error: {error_str}"
