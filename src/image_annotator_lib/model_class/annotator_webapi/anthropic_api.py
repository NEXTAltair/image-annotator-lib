from typing import override
import asyncio
from io import BytesIO

from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput
from ...core.utils import logger
from ...core.webapi_agent_cache import WebApiAgentCache, create_cache_key, create_config_hash
from ...exceptions.errors import (
    ModelNotFoundError,
    WebApiError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiTimeoutError,
    ApiServerError,
)
from .webapi_shared import BASE_PROMPT, SYSTEM_PROMPT


class AnthropicApiAnnotator(WebApiBaseAnnotator):
    """Anthropic Claude API を使用するアノテーター (PydanticAI Agent版)"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)
        self.agent: Agent | None = None
        self.api_key: SecretStr | None = None
        self.api_model_id: str | None = None

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._load_configuration()
        cache_key = create_cache_key(self.model_name, "anthropic", self.api_model_id or "")
        config_hash = self._get_config_hash()
        
        def creator_func() -> Agent:
            return self._create_agent()
        
        self.agent = WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        # Agent は Cache で管理されるため、ここでは何もしない
        pass

    def _load_configuration(self):
        """設定を読み込む"""
        self.api_key = SecretStr(config_registry.get(self.model_name, "api_key", default=""))
        self.api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
        
        if not self.api_key.get_secret_value():
            raise WebApiError("Anthropic API キーが設定されていません", provider_name="Anthropic")
        if not self.api_model_id:
            raise WebApiError("Anthropic API モデルIDが設定されていません", provider_name="Anthropic")

    def _create_agent(self) -> Agent:
        """新しいPydanticAI Agentを作成する"""
        try:
            provider = AnthropicProvider(api_key=self.api_key.get_secret_value())
            model = AnthropicModel(model_name=self.api_model_id, provider=provider)
            agent = Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
            return agent
        except Exception as e:
            logger.error(f"Anthropic Agent作成失敗: {e}")
            raise WebApiError(f"Anthropic Agent作成エラー: {e}", provider_name="Anthropic")

    def _get_config_hash(self) -> str:
        """設定のハッシュ値を生成する"""
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        }
        return create_config_hash(config_data)

    def _preprocess_images(self, images: list[Image.Image]) -> list[BinaryContent]:
        """画像を PydanticAI の BinaryContent 形式に変換する"""
        binary_contents = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
            binary_contents.append(binary_content)
        return binary_contents

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[RawOutput]:
        """PydanticAI Agent を使用して推論を実行する"""
        # PIL.Image リストを想定 (WebApiBaseAnnotator の基底実装)
        if not all(isinstance(item, (str, bytes)) for item in processed_images):
            logger.error("AnthropicApiAnnotator received invalid input for _run_inference")
            return [{"response": None, "error": "Invalid input type for Anthropic API"}] * len(
                processed_images
            )

        if self.agent is None:
            raise WebApiError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。",
                provider_name="Anthropic",
            )

        logger.debug(f"Anthropic API 呼び出しに使用するモデルID: {self.api_model_id}")

        # 画像前処理（PIL.Image → BinaryContent）
        # 基底クラスから渡されるのは base64 文字列の場合もあるので対応
        try:
            from PIL import Image
            import base64
            
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
            
            binary_contents = self._preprocess_images(pil_images)
        except Exception as e:
            logger.error(f"画像前処理エラー: {e}")
            return [{"response": None, "error": f"画像前処理エラー: {e}"}] * len(processed_images)

        results: list[RawOutput] = []
        for binary_content in binary_contents:
            try:
                self._wait_for_rate_limit()
                
                # 同期実行
                annotation = self._run_inference_sync(binary_content)
                results.append({"response": annotation, "error": None})

            except Exception as e:
                error_message = self._handle_api_error(e)
                results.append({"response": None, "error": error_message})

        return results

    def _run_inference_sync(self, binary_content: BinaryContent) -> AnnotationSchema:
        """同期版推論実行"""
        return asyncio.run(self._run_inference_async(binary_content))

    async def _run_inference_async(self, binary_content: BinaryContent) -> AnnotationSchema:
        """非同期推論実行"""
        # Anthropic 固有パラメータ
        temperature = config_registry.get(self.model_name, "temperature", default=0.7)
        max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
        
        model_params = {
            "temperature": float(temperature) if temperature is not None else 0.7,
            "max_tokens": int(max_tokens) if max_tokens is not None else 1800,
        }
        
        result = await self.agent.run(
            user_prompt="この画像を詳細に分析して、タグとキャプションを生成してください。",
            message_history=[binary_content],
            model_settings=model_params,
        )
        return result.data

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
