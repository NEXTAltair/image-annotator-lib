from typing import cast, override
import asyncio
from io import BytesIO

from openai import APIConnectionError, OpenAI
from openai._types import NOT_GIVEN
from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from image_annotator_lib.exceptions.errors import (
    ConfigurationError, 
    WebApiError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiTimeoutError,
    ApiServerError,
)

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput
from ...core.utils import logger
from ...core.webapi_agent_cache import WebApiAgentCache, create_cache_key, create_config_hash
from .webapi_shared import BASE_PROMPT, JSON_SCHEMA, SYSTEM_PROMPT


class OpenRouterApiAnnotator(WebApiBaseAnnotator):
    """OpenRouter API を使用して画像に注釈を付けるクラス (PydanticAI Agent版)"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.agent: Agent | None = None
        self.api_key: SecretStr | None = None
        self.api_model_id: str | None = None

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self._load_configuration()
        cache_key = create_cache_key(self.model_name, "openrouter", self.api_model_id or "")
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
            raise WebApiError("OpenRouter API キーが設定されていません", provider_name="OpenRouter")
        if not self.api_model_id:
            raise WebApiError("OpenRouter API モデルIDが設定されていません", provider_name="OpenRouter")

    def _create_agent(self) -> Agent:
        """新しいPydanticAI Agentを作成する"""
        try:
            # OpenRouter固有ヘッダー設定
            default_headers = {}
            referer = config_registry.get(self.model_name, "referer")
            app_name = config_registry.get(self.model_name, "app_name")
            
            if referer and isinstance(referer, str):
                default_headers["HTTP-Referer"] = referer
            if app_name and isinstance(app_name, str):
                default_headers["X-Title"] = app_name

            provider = OpenAIProvider(
                api_key=self.api_key.get_secret_value(),
                base_url="https://openrouter.ai/api/v1",
                default_headers=default_headers
            )
            model = OpenAIModel(model_name=self.api_model_id, provider=provider)
            agent = Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
            return agent
        except Exception as e:
            logger.error(f"OpenRouter Agent作成失敗: {e}")
            raise WebApiError(f"OpenRouter Agent作成エラー: {e}", provider_name="OpenRouter")

    def _get_config_hash(self) -> str:
        """設定のハッシュ値を生成する"""
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
            "json_schema_supported": config_registry.get(self.model_name, "json_schema_supported", default=False),
            "referer": config_registry.get(self.model_name, "referer"),
            "app_name": config_registry.get(self.model_name, "app_name"),
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
        if self.agent is None:
            raise WebApiError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。",
                provider_name="OpenRouter",
            )

        logger.debug(f"OpenRouter API 呼び出しに使用するモデルID: {self.api_model_id}")

        # 基底クラスから受け取るのは Base64 文字列のリスト
        # これを PIL.Image に変換してから BinaryContent にする
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
                    # 予期しない型
                    logger.error(f"予期しない入力型: {type(item)}")
                    return [{"response": None, "error": f"予期しない入力型: {type(item)}"}] * len(processed_images)
            
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
        # OpenRouter 固有パラメータ
        temperature = config_registry.get(self.model_name, "temperature", default=0.7)
        max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
        timeout = config_registry.get(self.model_name, "timeout", default=120)
        
        model_params = {
            "temperature": float(temperature) if temperature is not None else 0.7,
            "max_tokens": int(max_tokens) if max_tokens is not None else 1800,
            "timeout": float(timeout) if timeout is not None else 120.0,
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
        
        # OpenRouter/OpenAI 互換エラーパターン
        if "401" in error_str or "authentication" in error_str.lower():
            raise ApiAuthenticationError(f"OpenRouter API 認証エラー: {error_str}")
        elif "429" in error_str or "rate limit" in error_str.lower():
            raise ApiRateLimitError(f"OpenRouter API レート制限: {error_str}")
        elif "timeout" in error_str.lower():
            raise ApiTimeoutError(f"OpenRouter API タイムアウト: {error_str}")
        elif "500" in error_str or "server error" in error_str.lower():
            raise ApiServerError(f"OpenRouter API サーバーエラー: {error_str}")
        
        # 一般エラー
        logger.error(f"OpenRouter API エラー: {error}")
        return f"OpenRouter API Error: {error_str}"

