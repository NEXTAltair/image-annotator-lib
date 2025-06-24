import asyncio
import time
from io import BytesIO
from typing import Any, Self, override

from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput, WebApiFormattedOutput
from ...core.utils import logger
from ...core.webapi_agent_cache import WebApiAgentCache, create_cache_key, create_config_hash
from ...exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ConfigurationError,
    InsufficientCreditsError,
    WebApiError,
)
from .webapi_shared import BASE_PROMPT


class GoogleApiAnnotator(WebApiBaseAnnotator):
    """PydanticAI Agentベースの Google Gemini アノテーター実装

    既存の WebApiBaseAnnotator インターフェースとの互換性を保ちながら、
    PydanticAI Agent を使用して Google Gemini API と構造化出力を統合する。
    """

    def __init__(self, model_name: str):
        """Google アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
        """
        super().__init__(model_name)

        # PydanticAI Agent は __enter__ で作成される
        self.agent: Agent | None = None

    @override
    def __enter__(self) -> Self:
        """コンテキストマネージャーエントリ - PydanticAI Agentを準備する"""
        logger.info(f"PydanticAI Google アノテーター '{self.model_name}' のコンテキストに入ります...")

        try:
            # 設定値を読み込む
            self._load_configuration()

            # キャッシュからAgentを取得または新規作成
            cache_key = create_cache_key(self.model_name, "google", self.api_model_id)
            config_hash = self._get_config_hash()

            self.agent = WebApiAgentCache.get_agent(
                cache_key=cache_key, agent_creator=self._create_agent, config_hash=config_hash
            )

            if self.agent is None:
                raise ConfigurationError("PydanticAI Agent の取得に失敗しました")

            logger.info(
                f"PydanticAI Google Agent 準備完了 (model: {self.api_model_id}, cache_key: {cache_key})"
            )

        except Exception as e:
            logger.error(f"PydanticAI Google Agent の準備中にエラーが発生: {e}")
            self.agent = None
            raise ConfigurationError(f"Google Agent 準備中のエラー: {e}") from e

        return self

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャー終了 - agentリソースをクリーンアップする"""
        if self.agent:
            logger.debug(f"PydanticAI Google Agent のリソースを解放")
            self.agent = None

    def _load_configuration(self) -> None:
        """設定レジストリから設定を読み込む"""
        # model_path をAPI model IDとして設定（通常は同じ）
        self.api_model_id = self.model_path
        if not self.api_model_id:
            raise ConfigurationError(f"{self.model_name} の model_path が見つかりません")

        # エラーハンドリング用のプロバイダ名
        self.provider_name = "google"

        # 環境変数からAPIキーを取得
        import os

        api_key_str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key_str:
            raise ConfigurationError("Google API key が環境変数に設定されていません (GOOGLE_API_KEY または GEMINI_API_KEY)")

        self.api_key = SecretStr(api_key_str)
        logger.debug(f"Google API key を環境変数から読み込みました")

    def _get_config_hash(self) -> str:
        """Agent作成に影響する設定のハッシュを生成"""
        config_data = {
            "model_id": self.api_model_id,
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "top_p": config_registry.get(self.model_name, "top_p", default=1.0),
            "top_k": config_registry.get(self.model_name, "top_k", default=32),
            "max_output_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        }
        return create_config_hash(config_data)

    def _create_agent(self) -> Agent:
        """新しいPydanticAI Agentを作成する"""
        try:
            # Google Provider と Model を作成
            provider = GoogleProvider(api_key=self.api_key.get_secret_value())
            model = GoogleModel(model_name=self.api_model_id, provider=provider)

            # Agent を作成（構造化出力対応）
            agent = Agent(
                model=model,
                output_type=AnnotationSchema,
                system_prompt=BASE_PROMPT,
            )

            logger.debug(f"PydanticAI Google Agent を作成しました (model: {self.api_model_id})")
            return agent

        except Exception as e:
            logger.error(f"PydanticAI Google Agent の作成中にエラー: {e}")
            raise ConfigurationError(f"Google Agent 作成エラー: {e}") from e

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[BinaryContent]:
        """画像リストをPydanticAI BinaryContentのリストに変換する"""
        binary_contents = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            
            binary_content = BinaryContent(
                data=buffered.getvalue(),
                media_type="image/webp"
            )
            binary_contents.append(binary_content)
        
        return binary_contents

    @override
    def _run_inference(self, processed: list[BinaryContent]) -> list[RawOutput]:
        """PydanticAI Agentを使用して画像推論を実行する"""
        if self.agent is None:
            raise WebApiError("PydanticAI Agent が初期化されていません", provider_name="google")

        logger.debug(f"Google API 呼び出しに使用するモデルID: {self.api_model_id}")

        results: list[RawOutput] = []
        for binary_content in processed:
            annotation_schema: AnnotationSchema | None = None
            error_message: str | None = None
            
            try:
                self._wait_for_rate_limit()

                # PydanticAI Agent による推論実行（同期ラッパー）
                annotation_schema = self._run_inference_sync(binary_content)

            except Exception as e:
                error_message = self._handle_api_error(e)
                logger.error(f"Google API 推論エラー: {error_message}", exc_info=True)

            results.append(RawOutput(response=annotation_schema, error=error_message))

        return results

    def _run_inference_sync(self, binary_content: BinaryContent) -> AnnotationSchema:
        """同期的にPydanticAI Agent推論を実行する"""
        try:
            # 非同期推論を同期的に実行
            result = asyncio.run(self._run_inference_async(binary_content))
            return result
        except Exception as e:
            logger.error(f"Google Agent 推論実行エラー: {e}")
            raise

    async def _run_inference_async(self, binary_content: BinaryContent) -> AnnotationSchema:
        """非同期でPydanticAI Agent推論を実行する"""
        if self.agent is None:
            raise WebApiError("Agent が初期化されていません", provider_name="google")

        try:
            # Gemini固有パラメータ設定
            model_params = {
                "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
                "top_p": config_registry.get(self.model_name, "top_p", default=1.0),
                "top_k": config_registry.get(self.model_name, "top_k", default=32),
                "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
            }

            # Agent実行（画像と追加プロンプト）
            result = await self.agent.run(
                user_prompt="この画像を詳細に分析して、タグとキャプションを生成してください。",
                message_history=[binary_content],
                model_settings=model_params,
            )

            logger.debug(f"Google Agent 推論完了 (model: {self.api_model_id})")
            return result.data

        except Exception as e:
            logger.error(f"Google Agent 非同期推論エラー: {e}")
            raise

    def _handle_api_error(self, error: Exception) -> str:
        """APIエラーを統一的にハンドリングして、適切なエラーメッセージを返す"""
        # PydanticAI Google プロバイダー固有のエラーハンドリング
        error_message = f"Google API Error: {error!s}"
        
        # 具体的なエラータイプ別の処理
        if "authentication" in str(error).lower():
            raise ApiAuthenticationError(f"Google API認証エラー: {error}", provider_name="google")
        elif "rate limit" in str(error).lower() or "quota" in str(error).lower():
            raise ApiRateLimitError(f"Google APIレート制限: {error}", provider_name="google")
        elif "timeout" in str(error).lower():
            raise ApiTimeoutError(f"Google APIタイムアウト: {error}", provider_name="google")
        elif "server" in str(error).lower() or "5" in str(error)[:1]:
            raise ApiServerError(f"Google APIサーバーエラー: {error}", provider_name="google")
        else:
            return error_message
