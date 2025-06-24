import asyncio
import time
from io import BytesIO
from typing import Any, Self, override

from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

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


class OpenAIApiAnnotator(WebApiBaseAnnotator):
    """PydanticAI Agentベースの OpenAI アノテーター実装

    既存の WebApiBaseAnnotator インターフェースとの互換性を保ちながら、
    PydanticAI Agent を使用して OpenAI API と構造化出力を統合する。
    """

    def __init__(self, model_name: str):
        """OpenAI アノテーターをmodel_nameで初期化する

        Args:
            model_name: 設定ファイルからのモデル名
        """
        super().__init__(model_name)

        # PydanticAI Agent は __enter__ で作成される
        self.agent: Agent | None = None

    @override
    def __enter__(self) -> Self:
        """コンテキストマネージャーエントリ - PydanticAI Agentを準備する"""
        logger.info(f"PydanticAI OpenAI アノテーター '{self.model_name}' のコンテキストに入ります...")

        try:
            # 設定値を読み込む
            self._load_configuration()

            # キャッシュからAgentを取得または新規作成
            cache_key = create_cache_key(self.model_name, "openai", self.api_model_id)
            config_hash = self._get_config_hash()

            self.agent = WebApiAgentCache.get_agent(
                cache_key=cache_key, agent_creator=self._create_agent, config_hash=config_hash
            )

            if self.agent is None:
                raise ConfigurationError("PydanticAI Agent の取得に失敗しました")

            logger.info(
                f"PydanticAI OpenAI Agent 準備完了 (model: {self.api_model_id}, cache_key: {cache_key})"
            )

        except Exception as e:
            logger.error(f"PydanticAI OpenAI Agent の準備中にエラーが発生: {e}")
            self.agent = None
            raise ConfigurationError(f"OpenAI Agent 準備中のエラー: {e}") from e

        return self

    @override
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャー終了 - agentリソースをクリーンアップする"""
        if self.agent:
            logger.debug(f"PydanticAI OpenAI Agent のリソースを解放")
            self.agent = None

    def _load_configuration(self) -> None:
        """設定レジストリから設定を読み込む"""
        # model_path をAPI model IDとして設定（通常は同じ）
        self.api_model_id = self.model_path
        if not self.api_model_id:
            raise ConfigurationError(f"{self.model_name} の model_path が見つかりません")

        # エラーハンドリング用のプロバイダ名
        self.provider_name = "openai"

        # 環境変数からAPIキーを取得
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("環境変数に OPENAI_API_KEY が見つかりません")

        self.api_key = SecretStr(api_key)

    def _get_config_hash(self) -> str:
        """現在の設定からハッシュ値を生成する"""
        config_dict = {
            "api_model_id": self.api_model_id,
            "provider_name": self.provider_name,
            "temperature": config_registry.get(self.model_name, "temperature", 0.7),
            "max_output_tokens": config_registry.get(self.model_name, "max_output_tokens", 1800),
            # システムプロンプトも設定の一部として含める
            "system_prompt_hash": hash("画像解析AI画像annotation"),  # 簡易化
        }
        return create_config_hash(config_dict)

    def _create_agent(self) -> Agent:
        """OpenAI用のPydanticAI Agentを作成する"""
        if not self.api_model_id or not self.api_key:
            raise ConfigurationError("API model ID または API key が利用できません")

        # OpenAI モデルを作成
        provider = OpenAIProvider(api_key=self.api_key.get_secret_value())
        model = OpenAIModel(
            model_name=self.api_model_id,
            provider=provider,
        )

        # 構造化画像アノテーション用のシステムプロンプト
        system_prompt = """あなたは画像解析の専門AIです。提供された画像を分析して、構造化されたアノテーションを生成してください：

1. タグ: 画像の内容、オブジェクト、人物、スタイル、構成などについての包括的な説明タグのリスト
2. キャプション: 画像に見えるものを説明する記述的なキャプション
3. スコア: 0.0から1.0までの品質/美的スコア

分析は正確で、記述的で、包括的に行ってください。
レスポンススキーマで指定された正確な構造化フォーマットで出力してください。"""

        # 構造化出力でAgentを作成
        agent = Agent(
            model=model,
            output_type=AnnotationSchema,  # 構造化出力
            system_prompt=system_prompt,
        )

        return agent

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
        """PIL画像をPydanticAI Agent用のbytesに変換する"""
        processed_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            processed_images.append(buffered.getvalue())
        return processed_images

    @override
    def _run_inference(self, processed: list[bytes]) -> list[RawOutput]:
        """PydanticAI Agentを使用して推論を実行する（非同期操作の同期ラッパー）"""
        return asyncio.run(self._run_inference_async(processed))

    async def _run_inference_async(self, processed: list[bytes]) -> list[RawOutput]:
        """PydanticAI Agentを使用して推論を実行する（非同期実装）"""
        if self.agent is None:
            raise ConfigurationError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。"
            )

        results: list[RawOutput] = []

        for image_data in processed:
            try:
                # レート制限
                self._wait_for_rate_limit()

                # 画像とテキストでマルチモーダルプロンプトを作成
                prompt_parts = [
                    BinaryContent(data=image_data, media_type="image/webp"),
                    BASE_PROMPT,
                ]

                # PydanticAI Agentを実行
                response = await self.agent.run(prompt_parts)

                # 構造化出力を抽出
                if hasattr(response, "data"):
                    annotation = response.data
                else:
                    annotation = response

                # AnnotationSchemaが得られたことを検証
                if not isinstance(annotation, AnnotationSchema):
                    raise WebApiError(
                        f"AnnotationSchemaが期待されましたが、{type(annotation)}が得られました"
                    )

                results.append(RawOutput(response=annotation, error=None))

            except Exception as e:
                error_message = self._handle_api_error(e)
                results.append(RawOutput(response=None, error=error_message))

        return results

    def _wait_for_rate_limit(self) -> None:
        """レート制限の実装（WebApiBaseAnnotatorから保持）"""
        elapsed_time = time.time() - self.last_request_time
        wait_time = self.min_request_interval - elapsed_time
        if wait_time > 0:
            logger.debug(f"レート制限のため {wait_time:.2f} 秒待機します。")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception) -> str:
        """APIエラーを処理して適切なカスタム例外にマップする"""
        error_message = str(e)
        logger.error(f"OpenAI API エラーが発生しました: {error_message}")

        # HTTPステータスコードベースのエラーハンドリング
        if hasattr(e, "status_code"):
            status_code = getattr(e, "status_code", 0)
            if status_code == 401:
                raise ApiAuthenticationError(provider_name="openai") from e
            elif status_code == 402:
                raise InsufficientCreditsError(provider_name="openai") from e
            elif status_code == 429:
                retry_after = getattr(e, "retry_after", 60)
                raise ApiRateLimitError(provider_name="openai", retry_after=retry_after) from e
            elif status_code == 400:
                raise ApiRequestError(error_message, provider_name="openai") from e
            elif 500 <= status_code < 600:
                raise ApiServerError(error_message, provider_name="openai", status_code=status_code) from e

        # タイムアウトエラーの検出
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)) or "timeout" in error_message.lower():
            raise ApiTimeoutError(provider_name="openai") from e

        # 汎用WebAPIエラー
        raise WebApiError(
            f"OpenAI処理中に予期せぬエラーが発生しました: {error_message}", provider_name="openai"
        ) from e

    @override
    def _format_predictions(self, raw_outputs: list[RawOutput]) -> list[WebApiFormattedOutput]:
        """RawOutput を WebApiFormattedOutput にフォーマットする（既存インターフェースとの互換性）"""
        formatted_outputs: list[WebApiFormattedOutput] = []
        for output in raw_outputs:
            error = output.get("error")
            response_val = output.get("response")

            if error:
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error))
                continue

            if isinstance(response_val, AnnotationSchema):
                # AnnotationSchema を互換性のためdictに変換
                formatted_outputs.append(
                    WebApiFormattedOutput(annotation=response_val.model_dump(), error=None)
                )
            else:
                # None または予期しない型を処理
                error_message = (
                    f"無効なレスポンス型: {type(response_val)}"
                    if response_val is not None
                    else "レスポンスがNoneです"
                )
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error_message))

        return formatted_outputs

    @override
    def _generate_tags(self, formatted_output: WebApiFormattedOutput) -> list[str]:
        """フォーマット済み出力からタグを生成する（既存インターフェースとの互換性）"""
        if formatted_output.get("error") or not formatted_output.get("annotation"):
            return []

        annotation = formatted_output["annotation"]
        if isinstance(annotation, dict) and "tags" in annotation:
            tags = annotation["tags"]
            if isinstance(tags, list):
                return tags

        return []
