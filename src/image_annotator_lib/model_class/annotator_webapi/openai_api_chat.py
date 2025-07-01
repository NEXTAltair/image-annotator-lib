import base64
from io import BytesIO
from typing import override

from PIL import Image

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from image_annotator_lib.exceptions.errors import WebApiError

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin, PydanticAIProviderFactory
from ...core.types import RawOutput
from ...core.utils import logger


class OpenRouterApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """OpenRouter API を使用して画像に注釈を付けるクラス (Provider-level PydanticAI版)"""

    def __init__(self, model_name: str):
        WebApiBaseAnnotator.__init__(self, model_name)
        PydanticAIAnnotatorMixin.__init__(self, model_name)

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

    def run_with_model(self, images: list[Image.Image], model_id: str) -> list[RawOutput]:
        """指定されたモデルIDで推論を実行する"""
        if not self.agent:
            raise WebApiError(
                "Agent が初期化されていません。コンテキストマネージャーを使用してください。",
                provider_name="OpenRouter",
            )

        binary_contents = self._preprocess_images_to_binary(images)

        results: list[RawOutput] = []
        for binary_content in binary_contents:
            try:
                self._wait_for_rate_limit()

                # The result from agent.run is a ModelResponse object.
                # The actual content is in the first part of the response.
                response_content = self._run_inference_with_model(binary_content, model_id)
                results.append({"response": response_content, "error": None})

            except ModelHTTPError as e:
                # PydanticAI統一HTTPエラー処理
                error_message = f"OpenRouter HTTP {e.status_code}: {e.body or str(e)}"
                logger.error(f"OpenRouter API 推論エラー: {error_message}")
                results.append({"response": None, "error": error_message})

            except UnexpectedModelBehavior as e:
                # PydanticAI統一モデル動作エラー処理
                error_message = f"OpenRouter API Error: Unexpected model behavior: {str(e)}"
                logger.error(f"OpenRouter API 推論エラー: {error_message}")
                results.append({"response": None, "error": error_message})

            except Exception as e:
                # その他の予期しないエラー
                error_message = f"OpenRouter API Error: {str(e)}"
                logger.error(f"OpenRouter API 推論エラー: {error_message}")
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

