import base64
from io import BytesIO
from typing import override

from PIL import Image

from ...core.base import WebApiBaseAnnotator
from ...core.pydantic_ai_factory import PydanticAIAnnotatorMixin, _is_test_environment
from ...core.types import RawOutput
from ...core.utils import logger
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from ...core.types import AnnotationResult
from ...exceptions.errors import WebApiError


class AnthropicApiAnnotator(WebApiBaseAnnotator, PydanticAIAnnotatorMixin):
    """Anthropic Claude API を使用するアノテーター (Provider-level PydanticAI版)"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        WebApiBaseAnnotator.__init__(self, model_name)
        PydanticAIAnnotatorMixin.__init__(self, model_name)
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

                # The result from agent.run is a ModelResponse object.
                # The actual content is in the first part of the response.
                response_content = self._run_inference_with_model(binary_content, model_id)
                results.append({"response": response_content, "error": None})

            except ModelHTTPError as e:
                # PydanticAI統一HTTPエラー処理
                error_message = f"Anthropic HTTP {e.status_code}: {e.body or str(e)}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
                results.append({"response": None, "error": error_message})

            except UnexpectedModelBehavior as e:
                # PydanticAI統一モデル動作エラー処理
                error_message = f"Anthropic API Error: Unexpected model behavior: {str(e)}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
                results.append({"response": None, "error": error_message})

            except Exception as e:
                # その他の予期しないエラー
                error_message = f"Anthropic API Error: {str(e)}"
                logger.error(f"Anthropic API 推論エラー: {error_message}")
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

