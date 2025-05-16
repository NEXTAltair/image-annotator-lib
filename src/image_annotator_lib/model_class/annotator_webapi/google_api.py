from typing import Any, cast, override

from google.genai import errors
from PIL import Image

from image_annotator_lib.exceptions.errors import (
    WebApiError,
)

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput, WebApiInput
from ...core.utils import logger
from .webapi_shared import BASE_PROMPT, SYSTEM_PROMPT


class GoogleApiAnnotator(WebApiBaseAnnotator):
    """Google Gemini API を使用するアノテーター"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    @override
    def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
        """画像リストをバイトデータのリストに変換する"""
        from io import BytesIO

        encoded_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            encoded_images.append(buffered.getvalue())
        return encoded_images

    @override
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[RawOutput]:
        """画像データ (bytes) を Adapter の call_api に渡し、結果を取得する。"""
        if not all(isinstance(item, bytes) for item in processed):
            raise ValueError("Google API annotator requires byte inputs.")
        processed_bytes: list[bytes] = cast(list[bytes], processed)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません",
                provider_name=getattr(self, "provider_name", "Unknown"),
            )

        logger.debug(f"Google API 呼び出しに使用するモデルID (from annotator): {self.api_model_id}")

        results: list[RawOutput] = []
        for image_data in processed_bytes:
            annotation_schema: AnnotationSchema | None = None
            error_message: str | None = None
            try:
                self._wait_for_rate_limit()

                # WebApiInput を作成 (image_bytes を使用)
                web_api_input_for_image = self._create_web_api_input(image_data=image_data) 

                # API呼び出し用のパラメータを準備
                # これらは GoogleClientAdapter.call_api 内で解釈される
                api_params: dict[str, Any] = {
                    "prompt": BASE_PROMPT, # BaseAnnotator や WebApiBaseAnnotator の self.prompt を使うべきか検討
                    "system_prompt": SYSTEM_PROMPT, # 同上
                    "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
                    "top_p": config_registry.get(self.model_name, "top_p", default=1.0),
                    "top_k": config_registry.get(self.model_name, "top_k", default=32),
                    "max_output_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
                    # AnnotationSchema は Adapter 側でレスポンススキーマとして使われることを期待
                }

                # self.client は GoogleClientAdapter インスタンスなので、その call_api を呼び出す
                annotation_schema = self.client.call_api(
                    model_id=self.api_model_id, # Adapter はこの model_id を使う
                    web_api_input=web_api_input_for_image,
                    params=api_params,
                    output_schema=AnnotationSchema
                )

            except WebApiError as e: # Adapter から送出される WebApiError を捕捉
                error_message = f"Google API Adapter Error: {e.message}" # e.message を使用
                logger.error(error_message, exc_info=True)
            except errors.APIError as e: # google-genai SDK 固有のエラー (Adapter内でラップされなかった場合)
                error_message = f"Google GenAI SDK APIError: {e.code} {e.message}"
                logger.error(error_message, exc_info=True)
            except ValueError as e: # Adapter やここでのロジックに起因する ValueError
                 error_message = f"Google Annotator ValueError: {e!s}"
                 logger.error(error_message, exc_info=True)
            except Exception as e:
                error_message = f"Google Annotator Unexpected Error: {e!s}"
                logger.error(error_message, exc_info=True)

            results.append(RawOutput(response=annotation_schema, error=error_message))

        return results

    def _create_web_api_input(self, image_data: bytes) -> WebApiInput:
        """画像データから WebApiInput オブジェクトを作成するヘルパーメソッド。"""
        return WebApiInput(image_bytes=image_data)
