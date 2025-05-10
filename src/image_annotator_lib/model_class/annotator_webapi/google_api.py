import json
from typing import cast, override

from google.genai import errors
from google.genai import types as google_types
from PIL import Image
from pydantic import ValidationError

from image_annotator_lib.exceptions.errors import (
    WebApiError,
)

from ...core.base import WebApiAnnotationOutput, WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.utils import logger
from .webapi_shared import BASE_PROMPT, SYSTEM_PROMPT, AnnotationSchema, FormattedOutput


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
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[WebApiAnnotationOutput]:
        """画像データ (bytes) を Gemini API に送信し、結果を取得する。"""
        if not all(isinstance(item, bytes) for item in processed):
            raise ValueError("Google API annotator requires byte inputs.")
        processed_bytes: list[bytes] = cast(list[bytes], processed)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません",
                provider_name=getattr(self, "provider_name", "Unknown"),
            )

        logger.debug(f"Google API 呼び出しに使用するモデルID: {self.api_model_id}")

        results: list[WebApiAnnotationOutput] = []
        for image_data in processed_bytes:
            try:
                self._wait_for_rate_limit()

                contents = [
                    {"text": BASE_PROMPT},
                    {"inline_data": {"mime_type": "image/webp", "data": image_data}},
                ]

                temperature = config_registry.get(self.model_name, "temperature", default=0.7)
                top_p = config_registry.get(self.model_name, "top_p", default=1.0)
                top_k = config_registry.get(self.model_name, "top_k", default=32)
                max_output_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)

                response_schema = AnnotationSchema

                generation_config = google_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    temperature=temperature,
                    response_schema=response_schema,
                    top_p=top_p,
                    top_k=top_k,
                )

                response = self.client.models.generate_content(
                    model=self.api_model_id,
                    contents=contents,
                    config=generation_config,
                )
                # ここでAnnotationSchemaバリデーション
                try:
                    schema_obj = AnnotationSchema(**json.loads(response))
                    results.append({"annotation": schema_obj.model_dump(), "error": None})
                except ValidationError as ve:
                    logger.error(f"Google API: スキーマ不一致: {ve}")
                    results.append({"annotation": None, "error": f"スキーマ不一致: {ve}"})
            except json.JSONDecodeError as e:
                error_message = f"Google API: レスポンスJSONパース失敗: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"annotation": None, "error": error_message})
            except ValueError as e:
                error_message = f"Google API: パラメータ不正またはAPIキー未設定: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"annotation": None, "error": error_message})
            except RuntimeError as e:
                error_message = f"Google API: 実行時エラー: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"annotation": None, "error": error_message})
            except errors.APIError as e:
                error_message = f"Google API: APIError: {e.code} {e.message}"
                logger.error(error_message, exc_info=True)
                results.append({"annotation": None, "error": error_message})
            except Exception as e:
                error_message = f"Google API エラー: 処理中に予期せぬエラーが発生しました: {str(e)}"
                logger.error(error_message, exc_info=True)
                results.append({"annotation": None, "error": error_message})

        return results

    @override
    def _format_predictions(self, raw_outputs: list[WebApiAnnotationOutput]) -> list[FormattedOutput]:
        """Google Gemini API (google-genai SDK) からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            annotation = (
                AnnotationSchema(**output["annotation"]) if output["annotation"] is not None else None
            )
            formatted_outputs.append(FormattedOutput(annotation=annotation, error=output["error"]))
        return formatted_outputs