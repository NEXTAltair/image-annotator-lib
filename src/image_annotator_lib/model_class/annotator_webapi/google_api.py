import json
from typing import cast, override

from google.genai import errors
from google.genai import types as google_types
from PIL import Image
from pydantic import ValidationError

from image_annotator_lib.exceptions.errors import (
    WebApiError,
)

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput
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

        results: list[RawOutput] = []
        for image_data in processed_bytes:
            annotation = None # ループごとに初期化
            error = None      # ループごとに初期化
            try: # API呼び出しとレスポンス処理全体を try で囲む
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

                # --- レスポンス処理 --- #
                # 1. レスポンスが期待される AnnotationSchema 型か直接確認
                if isinstance(response, AnnotationSchema):
                    annotation = response
                # 2. GenerateContentResponse 型か確認
                elif isinstance(response, google_types.GenerateContentResponse):
                    # candidates が存在し、内容が期待通りか確認
                    if response.candidates and \
                       response.candidates[0].content and \
                       response.candidates[0].content.parts and \
                       hasattr(response.candidates[0].content.parts[0], 'text') and \
                       isinstance(response.candidates[0].content.parts[0].text, str):

                        content_text = response.candidates[0].content.parts[0].text
                        if content_text.strip():
                            try:
                                parsed_data = json.loads(content_text)
                                annotation = AnnotationSchema(**parsed_data) # バリデーション
                                # パース成功時はエラーメッセージを設定しない
                            except json.JSONDecodeError as je:
                                error = f"Google API: レスポンスJSONパース失敗: {je}(内容: {content_text!r})"
                                logger.error(error)
                            except ValidationError as ve:
                                error = f"Google API: スキーマ不一致 (GenerateContentResponse): {ve}"
                                logger.error(error)
                        else:
                            error = "Google API: GenerateContentResponse内のtextコンテンツが空です。"
                            logger.warning(error)
                    else:
                        # GenerateContentResponse だが期待する構造ではない場合
                        block_reason_msg = ""
                        finish_reason_msg = ""
                        if response.prompt_feedback and response.prompt_feedback.block_reason:
                            block_reason_msg = f" ブロック理由: {response.prompt_feedback.block_reason}."
                        if response.candidates and response.candidates[0].finish_reason:
                            finish_reason_msg = f" 終了理由: {response.candidates[0].finish_reason}."
                        error = f"Google API: GenerateContentResponseの構造が予期されるものではありませんでした。{block_reason_msg}{finish_reason_msg} 完全なレスポンス: {response}"
                        logger.warning(error)
                # 3. その他の予期しない型を処理
                else:
                    error = f"Google API: 未対応のレスポンス型: {type(response)}"
                    logger.error(error)
                # --- レスポンス処理ここまで --- #

            # --- 例外処理 --- #
            except errors.APIError as e:
                # Google SDKのAPIエラーをラップ
                error = f"Google API エラー: APIError: {e.code} {e.message}"
                logger.error(error, exc_info=True)
            except ValueError as e: # これは API キーがない場合のエラーだったはず
                 error = "Google API キーが見つかりません。環境変数 'GOOGLE_API_KEY' を設定してください。"
                 logger.error(error)
            except Exception as e:
                error = f"Google API エラー: 予期せぬエラー: {e!s}"
                logger.error(error, exc_info=True)
            # --- 例外処理ここまで --- #

            # 結果を追加 (annotation が None の場合は error がセットされているはず)
            results.append(RawOutput(response=annotation, error=error))

        return results
