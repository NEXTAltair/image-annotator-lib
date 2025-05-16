from typing import Any, cast, override

import openai
from openai import APIConnectionError
from pydantic import ValidationError

from image_annotator_lib.exceptions.errors import WebApiError

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput, WebApiInput
from ...core.utils import logger
from .webapi_shared import BASE_PROMPT, SYSTEM_PROMPT


class OpenAIApiAnnotator(WebApiBaseAnnotator):
    """OpenAI API の `client.responses.create` (`parse`) を使用するアノテーター"""

    # Pydanticモデルの定義
    # OpenAI SDKの `client.responses.parse` を使用し、構造化された出力を得るために定義。
    # 参照ドキュメント: OpenAI Structured Outputs (ユーザー提供のURLを元に記載)
    #   - URL: https://platform.openai.com/docs/guides/structured-outputs
    #   - 参照日時: 2025-05-10 (ユーザーがドキュメントを提示した日時を想定)
    #   - SDKバージョン: openai >= 1.0.0 (この機能が利用可能なバージョン)
    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    @override
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[RawOutput]:
        """OpenAI APIを使用して推論を実行し、構造化された応答を取得します。"""
        if not all(isinstance(item, str) for item in processed):
            raise WebApiError(
                "OpenAIApiAnnotator expects a list of base64 encoded image strings."
            )
        processed_str = cast(list[str], processed)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません"
            )
        # isinstance(self.client, openai.OpenAI) のチェックは削除。
        # self.client は OpenAIAdapter インスタンスであることを期待。

        max_output_tokens = config_registry.get(self.model_name, "max_output_tokens", default=2000)
        temperature = config_registry.get(self.model_name, "temperature", default=0.7)

        results: list[RawOutput] = []
        for image_data_base64 in processed_str:
            annotation: AnnotationSchema | None = None
            error_message: str | None = None
            try:
                self._wait_for_rate_limit()

                web_api_input = WebApiInput(image_b64=image_data_base64)

                api_params: dict[str, Any] = {
                    "prompt": BASE_PROMPT, # Adapter側でinputの一部として利用される想定
                    "system_prompt": SYSTEM_PROMPT, # 同上
                    "temperature": cast(float, temperature),
                    "max_output_tokens": max_output_tokens,
                    "use_responses_parse": True, # OpenAIAdapter に responses.parse を使うよう指示
                }

                # self.client は OpenAIAdapter インスタンス
                annotation = self.client.call_api(
                    model_id=self.api_model_id,
                    web_api_input=web_api_input,
                    params=api_params,
                    output_schema=AnnotationSchema # Adapter はこれを見てレスポンスをパースする
                )
                # call_api が None を返し、エラー情報を別途提供する設計も考えられるが、
                # ここでは成功時は AnnotationSchema を、失敗時は WebApiError を送出すると仮定。

            except WebApiError as e: # Adapterから送出されるエラー
                error_message = f"OpenAI Adapter Error: {e.message}"
                logger.error(error_message, exc_info=True)
            # Adapter が openai.APIError や ValidationError を WebApiError にラップすることを期待。
            # もしラップされないケースも考慮するなら、以下のようなキャッチも残す。
            except APIConnectionError as e:
                error_message = f"OpenAI API Connection Error: {e}"
                logger.error(error_message, exc_info=True)
            except openai.APIError as e: # APIError をキャッチ (RateLimitError などを含む)
                 error_message = f"OpenAI API Error: {e}"
                 logger.error(error_message, exc_info=True)
            except ValidationError as ve: # Pydanticのバリデーションエラー
                 error_message = f"OpenAI Response Validation Error: {ve}"
                 logger.error(error_message, exc_info=True)
            except Exception as e:
                error_message = f"OpenAI Annotator Unexpected Error: {e!s}"
                logger.error(error_message, exc_info=True)

            results.append(RawOutput(response=annotation, error=error_message))

        return results
