from typing import cast, override

import openai
from openai import APIConnectionError
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from image_annotator_lib.exceptions.errors import WebApiError

from ...core.base import WebApiAnnotationOutput, WebApiBaseAnnotator
from .webapi_shared import BASE_PROMPT, SYSTEM_PROMPT, AnnotationSchema, Responsedict


class OpenAIApiAnnotator(WebApiBaseAnnotator):
    """OpenAI API の `client.responses.parse` を使用するアノテーター"""

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
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[Responsedict]:
        """OpenAI APIを使用して推論を実行し、構造化された応答を取得します。"""
        # processed は base64エンコードされた画像の文字列のリスト
        if not all(isinstance(item, str) for item in processed):
            raise WebApiError(
                "OpenAIApiAnnotator expects a list of base64 encoded image strings."
            )
        processed_str = cast(list[str], processed)

        results: list[Responsedict] = []
        for image_data_base64 in processed_str:
            try:
                self._wait_for_rate_limit()
                messages = [
                    ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(type="text", text=BASE_PROMPT),
                            ChatCompletionContentPartImageParam(
                                type="image_url",
                                image_url={
                                    "url": f"data:image/jpeg;base64,{image_data_base64}",
                                    "detail": "auto"
                                },
                            ),
                        ],
                    ),
                ]
                response = self.client.responses.create(
                    model=self.api_model_id,
                    input=messages,
                    text_format=AnnotationSchema,
                    max_output_tokens=self.config.get("max_output_tokens", 2000),
                    temperature=float(self.config.get("temperature", 0.7)),
                )
                results.append(Responsedict(response=response.output_parsed, error=None))
            except (APIConnectionError, openai.APIError) as e:
                results.append(Responsedict(response=None, error=str(e)))
            except Exception as e:
                results.append(Responsedict(response=None, error=f"Unexpected error: {e}"))
        return results

    @override
    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[WebApiAnnotationOutput]:
        """OpenAI APIからの応答を標準形式にフォーマットします。"""
        formatted_results: list[WebApiAnnotationOutput] = []
        for output in raw_outputs:
            error = output.get("error")
            if error:
                formatted_results.append(WebApiAnnotationOutput(annotation=None, error=error))
                continue
            parsed = output.get("response")
            if not isinstance(parsed, AnnotationSchema):
                formatted_results.append(WebApiAnnotationOutput(annotation=None, error="No response data or invalid type"))
                continue
            annotation_data = {
                "tags": parsed.tags,
                "caption": parsed.captions[0] if parsed.captions else "",
                "score": float(parsed.score),
            }
            formatted_results.append(WebApiAnnotationOutput(annotation=annotation_data, error=None))
        return formatted_results
