from typing import cast, override

import openai
from openai import APIConnectionError
from pydantic import ValidationError

from image_annotator_lib.exceptions.errors import WebApiError

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.types import AnnotationSchema, RawOutput
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
        if not isinstance(self.client, openai.OpenAI):
             raise WebApiError(f"Invalid client type: {type(self.client)}. Expected openai.OpenAI.")

        # config_registry から直接取得する
        max_output_tokens = config_registry.get(self.model_name, "max_output_tokens", default=2000)
        temperature = config_registry.get(self.model_name, "temperature", default=0.7)

        results: list[RawOutput] = []
        for image_data_base64 in processed_str:
            annotation = None
            error = None
            try:
                self._wait_for_rate_limit()

                # client.responses.parse を使用して構造化レスポンスを取得
                response = self.client.responses.parse(
                    model=self.api_model_id,
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": SYSTEM_PROMPT},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": BASE_PROMPT},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_data_base64}", "detail": "auto"},
                            ]
                        }
                    ],
                    text_format=AnnotationSchema, # Pydanticモデルを指定
                    max_output_tokens=max_output_tokens,
                    temperature=cast(float, temperature),
                )

                # response.output_parsed から直接 AnnotationSchema を取得
                # response.refusal など、他の応答タイプも考慮する必要があるかもしれない
                if hasattr(response, 'output_parsed') and response.output_parsed:
                    if isinstance(response.output_parsed, AnnotationSchema):
                         annotation = response.output_parsed
                    else:
                         # 稀に output_parsed があっても型が違う場合 (SDKのバグ等)
                         error = f"OpenAIレスポンスのoutput_parsedが予期せぬ型です: {type(response.output_parsed)}"
                         logger.warning(error)
                elif hasattr(response, 'refusal') and response.refusal: # type: ignore
                     error = f"OpenAIがリクエストを拒否しました: {response.refusal}" # type: ignore
                     logger.warning(error)
                elif hasattr(response, 'error') and response.error: # APIレベルのエラー
                    error = f"OpenAI API Error response: {response.error}"
                    logger.error(error)
                else:
                     # output_parsed も refusal も error もない場合
                     error = f"OpenAIから予期せぬレスポンス形式: {response.to_dict() if hasattr(response, 'to_dict') else str(response)}"
                     logger.warning(error)

            except APIConnectionError as e:
                error = f"OpenAI API Connection Error: {e}"
                logger.error(error, exc_info=True)
            except openai.APIError as e: # APIError をキャッチ (RateLimitError などを含む)
                 error = f"OpenAI API Error: {e}"
                 logger.error(error, exc_info=True)
            except ValidationError as ve: # Pydanticのバリデーションエラー
                 error = f"OpenAIレスポンスのパース/バリデーション失敗 (Pydantic): {ve}\nRaw Response (potential): {response if 'response' in locals() else 'N/A'}"
                 logger.error(error, exc_info=True)
            except Exception as e:
                error = f"OpenAI: Unexpected error during parse: {e}"
                logger.error(error, exc_info=True)

            results.append(RawOutput(response=annotation, error=error))

        return results
