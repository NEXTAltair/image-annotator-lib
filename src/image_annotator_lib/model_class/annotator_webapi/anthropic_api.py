from typing import cast, override

import anthropic

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.utils import logger
from ...exceptions.errors import ModelNotFoundError, WebApiError
from .webapi_shared import (
    BASE_PROMPT,
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    AnnotationSchema,
    FormattedOutput,
    Responsedict,
)


class AnthropicApiAnnotator(WebApiBaseAnnotator):
    """Anthropic Claude API を使用するアノテーター"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[Responsedict]:
        """Anthropic API を使用して推論を実行する"""
        if not all(isinstance(item, str) for item in processed_images):
            logger.error("AnthropicApiAnnotator received non-string input for _run_inference")
            return [{"response": None, "error": "Invalid input type for Anthropic API"}] * len(processed_images)

        processed_images_str: list[str] = cast(list[str], processed_images)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません",
                provider_name=getattr(self, "provider_name", "Unknown"),
            )

        logger.debug(f"Anthropic API 呼び出しに使用するモデルID: {self.api_model_id}")

        results: list[Responsedict] = []
        for base64_image in processed_images_str:
            try:
                self._wait_for_rate_limit()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": BASE_PROMPT},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": base64_image,
                                },
                            },
                        ],
                    }
                ]
                tools = [
                    {
                        "name": "Annotatejson",
                        "description": "Parsing image annotation results to JSON",
                        "input_schema": JSON_SCHEMA,
                    }
                ]
                system_prompt = SYSTEM_PROMPT

                temperature_val = config_registry.get(self.model_name, "temperature", default=0.7)
                anthropic_temperature = float(temperature_val) if temperature_val is not None else None
                max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
                if max_tokens is None:
                    logger.warning(
                        f"モデル {self.model_name} の max_output_tokens が設定されていません。デフォルト1800を使用します。"
                    )
                    max_tokens = 1800

                response = self.client.messages.create(
                    model=self.api_model_id,
                    max_tokens=int(max_tokens),
                    system=system_prompt,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    temperature=anthropic_temperature, # type: ignore[arg-type]
                )
                annotation = None
                if response.content and type(response.content[0]).__name__ == "ToolUseBlock":
                    input_data = getattr(response.content[0], "input", None)
                    if isinstance(input_data, dict):
                        annotation = AnnotationSchema(**input_data)
                results.append({"response": annotation, "error": None})

            except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
                # 404エラーの場合はModelNotFoundErrorでラップ
                error_str = str(e)
                if "404" in error_str or "not_found_error" in error_str:
                    # モデル名を抽出(なければそのまま)
                    import re
                    m = re.search(r"model: ([\w\.\-\:]+)", error_str)
                    model_name = m.group(1) if m else "不明"
                    custom_error = ModelNotFoundError(model_name)
                    logger.error(f"Anthropic API モデル未検出: {custom_error}")
                    results.append({"response": None, "error": str(custom_error)})
                else:
                    logger.error(f"Anthropic API error: {e}")
                    results.append({"response": None, "error": f"Anthropic API error: {e}"})
            except Exception as e:
                logger.error(f"Anthropic API unknown error: {e}")
                results.append({"response": None, "error": f"Anthropic API unknown error: {e}"})
        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """Anthropic API からの応答をフォーマットする"""
        formatted_outputs: list[FormattedOutput] = []
        for output in raw_outputs:
            error = output.get("error")
            if error:
                formatted_outputs.append(FormattedOutput(annotation=None, error=error))
                continue

            response_val = output.get("response")
            # AnnotationSchema型ならそのまま
            if isinstance(response_val, AnnotationSchema):
                formatted_outputs.append(FormattedOutput(annotation=response_val, error=None))
                continue

            # それ以外は従来通り
            formatted_outputs.append(
                FormattedOutput(annotation=None, error=f"Invalid response type: {type(response_val)}")
            )
        return formatted_outputs