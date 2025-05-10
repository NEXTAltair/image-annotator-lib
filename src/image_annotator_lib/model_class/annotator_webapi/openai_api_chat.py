from typing import cast, override

from openai import APIConnectionError, OpenAI
from openai._types import NOT_GIVEN

from image_annotator_lib.exceptions.errors import ConfigurationError, WebApiError

from ...core.base import WebApiBaseAnnotator
from ...core.config import config_registry
from ...core.utils import logger
from .webapi_shared import (
    BASE_PROMPT,
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    AnnotationSchema,
    FormattedOutput,
    Responsedict,
)


class OpenRouterApiAnnotator(WebApiBaseAnnotator):
    """OpenRouter API と OpenAI.chat を使用して画像に注釈を付けるクラス"""

    def __init__(self, model_name: str):
        super().__init__(model_name)

    @override
    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[Responsedict]:
        if not all(isinstance(item, str) for item in processed_images):
            raise ValueError("OpenRouter API annotator requires string (base64) inputs.")
        processed_images_str: list[str] = cast(list[str], processed_images)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません"
            )
        if not isinstance(self.client, OpenAI):
            raise ConfigurationError(
                f"予期しないクライアントタイプ: {type(self.client)}"
            )

        logger.debug(f"OpenRouter API 呼び出しに使用するモデルID: {self.api_model_id}")

        referer = config_registry.get(self.model_name, "referer")
        app_name = config_registry.get(self.model_name, "app_name")
        extra_headers: dict[str, str] = {}
        if referer and isinstance(referer, str):
            extra_headers["HTTP-Referer"] = referer
        if app_name and isinstance(app_name, str):
            extra_headers["X-Title"] = app_name

        results: list[Responsedict] = []
        for base64_image in processed_images_str:
            try:
                self._wait_for_rate_limit()

                temperature_val = config_registry.get(self.model_name, "temperature", default=0.7)
                temperature = float(temperature_val) if temperature_val is not None else 0.7

                max_tokens_val = config_registry.get(self.model_name, "max_output_tokens", default=1800)
                max_tokens = int(max_tokens_val) if max_tokens_val is not None else 1800

                timeout_val = config_registry.get(self.model_name, "timeout", default=120)
                timeout = float(timeout_val) if timeout_val is not None else 60.0

                json_schema_supported = config_registry.get(self.model_name, "json_schema_supported", default=False)

                if json_schema_supported:
                    response = self._call_openrouter_with_json_schema(
                        base64_image, temperature, max_tokens, timeout, extra_headers
                    )
                else:
                    response = self._call_openrouter_without_json_schema(
                        base64_image, temperature, max_tokens, timeout, extra_headers
                    )

                # OpenAI/RouterのレスポンスからAnnotationSchemaを生成
                annotation = None
                error = None
                try:
                    # OpenRouterはOpenAI互換なのでchoices[0].message.contentをパース
                    if hasattr(response, "choices") and response.choices:
                        content_text = response.choices[0].message.content if response.choices[0].message else None
                        if content_text:
                            # コードブロック除去
                            if content_text.startswith("```json") and "```" in content_text:
                                json_start = content_text.find("\n", content_text.find("```json")) + 1
                                json_end = content_text.rfind("```")
                                if json_start > 0 and json_end > json_start:
                                    content_text = content_text[json_start:json_end].strip()
                            import json
                            annotation = AnnotationSchema(**json.loads(content_text))
                        else:
                            error = "OpenRouter: メッセージコンテンツが空です"
                    else:
                        error = f"OpenRouter: choicesが空または無効です ({type(response)})"
                except Exception as e:
                    logger.error(f"OpenRouter: AnnotationSchema変換失敗: {e}", exc_info=True)
                    error = f"OpenRouter: AnnotationSchema変換失敗: {e}"
                results.append({"response": annotation, "error": error})
            except (APIConnectionError, WebApiError) as e:
                error_message = str(e)
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except Exception as e:
                logger.error(f"予期しないエラー: {e}", exc_info=True)
                results.append({"response": None, "error": f"Unexpected error: {e}"})

        return results

    def _handle_inference_exception(self, e: Exception, results: list[Responsedict]):
        try:
            self._handle_api_error(e)
        except WebApiError as api_e:
            results.append({"response": None, "error": str(api_e)})

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        formatted_outputs = []
        for output in raw_outputs:
            error = output.get("error")
            if error:
                formatted_outputs.append(FormattedOutput(annotation=None, error=error))
                continue
            response_val = output.get("response")
            if not isinstance(response_val, AnnotationSchema):
                formatted_outputs.append(FormattedOutput(annotation=None, error=f"OpenRouter: Invalid response type: {type(response_val)}"))
                continue
            formatted_outputs.append(FormattedOutput(annotation=response_val, error=None))
        return formatted_outputs

    def _call_openrouter_with_json_schema(
        self, base64_image: str, temperature: float, max_tokens: int, timeout: float, extra_headers: dict[str, str]
    ):
        response_format = {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        }
        return self._call_openrouter_api(
            base64_image, temperature, max_tokens, timeout, response_format, extra_headers
        )

    def _call_openrouter_without_json_schema(
        self, base64_image: str, temperature: float, max_tokens: int, timeout: float, extra_headers: dict[str, str]
    ):
        return self._call_openrouter_api(
            base64_image, temperature, max_tokens, timeout, NOT_GIVEN, extra_headers
        )

    def _call_openrouter_api(
        self, base64_image, temperature, max_tokens, timeout, response_format, extra_headers
    ):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{base64_image}"},
                    },
                ],
            },
        ]
        return self.client.chat.completions.create(
            model=self.api_model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
            extra_headers=extra_headers,
        )
