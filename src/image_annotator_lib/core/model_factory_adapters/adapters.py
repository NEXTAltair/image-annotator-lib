"""
API Adapter classes for different providers (OpenAI, Anthropic, Google).

各プロバイダーのAPIクライアントをラップし、統一されたインターフェースを提供します。
"""

import io
import json
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import ToolChoiceParam, ToolParam
from google import genai
from google.genai import types
from openai import OpenAI
from openai.types.chat.chat_completion_named_tool_choice_param import ChatCompletionNamedToolChoiceParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from PIL import Image
from pydantic import ValidationError

from ...exceptions.errors import WebApiError
from ...model_class.annotator_webapi import webapi_shared
from ..types import AnnotationSchema, ApiClient, WebApiInput
from ..utils import logger


class OpenAIAdapter(ApiClient):
    def __init__(self, client: OpenAI, system_prompt: str | None = None, base_prompt: str | None = None):
        self._client = client
        self._system_prompt = system_prompt if system_prompt is not None else webapi_shared.SYSTEM_PROMPT
        self._base_prompt = base_prompt if base_prompt is not None else webapi_shared.BASE_PROMPT

    def call_api(
        self, model_id: str, web_api_input: WebApiInput, params: dict[str, Any]
    ) -> AnnotationSchema:
        logger.debug(f"OpenAIAdapter.call_api called with model_id: {model_id}, params: {params}")

        user_prompt_from_params = params.get("prompt")

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        final_user_text_prompt = user_prompt_from_params

        if web_api_input.image_b64:
            content_parts: list[dict[str, Any]] = []
            if final_user_text_prompt:
                content_parts.append({"type": "text", "text": final_user_text_prompt})

            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{web_api_input.image_b64}"},
                }
            )
            if content_parts:
                messages.append({"role": "user", "content": content_parts})

        elif final_user_text_prompt:
            messages.append({"role": "user", "content": final_user_text_prompt})

        if not messages or not any(msg.get("role") == "user" for msg in messages):
            if not web_api_input.image_b64 and not final_user_text_prompt:
                raise ValueError("User message requires either an image or a text prompt.")
            raise ValueError("User message is missing from the request.")

        try:
            response_format_param = params.get("response_format")
            tools_list: list[ChatCompletionToolParam] | None = None
            tool_choice_option: ChatCompletionNamedToolChoiceParam | None = None

            if (
                response_format_param
                and isinstance(response_format_param, dict)
                and response_format_param.get("type") == "json_object"
            ):
                if hasattr(AnnotationSchema, "model_json_schema"):
                    function_schema_dict = AnnotationSchema.model_json_schema()
                    if "name" not in function_schema_dict or "parameters" not in function_schema_dict:
                        logger.warning(
                            "AnnotationSchema.model_json_schema() does not return a valid FunctionDefinition structure."
                        )
                    else:
                        _fn_name = function_schema_dict.get("title", AnnotationSchema.__name__)
                        _fn_description = function_schema_dict.get("description")

                        _function_definition_payload: dict[str, Any] = {
                            "name": _fn_name,
                            "parameters": function_schema_dict,
                        }
                        if _fn_description:
                            _function_definition_payload["description"] = _fn_description

                        tools_list = [
                            {
                                "type": "function",
                                "function": cast(FunctionDefinition, _function_definition_payload),
                            }
                        ]
                        tool_choice_option = {"type": "function", "function": {"name": _fn_name}}

            completion_params: dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "max_tokens": params.get("max_output_tokens", 1800),
                "temperature": params.get("temperature", 0.7),
            }
            if tools_list:
                completion_params["tools"] = tools_list
            if tool_choice_option:
                completion_params["tool_choice"] = tool_choice_option

            completion = self._client.chat.completions.create(**completion_params)

            if completion.choices and completion.choices[0].message:
                message_content = completion.choices[0].message.content
                if message_content:
                    return AnnotationSchema.model_validate_json(message_content)
                elif completion.choices[0].message.tool_calls:
                    tool_call = completion.choices[0].message.tool_calls[0]
                    if tool_call.function.name == (function_schema_dict or {}).get("name"):
                        return AnnotationSchema.model_validate_json(tool_call.function.arguments)
            raise ValueError("OpenAI APIからのレスポンス形式が不正です。")
        except Exception as e:
            logger.error(f"OpenAIAdapter API呼び出しエラー: {e}")
            raise


class AnthropicAdapter(ApiClient):
    def __init__(self, client: Anthropic):
        self._client = client

    def call_api(
        self, model_id: str, web_api_input: WebApiInput, params: dict[str, Any]
    ) -> AnnotationSchema:
        logger.debug(f"AnthropicAdapter.call_api called with model_id: {model_id}, params: {params}")
        messages = []
        system_prompt = params.get("system_prompt", webapi_shared.SYSTEM_PROMPT)

        prompt = params.get("prompt", webapi_shared.BASE_PROMPT)

        if web_api_input.image_b64:
            user_content: list[dict[str, Any]] = []
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": web_api_input.image_b64,
                    },
                }
            )
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
        elif prompt:
            messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("User message requires either an image or a text prompt for Anthropic.")

        try:
            tools_list: list[ToolParam] | None = None
            tool_choice_option: ToolChoiceParam | None = None

            if hasattr(AnnotationSchema, "model_json_schema"):
                tool_name = "extract_image_annotations"
                tools_list = [
                    {
                        "name": tool_name,
                        "description": "Extracts tags, captions, and score from an image based on the provided schema.",
                        "input_schema": AnnotationSchema.model_json_schema(),
                    }
                ]
                tool_choice_option = {"type": "tool", "name": tool_name}

            message_create_params: dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "system": system_prompt,
                "max_tokens": params.get("max_output_tokens", 1800),
                "temperature": params.get("temperature", 0.7),
            }
            if tools_list:
                message_create_params["tools"] = tools_list
            if tool_choice_option:
                message_create_params["tool_choice"] = tool_choice_option

            response = self._client.messages.create(**message_create_params)

            if response.content:
                for block in response.content:
                    if block.type == "tool_use" and block.name == tool_name:
                        if isinstance(block.input, AnnotationSchema):
                            return block.input
                        elif isinstance(block.input, dict):
                            return AnnotationSchema.model_validate(block.input)
            raise ValueError(
                "Anthropic APIからのレスポンス形式が不正、またはtool_useブロックが見つかりません。"
            )
        except Exception as e:
            logger.error(f"AnthropicAdapter API呼び出しエラー: {e}")
            raise


class GoogleClientAdapter(ApiClient):
    def __init__(
        self, client: genai.Client, system_prompt: str | None = None, base_prompt: str | None = None
    ):
        self._client = client
        self._system_prompt = system_prompt if system_prompt is not None else webapi_shared.SYSTEM_PROMPT
        self._base_prompt = base_prompt if base_prompt is not None else webapi_shared.BASE_PROMPT

    def call_api(
        self, model_id: str, web_api_input: WebApiInput, params: dict[str, Any]
    ) -> AnnotationSchema:
        logger.debug(f"GoogleClientAdapter.call_api called with model_id: {model_id}, params: {params}")

        image_bytes_for_call: bytes | None = None
        if web_api_input.image_bytes:
            image_bytes_for_call = web_api_input.image_bytes
        elif web_api_input.image_b64:
            import base64

            try:
                image_bytes_for_call = base64.b64decode(web_api_input.image_b64)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image for Google API: {e}") from e

        final_text_prompt = params.get("prompt", self._base_prompt)

        contents: list[Any] = []
        if final_text_prompt:
            contents.append(final_text_prompt)
        if image_bytes_for_call:
            try:
                pil_image = Image.open(io.BytesIO(image_bytes_for_call))
                contents.append(pil_image)
            except Exception as img_e:
                raise ValueError(f"Failed to process image bytes for Google API: {img_e}") from img_e

        if not contents:
            raise ValueError("Google API call requires either text prompt or image data.")

        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)
        top_k = params.get("top_k", 32)
        max_output_tokens = params.get("max_output_tokens", 1800)

        gen_config_params: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        if self._system_prompt:
            gen_config_params["system_instruction"] = self._system_prompt

        generation_config = types.GenerateContentConfig(**gen_config_params)

        try:
            # client.models.generate_content を使用
            response = self._client.models.generate_content(
                model=model_id,  # モデル名を文字列で渡す
                contents=contents,
                config=generation_config,  # "generation_config" を "config" に修正
                # request_options=types.RequestOptions(timeout=...) # タイムアウト設定の方法を確認
            )

            if response and hasattr(response, "text") and response.text:
                try:
                    return AnnotationSchema.model_validate_json(response.text)
                except (json.JSONDecodeError, ValidationError) as e_text:
                    logger.warning(
                        f"Failed to validate response.text as AnnotationSchema (json): {e_text}. Content: {response.text[:200]}"
                    )
                    raise ValueError(
                        f"Google API response (from response.text) could not be parsed into AnnotationSchema: {e_text}. Content: {response.text[:200]}"
                    ) from e_text

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    part_text = candidate.content.parts[0].text
                    if part_text:
                        try:
                            return AnnotationSchema.model_validate_json(part_text)
                        except (json.JSONDecodeError, ValidationError) as e_part_text:
                            raise ValueError(
                                f"Google API response (from candidate part.text) could not be parsed into AnnotationSchema: {e_part_text}. Content: {part_text[:200]}"
                            ) from e_part_text

            error_detail = f"Response type: {type(response)}, Content: {str(response)[:200]}"
            if hasattr(response, "prompt_feedback"):
                pass

            raise ValueError(f"Google APIからのレスポンス形式が不正、または内容が空です。{error_detail}")

        except Exception as e:  # より具体的なエラー型をSDKから見つけられれば置き換える (例: genai.APIError)
            # 現状、types モジュールに GoogleAPIError のような具体的なエラー型は見当たらない。
            # google.api_core.exceptions も google-genai SDK で使われるかは不明。
            logger.error(f"GoogleClientAdapter API呼び出し中に予期せぬエラー: {e}")
            # WebApiError に元の例外を渡す (original_exception パラメータは削除し、from e でチェイン)
            raise WebApiError(
                f"Unexpected error during Google API call: {e}", provider_name="Google"
            ) from e
