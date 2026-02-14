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
        """OpenAI Chat Completions APIを呼び出してアノテーション結果を取得する。

        Args:
            model_id: 使用するモデルID。
            web_api_input: 画像データとプロンプトを含む入力。
            params: APIパラメータ（prompt, temperature, max_output_tokens等）。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: レスポンス形式が不正な場合。
        """
        logger.debug(f"OpenAIAdapter.call_api called with model_id: {model_id}, params: {params}")

        messages = self._build_messages(web_api_input, params.get("prompt"))
        tools_list, tool_choice_option = self._build_tools(params)
        completion_params = self._build_completion_params(model_id, messages, params, tools_list, tool_choice_option)

        try:
            completion = self._client.chat.completions.create(**completion_params)
            return self._parse_response(completion, tools_list)
        except Exception as e:
            logger.error(f"OpenAIAdapter API呼び出しエラー: {e}")
            raise

    def _build_messages(
        self, web_api_input: WebApiInput, user_prompt: str | None
    ) -> list[dict[str, Any]]:
        """OpenAI形式のメッセージリストを構築する。

        Args:
            web_api_input: 画像データを含む入力。
            user_prompt: ユーザーからのテキストプロンプト。

        Returns:
            Chat Completions APIに渡すmessagesリスト。

        Raises:
            ValueError: 画像もテキストプロンプトも無い場合。
        """
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        if web_api_input.image_b64:
            content_parts: list[dict[str, Any]] = []
            if user_prompt:
                content_parts.append({"type": "text", "text": user_prompt})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{web_api_input.image_b64}"},
                }
            )
            if content_parts:
                messages.append({"role": "user", "content": content_parts})
        elif user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        if not messages or not any(msg.get("role") == "user" for msg in messages):
            if not web_api_input.image_b64 and not user_prompt:
                raise ValueError("User message requires either an image or a text prompt.")
            raise ValueError("User message is missing from the request.")
        return messages

    def _build_tools(
        self, params: dict[str, Any]
    ) -> tuple[list[ChatCompletionToolParam] | None, ChatCompletionNamedToolChoiceParam | None]:
        """OpenAI function calling用のtool定義を構築する。

        response_formatがjson_objectの場合にAnnotationSchemaからtool定義を生成する。
        model_json_schema()はPydantic標準の{"title", "type", "properties", ...}形式を返すため、
        titleをfunction名、スキーマ全体をparametersとして使用する。

        Args:
            params: response_formatを含むAPIパラメータ。

        Returns:
            (tools_list, tool_choice_option) のタプル。
        """
        response_format_param = params.get("response_format")
        if not (
            response_format_param
            and isinstance(response_format_param, dict)
            and response_format_param.get("type") == "json_object"
            and hasattr(AnnotationSchema, "model_json_schema")
        ):
            return None, None

        function_schema_dict = AnnotationSchema.model_json_schema()

        fn_name = function_schema_dict.get("title", AnnotationSchema.__name__)
        fn_description = function_schema_dict.get("description")

        function_definition_payload: dict[str, Any] = {
            "name": fn_name,
            "parameters": function_schema_dict,
        }
        if fn_description:
            function_definition_payload["description"] = fn_description

        tools_list: list[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": cast(FunctionDefinition, function_definition_payload),
            }
        ]
        tool_choice_option: ChatCompletionNamedToolChoiceParam = {
            "type": "function",
            "function": {"name": fn_name},
        }
        return tools_list, tool_choice_option

    @staticmethod
    def _build_completion_params(
        model_id: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        tools_list: list[ChatCompletionToolParam] | None,
        tool_choice_option: ChatCompletionNamedToolChoiceParam | None,
    ) -> dict[str, Any]:
        """Chat Completions API呼び出しパラメータを構築する。

        Args:
            model_id: モデルID。
            messages: メッセージリスト。
            params: 元のAPIパラメータ。
            tools_list: ツール定義（オプション）。
            tool_choice_option: ツール選択指定（オプション）。

        Returns:
            API呼び出し用のパラメータ辞書。
        """
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
        return completion_params

    @staticmethod
    def _parse_response(
        completion: Any, tools_list: list[ChatCompletionToolParam] | None
    ) -> AnnotationSchema:
        """OpenAI APIレスポンスをパースしてAnnotationSchemaを返す。

        Args:
            completion: Chat Completions APIのレスポンス。
            tools_list: ツール定義（tool_callのname照合用）。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: レスポンス形式が不正な場合。
        """
        if completion.choices and completion.choices[0].message:
            message_content = completion.choices[0].message.content
            if message_content:
                return AnnotationSchema.model_validate_json(message_content)
            elif completion.choices[0].message.tool_calls:
                tool_call = completion.choices[0].message.tool_calls[0]
                # tools_listからfunction_schema_dictのnameを取得
                expected_name = None
                if tools_list:
                    expected_name = tools_list[0].get("function", {}).get("name")
                if expected_name and tool_call.function.name == expected_name:
                    return AnnotationSchema.model_validate_json(tool_call.function.arguments)
        raise ValueError("OpenAI APIからのレスポンス形式が不正です。")


class AnthropicAdapter(ApiClient):
    def __init__(self, client: Anthropic):
        self._client = client

    def call_api(
        self, model_id: str, web_api_input: WebApiInput, params: dict[str, Any]
    ) -> AnnotationSchema:
        """Anthropic Messages APIを呼び出してアノテーション結果を取得する。

        Args:
            model_id: 使用するモデルID。
            web_api_input: 画像データとプロンプトを含む入力。
            params: APIパラメータ（prompt, system_prompt, temperature等）。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: レスポンス形式が不正な場合。
        """
        logger.debug(f"AnthropicAdapter.call_api called with model_id: {model_id}, params: {params}")

        system_prompt = params.get("system_prompt", webapi_shared.SYSTEM_PROMPT)
        prompt = params.get("prompt", webapi_shared.BASE_PROMPT)
        messages = self._build_messages(web_api_input, prompt)
        tool_name, tools_list, tool_choice_option = self._build_tools()
        message_params = self._build_message_params(
            model_id, messages, system_prompt, params, tools_list, tool_choice_option
        )

        try:
            response = self._client.messages.create(**message_params)
            return self._parse_response(response, tool_name)
        except Exception as e:
            logger.error(f"AnthropicAdapter API呼び出しエラー: {e}")
            raise

    @staticmethod
    def _build_messages(web_api_input: WebApiInput, prompt: str | None) -> list[dict[str, Any]]:
        """Anthropic形式のメッセージリストを構築する。

        Args:
            web_api_input: 画像データを含む入力。
            prompt: テキストプロンプト。

        Returns:
            Messages APIに渡すmessagesリスト。

        Raises:
            ValueError: 画像もテキストプロンプトも無い場合。
        """
        messages: list[dict[str, Any]] = []
        if web_api_input.image_b64:
            user_content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": web_api_input.image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
            messages.append({"role": "user", "content": user_content})
        elif prompt:
            messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("User message requires either an image or a text prompt for Anthropic.")
        return messages

    @staticmethod
    def _build_tools() -> tuple[str, list[ToolParam] | None, ToolChoiceParam | None]:
        """Anthropic tool_use用のツール定義を構築する。

        Returns:
            (tool_name, tools_list, tool_choice_option) のタプル。
        """
        if not hasattr(AnnotationSchema, "model_json_schema"):
            return "extract_image_annotations", None, None

        tool_name = "extract_image_annotations"
        tools_list: list[ToolParam] = [
            {
                "name": tool_name,
                "description": "Extracts tags, captions, and score from an image based on the provided schema.",
                "input_schema": AnnotationSchema.model_json_schema(),
            }
        ]
        tool_choice_option: ToolChoiceParam = {"type": "tool", "name": tool_name}
        return tool_name, tools_list, tool_choice_option

    @staticmethod
    def _build_message_params(
        model_id: str,
        messages: list[dict[str, Any]],
        system_prompt: str,
        params: dict[str, Any],
        tools_list: list[ToolParam] | None,
        tool_choice_option: ToolChoiceParam | None,
    ) -> dict[str, Any]:
        """Messages API呼び出しパラメータを構築する。

        Args:
            model_id: モデルID。
            messages: メッセージリスト。
            system_prompt: システムプロンプト。
            params: 元のAPIパラメータ。
            tools_list: ツール定義（オプション）。
            tool_choice_option: ツール選択指定（オプション）。

        Returns:
            API呼び出し用のパラメータ辞書。
        """
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
        return message_create_params

    @staticmethod
    def _parse_response(response: Any, tool_name: str) -> AnnotationSchema:
        """Anthropic APIレスポンスをパースしてAnnotationSchemaを返す。

        Args:
            response: Messages APIのレスポンス。
            tool_name: 期待するツール名。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: レスポンス形式が不正な場合。
        """
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
        """Google Generative AI APIを呼び出してアノテーション結果を取得する。

        Args:
            model_id: 使用するモデルID。
            web_api_input: 画像データとプロンプトを含む入力。
            params: APIパラメータ（prompt, temperature, top_p, top_k等）。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: 入力データが不正な場合。
            WebApiError: API呼び出し中にエラーが発生した場合。
        """
        logger.debug(f"GoogleClientAdapter.call_api called with model_id: {model_id}, params: {params}")

        contents = self._build_contents(web_api_input, params.get("prompt", self._base_prompt))
        generation_config = self._build_generation_config(params)

        try:
            response = self._client.models.generate_content(
                model=model_id,
                contents=contents,
                config=generation_config,
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"GoogleClientAdapter API呼び出し中に予期せぬエラー: {e}")
            raise WebApiError(
                f"Unexpected error during Google API call: {e}", provider_name="Google"
            ) from e

    @staticmethod
    def _build_contents(web_api_input: WebApiInput, text_prompt: str | None) -> list[Any]:
        """Google API用のcontentsリストを構築する。

        Args:
            web_api_input: 画像データを含む入力。
            text_prompt: テキストプロンプト。

        Returns:
            generate_contentに渡すcontentsリスト。

        Raises:
            ValueError: base64デコード失敗、画像処理失敗、または入力が空の場合。
        """
        image_bytes: bytes | None = None
        if web_api_input.image_bytes:
            image_bytes = web_api_input.image_bytes
        elif web_api_input.image_b64:
            import base64

            try:
                image_bytes = base64.b64decode(web_api_input.image_b64)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image for Google API: {e}") from e

        contents: list[Any] = []
        if text_prompt:
            contents.append(text_prompt)
        if image_bytes:
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                contents.append(pil_image)
            except Exception as img_e:
                raise ValueError(f"Failed to process image bytes for Google API: {img_e}") from img_e

        if not contents:
            raise ValueError("Google API call requires either text prompt or image data.")
        return contents

    def _build_generation_config(self, params: dict[str, Any]) -> types.GenerateContentConfig:
        """Google API用の生成設定を構築する。

        Args:
            params: APIパラメータ（temperature, top_p, top_k, max_output_tokens）。

        Returns:
            GenerateContentConfig オブジェクト。
        """
        gen_config_params: dict[str, Any] = {
            "temperature": params.get("temperature", 0.7),
            "max_output_tokens": params.get("max_output_tokens", 1800),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", 32),
        }
        if self._system_prompt:
            gen_config_params["system_instruction"] = self._system_prompt
        return types.GenerateContentConfig(**gen_config_params)

    @staticmethod
    def _parse_response(response: Any) -> AnnotationSchema:
        """Google APIレスポンスをパースしてAnnotationSchemaを返す。

        Args:
            response: generate_contentのレスポンス。

        Returns:
            バリデーション済みのAnnotationSchema。

        Raises:
            ValueError: レスポンス形式が不正な場合。
        """
        # response.text から直接パース
        if response and hasattr(response, "text") and response.text:
            try:
                return AnnotationSchema.model_validate_json(response.text)
            except (json.JSONDecodeError, ValidationError) as e_text:
                logger.warning(
                    f"Failed to validate response.text as AnnotationSchema (json): {e_text}. "
                    f"Content: {response.text[:200]}"
                )
                raise ValueError(
                    f"Google API response (from response.text) could not be parsed into AnnotationSchema: "
                    f"{e_text}. Content: {response.text[:200]}"
                ) from e_text

        # candidates から取得を試みる
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                part_text = candidate.content.parts[0].text
                if part_text:
                    try:
                        return AnnotationSchema.model_validate_json(part_text)
                    except (json.JSONDecodeError, ValidationError) as e_part_text:
                        raise ValueError(
                            f"Google API response (from candidate part.text) could not be parsed "
                            f"into AnnotationSchema: {e_part_text}. Content: {part_text[:200]}"
                        ) from e_part_text

        error_detail = f"Response type: {type(response)}, Content: {str(response)[:200]}"
        raise ValueError(f"Google APIからのレスポンス形式が不正、または内容が空です。{error_detail}")
