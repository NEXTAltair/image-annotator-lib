"""WebAPIを利用したマルチモーダルアノテーター実装

このモジュールでは、外部のWebAPI(Google Gemini、OpenAI、Anthropic、OpenRouter)を
利用して画像アノテーションを行う具象クラスを提供します。
"""

import os
from typing import Any, Self, TypedDict

import anthropic
import openai
from dotenv import load_dotenv
from google import genai
from PIL import Image

from ..core.base import WebApiBaseAnnotator
from ..core.config import config_registry
from ..core.utils import logger
from ..exceptions.errors import ApiKeyMissingError, WebApiError


class Responsedict(TypedDict, total=False):
    """API応答を格納する辞書型"""

    response: (
        genai.types.GenerateContentResponse
        | openai.types.chat.ChatCompletion
        | anthropic.types.Message
        | dict[str, Any]  # OpenRouter は dict で返す想定
        | None
    )
    error: str | None


class FormattedOutput(TypedDict):
    """フォーマット済み出力を格納する辞書型"""

    annotation: dict[str, Any] | None  # {"tags": list[str], "captions": list[str], "score": float}
    error: str | None


BASE_PROMPT = """As an AI assistant specializing in image analysis, analyze images with particular attention to:
                    Character Details (if present):

                    Facing direction (left, right, front, back, three-quarter view)

                    Action or pose (standing, sitting, walking, etc.)

                    Hand positions and gestures

                    Gaze direction

                    Clothing details from top to bottom

                    Composition Elements:

                    Main subject position

                    Background elements and their placement

                    Lighting direction and effects

                    Color scheme and contrast

                    Depth and perspective

                    Technical Aspects and Scoring (1.00 to 10.00):

                    Score images based on these criteria:

                    Technical Quality (0-3 points):

                    Image clarity and resolution

                    Line quality and consistency

                    Color balance and harmony

                    Composition (0-3 points):

                    Layout and framing

                    Use of space

                    Balance of elements

                    Artistic Merit (0-4 points):

                    Creativity and originality

                    Emotional impact

                    Detail and complexity

                    Style execution

                    Examples of scoring:

                    9.50-10.00: Exceptional quality in all aspects

                    8.50-9.49: Excellent quality with minor imperfections

                    7.50-8.49: Very good quality with some room for improvement

                    6.50-7.49: Good quality with notable areas for improvement

                    5.50-6.49: Average quality with significant room for improvement

                    Below 5.50: Below average quality with major issues

                    Format score as a decimal with exactly two decimal places (e.g., 7.25, 8.90, 6.75)

                    Provide annotations in this exact format only:

                    tags: [30-50 comma-separated words identifying the above elements, maintaining left/right distinction]

                    caption: [Single 1-2 sentence objective description, explicitly noting direction and positioning]

                    score: [Single decimal number between 1.00 and 10.00, using exactly two decimal places]

                    Important formatting rules:

                    Use exactly these three sections in this order: tags, caption, score

                    Format score as a decimal number with exactly two decimal places (e.g., 8.50)

                    Do not add any additional text or commentary

                    Do not add any text after the score

                    Use standard tag conventions without underscores (e.g., \"blonde hair\" not \"blonde_hair\")

                    Always specify left/right orientation for poses, gazes, and positioning

                    Be precise about viewing angles and directions

                    Example output:
                    tags: 1girl, facing right, three quarter view, blonde hair, blue eyes, school uniform, sitting, right hand holding pencil, left hand on desk, looking down at textbook, classroom, desk, study materials, natural lighting from left window, serious expression, detailed background, realistic style

                    caption: A young student faces right in three-quarter view, sitting at a desk with her right hand holding a pencil while her left hand rests on the desk, looking down intently at a textbook in a sunlit classroom.

                    score: 5.50
                """

SYSTEM_PROMPT = """
                    You are an AI that MUST output ONLY valid JSON, with no additional text, markdown formatting, or explanations.

                    Output Structure:
                    {
                        "tags": ["tag1", "tag2", "tag3", ...],  // List of tags describing image features (max 150 tokens)
                        "captions": ["caption1", "caption2", ...],  // List of short descriptions explaining the image content (max 75 tokens)
                        "score": 0.85  // Quality evaluation of the image (decimal value between 0.0 and 1.0)
                    }

                    Rules:
                    1. ONLY output the JSON object - no other text or formatting
                    2. DO NOT use markdown code blocks (```) or any other formatting
                    3. DO NOT include any explanations or comments
                    4. Always return complete, valid, parseable JSON
                    5. Include all required fields: tags, captions, and score
                    6. Never truncate or leave incomplete JSON
                    7. DO NOT add any leading or trailing whitespace or newlines
                    8. DO NOT start with any introductory text like "Here is the analysis:"

                    Example of EXACT expected output format:
                    {"tags":["1girl","red_hair"],"captions":["A girl with long red hair"],"score":0.95}
                """

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "captions": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["tags", "captions", "score"],
    "propertyOrdering": ["tags", "captions", "score"],
}


class GoogleApiAnnotator(WebApiBaseAnnotator):
    """Google Gemini API を使用するアノテーター"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名
        """
        self.provider_name = "Google Gemini"
        self.client: genai.Client | None = None
        super().__init__(model_name)
        if self.model_name_on_provider is None:
            self.model_name_on_provider = "gemini-1.5-pro-latest"

    def __enter__(self) -> Self:
        """APIクライアントを初期化し、環境をセットアップする"""

        # クライアント初期化
        self.client = genai.Client(api_key=self.api_key)

        if self.client:
            return self
        else:
            raise WebApiError("Google GenAI Client の初期化に失敗しました", self.provider_name)

    def _load_api_key(self) -> str:
        """Google API キーを環境変数から読み込む"""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ApiKeyMissingError("GOOGLE_API_KEY", self.provider_name)
        return api_key

    def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
        """画像リストをバイトデータのリストに変換する"""
        from io import BytesIO

        encoded_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            encoded_images.append(buffered.getvalue())
        return encoded_images

    def _run_inference(self, processed_images: list[bytes]) -> list[Responsedict]:
        """Google Gemini API (クライアントベースSDK) を使用して推論を実行する"""
        if not self.client:
            raise WebApiError("API クライアントが初期化されていません", provider_name=self.provider_name)

        results: list[Responsedict] = []
        for image_data in processed_images:
            try:
                self._wait_for_rate_limit()

                # リクエスト準備
                content = genai.types.Content(
                    parts=[
                        genai.types.Part.from_text(text=BASE_PROMPT),
                        genai.types.Part.from_bytes(data=image_data, mime_type="image/webp"),
                    ],
                    role="user",
                )

                # システムインストラクション
                system_instruction_text = SYSTEM_PROMPT

                # JSON出力スキーマを定義
                response_schema = JSON_SCHEMA

                # 最新のクライアントベースAPIを使用
                response = self.client.models.generate_content(
                    model=self.model_name_on_provider,
                    contents=[content],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction_text,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                        temperature=0.4,
                        response_schema=response_schema,
                    ),
                )
                results.append({"response": response, "error": None})

            except Exception as e:
                try:
                    self._handle_api_error(e)  # 基底クラスのエラーハンドリングを試す
                except Exception as handled_e:
                    error_message = str(handled_e)
                    results.append({"error": error_message})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """Google Gemini API (google-genai SDK) からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"):
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"]))
                continue

            response = output.get("response")
            if response is None or not hasattr(response, "candidates") or not response.candidates:
                formatted_outputs.append(
                    FormattedOutput(annotation=None, error="応答コンテンツが無効または欠落している")
                )
                continue

            try:
                # 応答からテキストコンテンツを抽出
                text_content = response.candidates[0].content.parts[0].text
                # 共通ヘルパーメソッドを呼び出して解析
                formatted = self._parse_common_json_response(text_content)
                formatted_outputs.append(formatted)
            except (AttributeError, IndexError, TypeError) as e:
                error_message = f"Google Gemini API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
            except Exception as e:
                # _parse_common_json_response 内で処理されない予期せぬエラー
                error_message = f"Google Gemini応答のフォーマットの予期しないエラー: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))

        return formatted_outputs

    def _generate_tags(self, formatted_output: FormattedOutput) -> list[str]:
        """フォーマット済み出力からタグを生成する"""
        if formatted_output["error"] or formatted_output["annotation"] is None:
            return []

        annotation = formatted_output["annotation"]
        if "tags" in annotation and isinstance(annotation["tags"], list):
            return annotation["tags"]

        return []


class OpenAIApiAnnotator(WebApiBaseAnnotator):
    """OpenAI GPT-4o API を使用するアノテーター"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名
        """
        self.provider_name = "OpenAI"
        self.client: openai.OpenAI | None = None
        super().__init__(model_name)
        if self.model_name_on_provider is None:
            self.model_name_on_provider = "gpt-4o"

    def __enter__(self) -> Self:
        self.client = openai.OpenAI(api_key=self.api_key)
        return self

    def _load_api_key(self) -> str:
        """OpenAI API キーを環境変数から読み込む"""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ApiKeyMissingError("OPENAI_API_KEY", self.provider_name)
        return api_key

    def _run_inference(self, processed_images: list[str]) -> list[Responsedict]:
        """OpenAI API を使用して推論を実行する"""
        if not self.client:
            raise WebApiError("API クライアントが初期化されていません", provider_name=self.provider_name)

        results: list[Responsedict] = []
        for image_data in processed_images:
            try:
                self._wait_for_rate_limit()

                input = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": BASE_PROMPT + "\nEnsure the response is a valid JSON object.",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/webp;base64,{image_data}",
                            },
                        ],
                    },
                ]
                text = {
                    "format": {
                        "type": "json_schema",
                        "name": "Annotatejson",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "description": "タグの配列",
                                    "items": {"type": "string"},
                                },
                                "captions": {
                                    "type": "array",
                                    "description": "キャプションの配列",
                                    "items": {"type": "string"},
                                },
                                "score": {"type": "number", "description": "スコア(数値)"},
                            },
                            "required": ["tags", "captions", "score"],
                            "additionalProperties": False,
                        },
                    }
                }
                try:
                    timeout_float = float(self.timeout) if self.timeout is not None else 60.0
                except (ValueError, TypeError):
                    logger.warning(f"無効なタイムアウト値 '{self.timeout}', デフォルト60.0を使用します")
                    timeout_float = 60.0

                # model_name_on_providerがNoneでないことを表明
                assert self.model_name_on_provider is not None, (
                    "プロバイダーのモデル名を設定する必要があります"
                )
                response = self.client.responses.create(
                    model=self.model_name_on_provider,
                    input=input,
                    text=text,
                    temperature=1,
                    max_output_tokens=self.max_output_tokens,
                    timeout=timeout_float,  # Use the validated float value
                )
                results.append({"response": response, "error": None})
            except Exception as e:
                try:
                    self._handle_api_error(e)
                except Exception as handled_e:
                    error_message = str(handled_e)
                    results.append({"response": None, "error": error_message})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """OpenAI API からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"):
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"]))
                continue

            response = output.get("response")
            if response is None:
                formatted_outputs.append(FormattedOutput(annotation=None, error="応答が空です"))
                continue

            try:
                # 新しいAPIの応答形式から出力テキストを取得
                text_content = response.output_text
                # 共通ヘルパーメソッドを呼び出して解析
                formatted = self._parse_common_json_response(text_content)
                formatted_outputs.append(formatted)
            except AttributeError as e:
                error_message = f"OpenAI API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
            except Exception as e:
                error_message = f"OpenAI応答のフォーマットで予期しないエラー: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))

        return formatted_outputs

    def _generate_tags(self, formatted_output: FormattedOutput) -> list[str]:
        """フォーマット済み出力からタグを生成する"""
        if formatted_output["error"] or formatted_output["annotation"] is None:
            return []

        annotation = formatted_output["annotation"]
        # Extract tags directly from the parsed annotation data
        if "tags" in annotation and isinstance(annotation["tags"], list):
            return annotation["tags"]
        elif "tags" in annotation and isinstance(annotation["tags"], str):
            # Handle case where tags might be a single comma-separated string
            return [tag.strip() for tag in annotation["tags"].split(",")]

        return []


class AnthropicApiAnnotator(WebApiBaseAnnotator):
    """Anthropic Claude API を使用するアノテーター"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名
        """
        self.provider_name = "Anthropic"
        self.client: anthropic.Anthropic | None = None
        super().__init__(model_name)
        if self.model_name_on_provider is None:
            self.model_name_on_provider = "claude-3-opus-20240229"

    def __enter__(self) -> Self:
        self.client = anthropic.Anthropic(api_key=self.api_key)
        return self

    def _load_api_key(self) -> str:
        """Anthropic API キーを環境変数から読み込む"""
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ApiKeyMissingError("ANTHROPIC_API_KEY", self.provider_name)
        return api_key

    def _run_inference(self, processed_images: list[str]) -> list[Responsedict]:
        """Anthropic API を使用して推論を実行する"""
        if not self.client:
            raise WebApiError("API クライアントが初期化されていません", provider_name=self.provider_name)

        results: list[Responsedict] = []
        for encoded_image in processed_images:
            try:
                self._wait_for_rate_limit()

                assert self.model_name_on_provider is not None, "Model name on provider is not set"
                messages: list[anthropic.types.MessageParam] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": BASE_PROMPT},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": encoded_image,
                                },
                            },
                        ],
                    },
                ]
                tools = [
                    {
                        "type": "custom",
                        "name": "Annotatejson",
                        "description": "Parsing image annotation results to JSON",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "captions": {"type": "array", "items": {"type": "string"}},
                                "score": {"type": "number"},
                            },
                            "required": ["tags", "captions", "score"],
                        },
                    }
                ]
                system_prompt = SYSTEM_PROMPT
                response = self.client.messages.create(
                    model=self.model_name_on_provider,
                    max_tokens=self.max_output_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                )
                results.append({"response": response, "error": None})
            except Exception as e:
                try:
                    self._handle_api_error(e)
                except Exception as handled_e:
                    error_message = str(handled_e)
                    results.append({"response": None, "error": error_message})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """Anthropic API からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"):
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"]))
                continue

            response: anthropic.types.Message | None = output.get("response")
            if (
                response is None
                or not response.content
                or not isinstance(response.content[0], anthropic.types.ToolUseBlock)
            ):
                stop_reason = response.stop_reason if response and response.stop_reason else "unknown"
                error_msg = f"応答コンテンツが無効または欠落している。: {stop_reason}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_msg))
                continue

            try:
                # 応答からtoolで作成されたJSONを抽出
                dict_content = output["response"].content[0].input
                # 共通ヘルパーメソッドを呼び出して解析
                formatted = self._parse_common_json_response(dict_content)
                formatted_outputs.append(formatted)
            except (AttributeError, IndexError, TypeError) as e:
                error_message = f"Anthropic API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
            except Exception as e:
                error_message = f"Anthropic応答をフォーマットする予期しないエラー: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))

        return formatted_outputs

    def _generate_tags(self, formatted_output: FormattedOutput) -> list[str]:
        """フォーマット済み出力からタグを生成する"""
        if formatted_output["error"] or formatted_output["annotation"] is None:
            return []

        annotation = formatted_output["annotation"]
        # annotation 辞書から tags を抽出
        if "tags" in annotation and isinstance(annotation["tags"], list):
            return annotation["tags"]
        elif "tags" in annotation and isinstance(annotation["tags"], str):
            return [tag.strip() for tag in annotation["tags"].split(",")]

        return []


class OpenRouterApiAnnotator(WebApiBaseAnnotator):
    """OpenRouter API を使用するアノテーター (OpenAI互換APIクライアント)

    OpenAIApiAnnotator を継承し、APIエンドポイントと認証情報を OpenRouter 用に変更します。
    基本的な処理フロー (_run_inference, _format_predictions, _generate_tags) は
    OpenAIApiAnnotator の実装を再利用します。
    """

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名
        """
        self.provider_name = "OpenRouter"
        self.api_base = config_registry.get(
            model_name,
            "api_endpoint",
            "https://openrouter.ai/api/v1",
        )
        super().__init__(model_name)
        if self.model_name_on_provider is None:
            self.model_name_on_provider = "anthropic/claude-3-opus"

    def __enter__(self) -> Self:
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        if self.client:
            return self
        else:
            raise WebApiError(
                f"{self.provider_name} クライアントの初期化に失敗しました", self.provider_name
            )

    def _load_api_key(self) -> str:
        """OpenRouter API キーを環境変数から読み込む (オーバーライド)"""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ApiKeyMissingError("OPENROUTER_API_KEY", self.provider_name)
        return api_key

    def _run_inference(self, processed_images: list[str]) -> list[Responsedict]:
        """OpenRouter API を使用して推論を実行する"""
        if not self.client:
            raise WebApiError("API クライアントが初期化されていません", provider_name=self.provider_name)

        results: list[Responsedict] = []
        for encoded_image in processed_images:
            try:
                self._wait_for_rate_limit()

                assert self.model_name_on_provider is not None, (
                    "プロバイダーのモデル名を設定する必要があります"
                )

                # OpenRouter用のリクエスト形式
                response = self.client.chat.completions.create(
                    model=self.model_name_on_provider,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": BASE_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/webp;base64,{encoded_image}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=self.max_output_tokens,
                )
                results.append({"response": response, "error": None})
            except Exception as e:
                try:
                    self._handle_api_error(e)
                except Exception as handled_e:
                    error_message = str(handled_e)
                    results.append({"response": None, "error": error_message})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """OpenRouter API からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"):
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"]))
                continue

            response = output.get("response")
            if not response or not hasattr(response, "choices"):
                formatted_outputs.append(FormattedOutput(annotation=None, error="応答が空または無効です"))
                continue

            try:
                # OpenRouterのChoice型からメッセージコンテンツを取得
                choice = response.choices[0]
                if not choice.message or not choice.message.content:
                    formatted_outputs.append(
                        FormattedOutput(annotation=None, error="メッセージコンテンツが空です")
                    )
                    continue

                # メッセージコンテンツから直接JSONを解析
                formatted = self._parse_common_json_response(choice.message.content)
                formatted_outputs.append(formatted)
            except (AttributeError, IndexError) as e:
                error_message = f"OpenRouter API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
            except Exception as e:
                error_message = f"OpenRouter応答のフォーマットで予期しないエラー: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))

        return formatted_outputs
