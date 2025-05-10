"""WebAPIを利用したマルチモーダルアノテーター実装

このモジュールでは、外部のWebAPI(Google Gemini、OpenAI、Anthropic、OpenRouter)を
利用して画像アノテーションを行う具象クラスを提供します。
"""

from typing import Any, TypedDict, cast, override

import anthropic
import openai
from google.genai import types as google_types
from openai import APIConnectionError, OpenAI
from openai import types as openai_types
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types import chat as openai_chat_types
from PIL import Image
from pydantic import BaseModel, Field
import json
import requests

from image_annotator_lib.exceptions.errors import (
    ConfigurationError,
    WebApiError,
)

from ..core.base import WebApiAnnotationOutput, WebApiBaseAnnotator
from ..core.config import config_registry
from ..core.utils import logger

class Responsedict(TypedDict, total=False):
    """API応答を格納する辞書型
    # TODO: 具体的な内容の説明
    """

    response: (
        google_types.GenerateContentResponse
        | ChatCompletion
        | anthropic.types.Message
        | Any # OpenAIStructuredOutput.model_dump() を許容するために Any に変更
        | None
    )
    error: str | None


class FormattedOutput(TypedDict):
    """フォーマット済み出力を格納する辞書型
    # TODO: 具体的な内容の説明
    """

    annotation: dict[str, Any] | None
    error: str | None


class Google_Json_Schema(BaseModel):
    """Google API の応答を JSONクラスインスタンスにする

    example:
        my_Google_Json_Schema:
            [Google_Json_Schema(tags=['1girl', 'facing front', ...], captions=['A red-haired or ...', '...'], score=8.75)]
    """

    tags: list[str]
    captions: list[str]
    score: float


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

                    Use standard tag conventions without underscores (e.g., "blonde hair" not "blonde_hair")

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
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[Responsedict]:
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

        results: list[Responsedict] = []
        for image_data in processed_bytes:
            try:
                self._wait_for_rate_limit()

                # contentsをリストで渡す
                contents = [
                    {"text": BASE_PROMPT},
                    {"inline_data": {"mime_type": "image/webp", "data": image_data}},
                ]

                temperature = config_registry.get(self.model_name, "temperature", default=0.7)
                top_p = config_registry.get(self.model_name, "top_p", default=1.0)
                top_k = config_registry.get(self.model_name, "top_k", default=32)
                max_output_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)

                response_schema = Google_Json_Schema

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
                results.append({"response": response, "error": None})
            except json.JSONDecodeError as e:
                error_message = f"Google API: レスポンスJSONパース失敗: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except ValueError as e:
                error_message = f"Google API: パラメータ不正またはAPIキー未設定: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except RuntimeError as e:
                error_message = f"Google API: 実行時エラー: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except requests.exceptions.RequestException as e:
                error_message = f"Google API: 通信エラー: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except Exception as e:
                # 予期せぬエラーをログに記録し、エラー情報を結果に追加
                error_message = f"Google API エラー: 処理中に予期せぬエラーが発生しました: {str(e)}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})

        return results

    @override
    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """Google Gemini API (google-genai SDK) からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"): # type: ignore[typeddict-item]
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"])) # type: ignore[typeddict-item]
                continue

            response = output.get("response") # type: ignore[typeddict-item]
            if isinstance(response, google_types.GenerateContentResponse):
                parsed = getattr(response, "parsed", None)
                data = None
                if isinstance(parsed, Google_Json_Schema):
                    data = parsed
                elif isinstance(parsed, list) and parsed and isinstance(parsed[0], Google_Json_Schema):
                    data = parsed[0]

                if data is not None:
                    formatted_outputs.append(FormattedOutput(annotation=data.model_dump(), error=None))
                    continue

                # フォールバック: candidates.parts[0].textをパース
                try:
                    candidate = response.candidates[0] if response.candidates else None
                    part = candidate.content.parts[0] if candidate and candidate.content and candidate.content.parts else None
                    part_text = getattr(part, "text", None)
                    if not part_text or not part_text.strip():
                        formatted_outputs.append(FormattedOutput(annotation=None, error="Gemini API: candidates.parts[0].textが空文字列または無効です。プロンプトやAPI呼び出し内容を見直してください。"))
                        continue
                    json_data = json.loads(part_text)
                    schema_obj = Google_Json_Schema(**json_data)
                    formatted_outputs.append(FormattedOutput(annotation=schema_obj.model_dump(), error=None))
                    continue
                except Exception as e:
                    formatted_outputs.append(FormattedOutput(annotation=None, error=f"Gemini API: candidates.partsパース失敗: {e}\npart_text={part_text}"))
                    continue
            else:
                formatted_outputs.append(FormattedOutput(annotation=None, error="応答がGoogle GeminiのGenerateContentResponse型ではありません"))

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

    # Pydanticモデルの定義
    # OpenAI SDKの `client.responses.parse` を使用し、構造化された出力を得るために定義。
    # 参照ドキュメント: OpenAI Structured Outputs (ユーザー提供のURLを元に記載)
    #   - URL: https://platform.openai.com/docs/guides/structured-outputs
    #   - 参照日時: 2025-08-06 (ユーザーがドキュメントを提示した日時を想定)
    #   - SDKバージョン: openai >= 1.0.0 (この機能が利用可能なバージョン)
    class OpenAIStructuredOutput(BaseModel):
        tags: list[str]
        captions: list[str]
        score: float

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    @override
    def _run_inference(self, processed: list[str] | list[bytes]) -> list[Responsedict]:
        """OpenAI APIを使用して推論を実行し、構造化された応答を取得します。"""
        # processed は base64エンコードされた画像の文字列のリストを想定
        if not all(isinstance(item, str) for item in processed):
            raise WebApiError(
                "OpenAIApiAnnotator expects a list of base64 encoded image strings."
            )
        processed_str = cast(list[str], processed)

        results: list[Responsedict] = []
        # max_output_tokens と temperature の取得処理を復活
        max_output_tokens = self.config.get("max_output_tokens", 2000)
        temperature_config = self.config.get("temperature") # Noneの可能性あり
        temperature_to_use: float | object = float(temperature_config) if temperature_config is not None else NOT_GIVEN

        for image_data_base64 in processed_str:
            try:
                self._wait_for_rate_limit()

                # messages の型アノテーションを元に戻し、ignoreコメントでエラーを抑制
                messages: list[ChatCompletionMessageParam] = [ # type: ignore[assignment]
                    ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[ # type: ignore[arg-type]
                            {
                                "type": "input_text",
                                "text": BASE_PROMPT
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_data_base64}"
                            },
                        ],
                    ),
                ]

                logger.debug(
                    f"OpenAI API Request - Model: {self.api_model_id}, Method: responses.parse"
                )

                response_parsed = self.client.responses.parse(
                    model=self.api_model_id,
                    input=messages,  # type: ignore[arg-type] # Linterの一時的な抑制は残す (バージョン更新後の確認のため)
                    text_format=OpenAIApiAnnotator.OpenAIStructuredOutput,
                    # max_tokens と temperature を再度渡す
                    max_output_tokens=max_output_tokens,
                    temperature=temperature_to_use,  # type: ignore[arg-type] # Linterの一時的な抑制は残す
                )
                parsed_output: OpenAIApiAnnotator.OpenAIStructuredOutput = response_parsed.output_parsed

                # パースされた内容をログに出力
                parsed_output_dict = parsed_output.model_dump()
                logger.debug(f"OpenAI Parsed Output: {json.dumps(parsed_output_dict, indent=2, ensure_ascii=False)}")

                results.append(
                    Responsedict(
                        response=parsed_output_dict, # ログ出力した辞書をそのまま使用
                        error=None
                    )
                )

            except APIConnectionError as e:
                logger.error(f"OpenAI API Connection Error: {e}")
                results.append(Responsedict(response=None, error=f"API Connection Error: {e}"))
            except openai.APIError as e:
                logger.error(f"OpenAI API Error: {e}")
                results.append(Responsedict(response=None, error=f"OpenAI API Error: {e.message if hasattr(e, 'message') else str(e)}"))
            except Exception as e:
                logger.error(f"Error during OpenAI inference: {e}", exc_info=True)
                results.append(Responsedict(response=None, error=str(e)))
        return results

    @override
    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[WebApiAnnotationOutput]: # 戻り値型を合わせる
        """OpenAI APIからの応答を標準形式にフォーマットします。"""
        formatted_results: list[WebApiAnnotationOutput] = []
        for output in raw_outputs:
            if output.get("error"): # type: ignore[typeddict-item]
                formatted_results.append(
                    WebApiAnnotationOutput(annotation=None, error=output["error"]) # type: ignore[typeddict-item]
                )
                continue

            raw_response_data = output.get("response") # type: ignore[typeddict-item]

            if not raw_response_data or not isinstance(raw_response_data, dict):
                formatted_results.append(
                    WebApiAnnotationOutput(
                        annotation=None,
                        error="Invalid or empty response data from OpenAI API after parsing.",
                    )
                )
                continue
            
            try:
                if not all(k in raw_response_data for k in ["tags", "captions", "score"]):
                    raise ValueError("Parsed OpenAI response missing required keys: tags, captions, score")

                tags = raw_response_data["tags"]
                # caption は captions リストの最初の要素とするか、仕様に合わせて調整
                caption = raw_response_data["captions"][0] if raw_response_data["captions"] else ""
                score = raw_response_data["score"]

                if not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
                    raise ValueError("Invalid format for 'tags' in parsed OpenAI response.")
                if not isinstance(caption, str):
                    # captionsがリストなので、captionも文字列のリストになる想定。
                    # ここでは最初の要素を使うようにしたが、仕様に応じて変更。
                    # もしcaptionsが文字列のリストなら、captionの型もlist[str]にすべき。
                    # FormattedOutputやWebApiAnnotationOutputの'caption'の型と合わせる。
                    # 現在のWebApiAnnotationOutput.annotation.captionはstrを期待。
                    raise ValueError("Invalid format for 'caption' in parsed OpenAI response.")
                if not isinstance(score, (float, int)): # scoreはfloatのはず
                    raise ValueError("Invalid format for 'score' in parsed OpenAI response.")

                annotation_data = {
                    "tags": tags, # 文字列のリストをそのまま使用
                    "caption": caption,
                    "score": float(score),
                }
                formatted_results.append(
                    WebApiAnnotationOutput(annotation=annotation_data, error=None)
                )
            except Exception as e:
                logger.error(f"Error formatting OpenAI prediction: {e}", exc_info=True)
                formatted_results.append(
                    WebApiAnnotationOutput(annotation=None, error=str(e))
                )
        return formatted_results

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
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[Responsedict]:
        """Anthropic API を使用して推論を実行する"""
        if not all(isinstance(item, str) for item in processed_images):
            logger.error("AnthropicApiAnnotator received non-string input for _run_inference")
            return [{"response": None, "error": "Invalid input type for Anthropic API"}] * len(
                processed_images
            )

        processed_images_str: list[str] = cast(list[str], processed_images)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません",
                provider_name=getattr(self, "provider_name", "Unknown"),
            )
        if not isinstance(self.client, anthropic.Anthropic):
            raise ConfigurationError(
                f"予期しないクライアントタイプ: {type(self.client)}", provider_name=self.provider_name # self.provider_name を使用
            )

        logger.debug(f"Anthropic API 呼び出しに使用するモデルID: {self.api_model_id}")

        results: list[Responsedict] = []
        for base64_image in processed_images_str:
            try:
                self._wait_for_rate_limit()

                # model_name_on_provider は削除したので assert 不要
                # assert self.model_name_on_provider is not None, "Model name on provider is not set"

                # 型ヒントを削除し、辞書リテラルを使用
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
                                    "data": base64_image,  # 修正: encoded_image -> base64_image
                                },
                            },
                        ],
                    }
                ]
                # tools も辞書リテラルに変更
                tools = [
                    {
                        # "type": "custom", # Anthropic SDK は type を自動で付与する可能性があるので削除検討
                        "name": "Annotatejson",
                        "description": "Parsing image annotation results to JSON",
                        "input_schema": JSON_SCHEMA,
                    }
                ]
                system_prompt = SYSTEM_PROMPT

                # パラメータ取得
                temperature_val = config_registry.get(self.model_name, "temperature", default=0.7)
                # NotGivenを使用
                anthropic_temperature = float(temperature_val) if temperature_val is not None else NOT_GIVEN
                max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
                # max_tokens は None でないことを確認 (必須引数のため)
                if max_tokens is None:
                    logger.warning(
                        f"モデル {self.model_name} の max_output_tokens が設定されていません。デフォルト1800を使用します。"
                    )
                    max_tokens = 1800

                response = self.client.messages.create(
                    model=self.api_model_id,
                    max_tokens=int(max_tokens),  # int 型にキャスト
                    system=system_prompt,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    temperature=anthropic_temperature, # type: ignore[arg-type]
                )
                results.append({"response": response, "error": None})

            except APIConnectionError as e:
                error_message = f"Anthropic API サーバーへの接続に失敗しました: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except anthropic.RateLimitError as e:
                error_message = f"Anthropic API レート制限エラー: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except anthropic.APIStatusError as e:
                error_message = f"Anthropic API ステータスエラー: {e.status_code}, Response: {e.response}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except Exception as e:
                try:
                    self._handle_api_error(e)
                except WebApiError as api_e:
                    results.append({"response": None, "error": str(api_e)})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """Anthropic API からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"): # type: ignore[typeddict-item]
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"])) # type: ignore[typeddict-item]
                continue

            response_val: anthropic.types.Message | Any | None = output.get("response") # type: ignore[typeddict-item]

            if not isinstance(response_val, anthropic.types.Message):
                formatted_outputs.append(FormattedOutput(annotation=None, error=f"Invalid response type: {type(response_val)}"))
                continue

            # response_val は anthropic.types.Message 型であることが保証される
            if not response_val.content or not isinstance(response_val.content[0], anthropic.types.ToolUseBlock):
                stop_reason = response_val.stop_reason if response_val.stop_reason else "unknown"
                error_msg = f"応答コンテンツが無効または欠落している。: {stop_reason}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_msg))
                continue

            try:
                # 応答からtoolで作成されたJSONを抽出
                dict_content_unknown = response_val.content[0].input
                # dict_content が dict[str, Any] であることを期待
                if not isinstance(dict_content_unknown, dict):
                    raise ValueError(f"Expected dict from tool input, got {type(dict_content_unknown)}")
                dict_content = cast(dict[str, Any], dict_content_unknown)
                # 共通ヘルパーメソッドを呼び出して解析
                formatted = self._parse_common_json_response(dict_content)
                formatted_outputs.append(formatted)
            except (AttributeError, IndexError, TypeError, ValueError) as e: # ValueError を追加
                error_message = f"Anthropic API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
                continue

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
    """OpenRouter API を使用して画像に注釈を付けるクラス"""

    def __init__(self, model_name: str):
        """初期化

        Args:
            model_name: モデル名 (model_name_short)
        """
        super().__init__(model_name)

    def _run_inference(self, processed_images: list[str] | list[bytes]) -> list[Responsedict]:
        """OpenRouter API (OpenAI互換) を使用して推論を実行する"""
        if not all(isinstance(item, str) for item in processed_images):
            raise ValueError("OpenRouter API annotator requires string (base64) inputs.")
        processed_images_str: list[str] = cast(list[str], processed_images)

        if self.client is None or self.api_model_id is None:
            raise WebApiError(
                "API クライアントまたはモデル ID が初期化されていません",
                provider_name=getattr(self, "provider_name", "Unknown"),
            )
        if not isinstance(self.client, OpenAI):
            raise ConfigurationError(
                f"予期しないクライアントタイプ: {type(self.client)}", provider_name=self.provider_name # self.provider_name を使用
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

                # JSONスキーマ強制対応判定: TOMLやconfig_registryの設定値で判定
                json_schema_supported = config_registry.get(self.model_name, "json_schema_supported", default=False)
                if json_schema_supported:
                    response_format = {"type": "json_schema", "json_schema": JSON_SCHEMA}
                    logger.debug(
                        f"モデル {self.api_model_id} は JSON スキーマ強制に対応しているため、response_format を指定します。"
                    )
                else:
                    response_format = NOT_GIVEN
                    logger.debug(
                        f"モデル {self.api_model_id} は JSON スキーマ強制に未対応か不明なため、response_format を指定しません。"
                    )

                # 型ヒントを削除し、辞書リテラルを使用
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

                response = self.client.chat.completions.create(
                    model=self.api_model_id,
                    messages=messages,  # 辞書リストを直接渡す
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,  # type: ignore[arg-type]
                    timeout=timeout,
                    extra_headers=extra_headers,
                )
                results.append({"response": response, "error": None})

            except APIConnectionError as e:
                error_message = f"OpenRouter API サーバーへの接続に失敗しました: {e}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except openai.RateLimitError as e:
                error_message = f"OpenRouter API レート制限エラー: {e}"
                logger.error(error_message, exc_info=True)
            except openai.APIStatusError as e:
                error_message = f"OpenRouter API ステータスエラー: {e.status_code}, Response: {e.response}"
                logger.error(error_message, exc_info=True)
                results.append({"response": None, "error": error_message})
            except Exception as e:
                try:
                    self._handle_api_error(e)
                except WebApiError as api_e:
                    results.append({"response": None, "error": str(api_e)})

        return results

    def _format_predictions(self, raw_outputs: list[Responsedict]) -> list[FormattedOutput]:
        """OpenRouter API からの応答をフォーマットする"""
        formatted_outputs = []
        for output in raw_outputs:
            if output.get("error"): # type: ignore[typeddict-item]
                formatted_outputs.append(FormattedOutput(annotation=None, error=output["error"])) # type: ignore[typeddict-item]
                continue

            response_val = output.get("response") # type: ignore[typeddict-item]
            if not isinstance(response_val, openai_chat_types.ChatCompletion): # openai_chat_types を使用
                formatted_outputs.append(FormattedOutput(annotation=None, error=f"OpenRouter: Invalid response type: {type(response_val)}"))
                continue

            # response_val は ChatCompletion 型であることが保証される
            if not response_val.choices:
                formatted_outputs.append(FormattedOutput(annotation=None, error="OpenRouter: 応答が空または無効です (no choices)"))
                continue

            try:
                # OpenRouterのChoice型からメッセージコンテンツを取得
                choice = response_val.choices[0]
                if not choice.message or not choice.message.content:
                    formatted_outputs.append(
                        FormattedOutput(annotation=None, error="OpenRouter: メッセージコンテンツが空です")
                    )
                    continue

                content_text = choice.message.content
                
                # マークダウンコードブロック (`json) が含まれているか確認し、除去する
                if content_text.startswith("```json") and "```" in content_text:
                    # コードブロック内の実際のJSONを抽出
                    json_start = content_text.find("\n", content_text.find("```json")) + 1
                    json_end = content_text.rfind("```")
                    if json_start > 0 and json_end > json_start:
                        content_text = content_text[json_start:json_end].strip()
                        logger.debug(f"マークダウンコードブロックから抽出されたJSON: {content_text[:100]}...")
                
                # メッセージコンテンツから直接JSONを解析
                formatted = self._parse_common_json_response(content_text)
                formatted_outputs.append(formatted)
            except (AttributeError, IndexError) as e:
                error_message = f"OpenRouter API応答からテキストを抽出できませんでした: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))
            except Exception as e:
                error_message = f"OpenRouter応答のフォーマットで予期しないエラー: {e!s}"
                formatted_outputs.append(FormattedOutput(annotation=None, error=error_message))

        return formatted_outputs
