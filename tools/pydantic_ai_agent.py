"""
PydanticAI を使用したエージェントをテストする
現時点で OpenRouter が提供するモデルは ` TypeError: 'NoneType' object cannot be interpreted as an integer` エラー

"""

import asyncio
import os
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, SecretStr
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider

# from pydantic_ai.providers.openrouter import OpenRouterProvider

#
#  from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT
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

load_dotenv()


def get_openrouter_api_key() -> str:
    # model_factry.py get_api_keyより引用
    """プロバイダー名に基づいて環境変数からAPIキーを取得する。

    PydanticAI は `.env` `load_dotenv()` で環境変数にキーを登録すると自動で取得される
    OpenRouterは自動で取得しないので、ここで取得する

    .env ファイルのロードを試みる。

    Args:
        api_model_id: モデルID (e.g., "gemini-1.5-pro", "gemma-3-27b-it:free").
    Returns:
        APIキー文字列。

    Raises:
        ApiAuthenticationError: 対応する環境変数が見つからない場合。
        ConfigurationError: サポートされていないプロバイダー名の場合。
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("警告: 環境変数 'OPENROUTER_API_KEY' にAPIキーが見つかりませんでした。")
        return "APIキーが見つかりませんでした。"
    return api_key

# 1. 依存性定義 (エージェントが必要とする設定など)
class ImageAnnotatorDependencies(BaseModel):
    api_key: SecretStr
    api_model_id: str
    base_url: str | None = None
    timeout: int = 60

# 2. レスポンス定義 (エージェントが最終的に期待する出力型)
class Annotation(BaseModel):
    tags: list[str] = Field(default_factory=list, description="画像から抽出されたタグのリスト")
    captions: list[str] = Field(default_factory=list, description="画像を説明するキャプションのリスト")
    score: float = Field(description="画像の評価値")

# 画像前処理ヘルパー関数 (バイト列を返すように統一)
def _preprocess_image_to_bytes(image: Image.Image) -> bytes:
    """画像をwebp形式のバイトデータに変換する"""
    buffered = BytesIO()
    try:
        image.save(buffered, format="WEBP")
    except Exception as e:
        print(f"画像の保存中にエラー: {e}。PNGで再試行します。")
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
    return buffered.getvalue()

# 3. PydanticAI Agentの準備と実行 (メイン処理)
async def main_annotator_logic(
    images_to_annotate: list[Image.Image],
    provider_name: str,
    api_model_id: str,
):
    if not images_to_annotate:
        print("エラー: アノテーション対象の画像が提供されていません。")
        return
    if not images_to_annotate[0]:
        print("エラー: 提供された画像データが無効です。")
        return

    llm_provider: Any = None
    llm_client: Any = None

    try:
        if provider_name.lower() == "openai":
            llm_client = OpenAIModel(
                model_name=api_model_id,
                provider=OpenAIProvider()
            )
        elif provider_name.lower() == "google":
            llm_client = GeminiModel(
                model_name=api_model_id,
                provider=GoogleGLAProvider()
            )
        elif provider_name.lower() == "anthropic":
            llm_client = AnthropicModel(
                model_name=api_model_id,
                provider=AnthropicProvider()
            )
        # elif provider_name.lower() == "openrouter":
        #     llm_provider = OpenRouterProvider(
        #         api_key=get_openrouter_api_key(),
        #         base_url="https://openrouter.ai/api/v1"
        #     )
        #     llm_client = OpenAIModel(
        #         model_name=api_model_id,
        #         provider=llm_provider
        #     )
        else:
            print(f"エラー: 未対応のLLMプロバイダです: {provider_name}")
            return

    except Exception as e:
        print(f"LLMクライアント ({provider_name}) の初期化に失敗: {e}")
        import traceback
        traceback.print_exc()
        return


    agent_params = {
        "model": llm_client,
        "output_type": Annotation,
        "system_prompt": SYSTEM_PROMPT,
    }
    try:
        agent = Agent(**agent_params)
    except Exception as e:
        print(f"Agentの初期化に失敗しました: {e}")
        print("使用されたパラメータ:")
        for key, value in agent_params.items():
            if key == "model" and hasattr(value, 'model_name'):
                print(f"  {key}: {type(value).__name__} (model_name: {getattr(value, 'model_name', 'N/A')})")
            elif key == "model" and hasattr(value, 'name'): # GeminiModelの場合など
                 print(f"  {key}: {type(value).__name__} (name: {getattr(value, 'name', 'N/A')})")
            else:
                print(f"  {key}: {value}")
        import traceback
        traceback.print_exc()
        return

    print("--- エージェントのE2Eテストを開始します ---")
    first_image_bytes = _preprocess_image_to_bytes(images_to_annotate[0])

    simple_text_prompt = BASE_PROMPT

    print(f"\n--- ユーザープロンプト (テキストのみ): '{simple_text_prompt[:50]}...') ---")

    try:
        # agent.run に単純な文字列プロンプトを渡す
        response_container = await agent.run(
            user_prompt=[simple_text_prompt,
            BinaryContent(data=first_image_bytes, media_type="image/webp")]
        )
        print("\n--- エージェントの最終出力 (Annotation型を期待) ---")
        if response_container and hasattr(response_container, 'output'):
            final_output = response_container.output
            if isinstance(final_output, Annotation):
                print(f"  タグ: {final_output.tags}")
                print(f"  キャプション: {final_output.captions}")
                print(f"  スコア: {final_output.score}")
            else:
                print("  期待したAnnotation型ではありませんでした。")
                print(f"  実際の出力型: {type(final_output)}")
                print(f"  実際の出力内容: {final_output}")
        else:
            print("  エージェントからのレスポンスまたは出力が予期した形式ではありません。")
            print(f"  レスポンスコンテナ: {response_container}")
    except TypeError as e:
        print(f"TypeError が発生しました: {e}")
        print("--- TypeError発生時の関連変数 ---")
        print("  プロンプト内容:")
        if 'simple_text_prompt' in locals():
            print(f"    simple_text_prompt: {simple_text_prompt}")
        else:
            print("  プロンプト内容 (simple_text_prompt): <利用不可または未定義>")

        if 'llm_client' in locals() and hasattr(llm_client, 'model_settings'):
            client_settings = llm_client.model_settings if hasattr(llm_client, 'model_settings') else "N/A"
            print(f"  LLMクライアントのモデル設定: {client_settings}")
        elif 'llm_client' in locals() and hasattr(llm_client, '_model_settings'):
            client_settings = llm_client._model_settings if hasattr(llm_client, '_model_settings') else "N/A"
            print(f"  LLMクライアントのモデル設定 (プライベート): {client_settings}")
        else:
            print("  LLMクライアントのモデル設定: <取得不可>")

        print("----------------------------------")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("--------------------------------------")

if __name__ == "__main__":
    image_path = "tests/resources/img/1_img/file01.webp"
    img_to_process = Image.open(image_path)

    # default_provider_name = "OpenAI"
    # default_api_model_id = "gpt-4o-mini" # または "gpt-4-vision-preview" など、Vision対応モデル

    # default_provider_name = "Google"
    # default_api_model_id = "gemini-2.0-flash" # Vision対応のGeminiモデル

    default_provider_name = "OpenRouter"
    default_api_model_id = "meta-llama/llama-4-maverick:free"

    # default_provider_name = "Anthropic"
    # default_api_model_id = "claude-3-5-sonnet-20240620"


    asyncio.run(main_annotator_logic(
        images_to_annotate=[img_to_process],
        provider_name=default_provider_name,
        api_model_id=default_api_model_id
    ))
