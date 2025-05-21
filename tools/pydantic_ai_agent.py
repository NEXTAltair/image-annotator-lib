import asyncio
import os
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from openai.types import chat
from PIL import Image
from pydantic import BaseModel, Field, SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider

from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT

load_dotenv()

def get_api_key(provider_name: str, api_model_id: str) -> str:
    # model_factry.py より引用
    """プロバイダー名に基づいて環境変数からAPIキーを取得する。

    .env ファイルのロードを試みる。

    Args:
        provider_name: プロバイダー名 (e.g., "Google", "OpenAI", "Anthropic", "OpenRouter").
        api_model_id: モデルID (e.g., "gemini-1.5-pro", "gemma-3-27b-it:free").
    Returns:
        APIキー文字列。

    Raises:
        ApiAuthenticationError: 対応する環境変数が見つからない場合。
        ConfigurationError: サポートされていないプロバイダー名の場合。
    """
    env_var_map = {
        "Google": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
    }
    normalized_provider_name = provider_name.capitalize() if provider_name else ""
    env_var_name = env_var_map.get(normalized_provider_name)

    if normalized_provider_name == "Openrouter" or (env_var_name is None and ":" in api_model_id) :
        env_var_name = "OPENROUTER_API_KEY"
    elif not env_var_name:
        print(f"警告: プロバイダー '{provider_name}' に対応するAPIキー環境変数が不明です。OPENROUTER_API_KEY を試みます。")
        env_var_name = "OPENROUTER_API_KEY"

    api_key = os.getenv(env_var_name)
    if not api_key:
        print(f"警告: 環境変数 '{env_var_name}' にAPIキーが見つかりませんでした。")
        return "APIキーが見つかりませんでした。"
    return api_key


# カスタムOpenAIModelの定義
class CustomOpenAIModelWithFallback(OpenAIModel):
    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        # APIからのエラーレスポンスを最初にチェック
        # Linterエラー (ChatCompletion に error 属性がない) を無視するため、type: ignore を追加
        actual_error = getattr(response, "error", None) # type: ignore[attr-defined]
        if actual_error is not None:
            error_info = actual_error
            error_message = "API error during model response processing."
            if isinstance(error_info, dict):
                message = error_info.get('message', 'Unknown error')
                code = error_info.get('code')
                error_type = error_info.get('type')
                param = error_info.get('param')
                error_details = f"Message: {message}, Code: {code}, Type: {error_type}, Param: {param}"
                error_message = f"API Error: {error_details}"
            elif isinstance(error_info, str):
                error_message = f"API Error: {error_info}"

            print(f"エラーレスポンスを検知: {error_message}")
            # ここで pydantic-ai や openai ライブラリの適切な例外クラスを使うのが理想ですが、
            # 詳細が不明なため、汎用的な RuntimeError を使用します。
            # 必要に応じて、より具体的な例外タイプ (例: openai.APIStatusError) に変更してください。
            raise RuntimeError(error_message)

        return super()._process_response(response)

# 1. 依存性定義 (エージェントが必要とする設定など)
class ImageAnnotatorDependencies(BaseModel):
    api_key: SecretStr
    api_model_id: str
    base_url: str | None = None
    timeout: int = 60

# 2. レスポンス定義 (エージェントが最終的に期待する出力型)
class Annotation(BaseModel):
    tags: list[str] = Field(default_factory=list, description="画像から抽出されたタグのリスト")
    captions: list[str] = Field(default_factory=list, description="画像を説明するキャプション文")
    score: float | None = Field(None, description="アノテーションの信頼度スコア")

# 画像前処理ヘルパー関数 (バイト列を返すように統一)
def _preprocess_image_to_bytes(image: Image.Image) -> bytes:
    """画像をJPEGまたはPNG形式のバイトデータに変換する"""
    buffered = BytesIO()
    save_format = image.format if image.format in ["JPEG", "PNG"] else "JPEG"
    try:
        image.save(buffered, format=save_format)
    except Exception as e:
        print(f"画像の保存中にエラー: {e}。PNGで再試行します。")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
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
    api_key_for_provider: SecretStr | None = None

    try:
        api_key_value = get_api_key(provider_name, api_model_id)
        if api_key_value == "APIキーが見つかりませんでした。":
            print(f"警告: {provider_name} ({api_model_id}) のAPIキーが取得できませんでした。環境変数を確認してください。")
        else:
            api_key_for_provider = SecretStr(api_key_value)

        if provider_name.lower() == "openai":
            llm_provider = OpenAIProvider(api_key=api_key_for_provider.get_secret_value() if api_key_for_provider else None)
            llm_client = OpenAIModel(
                model_name=api_model_id,
                provider=llm_provider
            )
        elif provider_name.lower() == "google":
            llm_provider = GoogleGLAProvider(api_key=api_key_for_provider.get_secret_value() if api_key_for_provider else None)
            llm_client = GeminiModel(
                model_name=api_model_id,
                provider=llm_provider
            )
        elif provider_name.lower() == "anthropic":
            llm_provider = AnthropicProvider(api_key=api_key_for_provider.get_secret_value() if api_key_for_provider else None)
            llm_client = AnthropicModel(
                model_name=api_model_id,
                provider=llm_provider
            )
        elif provider_name.lower() == "openrouter":
            if not api_key_for_provider:
                print(f"エラー: OpenRouter ({api_model_id}) のAPIキーが取得できませんでした。処理を中止します。")
                return
            llm_provider = OpenAIProvider(
                api_key=api_key_for_provider.get_secret_value(),
                base_url="https://openrouter.ai/api/v1"
            )
            llm_client = OpenAIModel(
                model_name=api_model_id,
                provider=llm_provider
            )
        else:
            print(f"エラー: 未対応のLLMプロバイダです: {provider_name}")
            return
    except Exception as e:
        print(f"LLMクライアント ({provider_name}) の初期化に失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    annotator_deps = ImageAnnotatorDependencies(
        api_key=api_key_for_provider if api_key_for_provider else SecretStr(""),
        api_model_id=api_model_id
    )

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

    pil_image_format = Image.open(BytesIO(first_image_bytes)).format
    image_format = pil_image_format.lower() if pil_image_format else "jpeg"
    # mime_type は ImageUrl に直接指定するため、ここでは不要

    # base64_image = base64.b64encode(first_image_bytes).decode('utf-8') # 画像処理は一旦コメントアウト
    # image_data_url = f"data:image/{image_format};base64,{base64_image}" # 画像処理は一旦コメントアウト

    # アプローチ1：単純なテキストプロンプトのみを渡す  # noqa: RUF003
    simple_text_prompt = BASE_PROMPT

    print(f"\n--- ユーザープロンプト (テキストのみ): '{simple_text_prompt[:50]}...') ---")

    try:
        # agent.run に単純な文字列プロンプトを渡す
        response_container = await agent.run(simple_text_prompt)
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
    try:
        img_to_process = Image.open(image_path)
    except FileNotFoundError:
        print(f"エラー: 指定された画像ファイルが見つかりません: {image_path}")
        exit()
    except Exception as e:
        print(f"エラー: 画像ファイルの読み込み中に問題が発生しました: {e}")
        exit()

    # デフォルトはOpenAIでテスト
    # target_provider を変更してGoogleでもテスト可能
    # default_provider_name = "OpenAI"
    # default_api_model_id = "gpt-4o-mini" # または "gpt-4-vision-preview" など、Vision対応モデル

    # default_provider_name = "Google"
    # default_api_model_id = "gemini-2.0-flash" # Vision対応のGeminiモデル

    # default_provider_name = "OpenRouter"
    # default_api_model_id = "meta-llama/llama-4-maverick:free"

    default_provider_name = "Anthropic"
    default_api_model_id = "claude-3-5-sonnet-20240620"


    asyncio.run(main_annotator_logic(
        images_to_annotate=[img_to_process],
        provider_name=default_provider_name,
        api_model_id=default_api_model_id
    ))
