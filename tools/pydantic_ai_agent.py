import asyncio
from collections.abc import Sequence
from io import BytesIO
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field, SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider

from image_annotator_lib.core.model_factory import _get_api_key
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT


# 1. 依存性定義 (エージェントが必要とする設定など)
class ImageAnnotatorDependencies(BaseModel):
    api_key: SecretStr
    api_model_id: str
    base_url: str | None = None
    timeout: int = 60

# 2. レスポンス定義 (エージェントが最終的に期待する出力型)
class Annotation(BaseModel):
    tags: list[str] = Field(default_factory=list, description="画像から抽出されたタグのリスト")
    caption: str | None = Field(None, description="画像を説明するキャプション文")
    score: float | None = Field(None, description="アノテーションの信頼度スコア")

# 画像前処理ヘルパー関数 (バイト列を返すように統一)
def _preprocess_image_to_bytes(image: Image.Image) -> bytes:
    """画像をWEBP形式のバイトデータに変換する"""
    buffered = BytesIO()
    image.save(buffered, format="WEBP")
    return buffered.getvalue()

# 3. PydanticAI Agentの準備と実行 (メイン処理)
async def main(
    images_to_annotate: list[Image.Image],
    provider_name: str,
    api_model_id: str,
):
    if not images_to_annotate:
        print("エラー: アノテーション対象の画像が提供されていません。")
        return
    if not images_to_annotate[0]: # リストの最初の要素をチェック
        print("エラー: 提供された画像データが無効です。")
        return

    api_key_value = _get_api_key(provider_name, api_model_id)
    if not api_key_value:
        print(f"警告: {provider_name} ({api_model_id}) のAPIキーが取得できませんでした。")
        return

    # ImageAnnotatorDependenciesは現状Agentに直接渡すdepsとしては使われませんが、
    # APIキーやモデルIDを保持するコンテナとして利用
    annotator_deps = ImageAnnotatorDependencies(
        api_key=SecretStr(api_key_value),
        api_model_id=api_model_id
    )

    llm_provider: Any = None
    llm_client: Any = None
    try:
        if provider_name.lower() == "openai":
            llm_provider = OpenAIProvider(api_key=annotator_deps.api_key.get_secret_value())
            llm_client = OpenAIModel(
                model_name=annotator_deps.api_model_id,
                provider=llm_provider
            )
        elif provider_name.lower() == "google":
            llm_provider = GoogleGLAProvider(api_key=annotator_deps.api_key.get_secret_value())
            llm_client = GeminiModel(
                model_name=annotator_deps.api_model_id,
                provider=llm_provider
            )
        else:
            print(f"エラー: 未対応のLLMプロバイダです: {provider_name}")
            return
    except Exception as e:
        print(f"LLMクライアント ({provider_name}) の初期化に失敗: {e}")
        return

    # SYSTEM_PROMPTから期待される出力構造とルールを抽出
    expected_output_structure_example = ""
    strict_rules = ""
    try:
        if "Output Structure:" in SYSTEM_PROMPT and "Rules:" in SYSTEM_PROMPT:
            structure_text = SYSTEM_PROMPT.split("Output Structure:", 1)[1].split("Rules:", 1)[0].strip()
            # Annotationモデルに合わせて "captions" を "caption" に修正
            structure_text = structure_text.replace('    "captions": ["caption1", "caption2", ...],', '    "caption": "A descriptive caption for the image",')
            expected_output_structure_example = "Output Structure:\n" + structure_text
        if "Rules:" in SYSTEM_PROMPT:
            strict_rules = "Rules:" + SYSTEM_PROMPT.split("Rules:", 1)[1].strip()
        if not expected_output_structure_example or not strict_rules:
             raise ValueError("SYSTEM_PROMPTからの抽出失敗")
    except Exception as e:
        print(f"警告: SYSTEM_PROMPTの解析に失敗 ({e})。汎用指示を使用。")
        # Annotation モデルに合わせた汎用指示
        expected_output_structure_example = """Output Structure:
{
    "tags": ["tag1", "tag2", "tag3"],
    "caption": "A descriptive caption for the image",
    "score": 0.85
}"""
        strict_rules = """Rules:
1. ONLY output the JSON object - no other text or formatting.
2. DO NOT use markdown code blocks (```) or any other formatting.
3. DO NOT include any explanations or comments.
4. Always return complete, valid, parseable JSON.
5. Include all required fields as defined in the output_type (Annotation: tags, caption, score).
6. Never truncate or leave incomplete JSON.
7. DO NOT add any leading or trailing whitespace or newlines.
8. DO NOT start with any introductory text like "Here is the analysis:".
"""

    final_system_prompt = (
        expected_output_structure_example +
        "\n\n" +
        strict_rules
    )

    # Agentの初期化 (ツールを削除)
    # deps_type も現状は直接使われませんが、Agentの型定義のために残すことも可能です。
    # 不要であればNoneにするか、型引数から削除します。ここでは残してみます。
    agent: Agent[ImageAnnotatorDependencies, Annotation] = Agent(
        model=llm_client,
        tools=[], # ツールを削除
        deps_type=ImageAnnotatorDependencies, # Agentの型定義のために残す
        output_type=Annotation,
        system_prompt=final_system_prompt
    )

    print("--- エージェントのE2Eテストを開始します ---")

    # 画像データをバイト列に変換 (最初の画像のみを使用する例)
    # 複数画像に対応する場合は、プロンプトパーツのリストを適切に構築する必要があります。
    image_bytes = _preprocess_image_to_bytes(images_to_annotate[0])

    # マルチモーダルプロンプトの作成
    # BASE_PROMPT が画像に対する指示となる
    prompt_content_parts: Sequence[str | BinaryContent] = [
        BinaryContent(data=image_bytes, media_type="image/webp"), # Changed mime_type to media_type
        BASE_PROMPT  # テキストによる指示
    ]

    print(f"\n--- ユーザープロンプト (画像 + テキスト指示: '{BASE_PROMPT[:50]}...') ---")

    try:
        # depsはAgentのツールが使うものであり、agent.runに直接渡すdepsはツールのコンテキスト用。
        # ツールを削除したため、agent.runのdeps引数は不要かもしれない。
        # PydanticAIのAgentがdeps_typeで指定された依存性を内部的に解決するわけではないため、
        # ここでのannotator_depsは実質的に使われません。
        # もしAPIキー等をModelの初期化以外で渡す必要がある場合に意味を持ちます。
        response = await agent.run(prompt_content_parts, deps=annotator_deps) # Pass deps

        print("\n--- エージェントの最終出力 (Annotation型を期待) ---")
        if isinstance(response.output, Annotation):
            final_annotation: Annotation = response.output
            print(f"  タグ: {final_annotation.tags}")
            print(f"  キャプション: {final_annotation.caption}")
            print(f"  スコア: {final_annotation.score}")
        else:
            print("  期待したAnnotation型ではありませんでした。")
            print(f"  実際の出力型: {type(response.output)}")
            print(f"  実際の出力内容: {response.output}")

    except Exception as e:
        print(f"エージェント実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
    print("--------------------------------------")


if __name__ == "__main__":
    # 注意: このパスはスクリプトの実行場所からの相対パスです。
    # プロジェクトルートから実行する場合は 'tools/tests/resources/img/1_img/file01.webp' のようになる可能性があります。
    # より堅牢にするには、絶対パスを指定するか、__file__ を基準にしたパス操作を推奨します。
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

    default_provider_name = "Google"
    default_api_model_id = "gemini-2.0-flash" # Vision対応のGeminiモデル

    asyncio.run(main(
        images_to_annotate=[img_to_process], # 画像をリストで渡す
        provider_name=default_provider_name,
        api_model_id=default_api_model_id
    ))
