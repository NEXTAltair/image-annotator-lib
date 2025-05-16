"""
現時点では pydantic と pydantic_ai 用のクラスのみ
# TODO: 他のTypedDictも後で追加予定
"""
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field, SecretStr, ValidationInfo, field_validator
from pydantic_ai import Agent


# APIクライアント共通インターフェース
class ApiClient(Protocol):
    def call_api(self, *args: Any, **kwargs: Any) -> Any: ...

class WebApiComponents(TypedDict):
    """Web API アノテーターが `__enter__` で準備するコンポーネント。"""
    client: ApiClient  # 各プロバイダーのAPIクライアント
    api_model_id: str  # APIコールに使用する加工済みモデルID
    provider_name: str  # プロバイダー名を追加

class WebApiInput(BaseModel):
    image_b64: str | None = None   # base64文字列(Anthropic/OpenAI等)
    image_bytes: bytes | None = None  # バイト列(Google等)

    @field_validator('image_b64', 'image_bytes')
    @staticmethod
    def at_least_one_image(value, info: ValidationInfo):
        values = info.data
        if (
            value is None and
            (values.get('image_b64') is None if info.field_name == 'image_bytes'
             else values.get('image_bytes') is None)
        ):
            raise ValueError('image_b64またはimage_bytesのいずれか一方は必須です')
        return value


class AnnotationSchema(BaseModel):
    """画像アノテーションAPI共通のレスポンススキーマ"""
    tags: list[str]
    captions: list[str]
    score: float

class RawOutput(TypedDict, total=False):
    """構造化出力の結果に問題があれば、errorにエラーメッセージを格納する。"""
    response: AnnotationSchema | None
    error: str | None

# Web API Annotator 用の型定義を追加
class WebApiFormattedOutput(TypedDict):
    """
    WebApiBaseAnnotator._format_predictions の戻り値の型定義。

    _run_inferenceの戻り値(RawOutput型)を、後続処理･テスト･外部インターフェース用に整形した結果を格納します。

    Attributes:
        annotation: 解析されたアノテーション情報(タグ、キャプション、スコア等)を含むdict型。
                    解析に成功した場合はdict、失敗した場合はNone。
        error: 解析中にエラーが発生した場合のエラーメッセージ文字列。
               エラーがない場合はNone。
    """

    annotation: dict | None
    error: str | None

class BaseApiDependencies(BaseModel):
    """
    すべてのWeb APIアノテーターで共通の基本的な依存性を定義するモデル。
    """
    api_key: SecretStr = Field(description="APIキー(環境変数や設定ファイルから取得してもよい)")
    model_name: str = Field(description="設定ファイルで定義されたモデルの一意な名前 (例: my-openai-model)")
    api_model_id: str = Field(description="APIコール時に実際に使用するプロバイダー上のモデルID (例: gpt-4o-mini)")
    timeout: int = Field(default=60, description="APIコール時のタイムアウト(秒)")

# 共通パラメータを抽出
class CommonApiDependencies(BaseApiDependencies):
    temperature: float = Field(default=0.7, description="生成時のサンプリング温度")
    max_output_tokens: int = Field(default=1800, description="生成されるトークンの最大数")

class OpenAIAPIDependencies(CommonApiDependencies):
    """
    OpenAI Chat Completions/Responses API 共通依存性モデル
    """
    response_format: Any | None = Field(default=None, description="構造化出力用のPydanticモデルやJSONスキーマ")
    # OpenRouter等の拡張用
    json_schema_supported: bool = Field(default=False, description="OpenRouterでJSONスキーマ出力がサポートされているか")
    referer: str | None = Field(default=None, description="OpenRouter用のHTTPリファラ")
    app_name: str | None = Field(default=None, description="OpenRouter用のアプリケーション名 (X-Titleヘッダー)")

class GoogleApiDependencies(CommonApiDependencies):
    """Google Gemini API 用の依存性モデル"""
    top_p: float = Field(default=1.0, description="トップPサンプリング")
    top_k: int = Field(default=32, description="トップKサンプリング")
    # response_mime_type や response_schema は _run_inference 内で固定的に設定されているため、
    # 現状では依存性モデルに含めない。必要に応じて追加を検討。

class AnthropicApiDependencies(CommonApiDependencies):
    """Anthropic Claude API 用の依存性モデル"""
    # Anthropic SDKでは max_tokens_to_sample が推奨されるが、現在は max_tokens で渡している
    max_tokens: int = Field(default=1800, description="生成されるトークンの最大数 (APIパラメータ名は max_tokens)")
    # tools や tool_choice など、AnthropicのTool Use関連の設定を将来的に追加する可能性あり
