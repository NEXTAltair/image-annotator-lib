"""
現時点では pydantic と pydantic_ai 用のクラスのみ
# TODO: 他のTypedDictも後で追加予定
"""
from typing import Any, TypedDict

from pydantic import BaseModel, ValidationInfo, field_validator
from pydantic_ai import Agent


class WebApiComponents(TypedDict):
    """Web API アノテーターが `__enter__` で準備するコンポーネント。"""

    client: Any  # 各プロバイダーのAPIクライアント
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

    _run_inferenceの戻り値（RawOutput型）を、後続処理・テスト・外部インターフェース用に整形した結果を格納します。

    Attributes:
        annotation: 解析されたアノテーション情報（タグ、キャプション、スコア等）を含むdict型。
                    解析に成功した場合はdict、失敗した場合はNone。
        error: 解析中にエラーが発生した場合のエラーメッセージ文字列。
               エラーがない場合はNone。
    """

    annotation: dict | None
    error: str | None
