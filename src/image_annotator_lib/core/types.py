"""
型定義モジュール - pydantic、pydantic_ai、および基底クラス用の型定義
"""

from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, Union

import onnxruntime as ort
from pydantic import BaseModel, Field, SecretStr, ValidationInfo, field_validator
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.clip import CLIPModel, CLIPProcessor
from transformers.pipelines.base import Pipeline as TransformersPipelineObject

# --- 基底クラス用の型定義 (base.py から移動) ---


class TransformersComponents(TypedDict):
    model: Any
    processor: AutoProcessor


class TransformersPipelineComponents(TypedDict):
    pipeline: TransformersPipelineObject


class ONNXComponents(TypedDict):
    session: ort.InferenceSession
    csv_path: Path


class TensorFlowComponents(TypedDict):
    model_dir: Path | None
    model: Any


class CLIPComponents(TypedDict):
    model: Any
    processor: CLIPProcessor
    clip_model: CLIPModel


class AnnotationResult(TypedDict, total=False):
    """単一画像の標準化されたアノテーション結果。

    `BaseAnnotator.predict` メソッドの戻り値リストの要素型です。

    Attributes:
        phash: 画像の知覚ハッシュ (str)。計算失敗時は None。
        tags: アノテーション結果の主要な文字列リスト (list[str])。
               タガーの場合はタグ、スコアラーの場合はスコアタグ、
               キャプショナーの場合はキャプションが入ります。
        formatted_output: 整形済み出力 (Any)。`_format_predictions` の戻り値。
                          デバッグや詳細分析に使用できます。
        error: 処理中に発生したエラーメッセージ (str)。エラーがない場合は None。
    """

    phash: str | None
    tags: list[str]
    formatted_output: Any
    error: str | None


class TagConfidence(TypedDict):
    """タグとその信頼度、情報源を保持する型定義。"""

    confidence: float
    source: str


# --- Web API 関連の型定義 ---


# APIクライアント共通インターフェース
class ApiClient(Protocol):
    def call_api(self, *args: Any, **kwargs: Any) -> Any: ...


class WebApiComponents(TypedDict):
    """Web API アノテーターが `__enter__` で準備するコンポーネント。"""

    client: ApiClient  # 各プロバイダーのAPIクライアント
    api_model_id: str  # APIコールに使用する加工済みモデルID
    provider_name: str  # プロバイダー名を追加


# LoaderComponents型定義 (Union型)
LoaderComponents = (
    TransformersComponents
    | TransformersPipelineComponents
    | ONNXComponents
    | TensorFlowComponents
    | CLIPComponents
    | WebApiComponents
)


class WebApiInput(BaseModel):
    image_b64: str | None = None  # base64文字列(Anthropic/OpenAI等)
    image_bytes: bytes | None = None  # バイト列(Google等)

    @field_validator("image_b64", "image_bytes")
    @staticmethod
    def at_least_one_image(value: str | bytes | None, info: ValidationInfo) -> str | bytes | None:
        values = info.data
        if value is None and (
            values.get("image_b64") is None
            if info.field_name == "image_bytes"
            else values.get("image_bytes") is None
        ):
            raise ValueError("image_b64またはimage_bytesのいずれか一方は必須です")
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

    annotation: dict[str, Any] | None
    error: str | None


class BaseApiDependencies(BaseModel):
    """
    すべてのWeb APIアノテーターで共通の基本的な依存性を定義するモデル。
    """

    api_key: SecretStr = Field(description="APIキー(環境変数や設定ファイルから取得してもよい)")
    model_name: str = Field(description="設定ファイルで定義されたモデルの一意な名前 (例: my-openai-model)")
    api_model_id: str = Field(
        description="APIコール時に実際に使用するプロバイダー上のモデルID (例: gpt-4o-mini)"
    )
    timeout: int = Field(default=60, description="APIコール時のタイムアウト(秒)")


# 共通パラメータを抽出
class CommonApiDependencies(BaseApiDependencies):
    temperature: float = Field(default=0.7, description="生成時のサンプリング温度")
    max_output_tokens: int = Field(default=1800, description="生成されるトークンの最大数")


class OpenAIAPIDependencies(CommonApiDependencies):
    """
    OpenAI Chat Completions/Responses API 共通依存性モデル
    """

    response_format: Any | None = Field(
        default=None, description="構造化出力用のPydanticモデルやJSONスキーマ"
    )
    # OpenRouter等の拡張用
    json_schema_supported: bool = Field(
        default=False, description="OpenRouterでJSONスキーマ出力がサポートされているか"
    )
    referer: str | None = Field(default=None, description="OpenRouter用のHTTPリファラ")
    app_name: str | None = Field(
        default=None, description="OpenRouter用のアプリケーション名 (X-Titleヘッダー)"
    )


class GoogleApiDependencies(CommonApiDependencies):
    """Google Gemini API 用の依存性モデル"""

    top_p: float = Field(default=1.0, description="トップPサンプリング")
    top_k: int = Field(default=32, description="トップKサンプリング")
    # response_mime_type や response_schema は _run_inference 内で固定的に設定されているため、
    # 現状では依存性モデルに含めない。必要に応じて追加を検討。


class AnthropicApiDependencies(CommonApiDependencies):
    """Anthropic Claude API 用の依存性モデル"""

    # Anthropic SDKでは max_tokens_to_sample が推奨されるが、現在は max_tokens で渡している
    max_tokens: int = Field(
        default=1800, description="生成されるトークンの最大数 (APIパラメータ名は max_tokens)"
    )
    # tools や tool_choice など、AnthropicのTool Use関連の設定を将来的に追加する可能性あり


# --- 新バリデーションスキーマ (Pydantic V2) ---


class BaseAnnotationResult(BaseModel):
    """全モデルタイプ共通の基底クラス

    型安全なバリデーションスキーマの基底となるクラス。
    各モデルタイプに共通する最小限のフィールドのみを定義。
    """

    error: str | None = None
    model_name: str
    model_type: str


class WebApiAnnotationResult(BaseAnnotationResult):
    """Web APIモデル用（PydanticAIベース）

    OpenAI、Anthropic、Google等のWeb APIからの結果を格納。
    AI生成タグ、キャプション、信頼度スコア等を含む。
    """

    model_type: Literal["webapi"] = "webapi"
    tags: list[str] = Field(default_factory=list)  # AI生成タグ
    captions: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    provider_name: str  # "anthropic", "openai", "google"
    api_response: dict[str, Any] | None = None  # 生データ保持


class TaggerAnnotationResult(BaseAnnotationResult):
    """ローカルMLタガー用（ONNX/Transformers）

    WD-Tagger、DeepDanbooru等のローカルMLモデルからの結果を格納。
    カテゴリ別のスコア、閾値情報、フレームワーク情報等を含む。
    """

    model_type: Literal["tagger"] = "tagger"
    tags: list[str] = Field(default_factory=list)  # 閾値フィルタ後のタグ
    category_scores: dict[str, dict[str, float]]  # {"general": {"tag1": 0.85}}
    confidence_threshold: float
    total_tags_count: int
    framework: str  # "onnx", "transformers", "tensorflow"
    raw_predictions: list[float] | None = None  # 元のnumpy配列データ


class ScorerAnnotationResult(BaseAnnotationResult):
    """スコアラー用（CLIPベース）

    美的スコア、品質スコア等の数値評価モデルからの結果を格納。
    数値スコアがメインデータで、tagsフィールドは持たない。
    """

    model_type: Literal["scorer"] = "scorer"
    score_values: list[float]  # メインの数値スコア
    score_range: tuple[float, float] = (0.0, 10.0)
    score_format: str = "numeric"  # "numeric" | "tag_based"
    base_model: str  # "clip-vit-large-patch14"
    raw_scores: list[float] | None = None  # 元のテンソルデータ
    # tagsフィールドなし（数値スコアがメインデータ）


class CaptionerAnnotationResult(BaseAnnotationResult):
    """キャプション生成用（BLIP、CLIP-Caption等）

    BLIP、GIT等のキャプション生成モデルからの結果を格納。
    生成キャプションがメインデータで、tagsフィールドは持たない。
    """

    model_type: Literal["captioner"] = "captioner"
    captions: list[str]  # メインの生成キャプション
    confidence_scores: list[float] | None = None  # キャプションの信頼度
    generation_params: dict[str, Any] | None = None  # beam_size, temperature等
    base_model: str  # "blip-large", "clip-interrogator"
    raw_output: Any | None = None  # 元の生成結果
    # tagsフィールドなし（キャプションがメインデータ）


# Union型で型安全性確保
AnnotationResultV2 = Union[
    WebApiAnnotationResult,
    TaggerAnnotationResult,
    ScorerAnnotationResult,
    CaptionerAnnotationResult,
]


# --- capability-based統一バリデーションスキーマ (Plan対応) ---

from enum import Enum


class TaskCapability(str, Enum):
    """サポートするタスク能力（3つに限定）"""

    TAGS = "tags"
    CAPTIONS = "captions"
    SCORES = "scores"


class UnifiedAnnotationResult(BaseModel):
    """統一アノテーション結果（capability-based validation対応）

    マルチモーダルLLM対応のcapability-based統一スキーマ。
    1つのモデルが複数タスク（tags, captions, scores）を実行可能。
    """

    model_name: str
    capabilities: set[TaskCapability]
    error: str | None = None

    # マルチタスク対応フィールド（capabilityに応じて使用）
    tags: list[str] | None = None
    captions: list[str] | None = None
    scores: dict[str, float] | None = None

    # メタデータ（Optional）
    provider_name: str | None = None
    framework: str | None = None
    raw_output: dict[str, Any] | None = None

    # === 厳密なcapabilityバリデーション ===
    @field_validator("tags")
    @classmethod
    def validate_tags_capability(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        if v is not None:
            capabilities = info.data.get("capabilities", set())
            if TaskCapability.TAGS not in capabilities:
                raise ValueError(f"tags provided but TAGS not in capabilities: {capabilities}")
        return v

    @field_validator("captions")
    @classmethod
    def validate_captions_capability(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        if v is not None:
            capabilities = info.data.get("capabilities", set())
            if TaskCapability.CAPTIONS not in capabilities:
                raise ValueError(f"captions provided but CAPTIONS not in capabilities: {capabilities}")
        return v

    @field_validator("scores")
    @classmethod
    def validate_scores_capability(
        cls, v: dict[str, float] | None, info: ValidationInfo
    ) -> dict[str, float] | None:
        if v is not None:
            capabilities = info.data.get("capabilities", set())
            if TaskCapability.SCORES not in capabilities:
                raise ValueError(f"scores provided but SCORES not in capabilities: {capabilities}")
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities_not_empty(cls, v: set[TaskCapability]) -> set[TaskCapability]:
        if not v:
            raise ValueError("capabilities cannot be empty")
        return v


# === 新しい統一型システム ===
UnifiedPHashAnnotationResults = dict[str, dict[str, UnifiedAnnotationResult]]
