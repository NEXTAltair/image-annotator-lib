"""
型定義モジュール - pydantic、pydantic_ai、および基底クラス用の型定義
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# Lazy imports for heavy ML libraries (imported only during type checking)
if TYPE_CHECKING:
    import onnxruntime as ort
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
    metadata_path: Path


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
#
# ADR 0023 Phase 1 (Issue #35, PR #40): 旧 `WebApiBaseAnnotator` 系の dead types は
# 全て削除された。WebAPI 系の type は `WebApiAnnotator` (`webapi/annotator.py`)
# が直接 `UnifiedAnnotationResult` を構築するため、独立した type を必要としない。
#
# 削除済み:
# - `ApiClient` Protocol (旧 SDK adapter 共通インターフェース)
# - `WebApiComponents` TypedDict (旧 `WebApiBaseAnnotator.__enter__` 戻り値)
# - `WebApiInput` (旧 base64/bytes 入力 union、PydanticAI BinaryContent に置換)
# - `RawOutput` TypedDict (旧 `_run_inference` 戻り値)
# - `WebApiFormattedOutput` TypedDict (旧 `_format_predictions` 戻り値)
# - `BaseApiDependencies` / `CommonApiDependencies` / `OpenAIAPIDependencies` /
#   `GoogleApiDependencies` / `AnthropicApiDependencies` (旧 SDK adapter 設定 model)


# LoaderComponents型定義 (Union型) — ローカル ML 系のみ。WebAPI 経路は `WebApiAnnotator`
# が `self.components: None` で固定するため Union に含めない (Issue #35 PR #40)。
LoaderComponents = (
    TransformersComponents
    | TransformersPipelineComponents
    | ONNXComponents
    | TensorFlowComponents
    | CLIPComponents
)


class RatingPrediction(BaseModel):
    """Model-native rating prediction.

    LoRAIro などの consumer 固有 rating へは変換せず、モデルの label scheme を保持する。
    """

    raw_label: str
    confidence_score: float | None = None
    source_scheme: str


class AnnotationSchema(BaseModel):
    """画像アノテーションAPI共通のレスポンススキーマ"""

    tags: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    score: float | None = None
    ratings: list[RatingPrediction] = Field(default_factory=list)


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
    """Web APIモデル用(PydanticAIベース)

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
    """ローカルMLタガー用(ONNX/Transformers)

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
    """スコアラー用(CLIPベース)

    美的スコア、品質スコア等の数値評価モデルからの結果を格納。
    数値スコアがメインデータで、tagsフィールドは持たない。
    """

    model_type: Literal["scorer"] = "scorer"
    score_values: list[float]  # メインの数値スコア
    score_range: tuple[float, float] = (0.0, 10.0)
    score_format: str = "numeric"  # "numeric" | "tag_based"
    base_model: str  # "clip-vit-large-patch14"
    raw_scores: list[float] | None = None  # 元のテンソルデータ
    # tagsフィールドなし(数値スコアがメインデータ)


class CaptionerAnnotationResult(BaseAnnotationResult):
    """キャプション生成用(BLIP、CLIP-Caption等)

    BLIP、GIT等のキャプション生成モデルからの結果を格納。
    生成キャプションがメインデータで、tagsフィールドは持たない。
    """

    model_type: Literal["captioner"] = "captioner"
    captions: list[str]  # メインの生成キャプション
    confidence_scores: list[float] | None = None  # キャプションの信頼度
    generation_params: dict[str, Any] | None = None  # beam_size, temperature等
    base_model: str  # "blip-large", "clip-interrogator"
    raw_output: Any | None = None  # 元の生成結果
    # tagsフィールドなし(キャプションがメインデータ)


# Union型で型安全性確保
AnnotationResultV2 = (
    WebApiAnnotationResult | TaggerAnnotationResult | ScorerAnnotationResult | CaptionerAnnotationResult
)

# --- capability-based統一バリデーションスキーマ (Plan対応) ---

class TaskCapability(str, Enum):
    """サポートするタスク能力(ADR 0002: SCORE_LABELS 追加)。"""

    TAGS = "tags"
    CAPTIONS = "captions"
    SCORES = "scores"
    SCORE_LABELS = "score_labels"
    RATINGS = "ratings"


class UnifiedAnnotationResult(BaseModel):
    """統一アノテーション結果(capability-based validation対応)

    マルチモーダルLLM対応のcapability-based統一スキーマ。
    1つのモデルが複数タスク(tags, captions, scores)を実行可能。
    """

    model_name: str
    capabilities: set[TaskCapability]
    error: str | None = None

    # マルチタスク対応フィールド(capabilityに応じて使用)
    tags: list[str] | None = None
    captions: list[str] | None = None
    scores: dict[str, float] | None = None
    # ADR 0002: scorer 由来の categorical label (例: "very aesthetic", "aesthetic")。
    # content tag (WDTagger 等) と field レベルで分離。
    score_labels: list[str] | None = None
    # ADR 0003: rating / NSFW classifier 由来の model-native rating。
    ratings: list[RatingPrediction] | None = None

    # メタデータ(Optional)
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

    @field_validator("score_labels")
    @classmethod
    def validate_score_labels_capability(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        if v is not None:
            capabilities = info.data.get("capabilities", set())
            if TaskCapability.SCORE_LABELS not in capabilities:
                raise ValueError(
                    f"score_labels provided but SCORE_LABELS not in capabilities: {capabilities}"
                )
        return v

    @field_validator("ratings")
    @classmethod
    def validate_ratings_capability(
        cls, v: list[RatingPrediction] | None, info: ValidationInfo
    ) -> list[RatingPrediction] | None:
        if v is not None:
            capabilities = info.data.get("capabilities", set())
            if TaskCapability.RATINGS not in capabilities:
                raise ValueError(f"ratings provided but RATINGS not in capabilities: {capabilities}")
        return v

    @model_validator(mode="after")
    def validate_capabilities_not_empty(self) -> UnifiedAnnotationResult:
        """エラー結果以外はcapabilitiesが必須。"""
        if self.error is None and not self.capabilities:
            raise ValueError("capabilities cannot be empty for non-error results")
        return self


# === 新しい統一型システム ===
UnifiedPHashAnnotationResults = dict[str, dict[str, UnifiedAnnotationResult]]


# --- アノテーターメタデータ用の型定義 (Issue #19) ---


ModelType = Literal["tagger", "scorer", "captioner", "vision"]
"""アノテーターモデルの主用途分類。

`is_api` と直交した capability 軸の分類。WebAPI モデルでも tagger/scorer/captioner
の区別はあり得るため、"webapi" は含めない。

- "tagger": タグ予測モデル (WD-Tagger, DeepDanbooru 等)
- "scorer": 数値スコアモデル (aesthetic scorer 等)
- "captioner": キャプション生成モデル (BLIP, GIT 等)
- "vision": 汎用 vision モデル / 未分類 (汎用 VLM 等)
"""


@dataclass(frozen=True)
class AnnotatorInfo:
    """アノテーターモデルのメタデータ。

    Attributes:
        name: モデル名 (レジストリキー)。
        model_type: モデルの主用途分類 (tagger/scorer/captioner/vision)。
        capabilities: モデルが提供する出力種別の集合。
        is_local: ローカル実行モデルか。
        is_api: 外部 API を呼ぶモデルか (API キー必要)。
        device: ローカル実行時のデバイス指定。API モデルでは None。

    Invariants (検証はテスト側で担保):
        - is_local XOR is_api == True
        - is_api == True のとき device is None
    """

    name: str
    model_type: ModelType
    capabilities: frozenset[TaskCapability]
    is_local: bool
    is_api: bool
    device: str | None = None
    # --- Phase 2: 詳細メタデータ (Issue #19/#26, ADR 0005 責務境界: lib 側で統一提供) ---
    provider: str | None = None
    """プロバイダー名。ローカルモデルは "local"、API モデルは "openai"/"anthropic"/"google" 等。
    config_registry に未登録の PydanticAI 直接モデルは model_id の slash 前から推論。"""
    litellm_model_id: str | None = None
    """LiteLLM ID (例: "openai/gpt-4o")。ローカルモデルは None。

    ADR 0023 Phase 2 (Issue #41): 旧 `api_model_id` field は本 field にリネームされた。
    LoRAIro 側は ADR 0023 line 73 に従い互換シムを残さず一括破壊的変更で追従する。
    """
    estimated_size_gb: float | None = None
    """ローカル ML モデルのダウンロードサイズ推定値 (GB)。API モデルは None。"""
    discontinued_at: datetime.datetime | None = None
    """廃止日時。現役モデルは None (ADR 0021: LiteLLM 統合後は LiteLLM 由来に切替予定)。"""
    max_output_tokens: int | None = None
    """WebAPI 呼び出し時のトークン上限。API モデルのみ設定。ローカルモデルは None。"""


# --- pHash ベースの結果コンテナ (Issue #9) ---


class PHashAnnotationResults(dict[str, dict[str, UnifiedAnnotationResult]]):
    """統一バリデーションスキーマ用の画像pHashをキーとする評価結果辞書。

    Attributes:
        [phash]: 画像のpHashをキーとする辞書。
                 各キーの値は、モデル名をキーとする辞書。
                 各モデル名の値は、型安全なUnifiedAnnotationResult。
    """

    pass
