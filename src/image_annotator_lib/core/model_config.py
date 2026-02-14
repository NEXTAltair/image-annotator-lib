"""
型安全なモデル設定クラス - Pydantic v2ベース

このモジュールはconfig_registryの辞書ベース設定を型安全なPydanticモデルに変換します。

Phase 1A: Config Objects導入
- BaseModelConfig: 全モデルタイプ共通の基底クラス
- LocalMLModelConfig: ローカルMLモデル用
- WebAPIModelConfig: Web API用
- ModelConfigFactory: config_registry ↔ Config Objects変換
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions.errors import ConfigurationError
from .utils import logger


class BaseModelConfig(BaseModel):
    """全モデルタイプ共通の基底Configuration Object

    Pydantic v2最適化設定:
    - frozen=True: イミュータブル化 (+15% performance)
    - extra='forbid': 厳密なバリデーション（未定義フィールド拒否）
    - populate_by_name=True: エイリアスサポート
    - str_strip_whitespace=True: 文字列の前後空白削除
    - cache_strings=True: 文字列最適化
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        cache_strings=True,
    )

    # 全モデルタイプ共通フィールド
    model_name: str = Field(description="設定ファイルで定義されたモデルの一意な名前")
    class_name: str = Field(
        description="使用するAnnotatorクラス名", alias="class", serialization_alias="class"
    )
    device: str = Field(default="cuda", description="処理デバイス (cuda/cpu)")
    estimated_size_gb: float | None = Field(default=None, description="推定メモリサイズ (GB)")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """デバイス名の検証"""
        allowed_devices = {"cuda", "cpu"}
        if v not in allowed_devices:
            raise ValueError(f"device must be one of {allowed_devices}, got: {v}")
        return v

    @field_validator("estimated_size_gb")
    @classmethod
    def validate_size(cls, v: float | None) -> float | None:
        """メモリサイズの検証"""
        if v is not None and v <= 0:
            raise ValueError(f"estimated_size_gb must be positive, got: {v}")
        return v


class LocalMLModelConfig(BaseModelConfig):
    """ローカルMLモデル用Configuration Object

    ONNX、Transformers、TensorFlow、CLIPベースのモデル設定。
    """

    model_path: str = Field(description="モデルパス(HuggingFace repo/URL/local path)")
    model_type: str | None = Field(
        default=None,
        description="モデルタイプ (scorer/tagger/captioner)",
        alias="type",
        serialization_alias="type",
    )
    base_model: str | None = Field(default=None, description="CLIPベースモデル(CLIP系のみ)")
    activation_type: str | None = Field(default=None, description="活性化関数(一部モデルのみ)")
    final_activation_type: str | None = Field(default=None, description="最終層活性化関数(一部モデルのみ)")
    batch_size: int = Field(default=1, gt=0, description="バッチサイズ")
    gpu_memory_limit_gb: float | None = Field(default=None, gt=0, description="GPU メモリ上限 (GB)")

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """モデルパスの検証（空文字列拒否）"""
        if not v or not v.strip():
            raise ValueError("model_path cannot be empty")
        return v


class WebAPIModelConfig(BaseModelConfig):
    """Web APIモデル用Configuration Object

    OpenAI、Anthropic、Google等のWeb API設定。
    """

    api_model_id: str = Field(
        description="APIプロバイダー上の実際のモデルID", alias="model_name_on_provider"
    )
    timeout: int = Field(default=60, gt=0, le=300, description="APIタイムアウト (秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="リトライ回数")
    retry_delay: float = Field(default=1.0, ge=0.0, le=60.0, description="リトライ遅延 (秒)")
    min_request_interval: float = Field(default=1.0, ge=0.0, le=60.0, description="最小リクエスト間隔 (秒)")

    # プロンプト設定
    prompt_template: str = Field(default="Describe this image.", description="プロンプトテンプレート")

    # 生成パラメータ(PydanticAI用、Optional)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="サンプリング温度")
    max_output_tokens: int | None = Field(default=None, gt=0, le=8192, description="最大トークン数")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Top-Pサンプリング")
    top_k: int | None = Field(default=None, gt=0, description="Top-Kサンプリング")

    @field_validator("api_model_id")
    @classmethod
    def validate_api_model_id(cls, v: str) -> str:
        """APIモデルIDの検証（空文字列拒否）"""
        if not v or not v.strip():
            raise ValueError("api_model_id cannot be empty")
        return v


class ModelConfigFactory:
    """Config Objects <-> config_registry 変換Factory

    辞書ベースのconfig_registryとPydantic Config Objectsの相互変換を担当。
    Phase 1A: 基盤実装
    """

    @staticmethod
    def from_registry(
        model_name: str, registry_dict: dict[str, Any]
    ) -> LocalMLModelConfig | WebAPIModelConfig:
        """config_registryの辞書をConfig Objectに変換

        Args:
            model_name: モデル名（config_registryのキー）
            registry_dict: モデル設定辞書（config_registryの値）

        Returns:
            LocalMLModelConfig | WebAPIModelConfig: 型安全なConfig Object

        Raises:
            ConfigurationError: 変換に失敗した場合
        """
        try:
            # 辞書をコピーして model_name を追加
            config_dict = dict(registry_dict)
            config_dict["model_name"] = model_name

            # Web APIモデルかローカルMLモデルかを判定
            # Web APIモデルは model_name_on_provider フィールドを持つ
            if "model_name_on_provider" in config_dict:
                logger.debug(f"Web APIモデルとして '{model_name}' をパース")
                # Filter out api_key and other fields not in WebAPIModelConfig
                # api_key is retrieved separately from config_registry for security
                # api_model_id is aliased as model_name_on_provider, so filter it out to avoid duplication
                # referer/app_name are OpenRouter-specific, retrieved via config_registry.get()
                filtered_dict = {
                    k: v
                    for k, v in config_dict.items()
                    if k
                    not in (
                        "api_key",
                        "api_model_id",
                        "capabilities",
                        "max_retries",
                        "rate_limit_requests_per_minute",
                        "referer",
                        "app_name",
                    )
                }
                return WebAPIModelConfig(**filtered_dict)
            elif "model_path" in config_dict:
                logger.debug(f"ローカルMLモデルとして '{model_name}' をパース")
                # Filter out fields not in LocalMLModelConfig
                # capabilities are retrieved via config_registry.get() for dynamic capability support
                filtered_dict = {
                    k: v
                    for k, v in config_dict.items()
                    if k not in ("capabilities",)
                }
                return LocalMLModelConfig(**filtered_dict)
            else:
                error_msg = f"モデル '{model_name}' の設定に 'model_path' または 'model_name_on_provider' がありません"
                logger.error(error_msg)
                raise ConfigurationError(
                    message=error_msg,
                    details={"model_name": model_name, "config_dict": config_dict},
                )

        except Exception as e:
            error_msg = f"モデル '{model_name}' の設定オブジェクト変換に失敗: {e}"
            logger.error(error_msg)
            raise ConfigurationError(
                message=error_msg,
                details={"model_name": model_name, "error": str(e), "config_dict": registry_dict},
            ) from e

    @staticmethod
    def to_dict(config: LocalMLModelConfig | WebAPIModelConfig) -> dict[str, Any]:
        """Config Objectを辞書に変換（後方互換性用）

        Args:
            config: Config Object

        Returns:
            dict[str, Any]: config_registry互換の辞書

        Notes:
            - model_name は辞書から除外（config_registryのキーとして使用されるため）
            - Pydantic v2の model_dump() を使用
            - by_alias=True でエイリアス名（"class", "model_name_on_provider"）を使用
        """
        try:
            # Pydantic v2の model_dump() を使用
            result = config.model_dump(by_alias=True, exclude={"model_name"}, exclude_none=True)
            logger.debug(f"Config Object '{config.model_name}' を辞書に変換")
            return result

        except Exception as e:
            error_msg = f"Config Object '{config.model_name}' の辞書変換に失敗: {e}"
            logger.error(error_msg)
            raise ConfigurationError(
                message=error_msg, details={"model_name": config.model_name, "error": str(e)}
            ) from e
