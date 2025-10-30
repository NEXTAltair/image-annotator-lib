"""Tests for model configuration classes (Phase 1A: Config Objects).

型安全なPydantic v2ベースのConfiguration Objectsのテスト。
"""

import pytest
from pydantic import ValidationError

from image_annotator_lib.core.model_config import (
    BaseModelConfig,
    LocalMLModelConfig,
    ModelConfigFactory,
    WebAPIModelConfig,
)
from image_annotator_lib.exceptions.errors import ConfigurationError


class TestBaseModelConfig:
    """BaseModelConfigのテストケース"""

    def test_init_with_required_fields(self):
        """必須フィールドのみで初期化"""
        config = BaseModelConfig(model_name="test_model", class_name="TestClass")

        assert config.model_name == "test_model"
        assert config.class_name == "TestClass"
        assert config.device == "cuda"  # default
        assert config.estimated_size_gb is None

    def test_init_with_alias_class(self):
        """aliasフィールド 'class' で初期化"""
        config = BaseModelConfig(model_name="test_model", **{"class": "TestClass"})

        assert config.class_name == "TestClass"

    def test_init_with_all_fields(self):
        """全フィールドで初期化"""
        config = BaseModelConfig(
            model_name="test_model",
            class_name="TestClass",
            device="cpu",
            estimated_size_gb=1.5,
        )

        assert config.model_name == "test_model"
        assert config.class_name == "TestClass"
        assert config.device == "cpu"
        assert config.estimated_size_gb == 1.5

    def test_frozen_immutability(self):
        """frozen=True によるイミュータブル性"""
        config = BaseModelConfig(model_name="test_model", class_name="TestClass")

        with pytest.raises(ValidationError, match="Instance is frozen"):
            config.device = "cpu"  # type: ignore

    def test_extra_forbid(self):
        """extra='forbid' による未定義フィールド拒否"""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BaseModelConfig(
                model_name="test_model",
                class_name="TestClass",
                invalid_field="value",  # type: ignore
            )

    def test_device_validation_invalid(self):
        """デバイス名の検証（無効な値）"""
        with pytest.raises(ValidationError, match="device must be one of"):
            BaseModelConfig(model_name="test_model", class_name="TestClass", device="gpu")

    def test_device_validation_valid(self):
        """デバイス名の検証（有効な値）"""
        config_cuda = BaseModelConfig(model_name="test_model", class_name="TestClass", device="cuda")
        config_cpu = BaseModelConfig(model_name="test_model", class_name="TestClass", device="cpu")

        assert config_cuda.device == "cuda"
        assert config_cpu.device == "cpu"

    def test_estimated_size_validation_negative(self):
        """メモリサイズの検証（負の値）"""
        with pytest.raises(ValidationError, match="estimated_size_gb must be positive"):
            BaseModelConfig(model_name="test_model", class_name="TestClass", estimated_size_gb=-1.0)

    def test_estimated_size_validation_zero(self):
        """メモリサイズの検証（ゼロ）"""
        with pytest.raises(ValidationError, match="estimated_size_gb must be positive"):
            BaseModelConfig(model_name="test_model", class_name="TestClass", estimated_size_gb=0.0)

    def test_estimated_size_validation_positive(self):
        """メモリサイズの検証（正の値）"""
        config = BaseModelConfig(model_name="test_model", class_name="TestClass", estimated_size_gb=2.5)

        assert config.estimated_size_gb == 2.5

    def test_str_strip_whitespace(self):
        """str_strip_whitespace=True による前後空白削除"""
        config = BaseModelConfig(model_name="  test_model  ", class_name="  TestClass  ", device="  cuda  ")

        assert config.model_name == "test_model"
        assert config.class_name == "TestClass"
        assert config.device == "cuda"


class TestLocalMLModelConfig:
    """LocalMLModelConfigのテストケース"""

    def test_init_with_required_fields(self):
        """必須フィールドのみで初期化"""
        config = LocalMLModelConfig(
            model_name="test_model", class_name="TestClass", model_path="/path/to/model"
        )

        assert config.model_name == "test_model"
        assert config.class_name == "TestClass"
        assert config.model_path == "/path/to/model"
        assert config.device == "cuda"  # default
        assert config.batch_size == 1  # default

    def test_init_with_all_fields(self):
        """全フィールドで初期化"""
        config = LocalMLModelConfig(
            model_name="test_model",
            class_name="WDTagger",
            model_path="SmilingWolf/wd-v1-4-moat-tagger-v2",
            device="cuda",
            estimated_size_gb=0.456,
            base_model="openai/clip-vit-large-patch14",
            activation_type="ReLU",
            final_activation_type="Sigmoid",
            batch_size=4,
            gpu_memory_limit_gb=8.0,
        )

        assert config.model_name == "test_model"
        assert config.class_name == "WDTagger"
        assert config.model_path == "SmilingWolf/wd-v1-4-moat-tagger-v2"
        assert config.device == "cuda"
        assert config.estimated_size_gb == 0.456
        assert config.base_model == "openai/clip-vit-large-patch14"
        assert config.activation_type == "ReLU"
        assert config.final_activation_type == "Sigmoid"
        assert config.batch_size == 4
        assert config.gpu_memory_limit_gb == 8.0

    def test_model_path_validation_empty(self):
        """model_pathの検証（空文字列）"""
        with pytest.raises(ValidationError, match="model_path cannot be empty"):
            LocalMLModelConfig(model_name="test_model", class_name="TestClass", model_path="")

    def test_model_path_validation_whitespace(self):
        """model_pathの検証（空白のみ）"""
        with pytest.raises(ValidationError, match="model_path cannot be empty"):
            LocalMLModelConfig(model_name="test_model", class_name="TestClass", model_path="   ")

    def test_batch_size_validation_zero(self):
        """batch_sizeの検証（ゼロ）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            LocalMLModelConfig(
                model_name="test_model",
                class_name="TestClass",
                model_path="/path",
                batch_size=0,
            )

    def test_batch_size_validation_negative(self):
        """batch_sizeの検証（負の値）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            LocalMLModelConfig(
                model_name="test_model",
                class_name="TestClass",
                model_path="/path",
                batch_size=-1,
            )

    def test_gpu_memory_limit_validation_zero(self):
        """gpu_memory_limit_gbの検証（ゼロ）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            LocalMLModelConfig(
                model_name="test_model",
                class_name="TestClass",
                model_path="/path",
                gpu_memory_limit_gb=0.0,
            )

    def test_gpu_memory_limit_validation_negative(self):
        """gpu_memory_limit_gbの検証（負の値）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            LocalMLModelConfig(
                model_name="test_model",
                class_name="TestClass",
                model_path="/path",
                gpu_memory_limit_gb=-1.0,
            )


class TestWebAPIModelConfig:
    """WebAPIModelConfigのテストケース"""

    def test_init_with_required_fields(self):
        """必須フィールドのみで初期化"""
        config = WebAPIModelConfig(
            model_name="gpt-4o-mini",
            class_name="OpenAIApiAnnotator",
            model_name_on_provider="gpt-4o-mini-2024-07-18",
        )

        assert config.model_name == "gpt-4o-mini"
        assert config.class_name == "OpenAIApiAnnotator"
        assert config.api_model_id == "gpt-4o-mini-2024-07-18"
        assert config.timeout == 60  # default
        assert config.retry_count == 3  # default
        assert config.retry_delay == 1.0  # default
        assert config.min_request_interval == 1.0  # default

    def test_init_with_alias_model_name_on_provider(self):
        """aliasフィールド 'model_name_on_provider' で初期化"""
        config = WebAPIModelConfig(
            model_name="test_model",
            class_name="TestClass",
            model_name_on_provider="provider-model-id",
        )

        assert config.api_model_id == "provider-model-id"

    def test_init_with_all_fields(self):
        """全フィールドで初期化"""
        config = WebAPIModelConfig(
            model_name="claude-3-7-sonnet",
            class_name="AnthropicApiAnnotator",
            api_model_id="claude-3-7-sonnet-20250219",
            timeout=90,
            retry_count=5,
            retry_delay=2.0,
            min_request_interval=1.5,
            temperature=0.8,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=50,
        )

        assert config.model_name == "claude-3-7-sonnet"
        assert config.class_name == "AnthropicApiAnnotator"
        assert config.api_model_id == "claude-3-7-sonnet-20250219"
        assert config.timeout == 90
        assert config.retry_count == 5
        assert config.retry_delay == 2.0
        assert config.min_request_interval == 1.5
        assert config.temperature == 0.8
        assert config.max_output_tokens == 2048
        assert config.top_p == 0.95
        assert config.top_k == 50

    def test_api_model_id_validation_empty(self):
        """api_model_idの検証（空文字列）"""
        with pytest.raises(ValidationError, match="api_model_id cannot be empty"):
            WebAPIModelConfig(model_name="test_model", class_name="TestClass", api_model_id="")

    def test_api_model_id_validation_whitespace(self):
        """api_model_idの検証（空白のみ）"""
        with pytest.raises(ValidationError, match="api_model_id cannot be empty"):
            WebAPIModelConfig(model_name="test_model", class_name="TestClass", api_model_id="   ")

    def test_timeout_validation_zero(self):
        """timeoutの検証（ゼロ）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                timeout=0,
            )

    def test_timeout_validation_exceeds_limit(self):
        """timeoutの検証（上限超過）"""
        with pytest.raises(ValidationError, match="Input should be less than or equal to 300"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                timeout=301,
            )

    def test_retry_count_validation_negative(self):
        """retry_countの検証（負の値）"""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                retry_count=-1,
            )

    def test_retry_count_validation_exceeds_limit(self):
        """retry_countの検証（上限超過）"""
        with pytest.raises(ValidationError, match="Input should be less than or equal to 10"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                retry_count=11,
            )

    def test_temperature_validation_negative(self):
        """temperatureの検証（負の値）"""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                temperature=-0.1,
            )

    def test_temperature_validation_exceeds_limit(self):
        """temperatureの検証（上限超過）"""
        with pytest.raises(ValidationError, match="Input should be less than or equal to 2"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                temperature=2.1,
            )

    def test_max_output_tokens_validation_zero(self):
        """max_output_tokensの検証（ゼロ）"""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                max_output_tokens=0,
            )

    def test_max_output_tokens_validation_exceeds_limit(self):
        """max_output_tokensの検証（上限超過）"""
        with pytest.raises(ValidationError, match="Input should be less than or equal to 8192"):
            WebAPIModelConfig(
                model_name="test_model",
                class_name="TestClass",
                api_model_id="model-id",
                max_output_tokens=8193,
            )


class TestModelConfigFactory:
    """ModelConfigFactoryのテストケース"""

    def test_from_registry_local_ml_model(self):
        """config_registry → LocalMLModelConfig変換"""
        registry_dict = {
            "class": "WDTagger",
            "model_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
            "device": "cuda",
            "estimated_size_gb": 0.456,
        }

        config = ModelConfigFactory.from_registry("wd-tagger", registry_dict)

        assert isinstance(config, LocalMLModelConfig)
        assert config.model_name == "wd-tagger"
        assert config.class_name == "WDTagger"
        assert config.model_path == "SmilingWolf/wd-v1-4-moat-tagger-v2"
        assert config.device == "cuda"
        assert config.estimated_size_gb == 0.456

    def test_from_registry_webapi_model(self):
        """config_registry → WebAPIModelConfig変換"""
        registry_dict = {
            "class": "GoogleApiAnnotator",
            "model_name_on_provider": "gemini-1.5-pro",
            "timeout": 90,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        config = ModelConfigFactory.from_registry("gemini-1.5-pro", registry_dict)

        assert isinstance(config, WebAPIModelConfig)
        assert config.model_name == "gemini-1.5-pro"
        assert config.class_name == "GoogleApiAnnotator"
        assert config.api_model_id == "gemini-1.5-pro"
        assert config.timeout == 90

    def test_from_registry_missing_identifier_fields(self):
        """識別子フィールド欠落時のエラー"""
        registry_dict = {
            "class": "TestClass",
            "device": "cuda",
        }

        with pytest.raises(ConfigurationError, match=r"model_path.*model_name_on_provider"):
            ModelConfigFactory.from_registry("test_model", registry_dict)

    def test_from_registry_invalid_data(self):
        """不正なデータ形式のエラー"""
        registry_dict = {
            "class": "TestClass",
            "model_path": "/path",
            "device": "invalid_device",  # 不正なデバイス名
        }

        with pytest.raises(ConfigurationError):
            ModelConfigFactory.from_registry("test_model", registry_dict)

    def test_to_dict_local_ml_model(self):
        """LocalMLModelConfig → 辞書変換"""
        config = LocalMLModelConfig(
            model_name="wd-tagger",
            class_name="WDTagger",
            model_path="SmilingWolf/wd-v1-4-moat-tagger-v2",
            device="cuda",
            estimated_size_gb=0.456,
        )

        result = ModelConfigFactory.to_dict(config)

        assert "model_name" not in result  # 除外される
        assert result["class"] == "WDTagger"  # エイリアス名
        assert result["model_path"] == "SmilingWolf/wd-v1-4-moat-tagger-v2"
        assert result["device"] == "cuda"
        assert result["estimated_size_gb"] == 0.456

    def test_to_dict_webapi_model(self):
        """WebAPIModelConfig → 辞書変換"""
        config = WebAPIModelConfig(
            model_name="gemini-1.5-pro",
            class_name="GoogleApiAnnotator",
            api_model_id="gemini-1.5-pro",
            timeout=90,
            retry_count=3,
        )

        result = ModelConfigFactory.to_dict(config)

        assert "model_name" not in result  # 除外される
        assert result["class"] == "GoogleApiAnnotator"  # エイリアス名
        assert result["model_name_on_provider"] == "gemini-1.5-pro"  # エイリアス名
        assert result["timeout"] == 90
        assert result["retry_count"] == 3

    def test_to_dict_exclude_none(self):
        """Noneフィールドの除外"""
        config = LocalMLModelConfig(
            model_name="test_model",
            class_name="TestClass",
            model_path="/path",
            device="cuda",
            # estimated_size_gb=None (デフォルト)
        )

        result = ModelConfigFactory.to_dict(config)

        assert "estimated_size_gb" not in result  # None は除外される
        assert "base_model" not in result
        assert "activation_type" not in result

    def test_roundtrip_local_ml_model(self):
        """ラウンドトリップテスト: registry → Config → registry"""
        original_dict = {
            "class": "WDTagger",
            "model_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
            "device": "cuda",
            "estimated_size_gb": 0.456,
            "batch_size": 4,
        }

        # registry → Config
        config = ModelConfigFactory.from_registry("wd-tagger", original_dict)

        # Config → registry
        result_dict = ModelConfigFactory.to_dict(config)

        assert result_dict["class"] == original_dict["class"]
        assert result_dict["model_path"] == original_dict["model_path"]
        assert result_dict["device"] == original_dict["device"]
        assert result_dict["estimated_size_gb"] == original_dict["estimated_size_gb"]
        assert result_dict["batch_size"] == original_dict["batch_size"]

    def test_roundtrip_webapi_model(self):
        """ラウンドトリップテスト: registry → Config → registry"""
        original_dict = {
            "class": "GoogleApiAnnotator",
            "model_name_on_provider": "gemini-1.5-pro",
            "timeout": 90,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        # registry → Config
        config = ModelConfigFactory.from_registry("gemini-1.5-pro", original_dict)

        # Config → registry
        result_dict = ModelConfigFactory.to_dict(config)

        assert result_dict["class"] == original_dict["class"]
        assert result_dict["model_name_on_provider"] == original_dict["model_name_on_provider"]
        assert result_dict["timeout"] == original_dict["timeout"]
        assert result_dict["retry_count"] == original_dict["retry_count"]
        assert result_dict["retry_delay"] == original_dict["retry_delay"]
        assert result_dict["min_request_interval"] == original_dict["min_request_interval"]
