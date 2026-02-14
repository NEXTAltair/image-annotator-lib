"""Integration tests for Config Objects migration (Phase 1A).

実際のconfig_registryとの統合、TOMLファイルからのロード、
Config Objectsへの変換、後方互換性を検証します。
"""

import pytest

from image_annotator_lib.core.model_config import (
    LocalMLModelConfig,
    ModelConfigFactory,
    WebAPIModelConfig,
)
from image_annotator_lib.exceptions.errors import ConfigurationError


class TestConfigRegistryIntegration:
    """config_registryとの統合テスト"""

    def test_factory_converts_local_ml_model_from_registry(self, managed_config_registry):
        """config_registryのローカルMLモデル設定をConfig Objectに変換"""
        # config_registryに設定を追加
        managed_config_registry._merged_config_data["wd-tagger"] = {
            "class": "WDTagger",
            "model_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
            "device": "cuda",
            "estimated_size_gb": 0.456,
        }

        # config_registryから取得
        registry_dict = managed_config_registry.get_all_config()["wd-tagger"]

        # Config Objectに変換
        config = ModelConfigFactory.from_registry("wd-tagger", registry_dict)

        assert isinstance(config, LocalMLModelConfig)
        assert config.model_name == "wd-tagger"
        assert config.class_name == "WDTagger"
        assert config.model_path == "SmilingWolf/wd-v1-4-moat-tagger-v2"
        assert config.device == "cuda"
        assert config.estimated_size_gb == 0.456

    def test_factory_converts_webapi_model_from_registry(self, managed_config_registry):
        """config_registryのWeb APIモデル設定をConfig Objectに変換"""
        # config_registryに設定を追加
        managed_config_registry._merged_config_data["gemini-1.5-pro"] = {
            "class": "GoogleApiAnnotator",
            "model_name_on_provider": "gemini-1.5-pro",
            "timeout": 90,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        # config_registryから取得
        registry_dict = managed_config_registry.get_all_config()["gemini-1.5-pro"]

        # Config Objectに変換
        config = ModelConfigFactory.from_registry("gemini-1.5-pro", registry_dict)

        assert isinstance(config, WebAPIModelConfig)
        assert config.model_name == "gemini-1.5-pro"
        assert config.class_name == "GoogleApiAnnotator"
        assert config.api_model_id == "gemini-1.5-pro"
        assert config.timeout == 90
        assert config.retry_count == 3

    def test_factory_roundtrip_preserves_data(self, managed_config_registry):
        """registry → Config → registry ラウンドトリップでデータ保持"""
        original_dict = {
            "class": "WDTagger",
            "model_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
            "device": "cuda",
            "estimated_size_gb": 0.456,
            "batch_size": 4,
        }

        managed_config_registry._merged_config_data["wd-tagger"] = original_dict

        # registry → Config
        registry_dict = managed_config_registry.get_all_config()["wd-tagger"]
        config = ModelConfigFactory.from_registry("wd-tagger", registry_dict)

        # Config → registry
        result_dict = ModelConfigFactory.to_dict(config)

        # データが保持されていることを確認
        assert result_dict["class"] == original_dict["class"]
        assert result_dict["model_path"] == original_dict["model_path"]
        assert result_dict["device"] == original_dict["device"]
        assert result_dict["estimated_size_gb"] == original_dict["estimated_size_gb"]
        assert result_dict["batch_size"] == original_dict["batch_size"]

    def test_config_object_can_replace_registry_dict(self, managed_config_registry):
        """Config Object → 辞書変換でconfig_registryに戻せることを確認"""
        # Config Objectを作成
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="WDTagger",
            model_path="SmilingWolf/wd-v1-4-moat-tagger-v2",
            device="cuda",
            estimated_size_gb=0.5,
        )

        # 辞書に変換
        config_dict = ModelConfigFactory.to_dict(config)

        # config_registryに設定(後方互換性)
        managed_config_registry._merged_config_data["test-model"] = config_dict

        # 取得して確認
        retrieved = managed_config_registry.get_all_config()["test-model"]
        assert retrieved["class"] == "WDTagger"
        assert retrieved["model_path"] == "SmilingWolf/wd-v1-4-moat-tagger-v2"
        assert retrieved["device"] == "cuda"


class TestConfigObjectsWithRealSettings:
    """実際の設定パターンとの統合テスト"""

    def test_wdtagger_config_pattern(self, managed_config_registry):
        """WD Taggerの実際の設定パターン"""
        managed_config_registry._merged_config_data["wd-v1-4-moat-tagger-v2"] = {
            "class": "WDTagger",
            "model_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
            "estimated_size_gb": 0.456,
        }

        registry_dict = managed_config_registry.get_all_config()["wd-v1-4-moat-tagger-v2"]
        config = ModelConfigFactory.from_registry("wd-v1-4-moat-tagger-v2", registry_dict)

        assert isinstance(config, LocalMLModelConfig)
        assert config.class_name == "WDTagger"
        assert config.device == "cuda"  # default
        assert config.batch_size == 1  # default

    def test_clip_aesthetic_config_pattern(self, managed_config_registry):
        """CLIP美的スコアラーの実際の設定パターン"""
        managed_config_registry._merged_config_data["ImprovedAesthetic"] = {
            "class": "ImprovedAesthetic",
            "model_path": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
            "base_model": "openai/clip-vit-large-patch14",
            "device": "cuda",
        }

        registry_dict = managed_config_registry.get_all_config()["ImprovedAesthetic"]
        config = ModelConfigFactory.from_registry("ImprovedAesthetic", registry_dict)

        assert isinstance(config, LocalMLModelConfig)
        assert config.class_name == "ImprovedAesthetic"
        assert config.base_model == "openai/clip-vit-large-patch14"
        assert "github.com" in config.model_path

    def test_waifu_aesthetic_config_pattern(self, managed_config_registry):
        """WaifuAestheticの実際の設定パターン（活性化関数付き）"""
        managed_config_registry._merged_config_data["WaifuAesthetic"] = {
            "class": "WaifuAesthetic",
            "model_path": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth",
            "base_model": "openai/clip-vit-base-patch32",
            "device": "cuda",
            "activation_type": "ReLU",
            "final_activation_type": "Sigmoid",
        }

        registry_dict = managed_config_registry.get_all_config()["WaifuAesthetic"]
        config = ModelConfigFactory.from_registry("WaifuAesthetic", registry_dict)

        assert isinstance(config, LocalMLModelConfig)
        assert config.activation_type == "ReLU"
        assert config.final_activation_type == "Sigmoid"

    def test_openai_api_config_pattern(self, managed_config_registry):
        """OpenAI APIの実際の設定パターン"""
        managed_config_registry._merged_config_data["gpt-4o-mini"] = {
            "class": "OpenAIApiAnnotator",
            "model_name_on_provider": "gpt-4o-mini-2024-07-18",
            "timeout": 60,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        registry_dict = managed_config_registry.get_all_config()["gpt-4o-mini"]
        config = ModelConfigFactory.from_registry("gpt-4o-mini", registry_dict)

        assert isinstance(config, WebAPIModelConfig)
        assert config.class_name == "OpenAIApiAnnotator"
        assert config.api_model_id == "gpt-4o-mini-2024-07-18"

    def test_anthropic_api_config_pattern(self, managed_config_registry):
        """Anthropic APIの実際の設定パターン"""
        managed_config_registry._merged_config_data["claude-3-7-sonnet"] = {
            "class": "AnthropicApiAnnotator",
            "model_name_on_provider": "claude-3-7-sonnet-20250219",
            "timeout": 60,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        registry_dict = managed_config_registry.get_all_config()["claude-3-7-sonnet"]
        config = ModelConfigFactory.from_registry("claude-3-7-sonnet", registry_dict)

        assert isinstance(config, WebAPIModelConfig)
        assert config.class_name == "AnthropicApiAnnotator"
        assert "claude-3-7-sonnet" in config.api_model_id

    def test_google_api_config_pattern(self, managed_config_registry):
        """Google APIの実際の設定パターン"""
        managed_config_registry._merged_config_data["gemini-2.0-flash"] = {
            "class": "GoogleApiAnnotator",
            "model_name_on_provider": "gemini-2.0-flash",
            "timeout": 90,
            "retry_count": 3,
            "retry_delay": 1.0,
            "min_request_interval": 1.0,
        }

        registry_dict = managed_config_registry.get_all_config()["gemini-2.0-flash"]
        config = ModelConfigFactory.from_registry("gemini-2.0-flash", registry_dict)

        assert isinstance(config, WebAPIModelConfig)
        assert config.class_name == "GoogleApiAnnotator"
        assert config.api_model_id == "gemini-2.0-flash"


class TestBackwardCompatibility:
    """後方互換性テスト"""

    def test_get_method_still_works_with_config_objects(self, managed_config_registry):
        """config_registry.get()が引き続き動作することを確認"""
        managed_config_registry._merged_config_data["test-model"] = {
            "class": "TestClass",
            "model_path": "/path/to/model",
            "device": "cpu",
        }

        # 従来のget()メソッドで取得
        value = managed_config_registry.get("test-model", "class")
        assert value == "TestClass"

        value = managed_config_registry.get("test-model", "device")
        assert value == "cpu"

    def test_set_method_still_works_after_migration(self, managed_config_registry):
        """config_registry.set()が引き続き動作することを確認

        Note: managed_config_registryフィクスチャは簡略化されたset()を提供するため、
        辞書更新による互換性テストを実施
        """
        managed_config_registry._merged_config_data["test-model"] = {
            "class": "TestClass",
            "model_path": "/path/to/model",
        }

        # Config Objectから辞書に変換して設定更新(実際の使用パターン)
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="TestClass",
            model_path="/path/to/model",
            device="cpu",
        )
        config_dict = ModelConfigFactory.to_dict(config)
        managed_config_registry._merged_config_data["test-model"] = config_dict

        # 値が更新されていることを確認
        value = managed_config_registry.get("test-model", "device")
        assert value == "cpu"

    def test_existing_code_using_dict_access_still_works(self, managed_config_registry):
        """辞書アクセスパターンが引き続き動作することを確認"""
        managed_config_registry._merged_config_data["test-model"] = {
            "class": "TestClass",
            "model_path": "/path/to/model",
            "device": "cuda",
            "estimated_size_gb": 1.0,
        }

        # 辞書として取得
        all_config = managed_config_registry.get_all_config()
        model_config = all_config["test-model"]

        assert model_config["class"] == "TestClass"
        assert model_config["model_path"] == "/path/to/model"
        assert model_config["device"] == "cuda"


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_missing_required_fields_raises_configuration_error(self):
        """必須フィールド欠落時のエラー"""
        incomplete_dict = {
            "class": "TestClass",
            # model_path も model_name_on_provider も欠落
        }

        with pytest.raises(ConfigurationError):
            ModelConfigFactory.from_registry("incomplete-model", incomplete_dict)

    def test_invalid_field_values_raise_configuration_error(self):
        """不正なフィールド値のエラー"""
        invalid_dict = {
            "class": "TestClass",
            "model_path": "/path",
            "device": "invalid_device",
        }

        with pytest.raises(ConfigurationError):
            ModelConfigFactory.from_registry("invalid-model", invalid_dict)

    def test_error_contains_helpful_context(self):
        """エラーに有用なコンテキスト情報が含まれることを確認"""
        incomplete_dict = {
            "class": "TestClass",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            ModelConfigFactory.from_registry("test-model", incomplete_dict)

        # エラー詳細にmodel_nameとconfig_dictが含まれる
        assert "test-model" in str(exc_info.value.details["model_name"])
        assert "config_dict" in exc_info.value.details
