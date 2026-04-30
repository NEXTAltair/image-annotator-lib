"""BaseAnnotator Dependency Injection機能の単体テスト (Phase 1B)

このモジュールは、BaseAnnotatorのConfig Object注入機能をテストします。

テスト対象:
- Config Objectの注入
- 後方互換性(config=None時のフォールバック)
- 型安全性
- エラーハンドリング
"""

import pytest

from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.model_config import LocalMLModelConfig
from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult
from image_annotator_lib.exceptions.errors import ConfigurationError


# テスト用の具象アノテータークラス
class ConcreteTestAnnotator(BaseAnnotator):
    """テスト用の具象アノテータークラス"""

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        pass

    def _preprocess_images(self, images):
        """前処理(テスト用スタブ)"""
        return images

    def _run_inference(self, processed):
        """推論実行(テスト用スタブ)"""
        return processed

    def _format_predictions(self, raw_outputs):
        """結果整形(テスト用スタブ)。UnifiedAnnotationResult を返す。"""
        return [
            UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities={TaskCapability.TAGS},
                tags=["test_tag"],
            )
            for _ in raw_outputs
        ]


@pytest.fixture
def managed_config_registry():
    """テスト用のconfig_registryモック"""
    import copy

    from image_annotator_lib.core.config import config_registry

    # 元の状態を保存
    original_system = copy.deepcopy(config_registry._system_config_data)
    original_user = copy.deepcopy(config_registry._user_config_data)
    original_merged = copy.deepcopy(config_registry._merged_config_data)

    # テスト用にクリア
    config_registry._system_config_data.clear()
    config_registry._user_config_data.clear()
    config_registry._merged_config_data.clear()

    yield config_registry

    # 元の状態を復元
    config_registry._system_config_data = original_system
    config_registry._user_config_data = original_user
    config_registry._merged_config_data = original_merged


class TestConfigObjectInjection:
    """Config Object注入機能のテスト"""

    def test_direct_config_injection(self):
        """Config Objectを直接注入できることを確認"""
        # Config Objectを作成
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="ConcreteTestAnnotator",
            model_path="/path/to/model",
            device="cpu",
            estimated_size_gb=1.5,
        )

        # Config Objectを注入してアノテーター生成
        annotator = ConcreteTestAnnotator(model_name="test-model", config=config)

        # Config Objectの内容が正しく設定されていることを確認
        assert annotator._config == config
        assert annotator.model_name == "test-model"
        assert annotator.model_path == "/path/to/model"
        assert annotator.device == "cpu"

    def test_config_attributes_accessible(self):
        """Config Objectの属性にアクセスできることを確認"""
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="ConcreteTestAnnotator",
            model_path="/path/to/model",
            device="cuda",
            estimated_size_gb=2.0,
            batch_size=8,
        )

        annotator = ConcreteTestAnnotator(model_name="test-model", config=config)

        # 各属性に正しくアクセスできることを確認
        assert annotator._config.model_name == "test-model"
        assert annotator._config.class_name == "ConcreteTestAnnotator"
        assert annotator._config.device == "cuda"
        assert annotator._config.estimated_size_gb == 2.0
        assert annotator._config.batch_size == 8

    def test_config_immutability(self):
        """Config Objectがイミュータブルであることを確認"""
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="ConcreteTestAnnotator",
            model_path="/path/to/model",
            device="cpu",
        )

        annotator = ConcreteTestAnnotator(model_name="test-model", config=config)

        # Config Objectを変更しようとするとエラーになることを確認
        with pytest.raises(Exception):  # Pydantic ValidationError
            annotator._config.device = "cuda"


class TestBackwardCompatibility:
    """後方互換性のテスト"""

    def test_none_config_loads_from_registry(self, managed_config_registry, mock_cuda_available):
        """config=Noneの場合、config_registryから読み込むことを確認"""
        # config_registryにテストデータを設定
        managed_config_registry._merged_config_data["test-model"] = {
            "class": "ConcreteTestAnnotator",
            "model_path": "/registry/path",
            "device": "cuda",
            "estimated_size_gb": 1.0,
        }

        # config引数なしでアノテーター生成
        annotator = ConcreteTestAnnotator(model_name="test-model")

        # config_registryの設定が使用されていることを確認
        assert annotator.model_name == "test-model"
        assert annotator.model_path == "/registry/path"
        assert annotator.device == "cuda"
        assert annotator._config.estimated_size_gb == 1.0

    def test_legacy_code_pattern_still_works(self, managed_config_registry):
        """既存コードのパターン(引数なし)が引き続き動作することを確認"""
        managed_config_registry._merged_config_data["legacy-model"] = {
            "class": "ConcreteTestAnnotator",
            "model_path": "/legacy/path",
            "device": "cpu",
        }

        # 既存コードのパターン(model_nameのみ指定)
        annotator = ConcreteTestAnnotator("legacy-model")

        assert annotator.model_name == "legacy-model"
        assert annotator.model_path == "/legacy/path"
        assert annotator.device == "cpu"

    def test_explicit_none_triggers_registry_fallback(self, managed_config_registry):
        """config=Noneを明示的に指定した場合もフォールバックすることを確認"""
        managed_config_registry._merged_config_data["explicit-none"] = {
            "class": "ConcreteTestAnnotator",
            "model_path": "/explicit/path",
            "device": "cpu",
        }

        # config=Noneを明示的に指定
        annotator = ConcreteTestAnnotator(model_name="explicit-none", config=None)

        assert annotator.model_path == "/explicit/path"


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_missing_model_in_registry_raises_error(self, managed_config_registry):
        """config_registryに存在しないモデルを指定するとエラーになることを確認"""
        # 空のレジストリ(モデル未登録)
        managed_config_registry._merged_config_data = {}

        # 存在しないモデルを指定
        with pytest.raises(ValueError, match="not found in config_registry"):
            ConcreteTestAnnotator(model_name="non-existent-model")

    def test_invalid_registry_data_raises_configuration_error(self, managed_config_registry):
        """不正なconfig_registryデータの場合、ConfigurationErrorになることを確認"""
        # 不正なデータ(必須フィールド欠落)
        managed_config_registry._merged_config_data["invalid-model"] = {
            "class": "ConcreteTestAnnotator",
            # model_path が欠落
        }

        with pytest.raises(ConfigurationError):
            ConcreteTestAnnotator(model_name="invalid-model")


class TestMultipleInstances:
    """複数インスタンスのテスト"""

    def test_multiple_annotators_with_different_configs(self, mock_cuda_available):
        """異なるConfig Objectで複数のアノテーターを生成できることを確認"""
        config1 = LocalMLModelConfig(
            model_name="model-1",
            class_name="ConcreteTestAnnotator",
            model_path="/path/1",
            device="cuda",
        )

        config2 = LocalMLModelConfig(
            model_name="model-2",
            class_name="ConcreteTestAnnotator",
            model_path="/path/2",
            device="cpu",
        )

        annotator1 = ConcreteTestAnnotator(model_name="model-1", config=config1)
        annotator2 = ConcreteTestAnnotator(model_name="model-2", config=config2)

        # 各アノテーターが独立した設定を持つことを確認
        assert annotator1.model_path == "/path/1"
        assert annotator1.device == "cuda"
        assert annotator2.model_path == "/path/2"
        assert annotator2.device == "cpu"

    def test_config_object_shared_across_instances(self):
        """同じConfig Objectを複数のアノテーターで共有できることを確認"""
        shared_config = LocalMLModelConfig(
            model_name="shared-model",
            class_name="ConcreteTestAnnotator",
            model_path="/shared/path",
            device="cuda",
        )

        annotator1 = ConcreteTestAnnotator(model_name="shared-model", config=shared_config)
        annotator2 = ConcreteTestAnnotator(model_name="shared-model", config=shared_config)

        # 両方のアノテーターが同じConfig Objectを参照していることを確認
        assert annotator1._config is annotator2._config
        assert annotator1.model_path == annotator2.model_path


# ==============================================================================
# Phase A Task 4: BaseAnnotator DI Edge Cases (2025-12-03)
# ==============================================================================


class TestConfigOverrideScenarios:
    """Config override and precedence tests."""

    def test_direct_config_overrides_registry(self, managed_config_registry, mock_cuda_available):
        """Test that directly injected config takes precedence over registry.

        Scenario:
        - Set config in registry
        - Inject different config directly
        - Verify direct config is used

        Tests:
        - Config injection precedence
        - Direct config override
        """
        # Setup registry config
        managed_config_registry._merged_config_data["test-model"] = {
            "class": "ConcreteTestAnnotator",
            "model_path": "/registry/path",
            "device": "cpu",
        }

        # Create direct config with different settings
        direct_config = LocalMLModelConfig(
            model_name="test-model",
            class_name="ConcreteTestAnnotator",
            model_path="/direct/path",
            device="cuda",
        )

        # Create annotator with direct config
        annotator = ConcreteTestAnnotator(model_name="test-model", config=direct_config)

        # Verify direct config was used (not registry)
        assert annotator.model_path == "/direct/path"
        assert annotator.device == "cuda"

    def test_config_device_fallback_to_cpu(self, managed_config_registry, mock_cuda_unavailable):
        """Test device fallback to CPU when CUDA unavailable.

        Scenario:
        - Config specifies CUDA
        - CUDA not available
        - Verify fallback to CPU

        Tests:
        - Device availability checking
        - Automatic CPU fallback
        """
        config = LocalMLModelConfig(
            model_name="test-model",
            class_name="ConcreteTestAnnotator",
            model_path="/path/to/model",
            device="cuda",
        )

        # Create annotator (should fallback to CPU)
        annotator = ConcreteTestAnnotator(model_name="test-model", config=config)

        # Verify device fallback
        assert annotator.device == "cpu"


class TestConfigValidationEdgeCases:
    """Edge cases for config validation."""

    def test_config_with_minimal_fields(self):
        """Test config with only required fields.

        Scenario:
        - Create config with minimal required fields
        - Create annotator
        - Verify defaults are applied

        Tests:
        - Minimal config acceptance
        - Default value application
        """
        config = LocalMLModelConfig(
            model_name="minimal-model",
            class_name="ConcreteTestAnnotator",
            model_path="/minimal/path",
        )

        annotator = ConcreteTestAnnotator(model_name="minimal-model", config=config)

        # Verify required fields set
        assert annotator.model_name == "minimal-model"
        assert annotator.model_path == "/minimal/path"

        # Verify device default applied
        assert annotator.device in ["cpu", "cuda"]  # Should have a device

    def test_config_immutability_after_annotator_creation(self):
        """Test that config remains immutable after annotator creation.

        Scenario:
        - Create annotator with config
        - Attempt to modify config attributes
        - Verify modifications fail

        Tests:
        - Config immutability
        - Attribute protection
        """
        config = LocalMLModelConfig(
            model_name="immutable-test",
            class_name="ConcreteTestAnnotator",
            model_path="/immutable/path",
            device="cpu",
        )

        annotator = ConcreteTestAnnotator(model_name="immutable-test", config=config)

        # Attempt to modify config
        with pytest.raises(Exception):  # Pydantic ValidationError
            annotator._config.model_path = "/modified/path"

        # Verify original value unchanged
        assert annotator.model_path == "/immutable/path"

    def test_multiple_annotators_with_same_config_object(self):
        """Test creating multiple annotators sharing same config object.

        Scenario:
        - Create single config object
        - Create multiple annotators with same config
        - Verify all share config reference

        Tests:
        - Config object reusability
        - Shared config reference
        """
        shared_config = LocalMLModelConfig(
            model_name="shared-model",
            class_name="ConcreteTestAnnotator",
            model_path="/shared/path",
            device="cpu",
        )

        # Create multiple annotators
        annotator1 = ConcreteTestAnnotator(model_name="shared-model", config=shared_config)
        annotator2 = ConcreteTestAnnotator(model_name="shared-model", config=shared_config)
        annotator3 = ConcreteTestAnnotator(model_name="shared-model", config=shared_config)

        # Verify all share same config object
        assert annotator1._config is annotator2._config
        assert annotator2._config is annotator3._config
        assert id(annotator1._config) == id(annotator2._config) == id(annotator3._config)

        # Verify all have same settings
        assert annotator1.model_path == annotator2.model_path == annotator3.model_path
        assert annotator1.device == annotator2.device == annotator3.device
