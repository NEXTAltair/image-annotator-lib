"""WebApiBaseAnnotator Dependency Injection機能の単体テスト (Phase 1B)

このモジュールは、WebApiBaseAnnotatorのWebAPIModelConfig注入機能をテストします。

テスト対象:
- WebAPIModelConfig注入
- 型チェック(LocalMLModelConfigの拒否)
- API関連パラメータの設定
- 後方互換性
"""

import pytest

from image_annotator_lib.core.base.webapi import WebApiBaseAnnotator
from image_annotator_lib.core.model_config import LocalMLModelConfig, WebAPIModelConfig
from image_annotator_lib.exceptions.errors import ConfigurationError


# テスト用の具象WebAPIアノテータークラス
class ConcreteWebApiAnnotator(WebApiBaseAnnotator):
    """テスト用の具象WebAPIアノテータークラス"""

    def _run_inference(self, processed):
        """推論実行(テスト用スタブ)"""
        return [{"response": "test_response", "error": None}]

    def _preprocess_images(self, images):
        """前処理(テスト用スタブ)"""
        return ["encoded_image"] * len(images)

    def _format_predictions(self, raw_outputs):
        """結果整形(テスト用スタブ)"""
        from image_annotator_lib.core.types import UnifiedAnnotationResult

        return [
            UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=[],
                provider_name="test",
            )
        ] * len(raw_outputs)

    def _generate_tags(self, formatted_output):
        """タグ生成(テスト用スタブ)"""
        return ["test_tag"]


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


class TestWebAPIConfigInjection:
    """WebAPIModelConfig注入機能のテスト"""

    def test_direct_webapi_config_injection(self):
        """WebAPIModelConfigを直接注入できることを確認"""
        # WebAPIModelConfigを作成
        config = WebAPIModelConfig(
            model_name="test-api-model",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="gpt-4o-mini",
            timeout=90,
            retry_count=5,
            retry_delay=2.0,
            min_request_interval=0.5,
            prompt_template="Custom prompt",
            max_output_tokens=2000,
        )

        # Config Objectを注入してアノテーター生成
        annotator = ConcreteWebApiAnnotator(model_name="test-api-model", config=config)

        # Config Objectの内容が正しく設定されていることを確認
        assert annotator._config == config
        assert annotator.model_name == "test-api-model"
        assert annotator.timeout == 90
        assert annotator.retry_count == 5
        assert annotator.retry_delay == 2.0
        assert annotator.min_request_interval == 0.5
        assert annotator.prompt_template == "Custom prompt"
        assert annotator.max_output_tokens == 2000

    def test_webapi_config_attributes_accessible(self):
        """WebAPIModelConfigの各属性にアクセスできることを確認"""
        config = WebAPIModelConfig(
            model_name="test-model",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="claude-3-5-sonnet",
            timeout=120,
            retry_count=10,
            retry_delay=3.0,
            temperature=0.7,
            max_output_tokens=4096,
        )

        annotator = ConcreteWebApiAnnotator(model_name="test-model", config=config)

        # 各属性に正しくアクセスできることを確認
        assert annotator._config.api_model_id == "claude-3-5-sonnet"
        assert annotator._config.timeout == 120
        assert annotator._config.retry_count == 10
        assert annotator._config.temperature == 0.7
        assert annotator._config.max_output_tokens == 4096

    def test_webapi_config_defaults_applied(self):
        """WebAPIModelConfigのデフォルト値が正しく適用されることを確認"""
        # 最小限の設定で作成(デフォルト値に依存)
        config = WebAPIModelConfig(
            model_name="minimal-model",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="gemini-2.0-flash",
        )

        annotator = ConcreteWebApiAnnotator(model_name="minimal-model", config=config)

        # デフォルト値が設定されていることを確認
        assert annotator.timeout == 60  # default
        assert annotator.retry_count == 3  # default
        assert annotator.retry_delay == 1.0  # default
        assert annotator.min_request_interval == 1.0  # default
        assert annotator.prompt_template == "Describe this image."  # default


class TestTypeValidation:
    """型検証のテスト"""

    def test_local_ml_config_rejected(self):
        """LocalMLModelConfigを渡すとエラーになることを確認"""
        # LocalMLModelConfigを作成(WebAPIには不適切)
        config = LocalMLModelConfig(
            model_name="wrong-type",
            class_name="ConcreteWebApiAnnotator",
            model_path="/path/to/model",
            device="cuda",
        )

        # WebApiBaseAnnotatorにLocalMLModelConfigを渡すとエラー
        with pytest.raises(ConfigurationError, match="WebAPIModelConfigが必要"):
            ConcreteWebApiAnnotator(model_name="wrong-type", config=config)


class TestBackwardCompatibility:
    """後方互換性のテスト"""

    def test_none_config_loads_webapi_from_registry(self, managed_config_registry):
        """config=Noneの場合、config_registryからWebAPIModelConfigを読み込むことを確認"""
        # config_registryにWebAPIモデルデータを設定
        managed_config_registry._merged_config_data["test-webapi"] = {
            "class": "ConcreteWebApiAnnotator",
            "model_name_on_provider": "gpt-4o-mini",
            "timeout": 75,
            "retry_count": 4,
            "retry_delay": 1.5,
            "min_request_interval": 2.0,
            "prompt_template": "Registry prompt",
        }

        # config引数なしでアノテーター生成
        annotator = ConcreteWebApiAnnotator(model_name="test-webapi")

        # config_registryの設定が使用されていることを確認
        assert annotator.model_name == "test-webapi"
        assert annotator.timeout == 75
        assert annotator.retry_count == 4
        assert annotator.retry_delay == 1.5
        assert annotator.min_request_interval == 2.0
        assert annotator.prompt_template == "Registry prompt"

    def test_legacy_webapi_pattern_still_works(self, managed_config_registry):
        """既存コードのパターン(引数なし)が引き続き動作することを確認"""
        managed_config_registry._merged_config_data["legacy-api"] = {
            "class": "ConcreteWebApiAnnotator",
            "model_name_on_provider": "claude-3-haiku",
            "timeout": 60,
        }

        # 既存コードのパターン(model_nameのみ指定)
        annotator = ConcreteWebApiAnnotator("legacy-api")

        assert annotator.model_name == "legacy-api"
        assert annotator.timeout == 60


class TestAPIParameters:
    """API関連パラメータのテスト"""

    def test_api_timeout_configuration(self):
        """タイムアウト設定が正しく動作することを確認"""
        config = WebAPIModelConfig(
            model_name="timeout-test",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="test-model",
            timeout=180,
        )

        annotator = ConcreteWebApiAnnotator(model_name="timeout-test", config=config)

        assert annotator.timeout == 180

    def test_retry_configuration(self):
        """リトライ設定が正しく動作することを確認"""
        config = WebAPIModelConfig(
            model_name="retry-test",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="test-model",
            retry_count=7,
            retry_delay=5.0,
        )

        annotator = ConcreteWebApiAnnotator(model_name="retry-test", config=config)

        assert annotator.retry_count == 7
        assert annotator.retry_delay == 5.0

    def test_rate_limit_configuration(self):
        """レート制限設定が正しく動作することを確認"""
        config = WebAPIModelConfig(
            model_name="rate-test",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="test-model",
            min_request_interval=3.0,
        )

        annotator = ConcreteWebApiAnnotator(model_name="rate-test", config=config)

        assert annotator.min_request_interval == 3.0

    def test_generation_parameters_configuration(self):
        """生成パラメータ設定が正しく動作することを確認"""
        config = WebAPIModelConfig(
            model_name="gen-test",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="test-model",
            temperature=0.8,
            max_output_tokens=1024,
            top_p=0.9,
            top_k=50,
        )

        annotator = ConcreteWebApiAnnotator(model_name="gen-test", config=config)

        assert annotator._config.temperature == 0.8
        assert annotator.max_output_tokens == 1024
        assert annotator._config.top_p == 0.9
        assert annotator._config.top_k == 50


class TestMultipleWebAPIInstances:
    """複数WebAPIインスタンスのテスト"""

    def test_multiple_webapi_annotators_with_different_configs(self):
        """異なるWebAPIModelConfigで複数のアノテーターを生成できることを確認"""
        config1 = WebAPIModelConfig(
            model_name="api-1",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="gpt-4o-mini",
            timeout=60,
        )

        config2 = WebAPIModelConfig(
            model_name="api-2",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="claude-3-5-sonnet",
            timeout=120,
        )

        annotator1 = ConcreteWebApiAnnotator(model_name="api-1", config=config1)
        annotator2 = ConcreteWebApiAnnotator(model_name="api-2", config=config2)

        # 各アノテーターが独立した設定を持つことを確認
        assert annotator1._config.api_model_id == "gpt-4o-mini"
        assert annotator1.timeout == 60
        assert annotator2._config.api_model_id == "claude-3-5-sonnet"
        assert annotator2.timeout == 120

    def test_webapi_config_object_shared_across_instances(self):
        """同じWebAPIModelConfigを複数のアノテーターで共有できることを確認"""
        shared_config = WebAPIModelConfig(
            model_name="shared-api",
            class_name="ConcreteWebApiAnnotator",
            api_model_id="gemini-2.0-flash",
            timeout=90,
        )

        annotator1 = ConcreteWebApiAnnotator(model_name="shared-api", config=shared_config)
        annotator2 = ConcreteWebApiAnnotator(model_name="shared-api", config=shared_config)

        # 両方のアノテーターが同じConfig Objectを参照していることを確認
        assert annotator1._config is annotator2._config
        assert annotator1.timeout == annotator2.timeout
