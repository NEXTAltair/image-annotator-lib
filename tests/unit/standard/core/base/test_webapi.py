"""WebApiBaseAnnotator クラスのテスト"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.base.webapi import WebApiBaseAnnotator
from image_annotator_lib.core.types import (
    AnnotationSchema,
    TaskCapability,
    UnifiedAnnotationResult,
    WebApiComponents,
)
from image_annotator_lib.exceptions.errors import ApiAuthenticationError, ConfigurationError


@pytest.fixture(autouse=True)
def setup_test_model_config():
    """Setup test model configuration for all tests."""
    from image_annotator_lib.core.config import config_registry

    # Use unique model name to avoid conflicts with existing registry state
    test_model_name = "webapi_unittest_model"

    # Cleanup first to ensure no leftover settings
    try:
        config_registry._config.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass

    # Set up WebAPIModelConfig-compatible configuration (no model_path)
    config = {
        "device": "cpu",
        "class": "MockWebApiAnnotator",
        "api_model_id": "test-api-model-id",
        "model_name_on_provider": "test-provider-model",
    }
    for key, value in config.items():
        config_registry.add_default_setting(test_model_name, key, value)

    yield

    # Cleanup after test
    try:
        config_registry._config.pop(test_model_name, None)
    except (AttributeError, KeyError):
        pass


class MockWebApiAnnotator(WebApiBaseAnnotator):
    """テスト用の WebApiBaseAnnotator 実装"""

    def __init__(self, model_name: str = "webapi_unittest_model"):
        super().__init__(model_name)
        self.inference_called = False

    def _run_inference(self, processed: list[str] | list[bytes]) -> Any:
        """モック推論実装"""
        self.inference_called = True
        return [{"result": "test"}]


class TestWebApiBaseAnnotator:
    """WebApiBaseAnnotator クラスのテスト"""

    # ================================================================================
    # 初期化テスト - 設定値エラーハンドリング
    # ================================================================================
    # NOTE: Phase 1B でのPydantic config validation導入により、これらのテストは実行不可能
    # Pydanticがinvalid値を拒否するため、設定変換時にConfigurationErrorが発生
    # 今後、config validation層でのエラーハンドリングテストに置き換える必要がある

    @pytest.mark.skip(reason="Pydantic config validation makes invalid value testing infeasible")
    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_init_with_invalid_timeout(self, mock_logger):
        """無効なtimeout値でのデフォルト設定テスト"""
        from image_annotator_lib.core.config import config_registry

        # 一時的に無効な値を設定
        original_value = config_registry.get("test_model", "timeout", None)
        config_registry.add_default_setting("test_model", "timeout", "invalid_timeout")  # 不正な文字列

        try:
            annotator = MockWebApiAnnotator()

            assert annotator.timeout == 60  # デフォルト値
            mock_logger.warning.assert_called()
            assert "timeout に不正な値" in str(mock_logger.warning.call_args)
        finally:
            # 復元
            if original_value is not None:
                config_registry.add_default_setting("test_model", "timeout", original_value)

    @pytest.mark.skip(reason="Pydantic config validation makes invalid value testing infeasible")
    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_init_with_invalid_retry_count(self, mock_logger):
        """無効なretry_count値でのデフォルト設定テスト"""
        from image_annotator_lib.core.config import config_registry

        # 一時的に無効な値を設定
        original_value = config_registry.get("test_model", "retry_count", None)
        config_registry.add_default_setting("test_model", "retry_count", [1, 2, 3])  # 不正なリスト

        try:
            annotator = MockWebApiAnnotator()

            assert annotator.retry_count == 3  # デフォルト値
            mock_logger.warning.assert_called()
            assert "retry_count に不正な値" in str(mock_logger.warning.call_args)
        finally:
            # 復元
            if original_value is not None:
                config_registry.add_default_setting("test_model", "retry_count", original_value)

    @pytest.mark.skip(reason="Pydantic config validation makes invalid value testing infeasible")
    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_init_with_invalid_retry_delay(self, mock_logger):
        """無効なretry_delay値でのデフォルト設定テスト"""
        from image_annotator_lib.core.config import config_registry

        # 一時的に無効な値を設定
        original_value = config_registry.get("test_model", "retry_delay", None)
        config_registry.add_default_setting("test_model", "retry_delay", {"invalid": "dict"})  # 不正な辞書

        try:
            annotator = MockWebApiAnnotator()

            assert annotator.retry_delay == 1.0  # デフォルト値
            mock_logger.warning.assert_called()
            assert "retry_delay に不正な値" in str(mock_logger.warning.call_args)
        finally:
            # 復元
            if original_value is not None:
                config_registry.add_default_setting("test_model", "retry_delay", original_value)

    @pytest.mark.skip(reason="Pydantic config validation makes invalid value testing infeasible")
    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_init_with_invalid_min_request_interval(self, mock_logger):
        """無効なmin_request_interval値でのデフォルト設定テスト"""
        from image_annotator_lib.core.config import config_registry

        # 一時的に無効な値を設定
        original_value = config_registry.get("test_model", "min_request_interval", None)
        config_registry.add_default_setting("test_model", "min_request_interval", None)  # None値

        try:
            annotator = MockWebApiAnnotator()

            assert annotator.min_request_interval == 1.0  # デフォルト値
            # None値の場合はValueError/TypeErrorになる可能性がある
        finally:
            # 復元
            if original_value is not None:
                config_registry.add_default_setting("test_model", "min_request_interval", original_value)

    @pytest.mark.skip(reason="Pydantic config validation makes invalid value testing infeasible")
    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_init_with_invalid_max_output_tokens(self, mock_logger):
        """無効なmax_output_tokens値でのデフォルト設定テスト"""
        from image_annotator_lib.core.config import config_registry

        # 一時的に無効な値を設定
        original_value = config_registry.get("test_model", "max_output_tokens", None)
        config_registry.add_default_setting(
            "test_model", "max_output_tokens", "invalid_string"
        )  # 不正な文字列

        try:
            annotator = MockWebApiAnnotator()

            assert annotator.max_output_tokens is None  # エラー時はNone
            mock_logger.warning.assert_called()
            assert "max_output_tokens に不正な値" in str(mock_logger.warning.call_args)
        finally:
            # 復元
            if original_value is not None:
                config_registry.add_default_setting("test_model", "max_output_tokens", original_value)

    # ================================================================================
    # コンテキストマネージャーテスト (__enter__ / __exit__)
    # ================================================================================

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.prepare_web_api_components")
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_context_manager_enter_success(self, mock_logger, mock_prepare):
        """コンテキストマネージャー __enter__ 成功テスト"""
        # WebApiComponents の準備
        mock_components: WebApiComponents = {
            "client": Mock(),
            "api_model_id": "test-api-model-id",
            "provider_name": "test_provider",
        }
        mock_prepare.return_value = mock_components

        annotator = MockWebApiAnnotator()

        with annotator:
            assert annotator.client is not None
            assert annotator.api_model_id == "test-api-model-id"
            assert annotator.provider_name == "test_provider"
            assert annotator.components == mock_components

        mock_prepare.assert_called_once_with("webapi_unittest_model")
        mock_logger.info.assert_called()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.prepare_web_api_components")
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_context_manager_enter_configuration_error(self, mock_logger, mock_prepare):
        """コンテキストマネージャー __enter__ 設定エラーテスト"""
        mock_prepare.side_effect = ConfigurationError("Configuration failed")

        annotator = MockWebApiAnnotator()

        with pytest.raises(ConfigurationError):
            with annotator:
                pass

        # エラー時にクリーンアップされることを確認
        assert annotator.components is None
        assert annotator.client is None
        assert annotator.api_model_id is None
        assert annotator.provider_name is None
        mock_logger.error.assert_called()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.prepare_web_api_components")
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_context_manager_enter_authentication_error(self, mock_logger, mock_prepare):
        """コンテキストマネージャー __enter__ 認証エラーテスト"""
        mock_prepare.side_effect = ApiAuthenticationError("Authentication failed")

        annotator = MockWebApiAnnotator()

        with pytest.raises(ApiAuthenticationError):
            with annotator:
                pass

        assert annotator.components is None
        mock_logger.error.assert_called()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.prepare_web_api_components")
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_context_manager_enter_unexpected_error(self, mock_logger, mock_prepare):
        """コンテキストマネージャー __enter__ 予期せぬエラーテスト"""
        mock_prepare.side_effect = RuntimeError("Unexpected error")

        annotator = MockWebApiAnnotator()

        with pytest.raises(ConfigurationError) as exc_info:
            with annotator:
                pass

        assert "予期せぬエラー" in str(exc_info.value)
        assert annotator.components is None
        mock_logger.exception.assert_called()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.prepare_web_api_components")
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_context_manager_exit_with_client(self, mock_logger, mock_prepare):
        """コンテキストマネージャー __exit__ クライアント解放テスト"""
        mock_components: WebApiComponents = {
            "client": Mock(),
            "api_model_id": "test-api-model-id",
            "provider_name": "test_provider",
        }
        mock_prepare.return_value = mock_components

        annotator = MockWebApiAnnotator()

        with annotator:
            assert annotator.client is not None

        # __exit__後にクリーンアップされることを確認
        assert annotator.client is None
        assert annotator.components is None
        mock_logger.debug.assert_called()

    # ================================================================================
    # _preprocess_images テスト
    # ================================================================================

    @pytest.mark.standard
    def test_preprocess_images_success(self):
        """画像前処理（Base64エンコード）成功テスト"""
        # PIL Image モック
        mock_image = Mock(spec=Image.Image)
        images = [mock_image, mock_image]

        annotator = MockWebApiAnnotator()
        result = annotator._preprocess_images(images)

        assert len(result) == 2
        assert all(isinstance(encoded, str) for encoded in result)

    # ================================================================================
    # _parse_common_json_response テスト
    # ================================================================================

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_parse_common_json_response_with_dict_success(self, mock_logger):
        """_parse_common_json_response - 辞書形式成功テスト"""
        annotator = MockWebApiAnnotator()
        text_content = {
            "tags": ["tag1", "tag2"],
            "captions": ["caption1"],
            "score": 0.95,
        }

        result = annotator._parse_common_json_response(text_content)

        assert result["annotation"] is not None
        assert result["error"] is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_parse_common_json_response_with_invalid_dict(self, mock_logger):
        """_parse_common_json_response - 無効な辞書テスト"""
        annotator = MockWebApiAnnotator()
        text_content = {"invalid_key": "invalid_value"}  # AnnotationSchemaに合わない

        result = annotator._parse_common_json_response(text_content)

        assert result["annotation"] is None
        assert result["error"] is not None
        assert "does not match AnnotationSchema" in result["error"]

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_parse_common_json_response_json_decode_error(self, mock_logger):
        """_parse_common_json_response - JSON解析エラーテスト"""
        annotator = MockWebApiAnnotator()
        text_content = "{invalid json"  # 不正なJSON

        result = annotator._parse_common_json_response(text_content)

        assert result["annotation"] is None
        assert result["error"] is not None
        assert "JSON解析エラー" in result["error"]
        mock_logger.error.assert_called()

    @pytest.mark.standard
    @patch("image_annotator_lib.core.base.webapi.logger")
    def test_parse_common_json_response_unexpected_error(self, mock_logger):
        """_parse_common_json_response - 予期せぬエラーテスト"""
        annotator = MockWebApiAnnotator()

        # json.loads が予期せぬエラーを発生させるようにモック
        with patch("image_annotator_lib.core.base.webapi.json.loads") as mock_json_loads:
            mock_json_loads.side_effect = RuntimeError("Unexpected parsing error")

            result = annotator._parse_common_json_response('{"tags": ["tag1"]}')

            assert result["annotation"] is None
            assert result["error"] is not None
            assert "予期せぬエラー" in result["error"]
            mock_logger.exception.assert_called()

    # ================================================================================
    # _generate_tags テスト
    # ================================================================================

    @pytest.mark.standard
    def test_generate_tags_from_dict_with_tags(self):
        """_generate_tags - 辞書形式（tagsあり）テスト"""
        annotator = MockWebApiAnnotator()
        formatted_output = {
            "annotation": {"tags": ["tag1", "tag2", "tag3"]},
            "error": None,
        }

        result = annotator._generate_tags(formatted_output)

        assert result == ["tag1", "tag2", "tag3"]

    @pytest.mark.standard
    def test_generate_tags_from_dict_without_tags(self):
        """_generate_tags - 辞書形式（tagsなし）テスト"""
        annotator = MockWebApiAnnotator()
        formatted_output = {
            "annotation": {"captions": ["caption1"]},
            "error": None,
        }

        result = annotator._generate_tags(formatted_output)

        assert result == []

    @pytest.mark.standard
    def test_generate_tags_with_error(self):
        """_generate_tags - エラー時のテスト"""
        annotator = MockWebApiAnnotator()
        formatted_output = {
            "annotation": None,
            "error": "Some error",
        }

        result = annotator._generate_tags(formatted_output)

        assert result == []

    @pytest.mark.standard
    def test_generate_tags_with_annotation_none(self):
        """_generate_tags - annotation=None時のテスト"""
        annotator = MockWebApiAnnotator()
        formatted_output = {
            "annotation": None,
            "error": None,
        }

        result = annotator._generate_tags(formatted_output)

        assert result == []

    # ================================================================================
    # _format_predictions テスト
    # ================================================================================

    @pytest.mark.standard
    @patch("image_annotator_lib.core.utils.get_model_capabilities")
    def test_format_predictions_with_annotation_schema(self, mock_get_capabilities):
        """_format_predictions - AnnotationSchema成功テスト"""
        mock_get_capabilities.return_value = {
            TaskCapability.TAGS,
            TaskCapability.CAPTIONS,
            TaskCapability.SCORES,
        }

        annotator = MockWebApiAnnotator()
        annotator.provider_name = "test_provider"

        annotation_schema = AnnotationSchema(
            tags=["tag1", "tag2"],
            captions=["caption1"],
            score=0.95,
        )

        raw_outputs = [
            {"response": annotation_schema, "error": None},
        ]

        result = annotator._format_predictions(raw_outputs)

        assert len(result) == 1
        assert isinstance(result[0], UnifiedAnnotationResult)
        assert result[0].tags == ["tag1", "tag2"]
        assert result[0].captions == ["caption1"]
        assert result[0].scores == {"score": 0.95}
        assert result[0].error is None
        assert result[0].provider_name == "test_provider"

    @pytest.mark.standard
    @patch("image_annotator_lib.core.utils.get_model_capabilities")
    def test_format_predictions_with_error(self, mock_get_capabilities):
        """_format_predictions - エラーレスポンステスト"""
        mock_get_capabilities.return_value = {TaskCapability.TAGS}

        annotator = MockWebApiAnnotator()
        annotator.provider_name = "test_provider"

        raw_outputs = [
            {"response": None, "error": "API call failed"},
        ]

        result = annotator._format_predictions(raw_outputs)

        assert len(result) == 1
        assert result[0].error == "API call failed"
        assert result[0].tags is None

    @pytest.mark.standard
    @patch("image_annotator_lib.core.utils.get_model_capabilities")
    def test_format_predictions_with_none_response(self, mock_get_capabilities):
        """_format_predictions - response=None テスト"""
        mock_get_capabilities.return_value = {TaskCapability.TAGS}

        annotator = MockWebApiAnnotator()
        annotator.provider_name = "test_provider"

        raw_outputs = [
            {"response": None, "error": None},
        ]

        result = annotator._format_predictions(raw_outputs)

        assert len(result) == 1
        assert result[0].error == "Response is None"

    @pytest.mark.standard
    @patch("image_annotator_lib.core.utils.get_model_capabilities")
    def test_format_predictions_with_invalid_response_type(self, mock_get_capabilities):
        """_format_predictions - 無効なresponse型テスト"""
        mock_get_capabilities.return_value = {TaskCapability.TAGS}

        annotator = MockWebApiAnnotator()
        annotator.provider_name = "test_provider"

        raw_outputs = [
            {"response": "invalid_string_response", "error": None},
        ]

        result = annotator._format_predictions(raw_outputs)

        assert len(result) == 1
        assert "Invalid response type" in result[0].error
