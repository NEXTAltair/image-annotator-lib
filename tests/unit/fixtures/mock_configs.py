"""
設定管理の統一モック
"""

from unittest.mock import patch

import pytest

# 標準テスト設定
STANDARD_TEST_CONFIG = {
    "test_model": {
        "model_path": "/test/path/model",
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "class": "TestAnnotator",
    },
    "webapi_model": {
        "model_path": "gpt-4o",
        "device": "cpu",
        "estimated_size_gb": 0.0,
        "class": "OpenAIApiAnnotator",
        "api_model_id": "gpt-4o",
    },
    "transformers_model": {
        "model_path": "microsoft/blip-image-captioning-base",
        "device": "cuda",
        "estimated_size_gb": 2.5,
        "class": "BlipCaptioner",
    },
}


@pytest.fixture
def mock_config_registry():
    """統一された設定レジストリモック"""
    with patch("image_annotator_lib.core.config.config_registry") as mock:

        def mock_get(model_name: str, key: str, default=None):
            """モック用get実装"""
            if model_name in STANDARD_TEST_CONFIG:
                return STANDARD_TEST_CONFIG[model_name].get(key, default)
            return default

        mock.get.side_effect = mock_get
        mock.get_all_config.return_value = STANDARD_TEST_CONFIG
        mock.list_models.return_value = list(STANDARD_TEST_CONFIG.keys())
        yield mock


@pytest.fixture
def mock_config_registry_with_custom_data():
    """カスタムデータ用設定レジストリモック"""

    def _create_mock(custom_config: dict):
        with patch("image_annotator_lib.core.config.config_registry") as mock:

            def mock_get(model_name: str, key: str, default=None):
                if model_name in custom_config:
                    return custom_config[model_name].get(key, default)
                return default

            mock.get.side_effect = mock_get
            mock.get_all_config.return_value = custom_config
            mock.list_models.return_value = list(custom_config.keys())
            return mock

    return _create_mock


@pytest.fixture
def mock_empty_config():
    """空の設定レジストリモック"""
    with patch("image_annotator_lib.core.config.config_registry") as mock:
        mock.get.return_value = None
        mock.get_all_config.return_value = {}
        mock.list_models.return_value = []
        yield mock


# ADR 0023 Phase 1 (Issue #35, PR #40): `load_available_api_models` は廃止された。
# WebAPI モデル一覧の mock は `image_annotator_lib.webapi.api_model_discovery.discover_available_vision_models`
# を直接 monkeypatch するか、test fixture で `_register_webapi_models_from_discovery` を
# 差し替える形で対応する。本 fixture (`mock_api_models_config`) は削除。
