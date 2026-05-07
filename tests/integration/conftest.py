# tests/integration/conftest.py
import copy
import shutil
import tempfile

import pytest
from PIL import Image
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.config import config_registry


@pytest.fixture(scope="session", autouse=True)
def disable_real_api_requests():
    """
    Globally disable real API requests for all integration tests.
    This is a safeguard to prevent accidental API calls.
    """
    models.ALLOW_MODEL_REQUESTS = False
    yield


# ADR 0023 Phase 1: PydanticAIAgentFactory cache は廃止された。
# Agent / Provider / Model は推論呼び出しごとに新規作成されるため、テスト間の
# cache クリアは不要 (旧 clear_pydantic_ai_cache fixture は削除)。


@pytest.fixture(scope="function")
def managed_config_registry():
    """
    A fixture to manage the global config_registry for tests.
    It saves the original state, allows modification during a test,
    and restores the original state afterward.
    """
    original_system = copy.deepcopy(config_registry._system_config_data)
    original_user = copy.deepcopy(config_registry._user_config_data)
    original_merged = copy.deepcopy(config_registry._merged_config_data)

    # Provide a clean registry for the test
    config_registry._system_config_data.clear()
    config_registry._user_config_data.clear()
    config_registry._merged_config_data.clear()

    def _set_config(model_name: str, config: dict):
        """Helper to set config for a test."""
        config_registry._merged_config_data[model_name] = config
        config_registry._user_config_data[model_name] = config

    # Temporarily replace the set method for test isolation
    original_set = config_registry.set
    config_registry.set = _set_config

    yield config_registry

    # Restore original state
    config_registry._system_config_data = original_system
    config_registry._user_config_data = original_user
    config_registry._merged_config_data = original_merged
    config_registry.set = original_set


# ADR 0023 Phase 1 (Issue #35): WebAPI 系は `WebApiAnnotator` 1 種に統一されたため、
# 旧プロバイダー固有クラス (OpenAIApiAnnotator / AnthropicApiAnnotator / 等) を直接 import
# する `_ensure_test_class_mapping` ヘルパーは削除された。WebAPI モデルの test 登録は
# `_register_webapi_models_from_discovery` 経由か、`_MODEL_CLASS_OBJ_REGISTRY` への
# 直接 monkeypatch で対応する。


@pytest.fixture(scope="session")
def integration_test_config():
    """
    Provides an isolated test configuration for integration tests.
    Creates temporary directories and test configurations.
    """
    temp_dir = tempfile.mkdtemp(prefix="integration_test_")

    config = {"test_mode": True, "temp_dir": temp_dir, "timeout": 30, "max_retries": 2, "batch_size": 1}

    yield config

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def lightweight_test_images():
    """
    Provides standardized lightweight test images for integration tests.
    Each image is slightly different to ensure a unique pHash.
    """
    images = []
    for i, color in enumerate(["red", "green", "blue"]):
        img = Image.new("RGB", (64, 64), color)
        # Add a single different pixel to each image to ensure unique phash
        img.putpixel((i, i), (255, 255, 255))
        images.append(img)
    return images


@pytest.fixture
def mock_api_responses():
    """
    Provides realistic mock API responses for fast integration tests.
    Placeholder: Returns a dummy dictionary.
    """
    return {
        "google": {"tags": ["mock_tag_1", "mock_tag_2"]},
        "openai": {"caption": "a mock caption"},
    }


class ApiKeyManager:
    """A fixture for managing API keys for real API tests."""

    def __init__(self):
        self.dummy_keys = {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
            "google": "test-google-key",
        }

    def get_key(self, provider: str) -> str | None:
        """Get API key for provider"""
        return self.dummy_keys.get(provider)

    def has_real_key(self, provider: str) -> bool:
        """Check if real API key is available"""
        import os

        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        env_var = env_vars.get(provider)
        return env_var is not None and bool(os.getenv(env_var))


@pytest.fixture(scope="session")
def api_key_manager():
    """
    Manages API keys for tests that hit real APIs.
    """
    return ApiKeyManager()


@pytest.fixture
def test_image_variants():
    """
    Provides different image formats for testing robustness.
    """
    import base64
    import io

    # Create base test image
    base_image = Image.new("RGB", (32, 32), "red")

    # PNG bytes
    png_buffer = io.BytesIO()
    base_image.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    # JPEG bytes
    jpeg_buffer = io.BytesIO()
    base_image.save(jpeg_buffer, format="JPEG")
    jpeg_bytes = jpeg_buffer.getvalue()

    # Base64 encoded
    base64_data = base64.b64encode(png_bytes).decode("utf-8")

    return {
        "pil_image": base_image,
        "png_bytes": png_bytes,
        "jpeg_bytes": jpeg_bytes,
        "base64_data": base64_data,
        "formats": ["PNG", "JPEG"],
    }


@pytest.fixture
def pydantic_ai_test_model() -> TestModel:
    """Provides a PydanticAI TestModel for integration tests."""
    test_model = TestModel()
    # Set a default response that can be used by tests
    test_model.response = ModelResponse(parts=[TextPart('{"tags": ["test_model_tag"]}')])
    return test_model


@pytest.fixture
def pydantic_ai_function_model() -> FunctionModel:
    """Provides a PydanticAI FunctionModel for custom logic in integration tests."""

    def custom_logic(messages, info):
        return ModelResponse(parts=[TextPart('{"tags": ["custom_integration_tag"]}')])

    return FunctionModel(custom_logic)
