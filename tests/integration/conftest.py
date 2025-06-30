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
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory


@pytest.fixture(scope="session", autouse=True)
def disable_real_api_requests():
    """
    Globally disable real API requests for all integration tests.
    This is a safeguard to prevent accidental API calls.
    """
    models.ALLOW_MODEL_REQUESTS = False
    yield


@pytest.fixture(scope="function", autouse=True)
def clear_pydantic_ai_cache():
    """
    Automatically clear PydanticAI cache before and after each test
    to ensure test isolation.
    """
    PydanticAIProviderFactory.clear_cache()
    yield
    PydanticAIProviderFactory.clear_cache()


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


def _ensure_test_class_mapping(model_name: str, config: dict):
    """Ensure test model class mappings exist in the class registry."""
    from image_annotator_lib.core.registry import get_cls_obj_registry

    registry = get_cls_obj_registry()
    class_name = config.get("class")

    print(
        f"MAPPING DEBUG: model_name='{model_name}', class_name='{class_name}', already_in_registry={model_name in registry}"
    )

    if class_name and model_name not in registry:
        # For WebAPI models, directly import and register the classes
        if class_name in ["OpenAIApiAnnotator", "AnthropicApiAnnotator", "GoogleApiAnnotator"]:
            try:
                print(f"IMPORT DEBUG: Attempting to import {class_name}")
                if class_name == "OpenAIApiAnnotator":
                    # Use the correct OpenAI class
                    from image_annotator_lib.model_class.annotator_webapi.openai_api_response import (
                        OpenAIApiAnnotator,
                    )

                    registry[model_name] = OpenAIApiAnnotator
                    print(f"IMPORT SUCCESS: {model_name} -> OpenAIApiAnnotator")
                elif class_name == "AnthropicApiAnnotator":
                    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
                        AnthropicApiAnnotator,
                    )

                    registry[model_name] = AnthropicApiAnnotator
                    print(f"IMPORT SUCCESS: {model_name} -> AnthropicApiAnnotator")
                elif class_name == "GoogleApiAnnotator":
                    from image_annotator_lib.model_class.annotator_webapi.google_api import (
                        GoogleApiAnnotator,
                    )

                    registry[model_name] = GoogleApiAnnotator
                    print(f"IMPORT SUCCESS: {model_name} -> GoogleApiAnnotator")
                return
            except ImportError as e:
                print(f"IMPORT FAILED: {class_name} - {e}")
                pass
            except Exception as e:
                print(f"IMPORT ERROR: {class_name} - {e}")
                pass

        # For local models that need direct import
        if class_name == "WDTagger":
            try:
                print("IMPORT DEBUG: Attempting to import WDTagger")
                from image_annotator_lib.model_class.tagger_onnx import WDTagger

                registry[model_name] = WDTagger
                print(f"IMPORT SUCCESS: {model_name} -> WDTagger")
                return
            except ImportError as e:
                print(f"IMPORT FAILED: WDTagger - {e}")
                pass

        # For local models, try existing registry lookup
        if class_name in registry:
            registry[model_name] = registry[class_name]
            print(f"REGISTRY SUCCESS: {model_name} -> {class_name} (from existing registry)")
            return

        # Fallback: search by class name pattern
        for registered_name, class_obj in registry.items():
            if class_name.lower() in registered_name.lower():
                registry[model_name] = class_obj
                print(f"PATTERN SUCCESS: {model_name} -> {registered_name} (pattern match)")
                break
        else:
            print(f"MAPPING FAILED: No solution found for {model_name} with class {class_name}")


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
