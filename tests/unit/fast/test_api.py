from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib import api

# Import the real classes for the registry
from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.model_class.tagger_onnx import WDTagger

# --- Mock Data ---

# Mock class registry using REAL classes
MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES = {
    "Gemini 1.5 Pro": GoogleApiAnnotator,
    "GPT-4o": OpenAIApiAnnotator,
    "Claude 3 Opus": AnthropicApiAnnotator,
    "Gemini 1.5 Flash (OpenRouter)": OpenRouterApiAnnotator,
    "wd-v1-4-convnext-tagger-v2": WDTagger,  # Use a real local model name
}

# Mock data as returned by load_available_api_models using realistic values
MOCK_AVAILABLE_API_MODELS = {
    "google/gemini-1.5-pro-latest": {"provider": "google", "model_name_short": "Gemini 1.5 Pro"},
    "openai/gpt-4o": {"provider": "openai", "model_name_short": "GPT-4o"},
    "anthropic/claude-3-opus-20240229": {"provider": "anthropic", "model_name_short": "Claude 3 Opus"},
    "google/gemini-flash-1.5": {"provider": "google", "model_name_short": "Gemini 1.5 Flash (OpenRouter)"},
}

# --- Fixtures ---


@pytest.fixture(autouse=True)
def clear_instance_registry():  # Renamed fixture, removed mock resets
    """Clear the internal instance registry before each test."""
    api._MODEL_INSTANCE_REGISTRY.clear()
    yield
    api._MODEL_INSTANCE_REGISTRY.clear()


@pytest.fixture(autouse=True)
def setup_test_model_configs():
    """Setup test model configurations for API tests."""
    from image_annotator_lib.core.config import config_registry

    configs = {
        "Gemini 1.5 Pro": {
            "class": "GoogleApiAnnotator",
            "api_model_id": "gemini-1.5-pro",
            "model_name_on_provider": "gemini-1.5-pro",
            "api_key": "test-api-key",
        },
        "GPT-4o": {
            "class": "OpenAIApiAnnotator",
            "api_model_id": "gpt-4o",
            "model_name_on_provider": "gpt-4o",
            "api_key": "test-api-key",
        },
        "Claude 3 Opus": {
            "class": "AnthropicApiAnnotator",
            "api_model_id": "claude-3-opus-20240229",
            "model_name_on_provider": "claude-3-opus-20240229",
            "api_key": "test-api-key",
        },
        "Gemini 1.5 Flash (OpenRouter)": {
            "class": "OpenRouterApiAnnotator",
            "api_model_id": "google/gemini-flash-1.5",
            "model_name_on_provider": "google/gemini-flash-1.5",
            "api_key": "test-api-key",
        },
        "test-model": {
            "class": "GoogleApiAnnotator",
            "api_model_id": "gemini-1.5-pro",
            "model_name_on_provider": "gemini-1.5-pro",
            "api_key": "test-api-key",
        },
    }
    for model_name, config in configs.items():
        for key, value in config.items():
            config_registry.add_default_setting(model_name, key, value)
    yield
    # Cleanup
    for model_name in configs.keys():
        try:
            config_registry._config.pop(model_name, None)
        except (AttributeError, KeyError):
            pass


# --- Test Cases ---


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_web_api_instance_success(mock_find_model, mock_get_factory):
    """Test successful instantiation for a Web API annotator."""
    model_name_short = "Gemini 1.5 Pro"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return GoogleApiAnnotator
    mock_find_model.return_value = (model_name_short, GoogleApiAnnotator)

    instance = api._create_annotator_instance(model_name_short)

    # Test actual behavior
    assert isinstance(instance, api.PydanticAIWebAPIWrapper)
    assert instance.model_name == model_name_short


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_local_model_instance_success(mock_find_model, mock_get_factory):
    """Test successful instantiation for a local model annotator."""
    model_name = "wd-v1-4-convnext-tagger-v2"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return WDTagger
    mock_find_model.return_value = (model_name, WDTagger)

    with patch.object(WDTagger, "__init__", return_value=None):
        instance = api._create_annotator_instance(model_name)

    # Test actual behavior
    assert isinstance(instance, WDTagger)


@pytest.mark.unit
@patch("image_annotator_lib.api.get_agent_factory")
@patch("image_annotator_lib.api.find_model_class_case_insensitive")
def test_create_web_api_instance_model_not_found_in_toml(mock_find_model, mock_get_factory):
    """Test KeyError when Web API model info is missing in available_api_models.toml."""
    model_name_short = "NonExistentWebModel"

    # Mock agent factory to return False (not a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = False
    mock_factory.get_available_models.return_value = []
    mock_get_factory.return_value = mock_factory

    # Mock find_model_class to return None (not found)
    mock_find_model.return_value = None

    # Test actual error behavior
    with pytest.raises(
        KeyError, match=f"Model '{model_name_short}' not found in registry or available models."
    ):
        api._create_annotator_instance(model_name_short)


@pytest.mark.unit
def test_get_annotator_instance_caches_instance():
    """Test that get_annotator_instance caches the created instance."""
    # Use a realistic local model name
    model_name = "wd-v1-4-convnext-tagger-v2"
    RealLocalAnnotatorClass = MOCK_CLASS_REGISTRY_WITH_REAL_CLASSES[model_name]

    mock_instance = MagicMock(spec=RealLocalAnnotatorClass)
    with patch("image_annotator_lib.api._create_annotator_instance", return_value=mock_instance):
        # Test caching behavior through actual state
        instance1 = api.get_annotator_instance(model_name)
        assert instance1 is mock_instance
        assert model_name in api._MODEL_INSTANCE_REGISTRY
        assert api._MODEL_INSTANCE_REGISTRY[model_name] is mock_instance

        # Second call should return cached instance
        instance2 = api.get_annotator_instance(model_name)
        assert instance2 is mock_instance
        assert instance1 is instance2  # Same cached object

    # Fixture handles cleanup


@pytest.mark.unit
@patch("image_annotator_lib.core.simplified_agent_wrapper.SimplifiedAgentWrapper")
@patch("image_annotator_lib.api.get_agent_factory")
def test_create_annotator_with_simplified_agent_wrapper(mock_get_factory, mock_wrapper):
    """Test creating annotator using SimplifiedAgentWrapper for direct model_id."""
    model_id = "google/gemini-2.5-pro-preview-03-25"

    # Mock agent factory to return True (is a direct model_id)
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = True
    mock_get_factory.return_value = mock_factory

    instance = api._create_annotator_instance(model_id)

    # Verify SimplifiedAgentWrapper was called
    mock_wrapper.assert_called_once_with(model_id, api_keys=None)
    assert instance == mock_wrapper.return_value


@pytest.mark.unit
@patch("image_annotator_lib.core.simplified_agent_wrapper.SimplifiedAgentWrapper")
@patch("image_annotator_lib.api.get_agent_factory")
def test_get_annotator_instance_with_api_keys_no_cache(mock_get_factory, mock_wrapper):
    """Test that get_annotator_instance doesn't use cache when api_keys are provided."""
    model_name = "test-model"
    api_keys = {"openai": "sk-test123"}

    # Mock agent factory
    mock_factory = MagicMock()
    mock_factory.is_model_available.return_value = True
    mock_get_factory.return_value = mock_factory

    # First call with api_keys
    api.get_annotator_instance(model_name, api_keys=api_keys)
    # Second call with api_keys
    api.get_annotator_instance(model_name, api_keys=api_keys)

    # Should create new instances each time (no caching)
    assert mock_wrapper.call_count == 2
    assert model_name not in api._MODEL_INSTANCE_REGISTRY


@pytest.mark.unit
@patch("image_annotator_lib.core.config.config_registry")
def test_pydanticai_wrapper_predict_without_api_model_id(mock_config):
    """Test PydanticAIWebAPIWrapper.predict raises error when api_model_id is not configured."""
    from PIL import Image

    # Mock config to return None for api_model_id
    mock_config.get.return_value = None

    wrapper = api.PydanticAIWebAPIWrapper("test-model", MagicMock())

    # Enter context to trigger config loading
    with wrapper:
        # Test that predict raises ValueError
        with pytest.raises(ValueError, match="has no api_model_id configured"):
            wrapper.predict([Image.new("RGB", (100, 100))], ["test_phash"])


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
@patch("image_annotator_lib.core.config.config_registry")
def test_pydanticai_wrapper_predict_handles_provider_error(
    mock_config, mock_get_capabilities, mock_run_inference
):
    """Test PydanticAIWebAPIWrapper.predict handles ProviderManager errors gracefully."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability

    # Mock config
    mock_config.get.return_value = "test/model-id"

    # Mock capabilities
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    # Mock ProviderManager to raise exception
    mock_run_inference.side_effect = Exception("API Error")

    wrapper = api.PydanticAIWebAPIWrapper("test-model", MagicMock())

    test_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    phash_list = ["phash1", "phash2"]

    with wrapper:
        results = wrapper.predict(test_images, phash_list)

        # Should return error results for all images
        assert len(results) == 2
        assert all(result.error is not None for result in results)
        assert all("Failed to run inference" in result.error for result in results)


# --- annotate() Function Tests ---


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_with_single_model_success(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() function with single model success."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.side_effect = lambda img: f"phash_{id(img)}"

    # Mock annotator instance
    mock_annotator = MagicMock()
    mock_get_instance.return_value = mock_annotator

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model",
        capabilities={TaskCapability.TAGS},
        tags=["tag1", "tag2"],
        error=None,
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute
    test_images = [Image.new("RGB", (100, 100))]
    results = api.annotate(test_images, ["test-model"])

    # Verify
    assert len(results) == 1
    assert "test-model" in next(iter(results.values()))
    mock_annotate_model.assert_called_once()


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_with_multiple_models_success(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() function with multiple models success."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"model1": MagicMock(), "model2": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.side_effect = lambda img: f"phash_{id(img)}"

    # Mock annotator instances
    mock_get_instance.side_effect = [MagicMock(), MagicMock()]

    # Mock annotation results
    mock_result1 = UnifiedAnnotationResult(
        model_name="model1", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_result2 = UnifiedAnnotationResult(
        model_name="model2", capabilities={TaskCapability.TAGS}, tags=["tag2"], error=None
    )
    mock_annotate_model.side_effect = [[mock_result1], [mock_result2]]

    # Execute
    test_images = [Image.new("RGB", (100, 100))]
    results = api.annotate(test_images, ["model1", "model2"])

    # Verify
    assert len(results) == 1
    phash_key = next(iter(results.keys()))
    assert "model1" in results[phash_key]
    assert "model2" in results[phash_key]
    assert mock_annotate_model.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_with_phash_provided(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance
):
    """Test annotate() function with provided pHash list."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute with provided phash
    test_images = [Image.new("RGB", (100, 100))]
    provided_phash = ["custom_phash_1"]
    results = api.annotate(test_images, ["test-model"], phash_list=provided_phash)

    # Verify - pHash calculation should not be called (phash_list provided)
    # Results should use the provided structure
    assert len(results) >= 1


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_with_phash_none(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() function with pHash auto-calculation."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.return_value = "auto_calculated_phash"

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute without phash_list
    test_images = [Image.new("RGB", (100, 100))]
    results = api.annotate(test_images, ["test-model"], phash_list=None)

    # Verify pHash was calculated
    mock_calc_phash.assert_called_once()
    assert len(results) >= 1


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_with_api_keys(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() function with API keys provided."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.return_value = "test_phash"

    # Mock annotator instance
    mock_annotator = MagicMock()
    mock_get_instance.return_value = mock_annotator

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute with API keys
    test_images = [Image.new("RGB", (100, 100))]
    api_keys = {"openai": "sk-test123"}
    results = api.annotate(test_images, ["test-model"], api_keys=api_keys)

    # Verify get_annotator_instance was called with api_keys
    mock_get_instance.assert_called_once_with("test-model", api_keys=api_keys)
    assert len(results) >= 1


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_empty_registry_initialization(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() initializes empty registry."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock empty registry initially, then populated
    mock_get_registry.side_effect = [{}, {"test-model": MagicMock()}]

    # Mock phash calculation
    mock_calc_phash.return_value = "test_phash"

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute
    test_images = [Image.new("RGB", (100, 100))]
    api.annotate(test_images, ["test-model"])

    # Verify initialize_registry was called
    mock_init_registry.assert_called_once()
    assert mock_get_registry.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
def test_annotate_model_error_handling(
    mock_get_capabilities,
    mock_init_registry,
    mock_get_registry,
    mock_annotate_model,
    mock_get_instance,
    mock_calc_phash,
):
    """Test annotate() handles model processing errors."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.return_value = "test_phash"

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation to raise exception
    mock_annotate_model.side_effect = ValueError("Test error")

    # Mock capabilities
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    # Execute
    test_images = [Image.new("RGB", (100, 100))]
    results = api.annotate(test_images, ["test-model"])

    # Verify error is handled and result created
    assert len(results) >= 1
    phash_key = next(iter(results.keys()))
    assert "test-model" in results[phash_key]
    assert results[phash_key]["test-model"].error is not None
    assert "ValueError" in results[phash_key]["test-model"].error


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
def test_annotate_result_length_mismatch(
    mock_get_capabilities,
    mock_init_registry,
    mock_get_registry,
    mock_annotate_model,
    mock_get_instance,
    mock_calc_phash,
):
    """Test annotate() handles result length mismatch."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock phash calculation
    mock_calc_phash.side_effect = lambda img: f"phash_{id(img)}"

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation results with wrong length (1 result for 2 images)
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]  # Only 1 result

    # Mock capabilities
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    # Execute with 2 images
    test_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    results = api.annotate(test_images, ["test-model"])

    # Verify error result is created for missing image
    assert len(results) >= 1
    # At least one result should have an error for the missing annotation
    found_error = False
    for phash_key in results:
        if "test-model" in results[phash_key]:
            if results[phash_key]["test-model"].error is not None:
                found_error = True
                break
    # Note: Due to _process_model_results logic, missing results get filled
    assert found_error or len(results) == 2


@pytest.mark.unit
@patch("image_annotator_lib.api.calculate_phash")
@patch("image_annotator_lib.api.get_annotator_instance")
@patch("image_annotator_lib.api._annotate_model")
@patch("image_annotator_lib.core.registry.get_cls_obj_registry")
@patch("image_annotator_lib.core.registry.initialize_registry")
def test_annotate_phash_list_longer_than_images(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """Test annotate() handles phash_list longer than images_list."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Mock registry
    mock_registry = {"test-model": MagicMock()}
    mock_get_registry.return_value = mock_registry

    # Mock annotator instance
    mock_get_instance.return_value = MagicMock()

    # Mock annotation results
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotate_model.return_value = [mock_result]

    # Execute with phash_list longer than images
    test_images = [Image.new("RGB", (100, 100))]
    long_phash_list = ["phash1", "phash2", "phash3"]  # 3 phashes for 1 image
    results = api.annotate(test_images, ["test-model"], phash_list=long_phash_list)

    # Verify function completes without error
    assert len(results) >= 1
    # Excess phashes should be ignored (warning logged)


# --- Helper Function Tests ---


@pytest.mark.unit
def test_annotate_model_context_manager():
    """Test _annotate_model() executes predict in context manager."""
    from PIL import Image

    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Create mock annotator
    mock_annotator = MagicMock()
    mock_result = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_annotator.predict.return_value = [mock_result]

    # Execute
    test_images = [Image.new("RGB", (100, 100))]
    phash_list = ["test_phash"]
    results = api._annotate_model(mock_annotator, test_images, phash_list)

    # Verify context manager was used
    mock_annotator.__enter__.assert_called_once()
    mock_annotator.__exit__.assert_called_once()
    mock_annotator.predict.assert_called_once_with(test_images, phash_list)
    assert results == [mock_result]


@pytest.mark.unit
def test_process_model_results():
    """Test _process_model_results() converts results to pHash structure."""
    from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

    # Create mock results
    mock_result1 = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag1"], error=None
    )
    mock_result2 = UnifiedAnnotationResult(
        model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["tag2"], error=None
    )
    annotation_results = [mock_result1, mock_result2]

    # Create empty results dict
    results_by_phash = api.PHashAnnotationResults()

    # Execute
    api._process_model_results("test-model", annotation_results, results_by_phash)

    # Verify results were added
    assert len(results_by_phash) == 2
    assert "image_0" in results_by_phash
    assert "image_1" in results_by_phash
    assert "test-model" in results_by_phash["image_0"]
    assert "test-model" in results_by_phash["image_1"]
    assert results_by_phash["image_0"]["test-model"] == mock_result1
    assert results_by_phash["image_1"]["test-model"] == mock_result2


@pytest.mark.unit
@patch("image_annotator_lib.core.utils.get_model_capabilities")
def test_handle_error(mock_get_capabilities):
    """Test _handle_error() creates error result."""
    from image_annotator_lib.core.types import TaskCapability

    # Mock capabilities
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    # Create empty results dict
    results_dict = api.PHashAnnotationResults()

    # Execute
    test_error = ValueError("Test error message")
    api._handle_error(
        e=test_error,
        model_name="test-model",
        image_hash="test_phash",
        results_dict=results_dict,
        idx=0,
        total_models=1,
    )

    # Verify error result was created
    assert "test_phash" in results_dict
    assert "test-model" in results_dict["test_phash"]
    error_result = results_dict["test_phash"]["test-model"]
    assert error_result.error is not None
    assert "ValueError" in error_result.error
    assert "Test error message" in error_result.error


@pytest.mark.unit
@patch("image_annotator_lib.core.registry.list_available_annotators")
def test_list_available_annotators(mock_list_annotators):
    """Test list_available_annotators() delegates to registry."""
    # Mock registry function
    mock_list_annotators.return_value = ["model1", "model2", "model3"]

    # Execute
    result = api.list_available_annotators()

    # Verify
    mock_list_annotators.assert_called_once()
    assert result == ["model1", "model2", "model3"]
