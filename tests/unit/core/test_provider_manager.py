from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.exceptions.errors import WebApiError


@pytest.fixture(autouse=True)
def clear_provider_instances():
    """Clear provider instances before each test."""
    ProviderManager._provider_instances.clear()
    yield
    ProviderManager._provider_instances.clear()


@pytest.mark.unit
def test_get_provider_instance_creates_anthropic_instance():
    """Test creating a new Anthropic provider instance."""
    instance = ProviderManager.get_provider_instance("anthropic")

    assert instance is not None
    assert "anthropic" in ProviderManager._provider_instances
    assert ProviderManager._provider_instances["anthropic"] is instance


@pytest.mark.unit
def test_get_provider_instance_creates_openai_instance():
    """Test creating a new OpenAI provider instance."""
    instance = ProviderManager.get_provider_instance("openai")

    assert instance is not None
    assert "openai" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_creates_google_instance():
    """Test creating a new Google provider instance."""
    instance = ProviderManager.get_provider_instance("google")

    assert instance is not None
    assert "google" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_creates_openrouter_instance():
    """Test creating a new OpenRouter provider instance."""
    instance = ProviderManager.get_provider_instance("openrouter")

    assert instance is not None
    assert "openrouter" in ProviderManager._provider_instances


@pytest.mark.unit
def test_get_provider_instance_returns_cached_instance():
    """Test that get_provider_instance returns cached instance."""
    instance1 = ProviderManager.get_provider_instance("anthropic")
    instance2 = ProviderManager.get_provider_instance("anthropic")

    assert instance1 is instance2


@pytest.mark.unit
def test_get_provider_instance_unsupported_provider_raises_error():
    """Test that unsupported provider raises WebApiError."""
    with pytest.raises(WebApiError, match="Unsupported provider: unknown_provider"):
        ProviderManager.get_provider_instance("unknown_provider")


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_explicit_config(mock_config):
    """Test provider determination from explicit configuration."""
    mock_config.get.return_value = "google"

    provider = ProviderManager._determine_provider("test_model", "any_model_id")

    assert provider == "google"
    mock_config.get.assert_called_once_with("test_model", "provider")


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_id_prefix(mock_config):
    """Test provider determination from model ID prefix."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "openai:gpt-4")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_gpt(mock_config):
    """Test provider determination from GPT model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "gpt-4-turbo")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_claude(mock_config):
    """Test provider determination from Claude model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "claude-3-opus")

    assert provider == "anthropic"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_from_model_name_pattern_gemini(mock_config):
    """Test provider determination from Gemini model name pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "gemini-1.5-pro")

    assert provider == "google"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_fallback_to_api_key(mock_config):
    """Test provider determination fallback to API key configuration."""

    def config_get_side_effect(model_name, key, default=None):
        if key == "provider":
            return None
        elif key == "google_api_key":
            return "test_key"
        return default

    mock_config.get.side_effect = config_get_side_effect

    provider = ProviderManager._determine_provider("test_model", "custom_model")

    assert provider == "google"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_default_fallback(mock_config):
    """Test provider determination default fallback to OpenAI."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "unknown_model")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_converts_results_to_phash_dict(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model converts results to phash-based dict."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock successful response
    from image_annotator_lib.core.types import AnnotationSchema

    mock_response = AnnotationSchema(tags=["test_tag"], captions=["test caption"], score=0.9)
    mock_provider_instance.run_with_model.return_value = [{"response": mock_response}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["phash"] == "test_phash_123"
    assert results["test_phash_123"]["tags"] == ["test_tag"]
    assert results["test_phash_123"]["error"] is None


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_handles_error_response(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model handles error responses."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock error response
    mock_provider_instance.run_with_model.return_value = [{"error": "API Error"}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["error"] == "API Error"
    assert results["test_phash_123"]["tags"] == []


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_handles_no_response(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test that run_inference_with_model handles missing response."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock response with no "response" key
    mock_provider_instance.run_with_model.return_value = [{}]

    mock_calculate_phash.return_value = "test_phash_123"

    # Execute
    test_image = Image.new("RGB", (100, 100))
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3"
    )

    # Verify
    assert len(results) == 1
    assert "test_phash_123" in results
    assert results["test_phash_123"]["error"] == "No response from provider"
    assert results["test_phash_123"]["tags"] == []


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_with_api_keys_injection(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test run_inference_with_model with injected API keys."""
    # Setup mocks
    mock_determine_provider.return_value = "anthropic"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock successful response
    from image_annotator_lib.core.types import AnnotationSchema

    mock_response = AnnotationSchema(tags=["injected_key_tag"], captions=["test"], score=0.9)
    mock_provider_instance.run_with_model.return_value = [{"response": mock_response}]

    mock_calculate_phash.return_value = "test_phash_456"

    # Execute with API keys
    test_image = Image.new("RGB", (100, 100))
    api_keys = {"anthropic": "test_injected_key"}
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3", api_keys=api_keys
    )

    # Verify API keys were passed
    mock_provider_instance.run_with_model.assert_called_once_with(
        "test_model", [test_image], "claude-3", api_keys
    )
    assert len(results) == 1
    assert "test_phash_456" in results


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.ProviderManager.get_provider_instance")
@patch("image_annotator_lib.core.provider_manager.ProviderManager._determine_provider")
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
def test_run_inference_with_model_multiple_images(
    mock_calculate_phash, mock_determine_provider, mock_get_provider
):
    """Test run_inference_with_model with multiple images."""
    # Setup mocks
    mock_determine_provider.return_value = "openai"

    mock_provider_instance = MagicMock()
    mock_get_provider.return_value = mock_provider_instance

    # Mock responses for 3 images
    from image_annotator_lib.core.types import AnnotationSchema

    mock_response1 = AnnotationSchema(tags=["tag1"], captions=[], score=0.8)
    mock_response2 = AnnotationSchema(tags=["tag2"], captions=[], score=0.9)
    mock_response3 = AnnotationSchema(tags=["tag3"], captions=[], score=0.7)
    mock_provider_instance.run_with_model.return_value = [
        {"response": mock_response1},
        {"response": mock_response2},
        {"response": mock_response3},
    ]

    mock_calculate_phash.side_effect = ["phash1", "phash2", "phash3"]

    # Execute
    test_images = [Image.new("RGB", (100, 100)) for _ in range(3)]
    results = ProviderManager.run_inference_with_model(
        model_name="test_model", images_list=test_images, api_model_id="gpt-4"
    )

    # Verify
    assert len(results) == 3
    assert "phash1" in results
    assert "phash2" in results
    assert "phash3" in results
    assert results["phash1"]["tags"] == ["tag1"]
    assert results["phash2"]["tags"] == ["tag2"]
    assert results["phash3"]["tags"] == ["tag3"]


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_with_openrouter_prefix(mock_config):
    """Test provider determination for OpenRouter with prefix."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "openrouter:meta-llama")

    assert provider == "openrouter"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_o1_model_pattern(mock_config):
    """Test provider determination for o1 model pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "o1-preview")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_o3_model_pattern(mock_config):
    """Test provider determination for o3 model pattern."""
    mock_config.get.return_value = None

    provider = ProviderManager._determine_provider("test_model", "o3-mini")

    assert provider == "openai"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_anthropic_api_key_fallback(mock_config):
    """Test provider determination fallback to Anthropic API key."""

    def config_get_side_effect(model_name, key, default=None):
        if key == "provider":
            return None
        elif key == "anthropic_api_key":
            return "test_anthropic_key"
        return default

    mock_config.get.side_effect = config_get_side_effect

    provider = ProviderManager._determine_provider("test_model", "custom_model_xyz")

    assert provider == "anthropic"


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_determine_provider_openai_api_key_fallback(mock_config):
    """Test provider determination fallback to OpenAI API key."""

    def config_get_side_effect(model_name, key, default=None):
        if key == "provider":
            return None
        elif key == "anthropic_api_key":
            return None
        elif key == "openai_api_key":
            return "test_openai_key"
        return default

    mock_config.get.side_effect = config_get_side_effect

    provider = ProviderManager._determine_provider("test_model", "custom_model_abc")

    assert provider == "openai"


@pytest.mark.unit
def test_cleanup_context():
    """Test ProviderInstanceBase cleanup_context method."""
    from image_annotator_lib.core.provider_manager import ProviderInstanceBase

    # Create instance
    provider_base = ProviderInstanceBase()

    # Mock annotator and context
    mock_annotator = MagicMock()
    mock_context = MagicMock()
    provider_base._active_contexts["test_model"] = (mock_annotator, mock_context)

    # Execute cleanup
    provider_base.cleanup_context("test_model")

    # Verify __exit__ was called
    mock_annotator.__exit__.assert_called_once_with(None, None, None)
    assert "test_model" not in provider_base._active_contexts


@pytest.mark.unit
def test_cleanup_context_with_exception():
    """Test ProviderInstanceBase cleanup_context handles exceptions gracefully."""
    from image_annotator_lib.core.provider_manager import ProviderInstanceBase

    # Create instance
    provider_base = ProviderInstanceBase()

    # Mock annotator that raises exception on __exit__
    mock_annotator = MagicMock()
    mock_annotator.__exit__.side_effect = RuntimeError("Cleanup error")
    mock_context = MagicMock()
    provider_base._active_contexts["test_model"] = (mock_annotator, mock_context)

    # Execute cleanup (should not raise)
    provider_base.cleanup_context("test_model")

    # Verify context was still removed
    assert "test_model" not in provider_base._active_contexts


@pytest.mark.unit
def test_cleanup_context_nonexistent_model():
    """Test ProviderInstanceBase cleanup_context with non-existent model."""
    from image_annotator_lib.core.provider_manager import ProviderInstanceBase

    provider_base = ProviderInstanceBase()

    # Should not raise error
    provider_base.cleanup_context("non_existent_model")

    # Verify nothing in active contexts
    assert len(provider_base._active_contexts) == 0


@pytest.mark.unit
@pytest.mark.skip(reason="Config mock not working in Provider Instance - needs refactoring")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_anthropic_provider_instance_run_with_model_success(mock_is_test_env, mock_config, mock_factory):
    """Test AnthropicProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import AnthropicProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = "test_api_key"

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["anthropic_tag"], captions=[], score=0.85)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute
        provider = AnthropicProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model", images_list=[test_image], api_model_id="claude-3-opus", api_keys=None
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["anthropic_tag"]


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_anthropic_provider_instance_missing_api_key(mock_is_test_env, mock_config):
    """Test AnthropicProviderInstance.run_with_model with missing API key."""
    from image_annotator_lib.core.provider_manager import AnthropicProviderInstance

    # Setup mocks - no API key
    mock_is_test_env.return_value = False
    mock_config.get.return_value = ""

    # Execute
    provider = AnthropicProviderInstance()
    test_image = Image.new("RGB", (100, 100))
    results = provider.run_with_model(
        model_name="test_model", images_list=[test_image], api_model_id="claude-3", api_keys=None
    )

    # Verify error result
    assert len(results) == 1
    assert "error" in results[0]
    assert "API key not configured" in results[0]["error"]


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_anthropic_provider_instance_with_injected_api_keys(mock_is_test_env, mock_config, mock_factory):
    """Test AnthropicProviderInstance.run_with_model with injected API keys."""
    from image_annotator_lib.core.provider_manager import AnthropicProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = ""  # No config API key

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["injected"], captions=[], score=0.9)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute with injected API keys
        provider = AnthropicProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        api_keys = {"anthropic": "injected_key"}
        results = provider.run_with_model(
            model_name="test_model", images_list=[test_image], api_model_id="claude-3", api_keys=api_keys
        )

        # Verify API key was used
        mock_factory.get_cached_agent.assert_called_once_with(
            model_name="test_model", api_model_id="claude-3", api_key="injected_key"
        )
        assert len(results) == 1
        assert "response" in results[0]


@pytest.mark.unit
@pytest.mark.skip(reason="Config mock not working in Provider Instance - needs refactoring")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_openai_provider_instance_run_with_model_success(mock_is_test_env, mock_config, mock_factory):
    """Test OpenAIProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import OpenAIProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = "test_openai_key"

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["openai_tag"], captions=[], score=0.92)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute
        provider = OpenAIProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model", images_list=[test_image], api_model_id="gpt-4-vision", api_keys=None
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["openai_tag"]


@pytest.mark.unit
@pytest.mark.skip(reason="Config mock not working in Provider Instance - needs refactoring")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_google_provider_instance_run_with_model_success(mock_is_test_env, mock_config, mock_factory):
    """Test GoogleProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import GoogleProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = "test_google_key"

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["google_tag"], captions=[], score=0.88)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute
        provider = GoogleProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model", images_list=[test_image], api_model_id="gemini-1.5-pro", api_keys=None
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["google_tag"]


@pytest.mark.unit
@pytest.mark.skip(reason="Config mock not working in Provider Instance - needs refactoring")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_openrouter_provider_instance_run_with_model_success(mock_is_test_env, mock_config, mock_factory):
    """Test OpenRouterProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import OpenRouterProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = "test_openrouter_key"

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["openrouter_tag"], captions=[], score=0.9)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute
        provider = OpenRouterProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="openrouter:meta-llama",
            api_keys=None,
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["openrouter_tag"]


@pytest.mark.unit
@pytest.mark.skip(reason="Config mock not working in Provider Instance - needs refactoring")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_provider_instance_agent_execution_error(mock_is_test_env, mock_config, mock_factory):
    """Test provider instance handling of agent execution errors."""
    from image_annotator_lib.core.provider_manager import OpenAIProviderInstance

    # Setup mocks
    mock_is_test_env.return_value = False
    mock_config.get.return_value = "test_key"

    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock agent execution error
    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.side_effect = RuntimeError("Agent execution failed")

        # Execute
        provider = OpenAIProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model", images_list=[test_image], api_model_id="gpt-4", api_keys=None
        )

        # Verify error handling
        assert len(results) == 1
        assert "error" in results[0]
        assert "API Error" in results[0]["error"]
