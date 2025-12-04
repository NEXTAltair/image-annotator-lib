from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.exceptions.errors import WebApiError


@pytest.fixture(autouse=True)
def clear_provider_instances():
    """Clear provider instances before and after each test."""
    ProviderManager.clear_cache()
    yield
    ProviderManager.clear_cache()


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
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
def test_anthropic_provider_instance_run_with_model_success(mock_factory):
    """Test AnthropicProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import AnthropicProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["anthropic_tag"], captions=[], score=0.85)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute with api_keys to bypass config_registry
        provider = AnthropicProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="claude-3-opus",
            api_keys={"anthropic": "test_api_key"},  # Direct API key injection
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
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
def test_openai_provider_instance_run_with_model_success(mock_factory):
    """Test OpenAIProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import OpenAIProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["openai_tag"], captions=[], score=0.92)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute with api_keys to bypass config_registry
        provider = OpenAIProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="gpt-4-vision",
            api_keys={"openai": "test_openai_key"},  # Direct API key injection
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["openai_tag"]


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
def test_google_provider_instance_run_with_model_success(mock_factory):
    """Test GoogleProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import GoogleProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["google_tag"], captions=[], score=0.88)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute with api_keys to bypass config_registry
        provider = GoogleProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="gemini-1.5-pro",
            api_keys={"google": "test_google_key"},  # Direct API key injection
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["google_tag"]


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
def test_openrouter_provider_instance_run_with_model_success(mock_factory):
    """Test OpenRouterProviderInstance.run_with_model successful execution."""
    from image_annotator_lib.core.provider_manager import OpenRouterProviderInstance
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup mocks
    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock successful agent response
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["openrouter_tag"], captions=[], score=0.9)

    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.return_value = mock_result

        # Execute with api_keys to bypass config_registry
        provider = OpenRouterProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="openrouter:meta-llama",
            api_keys={"openrouter": "test_openrouter_key"},  # Direct API key injection
        )

        # Verify
        assert len(results) == 1
        assert "response" in results[0]
        assert results[0]["response"].tags == ["openrouter_tag"]


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory")
def test_provider_instance_agent_execution_error(mock_factory):
    """Test provider instance handling of agent execution errors."""
    from image_annotator_lib.core.provider_manager import OpenAIProviderInstance

    # Setup mocks
    mock_agent = MagicMock()
    mock_factory.get_cached_agent.return_value = mock_agent

    # Mock agent execution error
    with patch("image_annotator_lib.core.provider_manager.ProviderManager._run_agent_safely") as mock_run:
        mock_run.side_effect = RuntimeError("Agent execution failed")

        # Execute with api_keys to bypass config_registry
        provider = OpenAIProviderInstance()
        test_image = Image.new("RGB", (100, 100))
        results = provider.run_with_model(
            model_name="test_model",
            images_list=[test_image],
            api_model_id="gpt-4",
            api_keys={"openai": "test_key"},  # Direct API key injection
        )

        # Verify error handling
        assert len(results) == 1
        assert "error" in results[0]
        assert "API Error" in results[0]["error"]


@pytest.mark.unit
def test_run_agent_safely_event_loop_fallback():
    """Test _run_agent_safely with event loop fallback to new loop execution."""
    from pydantic_ai import BinaryContent

    from image_annotator_lib.core.provider_manager import ProviderManager
    from image_annotator_lib.core.types import AnnotationSchema

    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.run_sync.side_effect = RuntimeError("Event loop is closed")

    # Mock successful async execution in new loop
    mock_result = MagicMock()
    mock_result.data = AnnotationSchema(tags=["fallback_success"], captions=[], score=0.95)

    # Create async mock for agent.run()
    async def mock_async_run(*args, **kwargs):
        return mock_result

    mock_agent.run.return_value = mock_async_run()

    # Create binary content
    binary_content = BinaryContent(data=b"test_image_data", media_type="image/webp")

    # Execute with fallback
    result = ProviderManager._run_agent_safely(mock_agent, binary_content, "test-model-id")

    # Verify fallback was triggered
    assert mock_agent.run_sync.called
    assert mock_agent.run.called
    assert result.data.tags == ["fallback_success"]


@pytest.mark.unit
def test_run_agent_safely_both_sync_and_async_fail():
    """Test _run_agent_safely when both sync and async execution fail."""
    from pydantic_ai import BinaryContent

    from image_annotator_lib.core.provider_manager import ProviderManager

    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.run_sync.side_effect = RuntimeError("Event loop is closed")

    # Mock async execution failure
    async def mock_async_fail(*args, **kwargs):
        raise RuntimeError("Async execution failed")

    mock_agent.run.return_value = mock_async_fail()

    # Create binary content
    binary_content = BinaryContent(data=b"test_image_data", media_type="image/webp")

    # Execute and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        ProviderManager._run_agent_safely(mock_agent, binary_content, "test-model-id")

    # Verify error message contains both errors
    error_msg = str(exc_info.value)
    assert "Both sync and async execution failed" in error_msg
    assert "Event loop is closed" in error_msg
    assert "Async execution failed" in error_msg


@pytest.mark.unit
def test_run_agent_safely_timeout_handling():
    """Test _run_agent_safely timeout handling in ThreadPoolExecutor."""
    import concurrent.futures

    from pydantic_ai import BinaryContent

    from image_annotator_lib.core.provider_manager import ProviderManager

    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.run_sync.side_effect = RuntimeError("Event loop is closed")

    # Mock async execution that simulates timeout by raising TimeoutError
    # We patch the executor to raise TimeoutError without actually waiting
    binary_content = BinaryContent(data=b"test_image_data", media_type="image/webp")

    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError(
            "Execution timed out after 60 seconds"
        )
        mock_executor.submit.return_value = mock_future
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Execute and expect RuntimeError wrapping the timeout
        with pytest.raises(RuntimeError) as exc_info:
            ProviderManager._run_agent_safely(mock_agent, binary_content, "test-model-id")

        # Verify timeout error was wrapped
        error_msg = str(exc_info.value)
        assert "Both sync and async execution failed" in error_msg or "timed out" in error_msg.lower()


# ==============================================================================
# Phase A Task 3: Provider Cache Management and Edge Cases (2025-12-03)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_provider_cache_clear():
    """Test that provider cache can be cleared.

    Scenario:
    - Create multiple provider instances
    - Clear the cache
    - Verify cache is empty

    Tests:
    - Cache clearing functionality
    - Complete cache removal
    """
    # Create multiple providers
    ProviderManager.get_provider_instance("anthropic")
    ProviderManager.get_provider_instance("openai")
    ProviderManager.get_provider_instance("google")

    # Verify cache populated
    assert len(ProviderManager._provider_instances) == 3

    # Clear cache using proper API
    ProviderManager.clear_cache()

    # Verify cache empty
    assert len(ProviderManager._provider_instances) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_provider_instance_recreation_after_clear():
    """Test that provider instances can be recreated after cache clear.

    Scenario:
    - Create provider instance
    - Store reference to instance
    - Clear cache
    - Create new provider instance with same parameters
    - Verify new instance is created (different object)

    Tests:
    - Provider recreation after cache clear
    - Instance independence
    """
    # Create initial provider
    provider1 = ProviderManager.get_provider_instance("anthropic")
    provider1_id = id(provider1)

    # Clear cache using proper API
    ProviderManager.clear_cache()

    # Create provider again with same parameters
    provider2 = ProviderManager.get_provider_instance("anthropic")
    provider2_id = id(provider2)

    # Verify new instance created
    assert provider1_id != provider2_id
    assert provider1 is not provider2


@pytest.mark.unit
@pytest.mark.fast
def test_multiple_provider_types_coexist():
    """Test that multiple provider types can coexist in cache.

    Scenario:
    - Create instances of all provider types
    - Verify all are cached separately
    - Verify each has correct type

    Tests:
    - Multi-provider cache management
    - Provider type differentiation
    - Cache key uniqueness
    """
    # Clear cache first
    ProviderManager.clear_cache()

    # Create all provider types
    anthropic = ProviderManager.get_provider_instance("anthropic")
    openai = ProviderManager.get_provider_instance("openai")
    google = ProviderManager.get_provider_instance("google")
    openrouter = ProviderManager.get_provider_instance("openrouter")

    # Verify all cached
    assert len(ProviderManager._provider_instances) == 4

    # Verify correct types
    from image_annotator_lib.core.provider_manager import (
        AnthropicProviderInstance,
        GoogleProviderInstance,
        OpenAIProviderInstance,
        OpenRouterProviderInstance,
    )

    assert isinstance(anthropic, AnthropicProviderInstance)
    assert isinstance(openai, OpenAIProviderInstance)
    assert isinstance(google, GoogleProviderInstance)
    assert isinstance(openrouter, OpenRouterProviderInstance)

    # Cleanup
    ProviderManager.clear_cache()


@pytest.mark.unit
@pytest.mark.fast
def test_provider_cache_key_generation():
    """Test that cache keys are generated correctly for different provider types.

    Scenario:
    - Create different provider types
    - Verify they are cached with correct keys

    Tests:
    - Cache key uniqueness per provider type
    - Provider type differentiation in cache
    """
    # Clear cache first
    ProviderManager.clear_cache()

    # Create different providers
    provider1 = ProviderManager.get_provider_instance("anthropic")
    provider2 = ProviderManager.get_provider_instance("openai")

    # Verify both are cached (different providers = different cache entries)
    assert len(ProviderManager._provider_instances) == 2

    # Verify different instances
    assert provider1 is not provider2
    assert id(provider1) != id(provider2)

    # Verify correct cache keys exist
    assert "anthropic" in ProviderManager._provider_instances
    assert "openai" in ProviderManager._provider_instances

    # Cleanup
    ProviderManager.clear_cache()


@pytest.mark.unit
@pytest.mark.fast
def test_provider_instance_state_persistence():
    """Test that provider instance state persists across retrievals.

    Scenario:
    - Get provider instance and set custom attribute
    - Retrieve same provider from cache
    - Verify custom attribute still exists

    Tests:
    - Provider instance state persistence
    - Cache returns same object
    - Instance state is not reset
    """
    # Clear cache first
    ProviderManager.clear_cache()

    # Get provider and set custom attribute
    provider1 = ProviderManager.get_provider_instance("anthropic")
    provider1._test_attribute = "test_value"

    # Get provider again (should be same instance from cache)
    provider2 = ProviderManager.get_provider_instance("anthropic")

    # Verify same instance
    assert provider1 is provider2

    # Verify custom attribute persists
    assert hasattr(provider2, "_test_attribute")
    assert provider2._test_attribute == "test_value"

    # Cleanup
    ProviderManager.clear_cache()
