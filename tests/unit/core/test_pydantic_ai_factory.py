"""Unit tests for pydantic_ai_factory module."""

import os
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.pydantic_ai_factory import (
    PydanticAIAnnotatorMixin,
    PydanticAIProviderFactory,
    _is_test_environment,
)


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """Clear provider cache before each test."""
    PydanticAIProviderFactory._providers.clear()
    yield
    PydanticAIProviderFactory._providers.clear()


# ========================================
# _is_test_environment() tests
# ========================================


@pytest.mark.unit
def test_is_test_environment_detects_pytest():
    """Test _is_test_environment detects pytest execution."""
    result = _is_test_environment()
    assert result is True


@pytest.mark.unit
@patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_case"})
def test_is_test_environment_detects_pytest_env():
    """Test _is_test_environment detects PYTEST_CURRENT_TEST env var."""
    result = _is_test_environment()
    assert result is True


@pytest.mark.unit
@patch.dict(os.environ, {"TESTING": "1"})
def test_is_test_environment_detects_testing_env():
    """Test _is_test_environment detects TESTING env var."""
    result = _is_test_environment()
    assert result is True


@pytest.mark.unit
@patch("sys.argv", ["pytest", "test_file.py"])
def test_is_test_environment_detects_pytest_argv():
    """Test _is_test_environment detects pytest in sys.argv."""
    result = _is_test_environment()
    assert result is True


# ========================================
# PydanticAIProviderFactory.get_provider() tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_creates_new_provider(mock_infer):
    """Test get_provider creates new provider instance."""
    mock_provider_class = Mock()
    mock_provider_instance = Mock()
    mock_provider_class.return_value = mock_provider_instance
    mock_infer.return_value = mock_provider_class

    result = PydanticAIProviderFactory.get_provider("openai", api_key="test_key")

    mock_infer.assert_called_once_with("openai")
    mock_provider_class.assert_called_once_with(api_key="test_key")
    assert result == mock_provider_instance


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_returns_cached_instance(mock_infer):
    """Test get_provider returns cached provider instance for same params."""
    mock_provider_class = Mock()
    mock_provider_instance = Mock()
    mock_provider_class.return_value = mock_provider_instance
    mock_infer.return_value = mock_provider_class

    # First call creates provider
    result1 = PydanticAIProviderFactory.get_provider("anthropic", api_key="key1")
    # Second call with same params returns cached
    result2 = PydanticAIProviderFactory.get_provider("anthropic", api_key="key1")

    assert result1 == result2
    assert mock_provider_class.call_count == 1


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_different_params_creates_new_instance(mock_infer):
    """Test get_provider creates new instance for different params."""
    mock_provider_class = Mock()
    mock_provider_instance1 = Mock()
    mock_provider_instance2 = Mock()
    mock_provider_class.side_effect = [mock_provider_instance1, mock_provider_instance2]
    mock_infer.return_value = mock_provider_class

    result1 = PydanticAIProviderFactory.get_provider("openai", api_key="key1")
    result2 = PydanticAIProviderFactory.get_provider("openai", api_key="key2")

    assert result1 != result2
    assert mock_provider_class.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_unsupported_provider_raises_error(mock_infer):
    """Test get_provider raises ValueError for unsupported provider."""
    mock_infer.side_effect = ValueError("Unsupported provider")

    with pytest.raises(ValueError, match="Unsupported provider"):
        PydanticAIProviderFactory.get_provider("unknown_provider")


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_maps_openrouter_to_openai(mock_infer):
    """Test get_provider maps openrouter to openai provider."""
    mock_provider_class = Mock()
    mock_infer.return_value = mock_provider_class

    PydanticAIProviderFactory.get_provider("openrouter", api_key="test_key")

    mock_infer.assert_called_once_with("openai")


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_get_provider_maps_google_to_google_gla(mock_infer):
    """Test get_provider maps google to google-gla provider."""
    mock_provider_class = Mock()
    mock_infer.return_value = mock_provider_class

    PydanticAIProviderFactory.get_provider("google", api_key="test_key")

    mock_infer.assert_called_once_with("google-gla")


# ========================================
# PydanticAIProviderFactory.create_agent() tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_create_agent_success(mock_agent_class):
    """Test create_agent creates Agent successfully."""
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result = PydanticAIProviderFactory.create_agent(
        model_name="test_model", api_model_id="gpt-4", api_key="test_key"
    )

    assert result == mock_agent
    mock_agent_class.assert_called_once()


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
@patch("pydantic_ai.models.ALLOW_MODEL_REQUESTS", True)
@patch.dict(os.environ, {"ALLOW_MODEL_REQUESTS": "true"}, clear=True)
def test_create_agent_sets_openai_env_var(mock_agent_class):
    """Test create_agent sets OPENAI_API_KEY environment variable."""
    mock_agent_class.return_value = Mock()

    PydanticAIProviderFactory.create_agent(
        model_name="test_model", api_model_id="gpt-4", api_key="test_openai_key"
    )

    assert os.environ.get("OPENAI_API_KEY") == "test_openai_key"


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
@patch("pydantic_ai.models.ALLOW_MODEL_REQUESTS", True)
@patch.dict(os.environ, {"ALLOW_MODEL_REQUESTS": "true"}, clear=True)
def test_create_agent_sets_anthropic_env_var(mock_agent_class):
    """Test create_agent sets ANTHROPIC_API_KEY environment variable."""
    mock_agent_class.return_value = Mock()

    PydanticAIProviderFactory.create_agent(
        model_name="test_model", api_model_id="claude-3-opus", api_key="test_anthropic_key"
    )

    assert os.environ.get("ANTHROPIC_API_KEY") == "test_anthropic_key"


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
@patch("pydantic_ai.models.ALLOW_MODEL_REQUESTS", True)
@patch.dict(os.environ, {"ALLOW_MODEL_REQUESTS": "true"}, clear=True)
def test_create_agent_sets_google_env_var(mock_agent_class):
    """Test create_agent sets GOOGLE_API_KEY environment variable."""
    mock_agent_class.return_value = Mock()

    PydanticAIProviderFactory.create_agent(
        model_name="test_model", api_model_id="gemini-pro", api_key="test_google_key"
    )

    assert os.environ.get("GOOGLE_API_KEY") == "test_google_key"


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_create_agent_handles_exception(mock_agent_class):
    """Test create_agent handles Agent creation exception."""
    mock_agent_class.side_effect = Exception("Agent creation failed")

    with pytest.raises(Exception, match="Agent creation failed"):
        PydanticAIProviderFactory.create_agent(
            model_name="test_model", api_model_id="gpt-4", api_key="test_key"
        )


# ========================================
# PydanticAIProviderFactory.get_cached_agent() tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.create_openrouter_agent")
def test_get_cached_agent_delegates_to_openrouter_for_openrouter_model(mock_create_openrouter):
    """Test get_cached_agent delegates to create_openrouter_agent for openrouter: prefix."""
    mock_agent = Mock()
    mock_create_openrouter.return_value = mock_agent

    result = PydanticAIProviderFactory.get_cached_agent(
        model_name="test_model",
        api_model_id="openrouter:model-id",
        api_key="test_key",
        config_data={"referer": "http://example.com"},
    )

    assert result == mock_agent
    mock_create_openrouter.assert_called_once_with(
        "test_model", "openrouter:model-id", "test_key", {"referer": "http://example.com"}
    )


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.create_agent")
def test_get_cached_agent_uses_standard_creation_for_non_openrouter(mock_create_agent):
    """Test get_cached_agent uses create_agent for non-openrouter models."""
    mock_agent = Mock()
    mock_create_agent.return_value = mock_agent

    result = PydanticAIProviderFactory.get_cached_agent(
        model_name="test_model", api_model_id="gpt-4", api_key="test_key", config_data=None
    )

    assert result == mock_agent
    mock_create_agent.assert_called_once_with("test_model", "gpt-4", "test_key")


# ========================================
# PydanticAIProviderFactory.create_openrouter_agent() tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_create_openrouter_agent_in_test_environment(mock_agent_class, mock_is_test):
    """Test create_openrouter_agent uses TestModel in test environment."""
    mock_is_test.return_value = True
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result = PydanticAIProviderFactory.create_openrouter_agent(
        model_name="test_model", api_model_id="openrouter:test-model", api_key="test_key"
    )

    assert result == mock_agent
    # Verify Agent was called with TestModel
    call_args = mock_agent_class.call_args
    assert isinstance(call_args.kwargs["model"], TestModel)


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
@patch("image_annotator_lib.core.pydantic_ai_factory.OpenAIChatModel")
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_create_openrouter_agent_production_environment(
    mock_agent_class, mock_openai_model_class, mock_get_provider, mock_is_test
):
    """Test create_openrouter_agent creates OpenAI-based agent in production."""
    mock_is_test.return_value = False
    mock_provider = Mock()
    mock_get_provider.return_value = mock_provider
    mock_model = Mock()
    mock_openai_model_class.return_value = mock_model
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result = PydanticAIProviderFactory.create_openrouter_agent(
        model_name="test_model",
        api_model_id="openrouter:test-model-id",
        api_key="test_key",
        config_data={"referer": "http://example.com", "app_name": "TestApp"},
    )

    assert result == mock_agent
    # Verify provider was created with correct params
    mock_get_provider.assert_called_once()
    call_kwargs = mock_get_provider.call_args.kwargs
    assert call_kwargs["api_key"] == "test_key"
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert call_kwargs["default_headers"]["HTTP-Referer"] == "http://example.com"
    assert call_kwargs["default_headers"]["X-Title"] == "TestApp"


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_provider")
@patch("image_annotator_lib.core.pydantic_ai_factory.OpenAIChatModel")
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_create_openrouter_agent_without_custom_headers(
    mock_agent_class, mock_openai_model_class, mock_get_provider, mock_is_test
):
    """Test create_openrouter_agent works without custom headers."""
    mock_is_test.return_value = False
    mock_get_provider.return_value = Mock()
    mock_openai_model_class.return_value = Mock()
    mock_agent_class.return_value = Mock()

    PydanticAIProviderFactory.create_openrouter_agent(
        model_name="test_model", api_model_id="openrouter:test-model", api_key="test_key", config_data=None
    )

    # Verify provider was created without default_headers
    call_kwargs = mock_get_provider.call_args.kwargs
    assert "default_headers" not in call_kwargs


# ========================================
# PydanticAIProviderFactory.clear_cache() tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.infer_provider_class")
def test_clear_cache_removes_all_providers(mock_infer):
    """Test clear_cache removes all cached providers."""
    mock_provider_class = Mock()
    mock_infer.return_value = mock_provider_class

    # Create some providers
    PydanticAIProviderFactory.get_provider("openai", api_key="key1")
    PydanticAIProviderFactory.get_provider("anthropic", api_key="key2")

    assert len(PydanticAIProviderFactory._providers) == 2

    PydanticAIProviderFactory.clear_cache()

    assert len(PydanticAIProviderFactory._providers) == 0


# ========================================
# PydanticAIProviderFactory._extract_provider_name() tests
# ========================================


@pytest.mark.unit
def test_extract_provider_name_from_prefix():
    """Test _extract_provider_name extracts provider from prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("openai:gpt-4")
    assert result == "openai"


@pytest.mark.unit
def test_extract_provider_name_detects_gpt_model():
    """Test _extract_provider_name detects OpenAI from gpt prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("gpt-4-turbo")
    assert result == "openai"


@pytest.mark.unit
def test_extract_provider_name_detects_o1_model():
    """Test _extract_provider_name detects OpenAI from o1 prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("o1-preview")
    assert result == "openai"


@pytest.mark.unit
def test_extract_provider_name_detects_o3_model():
    """Test _extract_provider_name detects OpenAI from o3 prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("o3-mini")
    assert result == "openai"


@pytest.mark.unit
def test_extract_provider_name_detects_claude_model():
    """Test _extract_provider_name detects Anthropic from claude prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("claude-3-opus")
    assert result == "anthropic"


@pytest.mark.unit
def test_extract_provider_name_detects_gemini_model():
    """Test _extract_provider_name detects Google from gemini prefix."""
    result = PydanticAIProviderFactory._extract_provider_name("gemini-pro")
    assert result == "google"


@pytest.mark.unit
def test_extract_provider_name_returns_unknown_for_unrecognized():
    """Test _extract_provider_name returns unknown for unrecognized model."""
    result = PydanticAIProviderFactory._extract_provider_name("unknown-model-123")
    assert result == "unknown"


# ========================================
# PydanticAIAnnotatorMixin tests
# ========================================


class TestAnnotator(PydanticAIAnnotatorMixin):
    """Test class implementing PydanticAIAnnotatorMixin."""

    def __init__(self, model_name: str):
        super().__init__(model_name)


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_annotator_mixin_load_configuration_in_test_env(mock_is_test, mock_config):
    """Test _load_configuration skips validation in test environment."""
    mock_is_test.return_value = True
    mock_config.get.side_effect = lambda model, key, default=None: "" if key == "api_key" else None

    annotator = TestAnnotator("test_model")
    annotator._load_configuration()

    # Should not raise error in test environment
    assert annotator.api_key.get_secret_value() == ""
    assert annotator.api_model_id is None


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_annotator_mixin_load_configuration_validates_api_key(mock_is_test, mock_config):
    """Test _load_configuration validates API key in production."""
    mock_is_test.return_value = False
    mock_config.get.side_effect = lambda model, key, default=None: "" if key == "api_key" else "gpt-4"

    annotator = TestAnnotator("test_model")

    with pytest.raises(ValueError, match="API キーが設定されていません"):
        annotator._load_configuration()


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory._is_test_environment")
def test_annotator_mixin_load_configuration_validates_model_id(mock_is_test, mock_config):
    """Test _load_configuration validates model ID in production."""
    mock_is_test.return_value = False
    mock_config.get.side_effect = lambda model, key, default=None: "test_key" if key == "api_key" else None

    annotator = TestAnnotator("test_model")

    with pytest.raises(ValueError, match="API モデルIDが設定されていません"):
        annotator._load_configuration()


@pytest.mark.unit
def test_annotator_mixin_preprocess_images_to_binary():
    """Test _preprocess_images_to_binary converts PIL Images to BinaryContent."""
    annotator = TestAnnotator("test_model")

    test_image = Image.new("RGB", (100, 100), color="red")
    result = annotator._preprocess_images_to_binary([test_image])

    assert len(result) == 1
    assert isinstance(result[0], BinaryContent)
    assert result[0].media_type == "image/webp"
    assert len(result[0].data) > 0


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.config_registry")
@patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent")
def test_annotator_mixin_setup_agent(mock_get_agent, mock_config):
    """Test _setup_agent creates agent successfully."""
    mock_config.get.side_effect = lambda model, key, default=None: (
        "test_key" if key == "api_key" else "gpt-4" if key == "api_model_id" else default
    )
    mock_agent = Mock()
    mock_get_agent.return_value = mock_agent

    annotator = TestAnnotator("test_model")
    annotator._setup_agent()

    assert annotator.agent == mock_agent
    mock_get_agent.assert_called_once_with(
        model_name="test_model", api_model_id="gpt-4", api_key="test_key"
    )


# ==============================================================================
# Phase A Task 2: AdvancedAgentFactory Caching Tests (2025-12-03)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
@patch("image_annotator_lib.core.base.pydantic_ai_annotator.Agent")
def test_advanced_agent_factory_caches_agent(mock_agent_class):
    """Test that AdvancedAgentFactory caches and reuses Agent instances.

    Scenario:
    - Create agent with config A
    - Create agent again with same config A
    - Verify second call returns cached instance

    Tests:
    - Agent caching functionality
    - Cache hit detection
    - Instance reuse
    """
    from unittest.mock import Mock

    from image_annotator_lib.core.base.pydantic_ai_annotator import (
        AdvancedAgentFactory,
        AnnotationAgentConfig,
    )

    # Setup mock
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    # Clear cache first
    AdvancedAgentFactory.clear_cache()

    config = AnnotationAgentConfig(
        model_id="gpt-4",
        name="test_agent",
    )

    # First call - creates new agent
    agent1 = AdvancedAgentFactory.create_optimized_agent(config)

    # Second call - should return cached agent
    agent2 = AdvancedAgentFactory.create_optimized_agent(config)

    # Verify same instance
    assert agent1 is agent2
    assert id(agent1) == id(agent2)

    # Verify Agent was only created once (cached on second call)
    assert mock_agent_class.call_count == 1


@pytest.mark.unit
@pytest.mark.fast
@patch("image_annotator_lib.core.base.pydantic_ai_annotator.Agent")
def test_advanced_agent_factory_detects_config_change(mock_agent_class):
    """Test that configuration changes trigger new Agent creation.

    Scenario:
    - Create agent with config A (name="agent1")
    - Create agent with config B (name="agent2", same model_id)
    - Verify second call creates NEW agent (not cached)

    Tests:
    - Configuration change detection
    - Cache invalidation on config change
    - Config hash comparison
    """
    from unittest.mock import Mock

    from image_annotator_lib.core.base.pydantic_ai_annotator import (
        AdvancedAgentFactory,
        AnnotationAgentConfig,
    )

    # Setup mock to return different instances
    mock_agent1 = Mock()
    mock_agent2 = Mock()
    mock_agent_class.side_effect = [mock_agent1, mock_agent2]

    # Clear cache first
    AdvancedAgentFactory.clear_cache()

    config1 = AnnotationAgentConfig(
        model_id="gpt-4",
        name="test_agent",
    )

    config2 = AnnotationAgentConfig(
        model_id="gpt-3.5-turbo",  # Different model
        name="test_agent",
    )

    # First call
    agent1 = AdvancedAgentFactory.create_optimized_agent(config1)

    # Second call with different config
    agent2 = AdvancedAgentFactory.create_optimized_agent(config2)

    # Verify different instances (config changed)
    assert agent1 is not agent2
    assert id(agent1) != id(agent2)

    # Verify Agent was created twice (no cache hit)
    assert mock_agent_class.call_count == 2


@pytest.mark.unit
@pytest.mark.fast
@patch("image_annotator_lib.core.base.pydantic_ai_annotator.Agent")
def test_advanced_agent_factory_cache_key_generation(mock_agent_class):
    """Test cache key generation logic (model_id + config_hash).

    Scenario:
    - Create agents with same model_id but different configs
    - Verify separate cache entries

    Tests:
    - Cache key uniqueness
    - model_id + config_hash combination
    - Multiple cache entries
    """
    from unittest.mock import Mock

    from image_annotator_lib.core.base.pydantic_ai_annotator import (
        AdvancedAgentFactory,
        AnnotationAgentConfig,
    )

    # Setup mock to return different instances
    mock_agent1 = Mock()
    mock_agent2 = Mock()
    mock_agent_class.side_effect = [mock_agent1, mock_agent2]

    # Clear cache first
    AdvancedAgentFactory.clear_cache()

    config1 = AnnotationAgentConfig(
        model_id="gpt-4",
        name="test_agent",
    )

    config2 = AnnotationAgentConfig(
        model_id="gpt-3.5-turbo",  # Different model
        name="test_agent",
    )

    agent1 = AdvancedAgentFactory.create_optimized_agent(config1)
    agent2 = AdvancedAgentFactory.create_optimized_agent(config2)

    # Verify separate instances
    assert agent1 is not agent2

    # Verify cache contains both
    assert len(AdvancedAgentFactory._agent_cache) == 2

    # Verify both agents were created
    assert mock_agent_class.call_count == 2


@pytest.mark.unit
@pytest.mark.fast
@patch("image_annotator_lib.core.base.pydantic_ai_annotator.Agent")
def test_advanced_agent_factory_clear_cache(mock_agent_class):
    """Test clear_cache removes all cached agents.

    Scenario:
    - Create multiple agents
    - Call clear_cache()
    - Verify cache is empty

    Tests:
    - Cache clearing functionality
    - Complete cache removal
    - Hash cleanup
    """
    from unittest.mock import Mock

    from image_annotator_lib.core.base.pydantic_ai_annotator import (
        AdvancedAgentFactory,
        AnnotationAgentConfig,
    )

    # Setup mock to return new instances
    mock_agent_class.side_effect = [Mock(), Mock(), Mock()]

    # Clear cache first
    AdvancedAgentFactory.clear_cache()

    # Create multiple agents
    for i in range(3):
        config = AnnotationAgentConfig(
            model_id=f"gpt-{i}",
            name=f"agent_{i}",
        )
        AdvancedAgentFactory.create_optimized_agent(config)

    # Verify cache populated
    assert len(AdvancedAgentFactory._agent_cache) == 3
    assert len(AdvancedAgentFactory._config_hashes) == 3

    # Clear cache
    AdvancedAgentFactory.clear_cache()

    # Verify cache empty
    assert len(AdvancedAgentFactory._agent_cache) == 0
    assert len(AdvancedAgentFactory._config_hashes) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_advanced_agent_factory_config_hash_calculation():
    """Test config hash calculation produces consistent results.

    Scenario:
    - Calculate hash for same config twice
    - Calculate hash for different configs
    - Verify hash consistency and uniqueness

    Tests:
    - Hash calculation determinism
    - Hash uniqueness for different configs
    - Hash consistency for same config
    """
    from image_annotator_lib.core.base.pydantic_ai_annotator import (
        AdvancedAgentFactory,
        AnnotationAgentConfig,
    )

    config1 = AnnotationAgentConfig(
        model_id="gpt-4",
        name="test",
    )

    config2 = AnnotationAgentConfig(
        model_id="gpt-4",
        name="test",
    )

    config3 = AnnotationAgentConfig(
        model_id="gpt-3.5-turbo",  # Different model
        name="test",
    )

    # Calculate hashes
    hash1 = AdvancedAgentFactory._calculate_config_hash(config1)
    hash2 = AdvancedAgentFactory._calculate_config_hash(config2)
    hash3 = AdvancedAgentFactory._calculate_config_hash(config3)

    # Verify consistency
    assert hash1 == hash2  # Same config → same hash

    # Verify uniqueness
    assert hash1 != hash3  # Different config → different hash
