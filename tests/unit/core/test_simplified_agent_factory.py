"""Unit tests for SimplifiedAgentFactory.

Tests Agent creation, caching, model discovery, and provider filtering.

Mock Strategy (Phase C Level 1-2):
- Level 1 (Mock): API model discovery, PydanticAI Agent creation
- Level 2 (Mock): SimpleConfig loading
- Level 3 (Real): Cache key generation, provider filtering, settings merge
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import Agent

from image_annotator_lib.core.simplified_agent_factory import (
    SimplifiedAgentFactory,
    create_agent,
    get_agent_factory,
    get_available_models,
    is_model_deprecated,
    list_all_models,
)


_BASE_MODEL_IDS = [
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-1.5-pro",
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",
]


@pytest.fixture
def mock_model_discovery():
    """Mock discover_available_vision_models と load_available_api_models。

    Mock Strategy:
    - Mock: API discovery calls, TOML full data
    - Real: Model list structure

    Returns:
        Mock function returning model discovery result
    """
    mock_toml_data = {
        mid: {"provider": mid.split("/")[0].capitalize(), "deprecated_on": None}
        for mid in _BASE_MODEL_IDS
    }
    with (
        patch(
            "image_annotator_lib.core.simplified_agent_factory.discover_available_vision_models"
        ) as mock_discover,
        patch(
            "image_annotator_lib.core.simplified_agent_factory.load_available_api_models"
        ) as mock_load,
    ):
        mock_discover.return_value = {"models": _BASE_MODEL_IDS}
        mock_load.return_value = mock_toml_data
        yield mock_discover


@pytest.fixture
def mock_model_discovery_with_deprecated():
    """廃止済みモデルを含むモックデータ。"""
    mock_toml_data = {
        "openai/gpt-4o": {"provider": "OpenAI", "deprecated_on": None},
        "openai/gpt-3.5-turbo": {"provider": "OpenAI", "deprecated_on": "2025-01-01T00:00:00Z"},
        "anthropic/claude-3-5-sonnet-20241022": {"provider": "Anthropic", "deprecated_on": None},
    }
    with (
        patch(
            "image_annotator_lib.core.simplified_agent_factory.discover_available_vision_models"
        ) as mock_discover,
        patch(
            "image_annotator_lib.core.simplified_agent_factory.load_available_api_models"
        ) as mock_load,
    ):
        mock_discover.return_value = {"models": list(mock_toml_data.keys())}
        mock_load.return_value = mock_toml_data
        yield mock_discover, mock_load


@pytest.fixture
def mock_simple_config():
    """Mock get_model_settings.

    Mock Strategy:
    - Mock: SimpleConfig loading
    - Real: Settings dict structure

    Returns:
        Mock function returning model settings
    """
    with patch("image_annotator_lib.core.simplified_agent_factory.get_model_settings") as mock:
        mock.return_value = {
            "max_output_tokens": 1800,
            "timeout": 60,
            "temperature": 0.7,
            "retry_count": 3,  # Non-Agent parameter (should be filtered)
        }
        yield mock


@pytest.fixture
def mock_agent_creation():
    """Mock PydanticAI Agent creation.

    Mock Strategy:
    - Mock: Agent.__init__()
    - Real: Agent object

    Returns:
        Mock Agent class
    """
    with patch("image_annotator_lib.core.simplified_agent_factory.Agent") as mock_agent_class:
        # Use side_effect to return different instances for each call
        mock_agent_class.side_effect = lambda *args, **kwargs: MagicMock(spec=Agent)
        yield mock_agent_class


@pytest.fixture(autouse=True)
def reset_global_factory():
    """Reset global _agent_factory before/after each test.

    CRITICAL: SimplifiedAgentFactory uses module-level _agent_factory singleton.
    Without this fixture, tests can affect each other.
    """
    import image_annotator_lib.core.simplified_agent_factory as factory_module

    factory_module._agent_factory = None
    yield
    factory_module._agent_factory = None


# ==============================================================================
# Phase C Week 2: Simplified Agent Factory Tests (2025-12-07)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_agent_with_settings_merge(mock_model_discovery, mock_simple_config, mock_agent_creation):
    """Test create_agent() merges model settings with kwargs and filters parameters.

    Coverage: Lines 55-85 (create_agent implementation)

    REAL components:
    - Real settings dict merge logic
    - Real parameter filtering (Agent vs non-Agent params)
    - Real agent_params extraction

    MOCKED:
    - get_model_settings() returns base config
    - Agent() constructor

    Scenario:
    1. Base config from SimpleConfig: max_output_tokens=1800, timeout=60, retry_count=3
    2. Override with kwargs: temperature=0.5 (override), top_p=0.9 (new)
    3. Filter non-Agent params: retry_count should be excluded
    4. Agent() called with: max_output_tokens, timeout, temperature, top_p

    Assertions:
    - get_model_settings() called with model_id
    - kwargs override base settings
    - Non-Agent params filtered out (retry_count not passed)
    - Agent() receives correct result_type (AnnotationSchema)
    - Agent() receives merged agent_params
    """
    factory = SimplifiedAgentFactory()

    # Create agent with overrides
    agent = factory.create_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.5, top_p=0.9)

    # Assert: get_model_settings called
    mock_simple_config.assert_called_once_with("google/gemini-2.5-pro-preview-03-25")

    # Assert: Agent() called with correct parameters
    mock_agent_creation.assert_called_once()
    call_kwargs = mock_agent_creation.call_args[1]

    # Verify model ID
    assert call_kwargs["model"] == "google/gemini-2.5-pro-preview-03-25"

    # Verify result_type (AnnotationSchema)
    from image_annotator_lib.core.types import AnnotationSchema

    assert call_kwargs["result_type"] == AnnotationSchema

    # Verify agent_params (filtered and merged)
    assert call_kwargs["max_output_tokens"] == 1800  # From base config
    assert call_kwargs["timeout"] == 60  # From base config
    assert call_kwargs["temperature"] == 0.5  # Overridden by kwargs
    assert call_kwargs["top_p"] == 0.9  # New param from kwargs

    # Verify non-Agent params filtered out
    assert "retry_count" not in call_kwargs  # Filtered out

    # Verify agent returned
    assert agent is not None


@pytest.mark.unit
@pytest.mark.fast
def test_get_cached_agent_caching_behavior(mock_model_discovery, mock_simple_config, mock_agent_creation):
    """Test get_cached_agent() caches agents by cache_key.

    Coverage: Lines 87-103 (get_cached_agent implementation)

    REAL components:
    - Real cache_key generation (model_id:hash(kwargs))
    - Real cache hit/miss logic
    - Real frozenset(kwargs.items()) hashing

    MOCKED:
    - Agent creation (to verify call count)

    Scenario:
    1. Call get_cached_agent() twice with same model_id and kwargs
       → Agent() called once, second call returns cached instance
    2. Call get_cached_agent() with different kwargs
       → Agent() called again (different cache_key)
    3. Call get_cached_agent() with different model_id but same kwargs
       → Agent() called again (different cache_key)

    Assertions:
    - Same model_id + kwargs → Agent() called once
    - Different kwargs → new Agent() call
    - Different model_id → new Agent() call
    - Cache returns same Agent instance for same cache_key
    """
    factory = SimplifiedAgentFactory()

    # Test 1: Same model_id and kwargs (cache hit)
    agent1 = factory.get_cached_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.7)
    agent2 = factory.get_cached_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.7)

    # Assert: Agent() called once (cache hit on second call)
    assert mock_agent_creation.call_count == 1
    assert agent1 is agent2  # Same instance

    # Test 2: Different kwargs (new cache_key)
    agent3 = factory.get_cached_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.5)

    # Assert: Agent() called again (different cache_key)
    assert mock_agent_creation.call_count == 2
    assert agent3 is not agent1  # Different instance

    # Test 3: Different model_id (new cache_key)
    agent4 = factory.get_cached_agent("openai/gpt-4o", temperature=0.7)

    # Assert: Agent() called again (different model_id)
    assert mock_agent_creation.call_count == 3
    assert agent4 is not agent1  # Different instance

    # Test 4: Clear cache and verify re-creation
    factory.clear_cache()
    agent5 = factory.get_cached_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.7)

    # Assert: Agent() called again after cache clear
    assert mock_agent_creation.call_count == 4
    assert agent5 is not agent1  # New instance (cache cleared)


@pytest.mark.unit
@pytest.mark.fast
def test_get_models_by_provider_filtering(mock_model_discovery):
    """Test get_models_by_provider() filters models by provider prefix.

    Coverage: Lines 118-130 (get_models_by_provider implementation)

    REAL components:
    - Real provider prefix matching (model.startswith(f"{provider}/"))
    - Real list comprehension filtering

    MOCKED:
    - discover_available_vision_models() returns fixed model list

    Scenario:
    Available models:
    - google/gemini-2.5-pro-preview-03-25
    - google/gemini-1.5-pro
    - openai/gpt-4o
    - anthropic/claude-3-5-sonnet-20241022

    Test filters:
    1. provider="google" → 2 models
    2. provider="openai" → 1 model
    3. provider="anthropic" → 1 model
    4. provider="nonexistent" → 0 models

    Assertions:
    - Correct number of models returned for each provider
    - All returned models start with "{provider}/"
    - Empty list for non-existent provider
    - discover_available_vision_models() called to refresh models
    """
    factory = SimplifiedAgentFactory()

    # Test 1: Google models
    google_models = factory.get_models_by_provider("google")

    assert len(google_models) == 2
    assert "google/gemini-2.5-pro-preview-03-25" in google_models
    assert "google/gemini-1.5-pro" in google_models
    for model in google_models:
        assert model.startswith("google/")

    # Test 2: OpenAI models
    openai_models = factory.get_models_by_provider("openai")

    assert len(openai_models) == 1
    assert "openai/gpt-4o" in openai_models
    for model in openai_models:
        assert model.startswith("openai/")

    # Test 3: Anthropic models
    anthropic_models = factory.get_models_by_provider("anthropic")

    assert len(anthropic_models) == 1
    assert "anthropic/claude-3-5-sonnet-20241022" in anthropic_models
    for model in anthropic_models:
        assert model.startswith("anthropic/")

    # Test 4: Non-existent provider
    nonexistent_models = factory.get_models_by_provider("nonexistent")

    assert len(nonexistent_models) == 0

    # Test 5: Verify discovery called
    mock_model_discovery.assert_called()


@pytest.mark.unit
@pytest.mark.fast
def test_global_factory_singleton():
    """Test get_agent_factory() returns singleton instance.

    Coverage: Lines 142-147 (get_agent_factory implementation)

    REAL components:
    - Real global _agent_factory singleton logic
    - Real instance creation on first call
    - Real instance reuse on subsequent calls

    MOCKED:
    - None (tests real singleton behavior)

    Scenario:
    1. Call get_agent_factory() twice
    2. Verify both calls return same instance

    Assertions:
    - First call creates new instance
    - Second call returns same instance
    - Instance is SimplifiedAgentFactory type
    """
    # Get factory twice
    factory1 = get_agent_factory()
    factory2 = get_agent_factory()

    # Assert: Same instance
    assert factory1 is factory2
    assert isinstance(factory1, SimplifiedAgentFactory)


@pytest.mark.unit
@pytest.mark.fast
def test_convenience_functions(mock_model_discovery, mock_simple_config, mock_agent_creation):
    """Test convenience functions (create_agent, get_available_models).

    Coverage: Lines 150-171 (convenience function implementations)

    REAL components:
    - Real delegation to factory methods

    MOCKED:
    - Factory methods (via fixtures)

    Scenario:
    1. create_agent() calls factory.create_agent()
    2. get_available_models() calls factory.get_available_models()

    Assertions:
    - Convenience functions delegate correctly
    - Return values match factory methods
    """
    # Test create_agent convenience function
    agent = create_agent("google/gemini-2.5-pro-preview-03-25", temperature=0.8)

    assert agent is not None
    mock_simple_config.assert_called_with("google/gemini-2.5-pro-preview-03-25")
    mock_agent_creation.assert_called()

    # Test get_available_models convenience function
    models = get_available_models()

    assert len(models) == 4
    assert "google/gemini-2.5-pro-preview-03-25" in models
    mock_model_discovery.assert_called()


# ==============================================================================
# ISSUE C: deprecated_on フィルタとライフサイクル API テスト
# ==============================================================================


@pytest.mark.unit
def test_get_available_models_excludes_deprecated(mock_model_discovery_with_deprecated):
    """get_available_models() は deprecated_on が設定されたモデルを除外する。"""
    factory = SimplifiedAgentFactory()
    models = factory.get_available_models()

    assert "openai/gpt-4o" in models
    assert "anthropic/claude-3-5-sonnet-20241022" in models
    assert "openai/gpt-3.5-turbo" not in models


@pytest.mark.unit
def test_list_all_models_includes_deprecated(mock_model_discovery_with_deprecated):
    """list_all_models() は廃止済みモデルも含む全モデルを返す。"""
    factory = SimplifiedAgentFactory()
    models = factory.list_all_models()

    assert "openai/gpt-4o" in models
    assert "anthropic/claude-3-5-sonnet-20241022" in models
    assert "openai/gpt-3.5-turbo" in models


@pytest.mark.unit
def test_is_model_deprecated_true(mock_model_discovery_with_deprecated):
    """廃止済みモデルに is_model_deprecated() が True を返す。"""
    factory = SimplifiedAgentFactory()
    assert factory.is_model_deprecated("openai/gpt-3.5-turbo") is True


@pytest.mark.unit
def test_is_model_deprecated_false_for_active(mock_model_discovery_with_deprecated):
    """アクティブなモデルに is_model_deprecated() が False を返す。"""
    factory = SimplifiedAgentFactory()
    assert factory.is_model_deprecated("openai/gpt-4o") is False


@pytest.mark.unit
def test_is_model_deprecated_false_for_unknown(mock_model_discovery_with_deprecated):
    """未知のモデル ID に is_model_deprecated() が False を返す。"""
    factory = SimplifiedAgentFactory()
    assert factory.is_model_deprecated("unknown/nonexistent-model") is False


@pytest.mark.unit
def test_list_all_models_falls_back_to_discovered_ids_when_toml_empty():
    """TOML データが空（書き込み失敗等）の場合、list_all_models() は discovery 結果を返す。"""
    discovered_ids = ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-3-5-sonnet-20241022"]
    with (
        patch(
            "image_annotator_lib.core.simplified_agent_factory.discover_available_vision_models"
        ) as mock_discover,
        patch(
            "image_annotator_lib.core.simplified_agent_factory.load_available_api_models"
        ) as mock_load,
    ):
        mock_discover.return_value = {"models": discovered_ids}
        mock_load.return_value = {}  # TOML 書き込み失敗 → 空

        factory = SimplifiedAgentFactory()
        all_models = factory.list_all_models()

    assert set(all_models) == set(discovered_ids)
