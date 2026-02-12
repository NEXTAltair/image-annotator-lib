"""Unit tests for pydantic_ai_factory module."""

import os
from unittest.mock import Mock, patch

import pytest

from image_annotator_lib.core.pydantic_ai_factory import (
    PydanticAIAgentFactory,
    _is_test_environment,
)


@pytest.fixture(autouse=True)
def clear_agent_cache():
    """Clear agent cache before each test."""
    PydanticAIAgentFactory._agents.clear()
    yield
    PydanticAIAgentFactory._agents.clear()


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
# PydanticAIAgentFactory tests
# ========================================


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_get_or_create_agent_creates_new_agent(mock_agent_class):
    """Test get_or_create_agent creates new agent."""
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result = PydanticAIAgentFactory.get_or_create_agent("gpt-4", api_key="test_key")

    assert result == mock_agent
    mock_agent_class.assert_called_once()


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_get_or_create_agent_caches_agent(mock_agent_class):
    """Test get_or_create_agent caches agent for same model_id."""
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result1 = PydanticAIAgentFactory.get_or_create_agent("gpt-4", api_key="key1")
    result2 = PydanticAIAgentFactory.get_or_create_agent("gpt-4", api_key="key1")

    assert result1 is result2
    assert mock_agent_class.call_count == 1


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_get_or_create_agent_different_models_creates_new_agent(mock_agent_class):
    """Test get_or_create_agent creates new agent for different model."""
    mock_agent1 = Mock()
    mock_agent2 = Mock()
    mock_agent_class.side_effect = [mock_agent1, mock_agent2]

    result1 = PydanticAIAgentFactory.get_or_create_agent("gpt-4", api_key="key1")
    result2 = PydanticAIAgentFactory.get_or_create_agent(
        "claude-3-opus", api_key="key1"
    )

    assert result1 is not result2
    assert mock_agent_class.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.core.pydantic_ai_factory.Agent")
def test_get_or_create_agent_openrouter_prefix(mock_agent_class):
    """Test get_or_create_agent handles openrouter: prefix."""
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent

    result = PydanticAIAgentFactory.get_or_create_agent(
        "openrouter:mistral-7b", api_key="test_key"
    )

    assert result == mock_agent


@pytest.mark.unit
def test_clear_cache_removes_all_agents():
    """Test clear_cache removes all cached agents."""
    # Manually add some agents to the cache
    PydanticAIAgentFactory._agents["test_agent_1"] = Mock()
    PydanticAIAgentFactory._agents["test_agent_2"] = Mock()

    assert len(PydanticAIAgentFactory._agents) == 2

    PydanticAIAgentFactory.clear_cache()

    assert len(PydanticAIAgentFactory._agents) == 0
