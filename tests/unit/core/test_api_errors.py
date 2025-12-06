"""Unit tests for API error handling in PydanticAI-based annotators.

Tests error handling for common API failure scenarios:
- 401 Authentication errors
- 429 Rate limit errors
- Timeout errors

Mock Strategy:
- Mock: PydanticAI Agent and its run() method to raise exceptions
- Real: Error handling logic in annotator classes
"""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior


@pytest.mark.unit
def test_api_authentication_failure_401(managed_config_registry):
    """Test 401 authentication failure handling.

    Mock Strategy:
    - Mock: Agent.run() to raise ModelHTTPError with 401 status
    - Real: Error handling in AnthropicApiAnnotator.run_with_model()

    Verifies:
    - ModelHTTPError(401) is caught
    - Error message includes status code
    - Result contains error (not exception)
    - NO retry attempted (single call to agent.run)
    """
    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
        AnthropicApiAnnotator,
    )

    # Setup config
    config_dict = {
        "class": "AnthropicApiAnnotator",
        "model_name_on_provider": "claude-3-5-sonnet-20241022",
        "api_model_id": "claude-3-5-sonnet-20241022",
        "api_key": "test_key",
        "timeout": 30,
        "retry_count": 0,
        "capabilities": ["tags", "captions", "scores"],
    }
    managed_config_registry.set("test_auth", config_dict)

    # Create mock Agent with AsyncMock.run() that raises 401 error
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        side_effect=ModelHTTPError(status_code=401, model_name="test_auth", body="Invalid API key")
    )

    with patch(
        "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent",
        return_value=mock_agent,
    ):
        annotator = AnthropicApiAnnotator("test_auth")

        # Create test image
        test_image = Image.new("RGB", (100, 100), color="red")

        with annotator:
            # Call run_with_model (should catch error, not raise)
            results = annotator.run_with_model([test_image], "claude-3-5-sonnet-20241022")

        # Verify error was caught and recorded
        assert len(results) == 1
        result = results[0]
        assert result.error is not None
        assert "401" in result.error or "Unauthorized" in result.error
        assert result.provider_name == "anthropic"

        # Verify NO retry (single call)
        assert mock_agent.run.call_count == 1


@pytest.mark.unit
def test_api_rate_limit_429(managed_config_registry):
    """Test 429 rate limit error handling.

    Mock Strategy:
    - Mock: Agent.run() to raise ModelHTTPError with 429 status
    - Real: Error handling in AnthropicApiAnnotator.run_with_model()

    Verifies:
    - ModelHTTPError(429) is caught
    - Error message includes status code
    - Result contains error (not exception)
    - Current implementation: NO automatic retry (single call)

    Note:
        Current implementation catches 429 but does not implement
        exponential backoff retry logic. Test verifies error is caught
        and recorded correctly.
    """
    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
        AnthropicApiAnnotator,
    )

    # Setup config
    config_dict = {
        "class": "AnthropicApiAnnotator",
        "model_name_on_provider": "claude-3-5-sonnet-20241022",
        "api_model_id": "claude-3-5-sonnet-20241022",
        "api_key": "test_key",
        "timeout": 30,
        "retry_count": 0,
        "capabilities": ["tags", "captions", "scores"],
    }
    managed_config_registry.set("test_rate_limit", config_dict)

    # Create mock Agent with AsyncMock.run() that raises 429 error
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        side_effect=ModelHTTPError(
            status_code=429, model_name="test_rate_limit", body="Too many requests"
        )
    )

    with patch(
        "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent",
        return_value=mock_agent,
    ):
        annotator = AnthropicApiAnnotator("test_rate_limit")

        # Create test image
        test_image = Image.new("RGB", (100, 100), color="blue")

        with annotator:
            # Call run_with_model (should catch error, not raise)
            results = annotator.run_with_model([test_image], "claude-3-5-sonnet-20241022")

        # Verify error was caught and recorded
        assert len(results) == 1
        result = results[0]
        assert result.error is not None
        assert "429" in result.error or "Rate limit" in result.error
        assert result.provider_name == "anthropic"

        # Verify current behavior: NO retry (single call)
        assert mock_agent.run.call_count == 1


@pytest.mark.unit
def test_api_timeout_error(managed_config_registry):
    """Test async timeout error handling.

    Mock Strategy:
    - Mock: Agent.run() to raise asyncio.TimeoutError
    - Real: Error handling in AnthropicApiAnnotator.run_with_model()

    Verifies:
    - asyncio.TimeoutError is caught by generic Exception handler
    - Error message is recorded in result
    - Result contains error (not exception)
    - NO retry attempted (single call)
    """
    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
        AnthropicApiAnnotator,
    )

    # Setup config
    config_dict = {
        "class": "AnthropicApiAnnotator",
        "model_name_on_provider": "claude-3-5-sonnet-20241022",
        "api_model_id": "claude-3-5-sonnet-20241022",
        "api_key": "test_key",
        "timeout": 30,
        "retry_count": 0,
        "capabilities": ["tags", "captions", "scores"],
    }
    managed_config_registry.set("test_timeout", config_dict)

    # Create mock Agent with AsyncMock.run() that raises TimeoutError
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))

    with patch(
        "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent",
        return_value=mock_agent,
    ):
        annotator = AnthropicApiAnnotator("test_timeout")

        # Create test image
        test_image = Image.new("RGB", (100, 100), color="green")

        with annotator:
            # Call run_with_model (should catch error, not raise)
            results = annotator.run_with_model([test_image], "claude-3-5-sonnet-20241022")

        # Verify error was caught and recorded
        assert len(results) == 1
        result = results[0]
        assert result.error is not None
        assert "Error" in result.error or "timed out" in result.error.lower()
        assert result.provider_name == "anthropic"

        # Verify NO retry (single call)
        assert mock_agent.run.call_count == 1


@pytest.mark.unit
def test_unexpected_model_behavior_error(managed_config_registry):
    """Test UnexpectedModelBehavior error handling.

    Mock Strategy:
    - Mock: Agent.run() to raise UnexpectedModelBehavior exception
    - Real: Error handling in AnthropicApiAnnotator.run_with_model()

    Verifies:
    - UnexpectedModelBehavior is caught
    - Error message includes "Unexpected model behavior"
    - Result contains error (not exception)
    """
    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
        AnthropicApiAnnotator,
    )

    # Setup config
    config_dict = {
        "class": "AnthropicApiAnnotator",
        "model_name_on_provider": "claude-3-5-sonnet-20241022",
        "api_model_id": "claude-3-5-sonnet-20241022",
        "api_key": "test_key",
        "timeout": 30,
        "retry_count": 0,
        "capabilities": ["tags", "captions", "scores"],
    }
    managed_config_registry.set("test_unexpected", config_dict)

    # Create mock Agent with AsyncMock.run() that raises UnexpectedModelBehavior
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(
        side_effect=UnexpectedModelBehavior("Model returned invalid structure")
    )

    with patch(
        "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent",
        return_value=mock_agent,
    ):
        annotator = AnthropicApiAnnotator("test_unexpected")

        # Create test image
        test_image = Image.new("RGB", (100, 100), color="yellow")

        with annotator:
            # Call run_with_model (should catch error, not raise)
            results = annotator.run_with_model([test_image], "claude-3-5-sonnet-20241022")

        # Verify error was caught and recorded
        assert len(results) == 1
        result = results[0]
        assert result.error is not None
        assert "Unexpected model behavior" in result.error
        assert result.provider_name == "anthropic"

        # Verify single call
        assert mock_agent.run.call_count == 1


@pytest.mark.unit
def test_api_error_with_multiple_images(managed_config_registry):
    """Test API error handling with multiple images.

    Mock Strategy:
    - Mock: Agent.run() to raise ModelHTTPError on first call, succeed on second
    - Real: Error handling in AnthropicApiAnnotator.run_with_model()

    Verifies:
    - Errors are recorded per-image
    - Partial results returned (error for first, success for second)
    - Both images processed independently
    """
    from image_annotator_lib.model_class.annotator_webapi.anthropic_api import (
        AnthropicApiAnnotator,
    )
    from image_annotator_lib.core.types import AnnotationSchema

    # Setup config
    config_dict = {
        "class": "AnthropicApiAnnotator",
        "model_name_on_provider": "claude-3-5-sonnet-20241022",
        "api_model_id": "claude-3-5-sonnet-20241022",
        "api_key": "test_key",
        "timeout": 30,
        "retry_count": 0,
        "capabilities": ["tags", "captions", "scores"],
    }
    managed_config_registry.set("test_multi", config_dict)

    # Create mock Agent with side_effect: error on first call, success on second
    mock_agent = MagicMock()
    success_response = AnnotationSchema(
        tags=["test_tag"], captions=["test caption"], score=0.9
    )
    mock_agent.run = AsyncMock(
        side_effect=[
            ModelHTTPError(status_code=500, model_name="test_multi", body="Internal error"),
            MagicMock(output=success_response),  # ModelResponse with output
        ]
    )

    with patch(
        "image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent",
        return_value=mock_agent,
    ):
        annotator = AnthropicApiAnnotator("test_multi")

        # Create two test images
        test_images = [
            Image.new("RGB", (100, 100), color="red"),
            Image.new("RGB", (100, 100), color="blue"),
        ]

        with annotator:
            # Call run_with_model with multiple images
            results = annotator.run_with_model(test_images, "claude-3-5-sonnet-20241022")

        # Verify partial results: error for first, success for second
        assert len(results) == 2

        # First result: error
        assert results[0].error is not None
        assert "500" in results[0].error

        # Second result: success
        assert results[1].error is None
        assert results[1].tags == ["test_tag"]
        assert results[1].captions == ["test caption"]

        # Verify both images processed (2 calls)
        assert mock_agent.run.call_count == 2
