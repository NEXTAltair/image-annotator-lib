"""Unit tests for API error handling in Plan 1 architecture.

Tests error handling for common API failure scenarios through
ProviderManager -> PydanticAIAgentFactory -> Agent path:
- 401 Authentication errors
- 429 Rate limit errors
- Timeout errors
- Unexpected model behavior errors
- Partial failure with multiple images

Mock Strategy:
- Mock: PydanticAIAgentFactory.get_or_create_agent to return mock Agent
- Mock: Agent.run_sync() to raise exceptions
- Real: Error handling logic in ProviderManager.run_inference_with_model()
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.types import AnnotationSchema


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    ProviderManager.clear_cache()
    yield
    ProviderManager.clear_cache()


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
@patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
@patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_api_authentication_failure_401(
    mock_config, mock_factory, mock_preprocess, mock_phash
):
    """Test 401 authentication failure is caught and returned as error result.

    Verifies:
    - ModelHTTPError(401) is caught per-image
    - Error message includes status code information
    - Result contains error (not exception propagation)
    """
    mock_config.get.return_value = "test_api_key"
    mock_phash.return_value = "phash_auth"
    mock_preprocess.return_value = [MagicMock()]

    mock_agent = MagicMock()
    mock_factory.get_or_create_agent.return_value = mock_agent
    mock_agent.run_sync.side_effect = ModelHTTPError(
        status_code=401, model_name="claude-3-opus", body="Invalid API key"
    )

    test_image = Image.new("RGB", (100, 100), color="red")
    results = ProviderManager.run_inference_with_model(
        model_name="test_model",
        images_list=[test_image],
        api_model_id="claude-3-opus",
    )

    assert len(results) == 1
    result = results["phash_auth"]
    assert result["error"] is not None
    assert "401" in result["error"] or "Invalid" in result["error"]
    assert result["tags"] == []
    assert mock_agent.run_sync.call_count == 1


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
@patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
@patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_api_rate_limit_429(
    mock_config, mock_factory, mock_preprocess, mock_phash
):
    """Test 429 rate limit error is caught and returned as error result.

    Verifies:
    - ModelHTTPError(429) is caught per-image
    - Error message includes rate limit information
    - NO automatic retry (single call to agent)
    """
    mock_config.get.return_value = "test_api_key"
    mock_phash.return_value = "phash_rate"
    mock_preprocess.return_value = [MagicMock()]

    mock_agent = MagicMock()
    mock_factory.get_or_create_agent.return_value = mock_agent
    mock_agent.run_sync.side_effect = ModelHTTPError(
        status_code=429, model_name="claude-3-opus", body="Too many requests"
    )

    test_image = Image.new("RGB", (100, 100), color="blue")
    results = ProviderManager.run_inference_with_model(
        model_name="test_model",
        images_list=[test_image],
        api_model_id="claude-3-opus",
    )

    assert len(results) == 1
    result = results["phash_rate"]
    assert result["error"] is not None
    assert "429" in result["error"] or "Too many" in result["error"]
    assert result["tags"] == []
    assert mock_agent.run_sync.call_count == 1


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
@patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
@patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_api_timeout_error(
    mock_config, mock_factory, mock_preprocess, mock_phash
):
    """Test timeout error is caught and returned as error result.

    Verifies:
    - TimeoutError is caught by generic Exception handler
    - Error message is recorded in result
    - NO retry attempted (single call)
    """
    mock_config.get.return_value = "test_api_key"
    mock_phash.return_value = "phash_timeout"
    mock_preprocess.return_value = [MagicMock()]

    mock_agent = MagicMock()
    mock_factory.get_or_create_agent.return_value = mock_agent
    mock_agent.run_sync.side_effect = TimeoutError("Request timed out")

    test_image = Image.new("RGB", (100, 100), color="green")
    results = ProviderManager.run_inference_with_model(
        model_name="test_model",
        images_list=[test_image],
        api_model_id="gpt-4",
    )

    assert len(results) == 1
    result = results["phash_timeout"]
    assert result["error"] is not None
    assert "timed out" in result["error"].lower()
    assert result["tags"] == []
    assert mock_agent.run_sync.call_count == 1


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
@patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
@patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_unexpected_model_behavior_error(
    mock_config, mock_factory, mock_preprocess, mock_phash
):
    """Test UnexpectedModelBehavior error is caught and returned as error result.

    Verifies:
    - UnexpectedModelBehavior is caught
    - Error message includes description
    - Result contains error (not exception)
    """
    mock_config.get.return_value = "test_api_key"
    mock_phash.return_value = "phash_unexpected"
    mock_preprocess.return_value = [MagicMock()]

    mock_agent = MagicMock()
    mock_factory.get_or_create_agent.return_value = mock_agent
    mock_agent.run_sync.side_effect = UnexpectedModelBehavior(
        "Model returned invalid structure"
    )

    test_image = Image.new("RGB", (100, 100), color="yellow")
    results = ProviderManager.run_inference_with_model(
        model_name="test_model",
        images_list=[test_image],
        api_model_id="gpt-4",
    )

    assert len(results) == 1
    result = results["phash_unexpected"]
    assert result["error"] is not None
    assert "invalid structure" in result["error"].lower()
    assert result["tags"] == []


@pytest.mark.unit
@patch("image_annotator_lib.core.provider_manager.calculate_phash")
@patch("image_annotator_lib.core.provider_manager.preprocess_images_to_binary")
@patch("image_annotator_lib.core.provider_manager.PydanticAIAgentFactory")
@patch("image_annotator_lib.core.provider_manager.config_registry")
def test_api_error_with_multiple_images(
    mock_config, mock_factory, mock_preprocess, mock_phash
):
    """Test API error handling with multiple images (partial failure).

    Verifies:
    - Errors are recorded per-image independently
    - First image fails, second succeeds
    - Both images processed (no early termination)
    """
    mock_config.get.return_value = "test_api_key"
    mock_phash.side_effect = ["phash_fail", "phash_ok"]
    mock_preprocess.return_value = [MagicMock(), MagicMock()]

    mock_agent = MagicMock()
    mock_factory.get_or_create_agent.return_value = mock_agent

    # 1件目: エラー、2件目: 成功
    success_response = MagicMock()
    success_response.data = AnnotationSchema(
        tags=["test_tag"], captions=["test caption"], score=0.9
    )
    mock_agent.run_sync.side_effect = [
        ModelHTTPError(
            status_code=500, model_name="gpt-4", body="Internal error"
        ),
        success_response,
    ]

    test_images = [
        Image.new("RGB", (100, 100), color="red"),
        Image.new("RGB", (100, 100), color="blue"),
    ]
    results = ProviderManager.run_inference_with_model(
        model_name="test_model",
        images_list=test_images,
        api_model_id="gpt-4",
    )

    assert len(results) == 2

    # 1件目: エラー
    assert results["phash_fail"]["error"] is not None
    assert "500" in results["phash_fail"]["error"]
    assert results["phash_fail"]["tags"] == []

    # 2件目: 成功
    assert results["phash_ok"]["error"] is None
    assert results["phash_ok"]["tags"] == ["test_tag"]

    # 両方処理された
    assert mock_agent.run_sync.call_count == 2
