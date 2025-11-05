"""Unit tests for webapi_helpers module.

Web APIコンポーネント準備のヘルパー関数群をテスト。
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from image_annotator_lib.core.model_factory_adapters.webapi_helpers import (
    _find_model_entry_by_name,
    _get_api_key,
    _initialize_api_client,
    _process_model_id,
    prepare_web_api_components,
)
from image_annotator_lib.exceptions.errors import ApiAuthenticationError, ConfigurationError

# ==============================================================================
# Test _find_model_entry_by_name
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_find_model_entry_by_name_found():
    """Test _find_model_entry_by_name when model is found."""
    available_models = {
        "gemini-1.5-pro": {"model_name_short": "Gemini Pro", "provider": "Google"},
        "gpt-4": {"model_name_short": "GPT-4", "provider": "OpenAI"},
    }

    result = _find_model_entry_by_name("Gemini Pro", available_models)

    assert result is not None
    assert result[0] == "gemini-1.5-pro"
    assert result[1]["provider"] == "Google"


@pytest.mark.unit
@pytest.mark.fast
def test_find_model_entry_by_name_not_found():
    """Test _find_model_entry_by_name when model is not found."""
    available_models = {
        "gemini-1.5-pro": {"model_name_short": "Gemini Pro", "provider": "Google"},
    }

    result = _find_model_entry_by_name("Unknown Model", available_models)

    assert result is None


# ==============================================================================
# Test _get_api_key
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_api_key_openai_from_env():
    """Test _get_api_key for OpenAI from environment variable."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        with patch(
            "image_annotator_lib.core.model_factory_adapters.webapi_helpers.dotenv.dotenv_values"
        ) as mock_dotenv:
            mock_dotenv.return_value = {}

            result = _get_api_key("OpenAI", "gpt-4")

            assert result == "test-openai-key"


@pytest.mark.unit
@pytest.mark.fast
def test_get_api_key_google_from_dotenv():
    """Test _get_api_key for Google from .env file."""
    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.dotenv.dotenv_values"
    ) as mock_dotenv:
        mock_dotenv.return_value = {"GOOGLE_API_KEY": "test-google-key-from-env"}

        result = _get_api_key("Google", "gemini-1.5-pro")

        assert result == "test-google-key-from-env"


@pytest.mark.unit
@pytest.mark.fast
def test_get_api_key_openrouter_with_colon():
    """Test _get_api_key for OpenRouter model with colon."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-or-key"}):
        with patch(
            "image_annotator_lib.core.model_factory_adapters.webapi_helpers.dotenv.dotenv_values"
        ) as mock_dotenv:
            mock_dotenv.return_value = {}

            result = _get_api_key("Google", "gemma-3-27b-it:free")

            assert result == "test-or-key"


@pytest.mark.unit
@pytest.mark.fast
def test_get_api_key_not_found():
    """Test _get_api_key raises error when key not found."""
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "image_annotator_lib.core.model_factory_adapters.webapi_helpers.dotenv.dotenv_values"
        ) as mock_dotenv:
            mock_dotenv.return_value = {}

            with pytest.raises(ApiAuthenticationError, match="APIキー 'OPENAI_API_KEY' が.envファイル"):
                _get_api_key("OpenAI", "gpt-4")


# ==============================================================================
# Test _process_model_id
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_process_model_id_openai_with_prefix():
    """Test _process_model_id removes openai/ prefix."""
    result = _process_model_id("openai/gpt-4", "OpenAI")

    assert result == "gpt-4"


@pytest.mark.unit
@pytest.mark.fast
def test_process_model_id_google_with_prefix():
    """Test _process_model_id removes google/ prefix."""
    result = _process_model_id("google/gemini-1.5-pro", "Google")

    assert result == "gemini-1.5-pro"


@pytest.mark.unit
@pytest.mark.fast
def test_process_model_id_anthropic_with_prefix():
    """Test _process_model_id removes anthropic/ prefix."""
    result = _process_model_id("anthropic/claude-3-5-sonnet", "Anthropic")

    assert result == "claude-3-5-sonnet"


@pytest.mark.unit
@pytest.mark.fast
def test_process_model_id_openrouter_with_colon():
    """Test _process_model_id keeps OpenRouter model ID with colon."""
    result = _process_model_id("meta-llama/llama-3.3-70b-instruct:free", "OpenRouter")

    assert result == "meta-llama/llama-3.3-70b-instruct:free"


@pytest.mark.unit
@pytest.mark.fast
def test_process_model_id_no_prefix():
    """Test _process_model_id returns original ID when no prefix."""
    result = _process_model_id("gpt-4", "OpenAI")

    assert result == "gpt-4"


# ==============================================================================
# Test _initialize_api_client
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_openai():
    """Test _initialize_api_client for OpenAI."""
    model_config = {"api_key": "test-key", "system_prompt": "Test system", "base_prompt": "Test base"}

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        result = _initialize_api_client("openai", model_config, "test-model")

        assert result is not None
        mock_openai_class.assert_called_once_with(api_key="test-key")


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_google():
    """Test _initialize_api_client for Google."""
    model_config = {"api_key": SecretStr("test-google-key"), "system_prompt": "Test", "base_prompt": "Test"}

    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.genai.Client"
    ) as mock_genai_class:
        mock_client = MagicMock()
        mock_genai_class.return_value = mock_client

        result = _initialize_api_client("google", model_config, "gemini-test")

        assert result is not None
        mock_genai_class.assert_called_once_with(api_key="test-google-key")


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_anthropic():
    """Test _initialize_api_client for Anthropic."""
    model_config = {"api_key": "test-anthropic-key"}

    with patch("anthropic.Anthropic") as mock_anthropic_class:
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        result = _initialize_api_client("anthropic", model_config, "claude-test")

        assert result is not None
        mock_anthropic_class.assert_called_once_with(api_key="test-anthropic-key")


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_openrouter():
    """Test _initialize_api_client for OpenRouter."""
    model_config = {
        "api_key": "test-or-key",
        "api_model_id": "meta-llama/llama-3.3:free",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_site_url": "http://test.com",
        "openrouter_app_name": "test-app",
    }

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        result = _initialize_api_client("openrouter", model_config, "test-or-model")

        assert result is not None
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "test-or-key"
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert "HTTP-Referer" in call_kwargs["default_headers"]


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_unsupported_provider():
    """Test _initialize_api_client raises error for unsupported provider."""
    model_config = {"api_key": "test-key"}

    with pytest.raises(ConfigurationError, match="Unsupported Web API provider"):
        _initialize_api_client("unsupported", model_config, "test-model")


@pytest.mark.unit
@pytest.mark.fast
def test_initialize_api_client_missing_api_key():
    """Test _initialize_api_client raises error when API key missing."""
    model_config = {}

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ApiAuthenticationError, match="API key for model"):
            _initialize_api_client("openai", model_config, "test-model")


# ==============================================================================
# Test prepare_web_api_components
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_web_api_components_success():
    """Test prepare_web_api_components successful flow."""
    mock_available_models = {
        "gemini-1.5-pro": {
            "model_name_short": "Gemini Pro",
            "provider": "Google",
            "api_key": "test-key",
        }
    }

    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.config.load_available_api_models"
    ) as mock_load:
        mock_load.return_value = mock_available_models

        with patch(
            "image_annotator_lib.core.model_factory_adapters.webapi_helpers._initialize_api_client"
        ) as mock_init:
            mock_client = MagicMock()
            mock_init.return_value = mock_client

            result = prepare_web_api_components("Gemini Pro")

            assert result["client"] == mock_client
            assert result["api_model_id"] == "gemini-1.5-pro"
            assert result["provider_name"] == "Google"


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_web_api_components_model_not_found():
    """Test prepare_web_api_components when model not found."""
    mock_available_models = {"gemini-1.5-pro": {"model_name_short": "Gemini Pro", "provider": "Google"}}

    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.config.load_available_api_models"
    ) as mock_load:
        mock_load.return_value = mock_available_models

        with pytest.raises(
            ConfigurationError, match="に対応するエントリが available_api_models.toml に見つかりません"
        ):
            prepare_web_api_components("Unknown Model")


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_web_api_components_no_provider():
    """Test prepare_web_api_components when provider missing in config."""
    mock_available_models = {"model-id": {"model_name_short": "Test Model"}}

    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.config.load_available_api_models"
    ) as mock_load:
        mock_load.return_value = mock_available_models

        with pytest.raises(ConfigurationError, match="'provider' が含まれていません"):
            prepare_web_api_components("Test Model")


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_web_api_components_empty_available_models():
    """Test prepare_web_api_components when no models available."""
    with patch(
        "image_annotator_lib.core.model_factory_adapters.webapi_helpers.config.load_available_api_models"
    ) as mock_load:
        mock_load.return_value = {}

        with pytest.raises(ConfigurationError, match="利用可能なAPIモデル情報"):
            prepare_web_api_components("Any Model")
