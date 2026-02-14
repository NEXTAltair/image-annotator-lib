"""
PydanticAI Dependencies Models for WebAPI Annotators

Defines dependency injection models for PydanticAI-based WebAPI annotators,
following PydanticAI recommended patterns while maintaining compatibility
with existing configuration system.
"""

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class WebApiDependencies(BaseModel):
    """Base dependency model for all WebAPI annotators."""

    # Core identification
    model_name: str = Field(description="Model name from configuration")
    api_model_id: str = Field(description="Actual model ID used by provider API")
    provider_name: str = Field(description="API provider name (openai, google, anthropic, etc.)")

    # Authentication
    api_key: SecretStr = Field(description="API key for authentication")

    # Timeout and retry settings
    timeout: int = Field(default=60, description="API request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    # Rate limiting
    min_request_interval: float = Field(default=1.0, description="Minimum interval between requests")

    # Output settings
    max_output_tokens: int = Field(default=1800, description="Maximum tokens in API response")

    # Prompt template
    prompt_template: str = Field(default="Describe this image.", description="Base prompt template")


class OpenAIDependencies(WebApiDependencies):
    """Dependencies for OpenAI/OpenRouter API annotators."""

    temperature: float = Field(default=0.7, description="Sampling temperature")
    response_format: Any | None = Field(default=None, description="Structured output format")

    # OpenRouter specific
    json_schema_supported: bool = Field(
        default=False, description="Whether JSON schema output is supported"
    )
    referer: str | None = Field(default=None, description="HTTP referer for OpenRouter")
    app_name: str | None = Field(default=None, description="Application name for OpenRouter")


class GoogleDependencies(WebApiDependencies):
    """Dependencies for Google Gemini API annotators."""

    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    top_k: int = Field(default=32, description="Top-k sampling")


class AnthropicDependencies(WebApiDependencies):
    """Dependencies for Anthropic Claude API annotators."""

    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1800, description="Maximum tokens (Anthropic uses max_tokens)")


# Type alias for easier imports
WebApiDependenciesType = (
    WebApiDependencies | OpenAIDependencies | GoogleDependencies | AnthropicDependencies
)
