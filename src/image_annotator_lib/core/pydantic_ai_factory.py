"""PydanticAI Agent factory using infer_model() exclusively.

Plan 1: Simplified implementation relying on PydanticAI's built-in capabilities.
"""

import asyncio
import os
from io import BytesIO
from typing import Any, ClassVar

from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import infer_provider_class

from ..model_class.annotator_webapi.webapi_shared import BASE_PROMPT
from .config import config_registry
from .types import AnnotationSchema, UnifiedAnnotationResult
from .utils import logger


def _is_test_environment() -> bool:
    """Check if we're running in a test environment.

    Uses lightweight checks only (env vars + sys.modules).
    No inspect.stack() to avoid performance overhead.

    Returns:
        True if running in a test environment.
    """
    import sys

    if os.getenv("TESTING"):
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if "pytest" in sys.modules:
        return True
    return False


class PydanticAIAgentFactory:
    """Simplified PydanticAI Agent factory using infer_model() exclusively.

    Plan 1 Design:
    - Delegates all model inference to PydanticAI's infer_model()
    - Simple Agent caching by api_model_id
    - OpenRouter support via OpenAIProvider + OpenAIChatModel pattern
    """

    _agents: ClassVar[dict[str, Agent[None, AnnotationSchema]]] = {}

    @classmethod
    def get_or_create_agent(
        cls, api_model_id: str, api_key: str | None = None, config_data: dict[str, Any] | None = None
    ) -> Agent[None, AnnotationSchema]:
        """Get or create Agent, with OpenRouter special handling.

        Args:
            api_model_id: Model ID (e.g., "gpt-4", "claude-3-opus", "openrouter:model-name")
            api_key: API key for the provider
            config_data: Optional config dict for OpenRouter custom headers

        Returns:
            PydanticAI Agent ready for inference
        """
        # Check for OpenRouter special case
        if api_model_id.startswith("openrouter:"):
            return cls._create_openrouter_agent(api_model_id, api_key, config_data)

        # Standard agent creation for all other providers
        cache_key = api_model_id
        if cache_key in cls._agents:
            logger.debug(f"Agent cache hit: {cache_key}")
            return cls._agents[cache_key]

        # Set API key in environment if provided
        if api_key:
            cls._set_env_api_key(api_model_id, api_key)

        try:
            # Let PydanticAI infer everything: provider, model format, etc.
            agent = Agent(model=api_model_id, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
            cls._agents[cache_key] = agent
            logger.debug(f"Created and cached agent for: {api_model_id}")
            return agent
        except Exception as e:
            logger.error(
                f"Agent creation failed for {api_model_id}: {type(e).__name__}: {e}", exc_info=True
            )
            raise

    @classmethod
    def _create_openrouter_agent(
        cls, api_model_id: str, api_key: str | None = None, config_data: dict[str, Any] | None = None
    ) -> Agent[None, AnnotationSchema]:
        """Create OpenRouter agent using OpenAIProvider + OpenAIChatModel pattern.

        Args:
            api_model_id: Model ID in format "openrouter:provider/model-name"
            api_key: OpenRouter API key
            config_data: Optional config with referer and app_name

        Returns:
            PydanticAI Agent configured for OpenRouter
        """
        # Test environment fallback
        if _is_test_environment():
            from pydantic_ai.messages import ModelResponse, TextPart
            from pydantic_ai.models.test import TestModel

            test_model = TestModel()
            test_model.response = ModelResponse(  # type: ignore[attr-defined]
                parts=[TextPart('{"tags": ["test"], "captions": ["test"], "score": 0.95}')]
            )
            agent = Agent(model=test_model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
            logger.debug("Created TestModel Agent for OpenRouter test environment")
            return agent

        # Extract actual model ID (remove "openrouter:" prefix)
        actual_model_id = api_model_id.split(":", 1)[1]

        # Build OpenRouter provider kwargs
        provider_kwargs: dict[str, Any] = {
            "api_key": api_key or "",
            "base_url": "https://openrouter.ai/api/v1",
        }

        # Add custom headers if provided
        if config_data:
            headers = {}
            if config_data.get("referer"):
                headers["HTTP-Referer"] = config_data["referer"]
            if config_data.get("app_name"):
                headers["X-Title"] = config_data["app_name"]
            if headers:
                provider_kwargs["default_headers"] = headers

        # Create OpenAI provider for OpenRouter
        try:
            provider = infer_provider_class("openai")(**provider_kwargs)
            # Create model with provider using official PydanticAI pattern
            model = OpenAIChatModel(model_name=actual_model_id, provider=provider)
            agent = Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
            logger.debug(f"Created OpenRouter Agent: {actual_model_id}")
            return agent
        except Exception as e:
            logger.error(f"OpenRouter agent creation failed: {e}", exc_info=True)
            raise

    @classmethod
    def _set_env_api_key(cls, api_model_id: str, api_key: str) -> None:
        """Set API key in environment for PydanticAI's auto-detection."""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        # Determine provider from model ID
        if api_model_id.startswith(("gpt", "o1", "o3")):
            provider = "openai"
        elif api_model_id.startswith("claude"):
            provider = "anthropic"
        elif api_model_id.startswith("gemini"):
            provider = "google"
        else:
            logger.warning(f"Unable to determine provider for {api_model_id}")
            return

        env_var = env_var_map.get(provider)
        if env_var:
            os.environ[env_var] = api_key
            logger.debug(f"Set {env_var} from api_key parameter")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached agents."""
        cls._agents.clear()
        logger.debug("Cleared PydanticAI agent cache")


# Helper function for image preprocessing
def preprocess_images_to_binary(images: list[Image.Image]) -> list[BinaryContent]:
    """Convert PIL Images to PydanticAI BinaryContent format.

    Args:
        images: List of PIL Images

    Returns:
        List of BinaryContent objects ready for PydanticAI Agent
    """
    binary_contents = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        binary_content = BinaryContent(data=buffered.getvalue(), media_type="image/webp")
        binary_contents.append(binary_content)
    return binary_contents
