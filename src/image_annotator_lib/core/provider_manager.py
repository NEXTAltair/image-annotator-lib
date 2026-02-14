"""Simplified provider manager using PydanticAI Agent Factory directly.

Plan 1: Unified implementation eliminating 4 ProviderInstance classes.
"""

from io import BytesIO

from PIL import Image

from ..exceptions.errors import WebApiError
from .config import config_registry
from .pydantic_ai_factory import PydanticAIAgentFactory, preprocess_images_to_binary
from .types import AnnotationResult
from .utils import calculate_phash, logger


class ProviderManager:
    """Simplified provider manager using PydanticAI Agent Factory."""

    @classmethod
    def run_inference_with_model(
        cls,
        model_name: str,
        images_list: list[Image.Image],
        api_model_id: str,
        api_keys: dict[str, str] | None = None,
    ) -> dict[str, AnnotationResult]:
        """Run inference with specified model ID.

        Args:
            model_name: Model name from configuration
            images_list: List of PIL Images
            api_model_id: API model ID (e.g., "gpt-4", "claude-3-opus", "openrouter:model")
            api_keys: Optional dict of API keys by provider

        Returns:
            Dict mapping image phash to AnnotationResult
        """
        logger.debug(f"Running inference: model_name={model_name}, api_model_id={api_model_id}")

        # Get API key for this model
        api_key = cls._get_api_key(model_name, api_model_id, api_keys)

        try:
            # Get or create Agent using the factory
            agent = PydanticAIAgentFactory.get_or_create_agent(api_model_id, api_key)

            # Preprocess images to BinaryContent
            binary_contents = preprocess_images_to_binary(images_list)

            # Run inference on each image
            results = {}
            for i, binary_content in enumerate(binary_contents):
                image = images_list[i] if i < len(images_list) else None
                phash = calculate_phash(image) if image else f"unknown_image_{i}"

                try:
                    # Run agent inference
                    response = agent.run_sync(binary_content)

                    if response.data:
                        tags = getattr(response.data, "tags", [])
                        results[phash] = AnnotationResult(
                            phash=phash,
                            tags=tags,
                            formatted_output=response.data,
                            error=None,
                        )
                    else:
                        results[phash] = AnnotationResult(
                            phash=phash,
                            tags=[],
                            formatted_output=None,
                            error="Empty response from API",
                        )

                except Exception as e:
                    logger.error(
                        f"Inference failed: model={model_name}, api_id={api_model_id}, error={e}",
                        exc_info=True,
                    )
                    results[phash] = AnnotationResult(
                        phash=phash, tags=[], formatted_output=None, error=str(e)
                    )

            logger.debug(f"Inference complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(
                f"Provider manager error: model={model_name}, api_id={api_model_id}, error={e}",
                exc_info=True,
            )
            raise WebApiError(f"Inference failed: {e}", provider_name="Unknown") from e

    @classmethod
    def _get_api_key(
        cls, model_name: str, api_model_id: str, api_keys: dict[str, str] | None = None
    ) -> str | None:
        """Get API key from injected dict or configuration.

        Args:
            model_name: Model name
            api_model_id: API model ID to determine provider
            api_keys: Optional injected API keys

        Returns:
            API key string or None
        """
        # Determine provider from API model ID
        provider = cls._get_provider(api_model_id)

        # Check injected keys first
        if api_keys:
            # Handle provider prefixes
            if api_model_id.startswith("openrouter:"):
                provider = "openrouter"
            if provider in api_keys:
                return api_keys[provider]

        # Fall back to config registry
        api_key = config_registry.get(model_name, "api_key", default="")
        return api_key if api_key else None

    @classmethod
    def _get_provider(cls, api_model_id: str) -> str:
        """Determine provider from API model ID.

        Args:
            api_model_id: Model ID string

        Returns:
            Provider name string
        """
        if ":" in api_model_id:
            return api_model_id.split(":", 1)[0]

        if api_model_id.startswith(("gpt", "o1", "o3")):
            return "openai"
        elif api_model_id.startswith("claude"):
            return "anthropic"
        elif api_model_id.startswith("gemini"):
            return "google"
        else:
            return "unknown"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached agents and providers."""
        PydanticAIAgentFactory.clear_cache()
        logger.debug("Cleared provider manager cache")
