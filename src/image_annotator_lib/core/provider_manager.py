"""Provider-level instance manager for efficient PydanticAI usage."""

from typing import Any, ClassVar

from PIL import Image

from ..exceptions.errors import WebApiError
from .config import config_registry
from .types import AnnotationResult, RawOutput
from .utils import calculate_phash, logger


class ProviderManager:
    """Manages provider-level instances for efficient PydanticAI usage"""

    _provider_instances: ClassVar[dict[str, Any]] = {}

    @classmethod
    def get_provider_instance(cls, provider_name: str) -> Any:
        """Get or create provider-level instance"""

        if provider_name not in cls._provider_instances:
            # Create new provider-level instance
            if provider_name == "anthropic":
                cls._provider_instances[provider_name] = AnthropicProviderInstance()
            elif provider_name == "openai":
                cls._provider_instances[provider_name] = OpenAIProviderInstance()
            elif provider_name == "openrouter":
                cls._provider_instances[provider_name] = OpenRouterProviderInstance()
            elif provider_name == "google":
                cls._provider_instances[provider_name] = GoogleProviderInstance()
            else:
                raise WebApiError(f"Unsupported provider: {provider_name}")

            logger.debug(f"Created new provider instance: {provider_name}")

        return cls._provider_instances[provider_name]

    @classmethod
    def run_inference_with_model(
        cls, model_name: str, images_list: list[Image.Image], api_model_id: str
    ) -> dict[str, AnnotationResult]:
        """Run inference with specified model ID using provider sharing
        
        Returns:
            Dict mapping image phash to AnnotationResult
        """

        # Determine provider from model configuration or model ID
        provider_name = cls._determine_provider(model_name, api_model_id)

        # Get provider instance
        provider_instance = cls.get_provider_instance(provider_name)

        # Execute inference
        raw_outputs = provider_instance.run_with_model(model_name, images_list, api_model_id)
        
        # Convert to phash-based dict format expected by tests
        results = {}
        for i, raw_output in enumerate(raw_outputs):
            # Generate phash for each image
            image_phash = calculate_phash(images_list[i]) if i < len(images_list) else f"unknown_image_{i}"
            
            # Convert RawOutput to AnnotationResult format
            if raw_output.get("error"):
                annotation_result = AnnotationResult(
                    phash=image_phash,
                    tags=[],
                    formatted_output=None,
                    error=raw_output["error"]
                )
            else:
                response = raw_output.get("response")
                if response:
                    tags = getattr(response, "tags", []) if hasattr(response, "tags") else []
                    annotation_result = AnnotationResult(
                        phash=image_phash,
                        tags=tags,
                        formatted_output=response,
                        error=None
                    )
                else:
                    annotation_result = AnnotationResult(
                        phash=image_phash,
                        tags=[],
                        formatted_output=None,
                        error="No response from provider"
                    )
            
            results[image_phash] = annotation_result
        
        return results

    @classmethod
    def _determine_provider(cls, model_name: str, api_model_id: str) -> str:
        """Determine provider from model configuration or model ID"""

        # First check explicit provider configuration
        provider = config_registry.get(model_name, "provider")
        if provider:
            return provider.lower()

        # Auto-detect from model ID
        if ":" in api_model_id:
            return api_model_id.split(":", 1)[0]

        # Auto-detect from model name patterns
        if api_model_id.startswith(("gpt", "o1", "o3")):
            return "openai"
        elif api_model_id.startswith("claude"):
            return "anthropic"
        elif api_model_id.startswith("gemini"):
            return "google"
        else:
            # Default fallback based on model configuration structure
            if config_registry.get(model_name, "anthropic_api_key"):
                return "anthropic"
            elif config_registry.get(model_name, "openai_api_key"):
                return "openai"
            elif config_registry.get(model_name, "google_api_key"):
                return "google"
            else:
                return "openai"  # Default fallback


class ProviderInstanceBase:
    """Base class for provider instances"""

    def __init__(self):
        self._active_contexts = {}

    def run_with_model(
        self, model_name: str, images_list: list[Image.Image], api_model_id: str
    ) -> list[RawOutput]:
        """Run inference with specified model ID"""

        # Get or create context-managed annotator for this model_name
        if model_name not in self._active_contexts:
            annotator = self._create_annotator(model_name)
            context = annotator.__enter__()
            self._active_contexts[model_name] = (annotator, context)

        _, context = self._active_contexts[model_name]

        # Execute inference with specified model
        return context.run_with_model(images_list, api_model_id)

    def _create_annotator(self, model_name: str):
        """Create annotator instance - to be implemented by subclasses"""
        raise NotImplementedError

    def cleanup_context(self, model_name: str):
        """Clean up context for specific model"""
        if model_name in self._active_contexts:
            annotator, context = self._active_contexts[model_name]
            try:
                annotator.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during context cleanup for {model_name}: {e}")
            del self._active_contexts[model_name]


class AnthropicProviderInstance(ProviderInstanceBase):
    """Anthropic provider instance"""

    def _create_annotator(self, model_name: str):
        from ..model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator

        return AnthropicApiAnnotator(model_name)


class OpenAIProviderInstance(ProviderInstanceBase):
    """OpenAI provider instance"""

    def _create_annotator(self, model_name: str):
        from ..model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

        return OpenAIApiAnnotator(model_name)


class OpenRouterProviderInstance(ProviderInstanceBase):
    """OpenRouter provider instance"""

    def _create_annotator(self, model_name: str):
        from ..model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        return OpenRouterApiAnnotator(model_name)


class GoogleProviderInstance(ProviderInstanceBase):
    """Google provider instance"""

    def _create_annotator(self, model_name: str):
        from ..model_class.annotator_webapi.google_api import GoogleApiAnnotator

        return GoogleApiAnnotator(model_name)
