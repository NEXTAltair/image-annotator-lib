"""
PydanticAI-based WebAPI Annotator Implementation

A proof-of-concept implementation of WebApiBaseAnnotator using PydanticAI Agent
while maintaining compatibility with existing interfaces and behaviors.

This replaces the existing WebApiBaseAnnotator with PydanticAI integration
while preserving all valuable existing functionality.
"""

import asyncio
import time
from abc import abstractmethod
from io import BytesIO
from typing import Any, Self, override

from PIL import Image
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.types import AnnotationSchema, RawOutput, WebApiFormattedOutput
from image_annotator_lib.core.utils import logger

# Import from main package for prototype
from image_annotator_lib.exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ConfigurationError,
    InsufficientCreditsError,
    WebApiError,
)

# Local imports
from .base_annotator import BaseAnnotator
from .dependencies import WebApiDependencies, WebApiDependenciesType


class PydanticAIWebApiAnnotator(BaseAnnotator):
    """PydanticAI Agent-based WebAPI annotator base class.

    This class replaces WebApiBaseAnnotator with PydanticAI integration
    while maintaining the same external interface and behavior.
    """

    def __init__(self, dependencies: WebApiDependenciesType):
        """Initialize with dependency injection instead of model_name.

        Args:
            dependencies: PydanticAI dependency model containing all necessary configuration
        """
        # Initialize parent with model_name from dependencies
        super().__init__(dependencies.model_name)

        # Store dependencies
        self.deps = dependencies

        # Rate limiting state
        self.last_request_time = 0.0

        # Agent will be created by subclasses
        self.agent: Agent | None = None

    @classmethod
    def from_model_name(cls, model_name: str) -> Self:
        """Create annotator from model_name using existing config system.

        This provides backward compatibility with existing code that uses model_name.
        """
        dependencies = cls._create_dependencies_from_config(model_name)
        return cls(dependencies)

    @classmethod
    def _create_dependencies_from_config(cls, model_name: str) -> WebApiDependencies:
        """Convert existing config system to dependency model.

        This bridges the gap between old config-based system and new dependency injection.
        """
        # Extract configuration values
        model_path = config_registry.get(model_name, "model_path")
        provider_name = cls._determine_provider_from_model_path(model_path)

        # Get API key from environment or config
        api_key = cls._get_api_key_for_provider(provider_name)

        # Build dependency model
        dependencies = WebApiDependencies(
            model_name=model_name,
            api_model_id=model_path,  # In most cases, model_path is the API model ID
            provider_name=provider_name,
            api_key=SecretStr(api_key),
            timeout=config_registry.get(model_name, "timeout", 60),
            retry_count=config_registry.get(model_name, "retry_count", 3),
            retry_delay=config_registry.get(model_name, "retry_delay", 1.0),
            min_request_interval=config_registry.get(model_name, "min_request_interval", 1.0),
            max_output_tokens=config_registry.get(model_name, "max_output_tokens", 1800),
            prompt_template=config_registry.get(model_name, "prompt_template", "Describe this image."),
        )

        return dependencies

    @staticmethod
    def _determine_provider_from_model_path(model_path: str) -> str:
        """Determine API provider from model path."""
        if not model_path:
            raise ConfigurationError("model_path is required")

        # Simple heuristics based on model naming conventions
        if "gpt-" in model_path or "openai" in model_path.lower():
            return "openai"
        elif "gemini" in model_path.lower() or "google" in model_path.lower():
            return "google"
        elif "claude" in model_path.lower() or "anthropic" in model_path.lower():
            return "anthropic"
        elif "/" in model_path:  # OpenRouter format: provider/model
            return "openrouter"
        else:
            raise ConfigurationError(f"Cannot determine provider from model_path: {model_path}")

    @staticmethod
    def _get_api_key_for_provider(provider_name: str) -> str:
        """Get API key for the specified provider."""
        import os

        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        env_var = key_mapping.get(provider_name)
        if not env_var:
            raise ConfigurationError(
                f"No API key environment variable defined for provider: {provider_name}"
            )

        api_key = os.getenv(env_var)
        if not api_key:
            raise ConfigurationError(f"API key not found in environment variable: {env_var}")

        return api_key

    def __enter__(self) -> Self:
        """Context manager entry - create and validate agent."""
        logger.info(f"PydanticAI Web API アノテーター '{self.deps.model_name}' のコンテキストに入ります...")

        try:
            # Create agent (implemented by subclasses)
            self.agent = self._create_agent()

            if self.agent is None:
                raise ConfigurationError("Agent creation failed")

            logger.info(f"PydanticAI Agent 準備完了 ({self.deps.provider_name}, {self.deps.api_model_id})")

        except Exception as e:
            logger.error(f"PydanticAI Agent の準備中にエラーが発生: {e}")
            self.agent = None
            raise ConfigurationError(f"Agent 準備中のエラー: {e}") from e

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Context manager exit - cleanup agent resources."""
        if self.agent:
            logger.debug(f"PydanticAI Agent のリソースを解放 ({self.deps.provider_name})")
            self.agent = None

    @abstractmethod
    def _create_agent(self) -> Agent:
        """Create PydanticAI Agent for this annotator.

        Subclasses must implement this to create their specific Agent
        with appropriate model, tools, and system prompts.
        """
        raise NotImplementedError("Subclasses must implement _create_agent")

    def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
        """Convert PIL Images to bytes for PydanticAI Agent.

        Args:
            images: List of PIL Images to process

        Returns:
            List of image data as bytes (WEBP format)
        """
        processed_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="WEBP")
            processed_images.append(buffered.getvalue())
        return processed_images

    async def _run_inference(self, processed: list[bytes]) -> list[RawOutput]:
        """Run inference using PydanticAI Agent.

        Args:
            processed: List of processed image data (bytes)

        Returns:
            List of RawOutput containing AnnotationSchema or error information
        """
        if self.agent is None:
            raise ConfigurationError("Agent is not initialized. Use context manager.")

        results: list[RawOutput] = []

        for image_data in processed:
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Create multimodal prompt with image and text
                prompt_parts = [
                    BinaryContent(data=image_data, media_type="image/webp"),
                    self.deps.prompt_template,
                ]

                # Run PydanticAI Agent
                response = await self.agent.run(prompt_parts, deps=self.deps)

                # Convert response to AnnotationSchema
                if hasattr(response, "data"):
                    annotation = AnnotationSchema.model_validate(response.data)
                else:
                    # Fallback for different response formats
                    annotation = AnnotationSchema.model_validate(response)

                results.append(RawOutput(response=annotation, error=None))

            except Exception as e:
                error_message = self._handle_api_error(e)
                results.append(RawOutput(response=None, error=error_message))

        return results

    def _wait_for_rate_limit(self) -> None:
        """Rate limiting implementation (preserved from original)."""
        elapsed_time = time.time() - self.last_request_time
        wait_time = self.deps.min_request_interval - elapsed_time
        if wait_time > 0:
            logger.debug(f"レート制限のため {wait_time:.2f} 秒待機します。")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception) -> str:
        """Handle API errors and map to appropriate custom exceptions.

        Preserves existing comprehensive error handling logic.
        """
        error_message = str(e)
        logger.error(f"API エラーが発生しました: {error_message}")

        provider_name = self.deps.provider_name

        # HTTP status code based error handling
        if hasattr(e, "status_code"):
            status_code = getattr(e, "status_code", 0)
            if status_code == 401:
                raise ApiAuthenticationError(provider_name=provider_name) from e
            elif status_code == 402:
                raise InsufficientCreditsError(provider_name=provider_name) from e
            elif status_code == 429:
                retry_after = getattr(e, "retry_after", 60)
                raise ApiRateLimitError(provider_name=provider_name, retry_after=retry_after) from e
            elif status_code == 400:
                raise ApiRequestError(error_message, provider_name=provider_name) from e
            elif 500 <= status_code < 600:
                raise ApiServerError(
                    error_message, provider_name=provider_name, status_code=status_code
                ) from e

        # Timeout error detection
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)) or "timeout" in error_message.lower():
            raise ApiTimeoutError(provider_name=provider_name) from e

        # Generic WebAPI error
        raise WebApiError(
            f"処理中に予期せぬエラーが発生しました: {error_message}", provider_name=provider_name
        ) from e

    @override
    def _format_predictions(self, raw_outputs: list[RawOutput]) -> list[WebApiFormattedOutput]:
        """Format RawOutput to WebApiFormattedOutput (preserved from original)."""
        formatted_outputs: list[WebApiFormattedOutput] = []
        for output in raw_outputs:
            error = output.get("error")
            response_val = output.get("response")

            if error:
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error))
                continue

            if isinstance(response_val, AnnotationSchema):
                # Convert AnnotationSchema to dict using model_dump()
                formatted_outputs.append(
                    WebApiFormattedOutput(annotation=response_val.model_dump(), error=None)
                )
            else:
                # Handle None or unexpected types
                error_message = (
                    f"Invalid response type: {type(response_val)}"
                    if response_val is not None
                    else "Response is None"
                )
                formatted_outputs.append(WebApiFormattedOutput(annotation=None, error=error_message))

        return formatted_outputs

    def _generate_tags(self, formatted_output: WebApiFormattedOutput) -> list[str]:
        """Generate tags from formatted output (preserved from original)."""
        if formatted_output.get("error") or not formatted_output.get("annotation"):
            return []

        annotation = formatted_output["annotation"]
        if isinstance(annotation, dict) and "tags" in annotation:
            tags = annotation["tags"]
            if isinstance(tags, list):
                return tags

        return []

    # Add sync wrapper for async _run_inference to maintain compatibility
    def _run_inference_sync(self, processed: list[bytes]) -> list[RawOutput]:
        """Synchronous wrapper for async _run_inference to maintain BaseAnnotator compatibility."""
        return asyncio.run(self._run_inference(processed))

    # Override predict to handle async nature
    def predict(self, images: list[Image.Image], phash_list: list[str] | None = None) -> list[Any]:
        """Override predict to handle async inference properly."""
        if not images:
            logger.warning("空の画像リストが渡されました。アノテーションをスキップします。")
            return []

        try:
            # Preprocess images
            processed = self._preprocess_images(images)

            # Run async inference
            raw_outputs = self._run_inference_sync(processed)

            # Format predictions
            formatted_outputs = self._format_predictions(raw_outputs)

            # Generate results compatible with BaseAnnotator.predict
            results = []
            for i, (image, formatted_output) in enumerate(zip(images, formatted_outputs, strict=True)):
                try:
                    phash = (
                        phash_list[i]
                        if phash_list and i < len(phash_list)
                        else self._calculate_phash(image)
                    )
                    tags = self._generate_tags(formatted_output)

                    result = {
                        "phash": phash,
                        "tags": tags,
                        "formatted_output": formatted_output,
                        "error": formatted_output.get("error"),
                    }
                    results.append(result)

                except Exception as e:
                    logger.exception(f"画像 {i} の処理中にエラー: {e}")
                    err_result = {
                        "phash": phash_list[i] if phash_list and i < len(phash_list) else None,
                        "tags": [],
                        "formatted_output": None,
                        "error": f"タグ生成エラー: {e}",
                    }
                    results.append(err_result)

        except Exception as e:
            logger.exception(f"予期せぬエラー: {e}")
            # Return error results for all images
            results = []
            for i in range(len(images)):
                err_result = {
                    "phash": phash_list[i] if phash_list and i < len(phash_list) else None,
                    "tags": [],
                    "formatted_output": None,
                    "error": f"予期せぬエラー: {e}",
                }
                results.append(err_result)

        return results
