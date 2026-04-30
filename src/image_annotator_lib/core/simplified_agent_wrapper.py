"""Simplified wrapper for PydanticAI Agents to integrate with existing API."""

import asyncio
from io import BytesIO
from typing import Any, Self

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from .base.annotator import BaseAnnotator
from .simplified_agent_factory import get_agent_factory
from .types import TaskCapability, UnifiedAnnotationResult
from .utils import get_model_capabilities, logger


class SimplifiedAgentWrapper(BaseAnnotator):
    """Wrapper to integrate simplified PydanticAI Agents with existing BaseAnnotator API."""

    def __init__(self, model_id: str):
        """
        Initialize the wrapper with a model ID.

        Args:
            model_id: The model ID (e.g., "google/gemini-2.5-pro-preview-03-25")
        """
        # Initialize BaseAnnotator with model_id as model_name
        super().__init__(model_name=model_id)
        self.model_id = model_id
        self._agent: Agent | None = None
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Setup the PydanticAI Agent using the simplified factory."""
        try:
            agent_factory = get_agent_factory()
            self._agent = agent_factory.get_cached_agent(self.model_id)
            logger.info(f"SimplifiedAgentWrapper initialized for model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to setup agent for {self.model_id}: {e}")
            raise

    def __enter__(self) -> Self:
        """Context manager entry - agent is already initialized in __init__."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Context manager exit - cleanup if needed."""
        # PydanticAI agents don't require explicit cleanup
        pass

    def _preprocess_images(self, images: list[Image.Image]) -> list[BinaryContent]:
        """
        Preprocess images by converting to PydanticAI BinaryContent format.

        Args:
            images: List of PIL Images to preprocess

        Returns:
            List of BinaryContent objects for PydanticAI
        """
        return [self._pil_to_binary_content(img) for img in images]

    def _run_inference(self, processed: list[BinaryContent]) -> list[Any]:
        """
        Run inference on preprocessed data.

        Args:
            processed: List of BinaryContent from _preprocess_images

        Returns:
            List of raw inference results
        """
        if not self._agent:
            raise RuntimeError(f"Agent not initialized for model {self.model_id}")

        results = []
        for binary_content in processed:
            result = self._run_agent_inference(binary_content)
            results.append(result)
        return results

    def _format_predictions(self, raw_outputs: list[Any]) -> list[UnifiedAnnotationResult]:
        """
        Format raw inference results into UnifiedAnnotationResult.

        Args:
            raw_outputs: List of raw results from _run_inference

        Returns:
            List of UnifiedAnnotationResult instances
        """
        capabilities = get_model_capabilities(self.model_name)
        if not capabilities:
            # PydanticAI Agents は config に type が無いことがあるため TAGS をデフォルトとする
            capabilities = {TaskCapability.TAGS}

        formatted: list[UnifiedAnnotationResult] = []
        for output in raw_outputs:
            tags = list(output.tags) if hasattr(output, "tags") and output.tags else []

            formatted.append(
                UnifiedAnnotationResult(
                    model_name=self.model_id,
                    capabilities=capabilities,
                    tags=tags if (TaskCapability.TAGS in capabilities and tags) else None,
                    framework="pydantic_ai",
                    raw_output={"method": "simplified_pydantic_ai", "tag_count": len(tags)},
                )
            )
        return formatted

    def _pil_to_binary_content(self, image: Image.Image) -> BinaryContent:
        """Convert PIL Image to PydanticAI BinaryContent."""
        # Convert to bytes
        byte_buffer = BytesIO()
        image.save(byte_buffer, format="PNG")
        image_bytes = byte_buffer.getvalue()

        # Create BinaryContent
        return BinaryContent(data=image_bytes, media_type="image/png")

    def _run_agent_inference(self, binary_content: BinaryContent) -> Any:
        """Run PydanticAI Agent inference with proper async handling."""
        if not self._agent:
            raise RuntimeError(f"Agent not initialized for model {self.model_id}")

        try:
            # Try sync first
            result = self._agent.run_sync([binary_content])
            logger.debug(f"Agent {self.model_id} inference completed successfully")
            return result
        except RuntimeError as e:
            if "Event loop" in str(e) or "asyncio" in str(e):
                # Handle event loop issues
                logger.debug(f"Falling back to async execution for {self.model_id}")
                return self._run_async_with_new_loop(binary_content)
            else:
                raise

    def _run_async_with_new_loop(self, binary_content: BinaryContent) -> Any:
        """Run async inference with a new event loop."""
        if not self._agent:
            raise RuntimeError(f"Agent not initialized for model {self.model_id}")

        def run_with_new_loop() -> Any:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                if not self._agent:
                    raise RuntimeError(f"Agent not initialized for model {self.model_id}")
                return new_loop.run_until_complete(self._agent.run([binary_content]))
            finally:
                new_loop.close()

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_with_new_loop)
            return future.result()

