"""Simplified wrapper for PydanticAI Agents to integrate with existing API."""

from typing import Any, Dict, List
import asyncio
from io import BytesIO

from PIL import Image
from pydantic_ai.messages import BinaryContent

from .base.annotator import BaseAnnotator
from .simplified_agent_factory import get_agent_factory
from .types import AnnotationResult
from .utils import logger


class SimplifiedAgentWrapper(BaseAnnotator):
    """Wrapper to integrate simplified PydanticAI Agents with existing BaseAnnotator API."""
    
    def __init__(self, model_id: str):
        """
        Initialize the wrapper with a model ID.
        
        Args:
            model_id: The model ID (e.g., "google/gemini-2.5-pro-preview-03-25")
        """
        super().__init__()
        self.model_id = model_id
        self._agent = None
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
    
    def _generate_tags(self, image: Image.Image) -> List[str]:
        """
        Generate tags for an image using the PydanticAI Agent.
        
        Args:
            image: PIL Image to annotate
            
        Returns:
            List of generated tags
        """
        if not self._agent:
            raise RuntimeError(f"Agent not initialized for model {self.model_id}")
        
        try:
            # Convert PIL Image to BinaryContent for PydanticAI
            binary_content = self._pil_to_binary_content(image)
            
            # Run inference
            result = self._run_agent_inference(binary_content)
            
            # Extract tags from AnnotationSchema result
            if hasattr(result, 'tags') and result.tags:
                return result.tags
            else:
                logger.warning(f"No tags returned from {self.model_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating tags with {self.model_id}: {e}")
            return []
    
    def _pil_to_binary_content(self, image: Image.Image) -> BinaryContent:
        """Convert PIL Image to PydanticAI BinaryContent."""
        # Convert to bytes
        byte_buffer = BytesIO()
        image.save(byte_buffer, format='PNG')
        image_bytes = byte_buffer.getvalue()
        
        # Create BinaryContent
        return BinaryContent(
            data=image_bytes,
            media_type='image/png'
        )
    
    def _run_agent_inference(self, binary_content: BinaryContent) -> Any:
        """Run PydanticAI Agent inference with proper async handling."""
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
        def run_with_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._agent.run([binary_content])
                )
            finally:
                new_loop.close()
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_with_new_loop)
            return future.result()
    
    def _format_output(self, image: Image.Image, tags: List[str]) -> Dict[str, Any]:
        """
        Format output for compatibility with existing API.
        
        Args:
            image: Original PIL Image
            tags: Generated tags
            
        Returns:
            Formatted output dictionary
        """
        return {
            "model_id": self.model_id,
            "tags": tags,
            "tag_count": len(tags),
            "method": "simplified_pydantic_ai"
        }
    
    def run_inference(self, image: Image.Image) -> AnnotationResult:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image to annotate
            
        Returns:
            AnnotationResult with tags and formatted output
        """
        try:
            tags = self._generate_tags(image)
            formatted_output = self._format_output(image, tags)
            
            return AnnotationResult(
                tags=tags,
                formatted_output=formatted_output,
                error=None
            )
        except Exception as e:
            error_msg = f"Inference failed for {self.model_id}: {e}"
            logger.error(error_msg)
            return AnnotationResult(
                tags=[],
                formatted_output={"error": error_msg},
                error=error_msg
            )