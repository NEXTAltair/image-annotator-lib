"""
OpenAI Agent Annotator - PydanticAI implementation

Concrete implementation of PydanticAI-based annotator for OpenAI models,
demonstrating how to replace existing OpenAI annotator classes with
PydanticAI Agent pattern while maintaining compatibility.
"""

from typing import override

from pydantic_ai import Agent
from pydantic_ai.models import OpenAIModel

# Import from main package
from image_annotator_lib.core.types import AnnotationSchema

# Local imports
from .dependencies import OpenAIDependencies
from .pydanticai_webapi_annotator import PydanticAIWebApiAnnotator


class OpenAIAgentAnnotator(PydanticAIWebApiAnnotator):
    """PydanticAI-based OpenAI annotator implementation.

    This class demonstrates how to create a concrete PydanticAI Agent
    for OpenAI models while maintaining the same external interface
    as the original OpenAI annotator classes.
    """

    def __init__(self, dependencies: OpenAIDependencies):
        """Initialize OpenAI annotator with dependencies."""
        super().__init__(dependencies)
        self.openai_deps = dependencies  # Type-specific reference

    @classmethod
    def from_model_name(cls, model_name: str) -> "OpenAIAgentAnnotator":
        """Create OpenAI annotator from model_name."""
        base_deps = cls._create_dependencies_from_config(model_name)

        # Convert to OpenAI-specific dependencies
        openai_deps = OpenAIDependencies(
            **base_deps.model_dump(),
            temperature=0.7,  # OpenAI default
            response_format=None,  # Will be set to structured output
            json_schema_supported=True,  # OpenAI supports JSON schema
        )

        return cls(openai_deps)

    @override
    def _create_agent(self) -> Agent:
        """Create PydanticAI Agent for OpenAI models."""
        # Create OpenAI model
        model = OpenAIModel(
            model_name=self.openai_deps.api_model_id,
            api_key=self.openai_deps.api_key,
        )

        # System prompt for image annotation
        system_prompt = """You are an expert image analysis AI. Analyze the provided image and generate:

1. Tags: A list of descriptive tags about the image content, objects, style, etc.
2. Captions: Descriptive captions explaining what you see in the image
3. Score: A quality/aesthetic score from 0.0 to 1.0

Provide structured output in the exact format specified by the response schema.
Be accurate, descriptive, and comprehensive in your analysis."""

        # Create Agent with structured output
        agent = Agent(
            model=model,
            deps_type=OpenAIDependencies,
            output_type=AnnotationSchema,  # Structured output
            system_prompt=system_prompt,
        )

        return agent


# Example usage and testing function
async def test_openai_agent_annotator():
    """Test function to demonstrate OpenAI Agent Annotator usage."""

    from PIL import Image

    # Create test dependencies
    deps = OpenAIDependencies(
        model_name="test-gpt-4o-mini",
        api_model_id="gpt-4o-mini",
        provider_name="openai",
        api_key="sk-test-key",  # Use actual key in real testing
        temperature=0.7,
    )

    # Create annotator
    annotator = OpenAIAgentAnnotator(deps)

    # Create test image (small red square)
    test_image = Image.new("RGB", (100, 100), color="red")

    # Test with context manager
    with annotator:
        results = annotator.predict([test_image])
        print("Annotation results:", results)

    return results


if __name__ == "__main__":
    import asyncio

    # Run test
    asyncio.run(test_openai_agent_annotator())
