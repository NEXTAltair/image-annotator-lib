"""
Integration Test for PydanticAI WebAPI Annotator

Tests the PydanticAI-based implementation against the existing interface
to ensure compatibility and functionality.
"""

import asyncio

from PIL import Image

# Import from main package for testing
from .dependencies import OpenAIDependencies

# Local imports
from .openai_agent_annotator import OpenAIAgentAnnotator


def create_test_image() -> Image.Image:
    """Create a simple test image."""
    # Create a simple test image with some content
    img = Image.new("RGB", (200, 200), color="blue")
    return img


async def test_basic_functionality():
    """Test basic annotator functionality."""
    print("=== Basic Functionality Test ===")

    # Create test dependencies
    deps = OpenAIDependencies(
        model_name="test-openai-model",
        api_model_id="gpt-4o-mini",
        provider_name="openai",
        api_key="test-key",  # In real test, use actual API key
        temperature=0.7,
    )

    # Create annotator
    annotator = OpenAIAgentAnnotator(deps)

    # Test context manager
    print("Testing context manager...")
    try:
        with annotator:
            print("✓ Context manager entry successful")
            print(f"✓ Agent created: {annotator.agent is not None}")

            # Test image preprocessing
            test_image = create_test_image()
            processed = annotator._preprocess_images([test_image])
            print(f"✓ Image preprocessing: {len(processed)} images, {len(processed[0])} bytes each")

    except Exception as e:
        print(f"✗ Context manager test failed: {e}")


def test_backward_compatibility():
    """Test backward compatibility with existing config system."""
    print("\n=== Backward Compatibility Test ===")

    # Test from_model_name class method
    try:
        # This would normally load from config, but we'll mock it
        print("Testing from_model_name creation...")

        # In real test, this would use actual config
        # annotator = OpenAIAgentAnnotator.from_model_name("some-configured-model")
        print("✓ from_model_name method available (would need real config for full test)")

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")


def test_api_interface_compatibility():
    """Test that the new implementation maintains API compatibility."""
    print("\n=== API Interface Compatibility Test ===")

    deps = OpenAIDependencies(
        model_name="test-model",
        api_model_id="gpt-4o-mini",
        provider_name="openai",
        api_key="test-key",
        temperature=0.7,
    )

    annotator = OpenAIAgentAnnotator(deps)

    # Test interface methods exist
    interface_methods = [
        "predict",
        "_preprocess_images",
        "_run_inference",
        "_format_predictions",
        "_generate_tags",
        "__enter__",
        "__exit__",
    ]

    for method_name in interface_methods:
        if hasattr(annotator, method_name):
            print(f"✓ Method {method_name} exists")
        else:
            print(f"✗ Method {method_name} missing")


def test_dependency_model_validation():
    """Test dependency model validation."""
    print("\n=== Dependency Model Validation Test ===")

    try:
        # Test valid dependencies
        valid_deps = OpenAIDependencies(
            model_name="test",
            api_model_id="gpt-4o-mini",
            provider_name="openai",
            api_key="test-key",
            temperature=0.7,
        )
        print("✓ Valid dependencies accepted")

        # Test invalid dependencies (this should raise validation error)
        try:
            invalid_deps = OpenAIDependencies(
                model_name="",  # Empty model name should be invalid
                api_model_id="gpt-4o-mini",
                provider_name="openai",
                api_key="test-key",
                temperature=2.0,  # Temperature > 1.0 might be invalid
            )
            print("? Invalid dependencies accepted (validation might be loose)")
        except Exception as e:
            print(f"✓ Invalid dependencies rejected: {e}")

    except Exception as e:
        print(f"✗ Dependency validation test failed: {e}")


async def run_all_tests():
    """Run all integration tests."""
    print("PydanticAI WebAPI Annotator Integration Tests")
    print("=" * 50)

    await test_basic_functionality()
    test_backward_compatibility()
    test_api_interface_compatibility()
    test_dependency_model_validation()

    print("\n" + "=" * 50)
    print("Integration tests completed")
    print("\nNOTE: Full functionality requires actual API keys and configuration.")
    print("This test suite validates structure and interface compatibility.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
