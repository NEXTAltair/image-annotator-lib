#!/usr/bin/env python3
"""Manual test for simplified PydanticAI configuration."""

import os
import sys

from PIL import Image

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_annotator_lib import create_agent, get_available_models
from image_annotator_lib.core.simple_config import get_default_settings, get_model_settings


def test_simplified_config():
    """Test the simplified configuration system."""
    print("=== Testing Simplified Configuration ===")

    # Test default settings
    print("\n1. Testing default settings:")
    defaults = get_default_settings()
    print(f"Default settings: {defaults}")

    # Test model-specific settings
    print("\n2. Testing model-specific settings:")
    test_models = [
        "google/gemini-2.5-pro-preview-03-25",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "nonexistent-model",
    ]

    for model_id in test_models:
        settings = get_model_settings(model_id)
        print(f"Settings for {model_id}: {settings}")


def test_available_models():
    """Test model discovery functionality."""
    print("\n=== Testing Model Discovery ===")

    # Get available models
    try:
        available_models = get_available_models()
        print(f"Available models count: {len(available_models)}")
        print("Sample models:")
        for model in available_models[:5]:
            print(f"  - {model}")
    except Exception as e:
        print(f"Error getting available models: {e}")


def test_agent_creation():
    """Test simplified agent creation."""
    print("\n=== Testing Agent Creation ===")

    # Test with a sample model (if available)
    try:
        available_models = get_available_models()
        if available_models:
            test_model = available_models[0]
            print(f"Testing with model: {test_model}")

            # Create agent
            agent = create_agent(test_model)
            print(f"Agent created successfully: {type(agent)}")
        else:
            print("No models available for testing")
    except Exception as e:
        print(f"Error creating agent: {e}")


def test_simplified_wrapper():
    """Test the simplified agent wrapper."""
    print("\n=== Testing Simplified Wrapper ===")

    try:
        # Create a simple test image
        Image.new("RGB", (100, 100), color="red")

        # Test with API discovery
        from image_annotator_lib.core.simplified_agent_factory import get_agent_factory

        factory = get_agent_factory()

        # Try to refresh models
        models = factory.refresh_available_models(force_refresh=False)
        print(f"Factory found {len(models)} models")

        if models:
            test_model = models[0]
            print(f"Testing wrapper with: {test_model}")

            # Test wrapper creation
            from image_annotator_lib.core.simplified_agent_wrapper import SimplifiedAgentWrapper

            wrapper = SimplifiedAgentWrapper(test_model)
            print(f"Wrapper created: {type(wrapper)}")

            # Note: Don't run actual inference without proper API keys
            print("Wrapper initialization successful (inference skipped)")
        else:
            print("No models available for wrapper testing")

    except Exception as e:
        print(f"Error testing wrapper: {e}")


def main():
    """Run all tests."""
    print("Starting Simplified PydanticAI Configuration Tests")
    print("=" * 50)

    try:
        test_simplified_config()
        test_available_models()
        test_agent_creation()
        test_simplified_wrapper()

        print("\n" + "=" * 50)
        print("All tests completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
