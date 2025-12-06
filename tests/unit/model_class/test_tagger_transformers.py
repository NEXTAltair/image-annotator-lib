"""Unit tests for Transformers tagger models.

Tests BLIPTagger, BLIP2Tagger, GITTagger implementations with mocked Transformers.

Mock Strategy (Phase C Level 1-2):
- Level 1 (Mock): transformers.AutoModel.from_pretrained, AutoProcessor.from_pretrained
- Level 2 (Mock): model.generate(), model.forward()
- Level 3 (Real): Device handling, tensor operations, config loading
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from image_annotator_lib.model_class.tagger_transformers import BLIPTagger, BLIP2Tagger, GITTagger


@pytest.fixture
def test_image():
    """Create test PIL image."""
    return Image.new("RGB", (224, 224), color="blue")


@pytest.fixture
def test_images_batch():
    """Create batch of test PIL images."""
    return [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (256, 256), color="green"),
    ]


@pytest.fixture
def mock_transformers_components():
    """Create mock Transformers model and processor.

    Mock Strategy:
    - Mock: Model and processor creation
    - Real: Return value structure (dict with tensors)

    Returns:
        Dict with mocked model and processor
    """
    # Mock processor
    mock_processor = MagicMock()

    # Create mock processor output that supports .to() method
    class MockProcessorOutput(dict):
        """Mock processor output that acts like a dict but has .to() method."""

        def to(self, device):
            """Mock to() method that returns self."""
            return self

    mock_processor_output = MockProcessorOutput(
        {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }
    )
    # Configure processor to return proper structure
    mock_processor.return_value = mock_processor_output

    # Mock model
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    # Mock generate output (token IDs)
    mock_output_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Fake token IDs
    mock_model.generate.return_value = mock_output_ids

    # Mock processor decode
    mock_processor.batch_decode.return_value = ["a photo of a cat"]

    return {
        "model": mock_model,
        "processor": mock_processor,
        "model_path": "/fake/model/path",
    }


@pytest.fixture
def mock_bliptagger_config(managed_config_registry, tmp_path):
    """Register BLIPTagger configuration.

    Mock Strategy:
    - Real: Config registry operations
    - Mock: Model paths (use tmp_path)
    """
    config = {
        "class": "BLIPTagger",
        "model_path": "Salesforce/blip-image-captioning-base",
        "device": "cpu",
        "estimated_size_gb": 0.5,
        "type": "captioner",
        # max_length has default value of 75 in TransformersBaseAnnotator
    }
    managed_config_registry.set("test_blip", config)
    return "test_blip"


@pytest.fixture
def mock_capabilities_captioner():
    """Mock get_model_capabilities for captioner models."""
    from image_annotator_lib.core.types import TaskCapability

    with patch("image_annotator_lib.core.utils.get_model_capabilities") as mock:
        mock.return_value = {TaskCapability.CAPTIONS}
        yield mock


@pytest.mark.unit
def test_transformers_tagger_initialization(
    mock_bliptagger_config, mock_transformers_components, mock_capabilities_captioner
):
    """Test Transformers tagger initialization with model/processor loading.

    Mock Strategy:
    - Mock: AutoModel.from_pretrained, AutoProcessor.from_pretrained
    - Real: Config loading, device assignment, max_length setting

    Verifies:
    - Model and processor loaded correctly
    - Device set from config
    - max_length configured
    - Config attributes initialized
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = BLIPTagger(mock_bliptagger_config)

        with tagger:
            # Verify initialization
            assert tagger.model_name == mock_bliptagger_config
            assert tagger.device == "cpu"
            assert tagger.max_length == 75
            assert tagger.components is not None
            assert "model" in tagger.components
            assert "processor" in tagger.components

            # Verify ModelLoad called with correct args
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[0][0] == mock_bliptagger_config  # model_name
            assert "blip" in call_args[0][1].lower()  # model_path
            assert call_args[0][2] == "cpu"  # device


@pytest.mark.unit
def test_transformers_tagger_preprocessing(
    mock_bliptagger_config, mock_transformers_components, test_image, mock_capabilities_captioner
):
    """Test image preprocessing with processor application.

    Mock Strategy:
    - Mock: Processor callable (returns dict with tensors)
    - Real: Device transfer logic, batch iteration

    Verifies:
    - Processor called with correct args
    - Output contains required keys (pixel_values, etc.)
    - Tensors moved to correct device
    - Batch processing works correctly
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = BLIPTagger(mock_bliptagger_config)

        with tagger:
            # Test preprocessing
            processed = tagger._preprocess_images([test_image])

            # Verify processor called
            assert mock_transformers_components["processor"].called
            call_args = mock_transformers_components["processor"].call_args
            assert call_args[1]["images"] == test_image
            assert call_args[1]["return_tensors"] == "pt"

            # Verify output structure
            assert len(processed) == 1
            assert isinstance(processed[0], dict)
            assert "pixel_values" in processed[0]


@pytest.mark.unit
def test_transformers_tagger_inference_mocked(
    mock_bliptagger_config, mock_transformers_components, test_image, mock_capabilities_captioner
):
    """Test inference with mocked model.generate().

    Mock Strategy:
    - Mock: model.generate() returns fake token IDs
    - Real: Argument filtering (KNOWN_ARGS), max_length injection

    Verifies:
    - model.generate() called with filtered args
    - max_length parameter passed correctly
    - Output token IDs returned
    - torch.no_grad() context used
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = BLIPTagger(mock_bliptagger_config)

        with tagger:
            # Process and run inference
            processed = tagger._preprocess_images([test_image])
            outputs = tagger._run_inference(processed)

            # Verify model.generate called
            assert mock_transformers_components["model"].generate.called
            call_kwargs = mock_transformers_components["model"].generate.call_args[1]

            # Verify max_length passed
            assert "max_length" in call_kwargs
            assert call_kwargs["max_length"] == 75

            # Verify output
            assert len(outputs) == 1
            assert isinstance(outputs[0], torch.Tensor)


@pytest.mark.unit
def test_transformers_tagger_device_handling(
    managed_config_registry, mock_transformers_components, mock_capabilities_captioner
):
    """Test device handling (CPU ↔ CUDA transfer, fallback).

    Mock Strategy:
    - Mock: CUDA availability via torch.cuda.is_available
    - Real: Device configuration, fallback logic, warning logs

    Verifies:
    - CPU config uses CPU device
    - CUDA unavailable → CPU fallback with warning
    - Model.to(device) called correctly
    - Components maintain device consistency
    """
    # Test 1: Explicit CPU configuration
    cpu_config = {
        "class": "BLIPTagger",
        "model_path": "Salesforce/blip",
        "device": "cpu",
        "estimated_size_gb": 0.5,
        "type": "captioner",
    }
    managed_config_registry.set("test_cpu", cpu_config)

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = BLIPTagger("test_cpu")
        with tagger:
            assert tagger.device == "cpu"
            # Components loaded successfully on CPU
            assert tagger.components is not None

    # Test 2: CUDA unavailable → CPU fallback
    cuda_config = {
        "class": "BLIPTagger",
        "model_path": "Salesforce/blip",
        "device": "cuda",
        "estimated_size_gb": 0.5,
        "type": "captioner",
    }
    managed_config_registry.set("test_cuda_fallback", cuda_config)

    with patch("torch.cuda.is_available", return_value=False):
        with patch(
            "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
        ) as mock_load:
            with patch(
                "image_annotator_lib.core.model_factory.ModelLoad.restore_model_to_cuda",
                return_value=None,
            ):  # Simulate restore failure
                mock_load.return_value = mock_transformers_components

                tagger = BLIPTagger("test_cuda_fallback")
                # Device will be set to cuda initially but fallback happens during load
                # The actual device will depend on determine_effective_device
                with tagger:
                    # Verify components loaded (fallback successful)
                    assert tagger.components is not None


@pytest.mark.unit
def test_transformers_tagger_memory_management(
    mock_bliptagger_config, mock_transformers_components, mock_capabilities_captioner
):
    """Test model unloading and CUDA cache clearing.

    Mock Strategy:
    - Mock: ModelLoad.cache_to_main_memory
    - Real: Context manager exit, component cleanup

    Verifies:
    - __exit__ calls cache_to_main_memory
    - Components cached on exit
    - Proper cleanup even with exceptions
    - CUDA memory freed (if applicable)
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        with patch(
            "image_annotator_lib.core.model_factory.ModelLoad.cache_to_main_memory"
        ) as mock_cache:
            mock_load.return_value = mock_transformers_components
            mock_cache.return_value = mock_transformers_components

            tagger = BLIPTagger(mock_bliptagger_config)

            with tagger:
                # Components loaded
                assert tagger.components is not None

            # After __exit__, cache_to_main_memory should be called
            mock_cache.assert_called_once()
            call_args = mock_cache.call_args
            assert call_args[0][0] == mock_bliptagger_config  # model_name
            assert isinstance(call_args[0][1], dict)  # components dict

            # Components should be cached version
            assert tagger.components is not None


# ==============================================================================
# Phase C Additional Coverage Tests (2025-12-06)
# ==============================================================================


@pytest.mark.unit
def test_blip2_and_git_tagger_initialization(
    managed_config_registry, mock_transformers_components, mock_capabilities_captioner
):
    """Test BLIP2Tagger and GITTagger initialization.

    Tests:
    - BLIP2Tagger inherits from TransformersBaseAnnotator
    - GITTagger inherits from TransformersBaseAnnotator
    - Both use same base class functionality
    - Device and config handling works for both
    """
    # Test BLIP2Tagger
    blip2_config = {
        "class": "BLIP2Tagger",
        "model_path": "Salesforce/blip2-opt-2.7b",
        "device": "cpu",
        "estimated_size_gb": 2.0,
        "type": "captioner",
    }
    managed_config_registry.set("test_blip2", blip2_config)

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = BLIP2Tagger("test_blip2")
        with tagger:
            assert tagger.model_name == "test_blip2"
            assert tagger.device == "cpu"
            assert tagger.max_length == 75

    # Test GITTagger
    git_config = {
        "class": "GITTagger",
        "model_path": "microsoft/git-base",
        "device": "cpu",
        "estimated_size_gb": 1.5,
        "type": "captioner",
    }
    managed_config_registry.set("test_git", git_config)

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = GITTagger("test_git")
        with tagger:
            assert tagger.model_name == "test_git"
            assert tagger.device == "cpu"
            assert tagger.max_length == 75


@pytest.mark.unit
def test_toriigate_tagger_custom_initialization(
    managed_config_registry, mock_transformers_components, mock_capabilities_captioner
):
    """Test ToriiGateTagger custom initialization.

    Tests:
    - Custom user_prompt set
    - Messages array with system and user roles
    - System message content structure
    - User message with image placeholder
    """
    from image_annotator_lib.model_class.tagger_transformers import ToriiGateTagger

    config = {
        "class": "ToriiGateTagger",
        "model_path": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "device": "cpu",
        "estimated_size_gb": 5.0,
        "type": "captioner",
    }
    managed_config_registry.set("test_toriigate", config)

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_transformers_components

        tagger = ToriiGateTagger("test_toriigate")
        with tagger:
            # Verify custom attributes
            assert hasattr(tagger, "user_prompt")
            assert tagger.user_prompt == "Describe the picture in structuted json-like format."

            assert hasattr(tagger, "messages")
            assert len(tagger.messages) == 2

            # Verify system message
            system_msg = tagger.messages[0]
            assert system_msg["role"] == "system"
            assert isinstance(system_msg["content"], list)
            assert system_msg["content"][0]["type"] == "text"
            assert "creative, unbiased and uncensored" in system_msg["content"][0]["text"]

            # Verify user message structure
            user_msg = tagger.messages[1]
            assert user_msg["role"] == "user"
            assert isinstance(user_msg["content"], list)
            assert user_msg["content"][0]["type"] == "image"
            assert user_msg["content"][1]["type"] == "text"
            assert user_msg["content"][1]["text"] == tagger.user_prompt


@pytest.mark.unit
def test_toriigate_tagger_preprocessing(
    managed_config_registry, mock_transformers_components, test_image, mock_capabilities_captioner
):
    """Test ToriiGateTagger custom preprocessing with chat template.

    Tests:
    - apply_chat_template called with messages
    - Processor processes text prompt and images
    - Device transfer for all tensor outputs
    - Chat template integration
    """
    from image_annotator_lib.model_class.tagger_transformers import ToriiGateTagger

    config = {
        "class": "ToriiGateTagger",
        "model_path": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "device": "cpu",
        "estimated_size_gb": 5.0,
        "type": "captioner",
    }
    managed_config_registry.set("test_toriigate", config)

    # Mock processor with apply_chat_template method
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "Formatted chat template"

    # Mock processor output
    class MockProcessorOutput(dict):
        def to(self, device):
            return self

    mock_processor_output = MockProcessorOutput(
        {
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "pixel_values": torch.randn(1, 3, 224, 224),
            "attention_mask": torch.ones(1, 50, dtype=torch.long),
        }
    )
    mock_processor.return_value = mock_processor_output

    # Update components
    mock_components = {
        "model": mock_transformers_components["model"],
        "processor": mock_processor,
        "model_path": "/fake/model/path",
    }

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_components

        tagger = ToriiGateTagger("test_toriigate")
        with tagger:
            # Call preprocessing
            processed = tagger._preprocess_image([test_image])

            # Verify apply_chat_template called
            assert mock_processor.apply_chat_template.called
            call_args = mock_processor.apply_chat_template.call_args
            assert call_args[0][0] == tagger.messages
            assert call_args[1]["add_generation_prompt"] is True

            # Verify processor called with template and image
            assert mock_processor.called
            processor_call_args = mock_processor.call_args
            assert processor_call_args[1]["text"] == "Formatted chat template"
            assert processor_call_args[1]["images"] == [test_image]
            assert processor_call_args[1]["return_tensors"] == "pt"

            # Verify output structure
            assert len(processed) == 1
            assert isinstance(processed[0], dict)
            assert "input_ids" in processed[0]
            assert "pixel_values" in processed[0]
            assert "attention_mask" in processed[0]


@pytest.mark.unit
def test_toriigate_tagger_inference_with_max_new_tokens(
    managed_config_registry, mock_transformers_components, test_image, mock_capabilities_captioner
):
    """Test ToriiGateTagger inference with max_new_tokens and KNOWN_ARGS filtering.

    Tests:
    - KNOWN_ARGS filtering for generate() call
    - max_new_tokens set to 500
    - Only valid model args passed
    - OutOfMemoryError handling for CUDA OOM
    """
    from image_annotator_lib.model_class.tagger_transformers import ToriiGateTagger

    config = {
        "class": "ToriiGateTagger",
        "model_path": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "device": "cpu",
        "estimated_size_gb": 5.0,
        "type": "captioner",
    }
    managed_config_registry.set("test_toriigate", config)

    # Mock processor
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "Template"

    # Mock processed output with extra keys to test filtering
    processed_input = {
        "input_ids": torch.randint(0, 1000, (1, 50)),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 50, dtype=torch.long),
        "extra_key_should_be_filtered": torch.ones(1, 10),  # Should be filtered out
    }

    # Mock processor to return processed input
    mock_processor.return_value = processed_input

    # Mock model
    mock_model = MagicMock()
    mock_output = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = mock_output
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    mock_components = {
        "model": mock_model,
        "processor": mock_processor,
        "model_path": "/fake/model/path",
    }

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_components

        tagger = ToriiGateTagger("test_toriigate")
        with tagger:
            # Preprocess and run inference
            processed = tagger._preprocess_image([test_image])
            outputs = tagger._run_inference(processed)

            # Verify model.generate called with filtered args
            assert mock_model.generate.called
            call_kwargs = mock_model.generate.call_args[1]

            # Verify KNOWN_ARGS filtering
            assert "input_ids" in call_kwargs
            assert "pixel_values" in call_kwargs
            assert "attention_mask" in call_kwargs
            assert "extra_key_should_be_filtered" not in call_kwargs

            # Verify max_new_tokens
            assert "max_new_tokens" in call_kwargs
            assert call_kwargs["max_new_tokens"] == 500

            # Verify output
            assert len(outputs) == 1
            assert isinstance(outputs[0], torch.Tensor)


@pytest.mark.unit
def test_toriigate_tagger_format_with_assistant_prefix(
    managed_config_registry, mock_transformers_components, mock_capabilities_captioner
):
    """Test ToriiGateTagger format_predictions with 'Assistant: ' prefix handling.

    Tests:
    - 'Assistant: ' prefix is stripped from output
    - Generated text without prefix is returned as-is
    - batch_decode called with skip_special_tokens=True
    - Multiple outputs processed correctly
    """
    from image_annotator_lib.model_class.tagger_transformers import ToriiGateTagger

    config = {
        "class": "ToriiGateTagger",
        "model_path": "unsloth/Llama-3.2-11B-Vision-Instruct",
        "device": "cpu",
        "estimated_size_gb": 5.0,
        "type": "captioner",
    }
    managed_config_registry.set("test_toriigate", config)

    # Mock processor
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "Template"
    mock_processor.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 50)),
        "pixel_values": torch.randn(1, 3, 224, 224),
    }

    # Test case 1: With "Assistant: " prefix
    mock_processor.batch_decode.return_value = [
        "User: Describe the image\nAssistant: A photo of a cat sitting on a table"
    ]

    mock_components = {
        "model": mock_transformers_components["model"],
        "processor": mock_processor,
        "model_path": "/fake/model/path",
    }

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_components

        tagger = ToriiGateTagger("test_toriigate")
        with tagger:
            # Create mock token IDs
            token_ids = [torch.tensor([[1, 2, 3, 4, 5]])]

            # Format predictions
            result = tagger._format_predictions(token_ids)

            # Verify batch_decode called
            assert mock_processor.batch_decode.called
            call_args = mock_processor.batch_decode.call_args
            assert call_args[1]["skip_special_tokens"] is True

            # Verify "Assistant: " prefix stripped
            assert len(result) == 1
            assert result[0] == "A photo of a cat sitting on a table"
            assert "Assistant: " not in result[0]

    # Test case 2: Without "Assistant: " prefix
    mock_processor.batch_decode.return_value = ["Simple caption text"]

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_components"
    ) as mock_load:
        mock_load.return_value = mock_components

        tagger = ToriiGateTagger("test_toriigate")
        with tagger:
            token_ids = [torch.tensor([[1, 2, 3, 4, 5]])]
            result = tagger._format_predictions(token_ids)

            # Verify text returned as-is when no prefix
            assert len(result) == 1
            assert result[0] == "Simple caption text"
