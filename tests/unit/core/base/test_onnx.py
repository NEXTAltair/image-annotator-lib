"""Unit tests for core/base/onnx.py ONNXBaseAnnotator class.

このモジュールではONNXBaseAnnotatorの各機能を段階的にテストします。
- Context manager lifecycle
- Tag extraction
- Image preprocessing
- Model inference
- Result formatting
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from image_annotator_lib.core.base.onnx import ONNXBaseAnnotator

# ==============================================================================
# Test Helpers
# ==============================================================================


class ConcreteONNXAnnotator(ONNXBaseAnnotator):
    """Concrete implementation of ONNXBaseAnnotator for testing."""

    def __init__(self, model_name: str, tags: list[str] | None = None):
        super().__init__(model_name)
        self._test_tags = tags or []

    def _load_tags(self) -> None:
        """Implementation of abstract method for testing."""
        self.all_tags = self._test_tags


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX InferenceSession."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input", shape=[1, 3, 448, 448])]
    session.get_outputs.return_value = [MagicMock(name="output")]
    return session


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def sample_tags():
    """Sample tags data."""
    return ["tag1", "tag2", "tag3", "category:tag4"]


# ==============================================================================
# Category 1: Initialization and Context Manager Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_onnx_annotator_init(sample_tags):
    """Test ONNXBaseAnnotator initialization."""
    annotator = ConcreteONNXAnnotator("test-model", tags=sample_tags)

    assert annotator.model_name == "test-model"
    assert annotator.components is None
    assert annotator.all_tags == []  # Not yet loaded


@pytest.mark.unit
@pytest.mark.fast
def test_onnx_annotator_context_manager_enter(mock_onnx_session, sample_tags):
    """Test __enter__ method loads resources correctly."""
    annotator = ConcreteONNXAnnotator("test-model", tags=sample_tags)

    # Set model_path on annotator before calling __enter__
    annotator.model_path = "/path/to/model.onnx"
    annotator.device = "cpu"

    # Mock input shape for _analyze_model_input_format
    mock_input = MagicMock()
    mock_input.shape = [1, 3, 448, 448]
    mock_onnx_session.get_inputs.return_value = [mock_input]

    with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
        mock_load.return_value = {"session": mock_onnx_session}

        result = annotator.__enter__()

        assert result is annotator
        assert annotator.components == {"session": mock_onnx_session}
        assert annotator.all_tags == sample_tags
        assert annotator.target_size == (448, 448)
        assert annotator.is_nchw_expected is True
        mock_load.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_onnx_annotator_context_manager_exit():
    """Test __exit__ method cleans up resources."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = {"session": MagicMock()}

    with patch("image_annotator_lib.core.model_factory.ModelLoad.release_model_components") as mock_release:
        mock_release.return_value = {}

        annotator.__exit__(None, None, None)

        # release_model_components returns empty dict, which is cast to ONNXComponents
        assert annotator.components == {}
        mock_release.assert_called_once()


# ==============================================================================
# Category 2: Tag Extraction Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_extract_category_tags():
    """Test _extract_category_tags extracts category-prefixed tags."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.all_tags = ["tag1", "tag2", "tag3"]
    annotator.character_indexes = [0, 2]  # Attribute name for category indexes

    tags_with_probs = [("tag1", 0.9), ("tag2", 0.5), ("tag3", 0.8)]

    result = annotator._extract_category_tags("character_indexes", tags_with_probs)

    assert isinstance(result, dict)
    assert "tag1" in result
    assert result["tag1"] == 0.9
    assert "tag3" in result
    assert result["tag3"] == 0.8
    assert "tag2" not in result


# ==============================================================================
# Category 3: Image Preprocessing Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_analyze_model_input_format_nchw():
    """Test _analyze_model_input_format detects NCHW format."""
    annotator = ConcreteONNXAnnotator("test-model")
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.shape = [1, 3, 448, 448]
    mock_session.get_inputs.return_value = [mock_input]
    annotator.components = {"session": mock_session}

    annotator._analyze_model_input_format()

    assert annotator.target_size == (448, 448)
    assert annotator.is_nchw_expected is True


@pytest.mark.unit
@pytest.mark.fast
def test_analyze_model_input_format_nhwc():
    """Test _analyze_model_input_format detects NHWC format."""
    annotator = ConcreteONNXAnnotator("test-model")
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.shape = [1, 448, 448, 3]
    mock_session.get_inputs.return_value = [mock_input]
    annotator.components = {"session": mock_session}

    annotator._analyze_model_input_format()

    assert annotator.target_size == (448, 448)
    assert annotator.is_nchw_expected is False


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_single_image(sample_image, mock_onnx_session):
    """Test _preprocess_images with single image."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = {"session": mock_onnx_session}
    annotator.target_size = (448, 448)
    annotator.is_nchw_expected = True

    result = annotator._preprocess_images([sample_image])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (1, 3, 448, 448)
    assert result[0].dtype == np.float32


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_multiple_images(mock_onnx_session):
    """Test _preprocess_images with multiple images."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = {"session": mock_onnx_session}
    annotator.target_size = (448, 448)
    annotator.is_nchw_expected = True

    images = [Image.new("RGB", (224, 224), color="red") for _ in range(3)]
    result = annotator._preprocess_images(images)

    assert isinstance(result, list)
    assert len(result) == 3
    for arr in result:
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 3, 448, 448)


# ==============================================================================
# Category 4: Inference Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_success(mock_onnx_session):
    """Test _run_inference executes model successfully."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = {"session": mock_onnx_session}

    input_data = [np.random.randn(1, 3, 448, 448).astype(np.float32)]
    expected_output = np.random.randn(1, 100).astype(np.float32)
    mock_onnx_session.run.return_value = [expected_output]

    result = annotator._run_inference(input_data)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == expected_output.shape
    mock_onnx_session.run.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_batch(mock_onnx_session):
    """Test _run_inference with batch input."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = {"session": mock_onnx_session}

    batch_size = 4
    input_data = [np.random.randn(1, 3, 448, 448).astype(np.float32) for _ in range(batch_size)]
    expected_output = np.random.randn(1, 100).astype(np.float32)
    mock_onnx_session.run.return_value = [expected_output]

    result = annotator._run_inference(input_data)

    assert isinstance(result, list)
    assert len(result) == batch_size


# ==============================================================================
# Category 5: Result Formatting Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_single(sample_tags, mock_onnx_session):
    """Test _format_predictions_single formats single prediction."""
    annotator = ConcreteONNXAnnotator("test-model", tags=sample_tags)
    annotator.all_tags = sample_tags
    annotator.tag_threshold = 0.5
    annotator.components = {"session": mock_onnx_session}

    # Create prediction with 4 values matching sample_tags length
    predictions = np.array([0.7, 0.3, 0.8, 0.4])

    with patch("image_annotator_lib.core.utils.get_model_capabilities") as mock_caps:
        from image_annotator_lib.core.types import TaskCapability

        mock_caps.return_value = [TaskCapability.TAGS]
        result = annotator._format_predictions_single(predictions)

        assert result.tags is not None
        assert isinstance(result.tags, list)


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_batch(sample_tags, mock_onnx_session):
    """Test _format_predictions formats batch predictions."""
    annotator = ConcreteONNXAnnotator("test-model", tags=sample_tags)
    annotator.all_tags = sample_tags
    annotator.tag_threshold = 0.5
    annotator.components = {"session": mock_onnx_session}

    batch_predictions = [np.array([0.7, 0.3, 0.8, 0.4]), np.array([0.2, 0.6, 0.5, 0.9])]

    with patch("image_annotator_lib.core.utils.get_model_capabilities") as mock_caps:
        from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult

        mock_caps.return_value = [TaskCapability.TAGS]
        with patch.object(annotator, "_format_predictions_single") as mock_format:
            mock_format.return_value = UnifiedAnnotationResult(
                model_name="test-model",
                capabilities=[TaskCapability.TAGS],
                tags=["tag1"],
                framework="onnx",
            )
            result = annotator._format_predictions(batch_predictions)

            assert len(result) == 2
            assert mock_format.call_count == 2


# ==============================================================================
# Category 6: Error Handling Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_session_not_initialized():
    """Test _run_inference fails when session not initialized."""
    annotator = ConcreteONNXAnnotator("test-model")
    annotator.components = None

    input_data = [np.random.randn(1, 3, 448, 448).astype(np.float32)]

    with pytest.raises(RuntimeError, match="ONNX セッションがロードされていません"):
        annotator._run_inference(input_data)
