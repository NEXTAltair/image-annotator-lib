"""Unit tests for TensorflowBaseAnnotator class.

TensorFlow基底アノテータの主要機能をテスト。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from image_annotator_lib.core.base.tensorflow import TensorflowBaseAnnotator
from image_annotator_lib.exceptions.errors import OutOfMemoryError

# ==============================================================================
# Test Helper
# ==============================================================================


class ConcreteTensorflowAnnotator(TensorflowBaseAnnotator):
    """Concrete implementation of TensorflowBaseAnnotator for testing."""

    def __init__(self, model_name: str, tags: list[str] | None = None):
        # Mock config_registry before calling super().__init__
        with patch("image_annotator_lib.core.base.tensorflow.config_registry") as mock_registry:
            mock_registry.get.return_value = "h5"
            super().__init__(model_name)
            self._test_tags = tags or []
            self.all_tags: list[str] = []

    def _load_tags(self) -> None:
        """Implementation of abstract method for testing."""
        self.all_tags = self._test_tags

    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray:
        """Implementation of abstract method for testing."""
        return np.random.randn(len(images), 224, 224, 3).astype(np.float32)

    def _format_predictions(self, raw_output: tf.Tensor) -> list[dict]:
        """Implementation of abstract method for testing."""
        return [{"tags": ["test"]} for _ in range(raw_output.shape[0])]


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_tf_model():
    """Mock TensorFlow model."""
    model = MagicMock()
    model.return_value = tf.constant([[0.8, 0.3, 0.9, 0.2]])
    return model


@pytest.fixture
def sample_tags():
    """Sample tags data."""
    return ["tag1", "tag2", "tag3", "tag4"]


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (224, 224), color="red")


# ==============================================================================
# Test Initialization
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_tensorflow_annotator_init():
    """Test TensorflowBaseAnnotator initialization."""
    with patch("image_annotator_lib.core.base.tensorflow.config_registry") as mock_registry:
        mock_registry.get.return_value = "h5"

        annotator = ConcreteTensorflowAnnotator("test-model")

        assert annotator.model_name == "test-model"
        assert annotator.model_format == "h5"
        assert annotator.components is None


# ==============================================================================
# Test Context Manager
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_tensorflow_annotator_context_manager_enter(sample_tags):
    """Test __enter__ method loads resources correctly."""
    annotator = ConcreteTensorflowAnnotator("test-model", tags=sample_tags)
    annotator.model_path = "/path/to/model.h5"
    annotator.device = "cpu"

    with patch("image_annotator_lib.core.model_factory.ModelLoad.load_tensorflow_components") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = {"model": mock_model}

        result = annotator.__enter__()

        assert result is annotator
        assert annotator.components == {"model": mock_model}
        assert annotator.all_tags == sample_tags
        mock_load.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_tensorflow_annotator_context_manager_enter_no_path():
    """Test __enter__ raises error when model_path is None."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.model_path = None

    with pytest.raises(ValueError, match="model_path が設定されていません"):
        annotator.__enter__()


@pytest.mark.unit
@pytest.mark.fast
def test_tensorflow_annotator_context_manager_exit():
    """Test __exit__ method cleans up resources."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.components = {"model": MagicMock()}

    with patch("image_annotator_lib.core.model_factory.ModelLoad.release_model_components") as mock_release:
        mock_release.return_value = {}
        with patch("tensorflow.keras.backend.clear_session") as mock_clear:
            annotator.__exit__(None, None, None)

            assert annotator.components is None
            mock_release.assert_called_once()
            mock_clear.assert_called_once()


# ==============================================================================
# Test Tag Loading
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_load_tag_file_success(tmp_path):
    """Test _load_tag_file with valid tags file."""
    tags_file = tmp_path / "tags.txt"
    tags_file.write_text("tag1\ntag2\ntag3\n", encoding="utf-8")

    annotator = ConcreteTensorflowAnnotator("test-model")
    tags = annotator._load_tag_file(tags_file)

    assert tags == ["tag1", "tag2", "tag3"]


@pytest.mark.unit
@pytest.mark.fast
def test_load_tag_file_not_found(tmp_path):
    """Test _load_tag_file with non-existent file."""
    tags_file = tmp_path / "nonexistent.txt"

    annotator = ConcreteTensorflowAnnotator("test-model")
    tags = annotator._load_tag_file(tags_file)

    assert tags == []


# ==============================================================================
# Test Inference
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_tf_success(mock_tf_model):
    """Test _run_inference_tf executes model successfully."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.components = {"model": mock_tf_model}

    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
    result = annotator._run_inference_tf(input_data)

    assert isinstance(result, tf.Tensor)
    mock_tf_model.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_tf_no_model():
    """Test _run_inference_tf raises error when model not loaded."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.components = None

    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)

    with pytest.raises(RuntimeError, match="TensorFlow モデルがロードされていません"):
        annotator._run_inference_tf(input_data)


@pytest.mark.unit
@pytest.mark.fast
def test_run_inference_tf_resource_exhausted():
    """Test _run_inference_tf handles OOM error."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    mock_model = MagicMock()
    mock_model.side_effect = tf.errors.ResourceExhaustedError(None, None, "OOM")
    annotator.components = {"model": mock_model}

    input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)

    with pytest.raises(OutOfMemoryError, match="TensorFlow リソース枯渇"):
        annotator._run_inference_tf(input_data)


# ==============================================================================
# Test Tag Extraction
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_extract_category_tags():
    """Test _extract_category_tags extracts category tags."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.all_tags = ["tag1", "tag2", "tag3"]
    annotator.character_indexes = [0, 2]

    tags_with_probs = [("tag1", 0.9), ("tag2", 0.5), ("tag3", 0.8)]

    result = annotator._extract_category_tags("character_indexes", tags_with_probs)

    assert isinstance(result, dict)
    assert "tag1" in result
    assert result["tag1"] == 0.9
    assert "tag3" in result
    assert result["tag3"] == 0.8
    assert "tag2" not in result


# ==============================================================================
# Test Prediction Formatting
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_single_with_numpy(sample_tags):
    """Test _format_predictions_single with numpy array."""
    annotator = ConcreteTensorflowAnnotator("test-model", tags=sample_tags)
    annotator.all_tags = sample_tags

    predictions = np.array([0.7, 0.3, 0.8, 0.4])

    result = annotator._format_predictions_single(predictions)

    assert isinstance(result, dict)
    assert "general" in result


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_single_with_tensor(sample_tags):
    """Test _format_predictions_single with TensorFlow tensor."""
    annotator = ConcreteTensorflowAnnotator("test-model", tags=sample_tags)
    annotator.all_tags = sample_tags

    predictions = tf.constant([[0.7, 0.3, 0.8, 0.4]])

    result = annotator._format_predictions_single(predictions)

    assert isinstance(result, dict)
    assert "general" in result


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_single_no_tags():
    """Test _format_predictions_single when tags not loaded."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.all_tags = []

    predictions = np.array([0.7, 0.3, 0.8, 0.4])

    result = annotator._format_predictions_single(predictions)

    assert "error" in result


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_single_mismatch():
    """Test _format_predictions_single with tag/prediction count mismatch."""
    annotator = ConcreteTensorflowAnnotator("test-model", tags=["tag1", "tag2"])
    annotator.all_tags = ["tag1", "tag2"]

    predictions = np.array([0.7, 0.3, 0.8, 0.4])

    result = annotator._format_predictions_single(predictions)

    assert "error" in result


# ==============================================================================
# Test Tag Generation
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_generate_tags_single():
    """Test _generate_tags_single generates tag list."""
    annotator = ConcreteTensorflowAnnotator("test-model")
    annotator.tag_threshold = 0.5

    formatted_output = {
        "general": {"tag1": 0.7, "tag2": 0.3, "tag3": 0.8},
        "character": {"tag4": 0.6},
    }

    result = annotator._generate_tags_single(formatted_output)

    assert isinstance(result, list)
    assert "tag3" in result
    assert "tag1" in result
    assert "tag4" in result
    assert "tag2" not in result


@pytest.mark.unit
@pytest.mark.fast
def test_generate_tags_single_error():
    """Test _generate_tags_single with error output."""
    annotator = ConcreteTensorflowAnnotator("test-model")

    formatted_output = {"error": {}}

    result = annotator._generate_tags_single(formatted_output)

    assert result == []
