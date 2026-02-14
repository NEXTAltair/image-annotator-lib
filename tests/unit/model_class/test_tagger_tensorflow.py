"""Unit tests for DeepDanbooruTagger class.

DeepDanbooruタガーの主要機能をテスト。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from image_annotator_lib.model_class.tagger_tensorflow import DeepDanbooruTagger

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_config():
    """Create a mock config object for testing."""
    config = MagicMock()
    config.model_name = "deepdanbooru-v3"
    config.model_path = "/path/to/model"
    config.device = "cpu"
    config.estimated_size_gb = 1.0
    return config


@pytest.fixture
def sample_tags():
    """Sample tags data for testing."""
    return ["tag1", "tag2", "tag3", "tag4"]


@pytest.fixture
def sample_character_tags():
    """Sample character tags."""
    return ["char1", "char2"]


@pytest.fixture
def sample_general_tags():
    """Sample general tags."""
    return ["general1", "general2"]


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (512, 512), color="red")


@pytest.fixture
def mock_model_dir(tmp_path, sample_tags, sample_character_tags, sample_general_tags):
    """Create a mock model directory with tag files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Write tags.txt
    (model_dir / "tags.txt").write_text("\n".join(sample_tags), encoding="utf-8")

    # Write tags-character.txt
    (model_dir / "tags-character.txt").write_text("\n".join(sample_character_tags), encoding="utf-8")

    # Write tags-general.txt
    (model_dir / "tags-general.txt").write_text("\n".join(sample_general_tags), encoding="utf-8")

    return model_dir


@pytest.fixture
def mock_tf_model():
    """Mock TensorFlow model."""
    model = MagicMock()
    model.return_value = tf.constant([[0.8, 0.3, 0.9, 0.2]])
    return model


# ==============================================================================
# Test Initialization
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_deepdanbooru_tagger_init(mock_config):
    """Test DeepDanbooruTagger initialization."""
    with patch("image_annotator_lib.model_class.tagger_tensorflow.config_registry") as mock_registry:
        mock_registry.get.return_value = 0.5
        tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    assert tagger.model_name == "deepdanbooru-v3"
    assert tagger.tag_threshold == 0.5


@pytest.mark.unit
@pytest.mark.fast
def test_deepdanbooru_tagger_init_default_threshold(mock_config):
    """Test DeepDanbooruTagger initialization with default threshold."""
    config = MagicMock()
    config.model_name = "deepdanbooru-v3"
    config.model_path = "/path/to/model"
    config.device = "cpu"
    config.estimated_size_gb = 1.0

    with patch("image_annotator_lib.model_class.tagger_tensorflow.config_registry") as mock_registry:
        # Simulate config_registry returning None for tag_threshold
        mock_registry.get.return_value = None

        tagger = DeepDanbooruTagger("deepdanbooru-v3", config=config)

        # Should use default value 0.35
        assert tagger.tag_threshold == 0.35


# ==============================================================================
# Test Tag Loading
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_load_tags_success(
    mock_config, mock_model_dir, sample_tags, sample_character_tags, sample_general_tags
):
    """Test _load_tags successfully loads all tag files."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {"model_dir": mock_model_dir}

    tagger._load_tags()

    assert tagger.components["all_tags"] == sample_tags
    assert tagger.components["tags_character"] == sample_character_tags
    assert tagger.components["tags_general"] == sample_general_tags


@pytest.mark.unit
@pytest.mark.fast
def test_load_tags_missing_model_dir(mock_config):
    """Test _load_tags raises error when model_dir is missing."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {}

    with pytest.raises(FileNotFoundError, match="モデルディレクトリが見つかりません"):
        tagger._load_tags()


@pytest.mark.unit
@pytest.mark.fast
def test_load_tags_missing_tags_file(mock_config, tmp_path):
    """Test _load_tags raises error when tags.txt is missing."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tagger.components = {"model_dir": model_dir}

    with pytest.raises(FileNotFoundError, match="tags.txt が見つからないか、空です"):
        tagger._load_tags()


@pytest.mark.unit
@pytest.mark.fast
def test_load_tags_optional_files_missing(mock_config, tmp_path, sample_tags):
    """Test _load_tags handles missing optional tag files."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Only create tags.txt
    (model_dir / "tags.txt").write_text("\n".join(sample_tags), encoding="utf-8")

    tagger.components = {"model_dir": model_dir}

    tagger._load_tags()

    assert tagger.components["all_tags"] == sample_tags
    assert tagger.components["tags_character"] == []
    assert tagger.components["tags_general"] == []


# ==============================================================================
# Test Image Preprocessing
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_v1_model():
    """Test _preprocess_images with v1 model (299x299)."""
    config = MagicMock()
    config.model_name = "deepdanbooru-v1-resnet50"
    config.model_path = "/path/to/model"
    config.device = "cpu"
    config.estimated_size_gb = 1.0

    with patch("image_annotator_lib.model_class.tagger_tensorflow.config_registry") as mock_registry:
        mock_registry.get.return_value = 0.5
        tagger = DeepDanbooruTagger("deepdanbooru-v1-resnet50", config=config)

    image = Image.new("RGB", (512, 512), color="red")
    result = tagger._preprocess_images([image])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 299, 299, 3)
    assert result.dtype == np.float32
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_v3_model(mock_config):
    """Test _preprocess_images with v3 model (512x512)."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    image = Image.new("RGB", (300, 300), color="blue")
    result = tagger._preprocess_images([image])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 3)
    assert result.dtype == np.float32
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_v4_model():
    """Test _preprocess_images with v4 model (512x512)."""
    config = MagicMock()
    config.model_name = "deepdanbooru-v4-20200814-sgd-e30"
    config.model_path = "/path/to/model"
    config.device = "cpu"
    config.estimated_size_gb = 1.0

    with patch("image_annotator_lib.model_class.tagger_tensorflow.config_registry") as mock_registry:
        mock_registry.get.return_value = 0.5
        tagger = DeepDanbooruTagger("deepdanbooru-v4-20200814-sgd-e30", config=config)

    image = Image.new("RGB", (400, 400), color="green")
    result = tagger._preprocess_images([image])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 3)
    assert result.dtype == np.float32


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_converts_rgba_to_rgb(mock_config):
    """Test _preprocess_images converts RGBA images to RGB."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    # Create RGBA image
    image = Image.new("RGBA", (512, 512), color=(255, 0, 0, 128))
    result = tagger._preprocess_images([image])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 3)


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_multiple_images(mock_config):
    """Test _preprocess_images with multiple images."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    images = [
        Image.new("RGB", (512, 512), color="red"),
        Image.new("RGB", (300, 300), color="blue"),
        Image.new("RGB", (600, 600), color="green"),
    ]

    result = tagger._preprocess_images(images)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 512, 512, 3)


# ==============================================================================
# Test Result Formatting
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_success(mock_config, sample_tags, sample_character_tags, sample_general_tags):
    """Test _format_predictions successfully formats predictions."""
    from image_annotator_lib.core.types import UnifiedAnnotationResult

    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": sample_tags,
        "tags_character": sample_character_tags,
        "tags_general": sample_general_tags,
    }

    # Create predictions matching sample_tags length
    raw_output = tf.constant([[0.7, 0.3, 0.8, 0.4]])

    result = tagger._format_predictions(raw_output)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], UnifiedAnnotationResult)
    assert result[0].raw_output is not None
    assert "general" in result[0].raw_output
    assert "character" in result[0].raw_output
    assert "other" in result[0].raw_output


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_categorizes_tags_correctly(mock_config):
    """Test _format_predictions correctly categorizes tags."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    # Create tags where we know which are general/character
    all_tags = ["general1", "general2", "other1", "char1", "char2", "other2"]
    tags_character = ["char1", "char2"]
    tags_general = ["general1", "general2"]

    tagger.components = {
        "all_tags": all_tags,
        "tags_character": tags_character,
        "tags_general": tags_general,
    }

    # Predictions: all tags have high scores
    predictions = tf.constant([[0.9, 0.8, 0.7, 0.85, 0.75, 0.6]])

    result = tagger._format_predictions(predictions)

    assert len(result) == 1
    formatted = result[0].raw_output

    # Check that character tags are in character category
    assert "char1" in formatted["character"]
    assert "char2" in formatted["character"]

    # Check that general tags are in general category
    assert "general1" in formatted["general"]
    assert "general2" in formatted["general"]

    # Check that other tags are in other category
    assert "other1" in formatted["other"]
    assert "other2" in formatted["other"]


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_sorts_by_score(
    mock_config, sample_tags, sample_character_tags, sample_general_tags
):
    """Test _format_predictions sorts tags by score in descending order."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": sample_tags,
        "tags_character": sample_character_tags,
        "tags_general": sample_general_tags,
    }

    # Create predictions with known scores
    predictions = tf.constant([[0.3, 0.9, 0.5, 0.7]])  # tag2 > tag4 > tag3 > tag1

    result = tagger._format_predictions(predictions)
    formatted = result[0].raw_output

    # Check that tags in each category are sorted by score
    for category in ["general", "character", "other"]:
        tags_in_category = list(formatted[category].keys())
        scores = [formatted[category][tag] for tag in tags_in_category]

        # Verify scores are in descending order
        assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_batch(mock_config):
    """Test _format_predictions with batch predictions."""
    from image_annotator_lib.core.types import UnifiedAnnotationResult

    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": ["tag1", "tag2"],
        "tags_character": ["tag1"],
        "tags_general": ["tag2"],
    }

    # Batch of 3 predictions
    predictions = tf.constant(
        [
            [0.8, 0.6],
            [0.7, 0.9],
            [0.5, 0.4],
        ]
    )

    result = tagger._format_predictions(predictions)

    assert isinstance(result, list)
    assert len(result) == 3

    for unified_result in result:
        assert isinstance(unified_result, UnifiedAnnotationResult)
        formatted = unified_result.raw_output
        assert "general" in formatted
        assert "character" in formatted
        assert "other" in formatted


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_shape_mismatch(mock_config):
    """Test _format_predictions handles tag/prediction count mismatch."""
    from image_annotator_lib.core.types import UnifiedAnnotationResult

    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": ["tag1", "tag2"],  # 2 tags
        "tags_character": [],
        "tags_general": [],
    }

    # But predictions have 4 values - mismatch!
    predictions = tf.constant([[0.8, 0.6, 0.7, 0.5]])

    result = tagger._format_predictions(predictions)

    # Should return error result
    assert len(result) == 1
    assert isinstance(result[0], UnifiedAnnotationResult)
    assert result[0].error is not None


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_empty_tags(mock_config):
    """Test _format_predictions handles empty tags list."""
    from image_annotator_lib.core.types import UnifiedAnnotationResult

    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": [],
        "tags_character": [],
        "tags_general": [],
    }

    predictions = tf.constant([[0.8, 0.6]])

    result = tagger._format_predictions(predictions)

    assert len(result) == 1
    assert isinstance(result[0], UnifiedAnnotationResult)
    assert result[0].error is not None


# ==============================================================================
# Test Edge Cases
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_preprocess_images_empty_list(mock_config):
    """Test _preprocess_images with empty image list."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)

    result = tagger._preprocess_images([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 512, 512, 3)


@pytest.mark.unit
@pytest.mark.fast
def test_format_predictions_with_numpy_array(
    mock_config, sample_tags, sample_character_tags, sample_general_tags
):
    """Test _format_predictions accepts numpy array input."""
    tagger = DeepDanbooruTagger("deepdanbooru-v3", config=mock_config)
    tagger.components = {
        "all_tags": sample_tags,
        "tags_character": sample_character_tags,
        "tags_general": sample_general_tags,
    }

    # Use numpy array instead of tf.Tensor
    raw_output = np.array([[0.7, 0.3, 0.8, 0.4]])

    result = tagger._format_predictions(raw_output)

    assert isinstance(result, list)
    assert len(result) == 1
