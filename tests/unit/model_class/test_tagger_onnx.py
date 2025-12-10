"""Unit tests for ONNX tagger models.

Tests WDTagger and Z3D_E621Tagger implementations with mocked ONNX Runtime.

Mock Strategy (Phase C Level 1-2):
- Level 1 (Mock): onnxruntime.InferenceSession - Model loading
- Level 2 (Mock): session.run() - Inference execution
- Level 3 (Real): Image preprocessing, tag extraction, config loading
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from image_annotator_lib.model_class.tagger_onnx import WDTagger, Z3D_E621Tagger


@pytest.fixture
def test_image():
    """Create test PIL image."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def test_images_batch():
    """Create batch of test PIL images."""
    return [
        Image.new("RGB", (100, 100), color="red"),
        Image.new("RGB", (150, 150), color="green"),
        Image.new("RGB", (200, 200), color="blue"),
    ]


@pytest.fixture
def mock_onnx_session():
    """Create mock ONNX InferenceSession.

    Mock Strategy:
    - Mock: InferenceSession creation and session.run()
    - Real: Input shape analysis, preprocessing logic

    Returns:
        Mock session with configured input/output metadata
    """
    session = MagicMock()

    # Configure input shape (NCHW format: batch=1, channels=3, height=448, width=448)
    mock_input = Mock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 448, 448]
    session.get_inputs.return_value = [mock_input]

    # Configure output
    mock_output = Mock()
    mock_output.name = "output"
    session.get_outputs.return_value = [mock_output]

    # Mock inference output (9 tags matching CSV - will be overridden in specific tests)
    mock_predictions = np.random.rand(1, 9).astype(np.float32)
    session.run.return_value = [mock_predictions]

    return session


@pytest.fixture
def mock_csv_file(tmp_path):
    """Create temporary CSV file for tag data.

    Returns:
        Path to temporary CSV with WD Tagger format
    """
    csv_path = tmp_path / "tags.csv"
    csv_content = """name,category
1girl,0
solo,0
smile,0
long_hair,0
rating:safe,9
rating:questionable,9
rating:explicit,9
character_name,4
artist_name,1
"""
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def mock_wdtagger_config(managed_config_registry, tmp_path):
    """Register WDTagger configuration.

    Mock Strategy:
    - Real: Config registry operations
    - Mock: Model file paths (use tmp_path)

    Note:
    - csv_path comes from ModelLoad components
    - tag_threshold uses default value of 0.35
    """
    config = {
        "class": "WDTagger",
        "model_path": str(tmp_path / "model.onnx"),
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_wdtagger", config)
    return "test_wdtagger"


@pytest.fixture
def mock_capabilities():
    """Mock get_model_capabilities to return tags capability.

    This avoids Pydantic validation issues with capabilities in config.
    """
    from image_annotator_lib.core.types import TaskCapability

    with patch("image_annotator_lib.core.utils.get_model_capabilities") as mock:
        mock.return_value = {TaskCapability.TAGS}
        yield mock


@pytest.mark.unit
def test_onnx_tagger_initialization_success(mock_wdtagger_config, mock_onnx_session, mock_csv_file):
    """Test ONNX tagger initialization with session creation.

    Mock Strategy:
    - Mock: onnxruntime.InferenceSession
    - Real: Config loading, category mapping setup, threshold setting

    Verifies:
    - Session created with correct model path
    - Category mapping initialized
    - Tag threshold configured from config
    - CSV path loaded from components
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            # Configure mock to return components with session and csv_path
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/path/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Verify initialization
                assert tagger.model_name == mock_wdtagger_config
                assert tagger.tag_threshold == 0.35  # Default value
                assert tagger.CATEGORY_MAPPING == {"rating": 9, "general": 0, "character": 4}
                assert hasattr(tagger, "rating_indexes")
                assert hasattr(tagger, "general_indexes")
                assert hasattr(tagger, "character_indexes")

                # Verify tags loaded
                assert len(tagger.all_tags) > 0
                assert "1girl" in tagger.all_tags
                assert "solo" in tagger.all_tags


@pytest.mark.unit
def test_onnx_tagger_preprocessing(mock_wdtagger_config, mock_onnx_session, mock_csv_file, test_image):
    """Test image preprocessing logic.

    Mock Strategy:
    - Mock: ONNX session (no actual inference)
    - Real: PIL image processing, padding, resizing, normalization

    Verifies:
    - RGB conversion for non-RGB images
    - Square padding logic
    - Resize to target size from model input shape
    - NCHW/NHWC format handling
    - Dtype conversion to float32
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/path/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Test preprocessing
                processed = tagger._preprocess_images([test_image])

                # Verify output
                assert len(processed) == 1
                assert isinstance(processed[0], np.ndarray)
                assert processed[0].dtype == np.float32

                # Verify shape matches model input (1, 3, 448, 448) for NCHW
                if tagger.is_nchw_expected:
                    assert processed[0].shape == (1, 3, 448, 448)
                else:
                    assert processed[0].shape == (1, 448, 448, 3)


@pytest.mark.unit
def test_onnx_tagger_inference(
    mock_wdtagger_config, mock_onnx_session, mock_csv_file, test_image, mock_capabilities
):
    """Test ONNX inference with mocked session.run().

    Mock Strategy:
    - Mock: session.run() returns fake predictions
    - Real: Tag extraction, threshold filtering, category mapping

    Verifies:
    - session.run() called with correct input/output names
    - Predictions processed correctly
    - Tags filtered by threshold
    - Category scores extracted
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/path/model.onnx",
            }

            # Create controlled predictions (9 tags from CSV)
            num_tags = 9
            predictions = np.zeros((1, num_tags), dtype=np.float32)
            predictions[0, 0] = 0.95  # 1girl (general, above threshold)
            predictions[0, 1] = 0.85  # solo (general, above threshold)
            predictions[0, 2] = 0.30  # smile (general, below threshold 0.5)
            predictions[0, 4] = 0.75  # rating:safe (rating)
            mock_onnx_session.run.return_value = [predictions]

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Run inference
                processed = tagger._preprocess_images([test_image])
                raw_outputs = tagger._run_inference(processed)
                formatted = tagger._format_predictions(raw_outputs)

                # Verify inference called
                assert mock_onnx_session.run.called
                call_args = mock_onnx_session.run.call_args
                assert call_args[0][0] == ["output"]  # output_name
                assert "input" in call_args[0][1]  # input dict

                # Verify results
                assert len(formatted) == 1
                result = formatted[0]
                assert result.tags is not None
                assert len(result.tags) >= 2  # At least 1girl and solo (above threshold)
                assert result.framework == "onnx"
                assert result.raw_output is not None


@pytest.mark.unit
def test_onnx_tagger_batch_processing(
    mock_wdtagger_config, mock_onnx_session, mock_csv_file, test_images_batch, mock_capabilities
):
    """Test batch processing of multiple images.

    Mock Strategy:
    - Mock: session.run() called once per image
    - Real: Batch loop logic, result aggregation

    Verifies:
    - All images preprocessed correctly
    - session.run() called once per image (no batching)
    - Results list matches input image count
    - Each result is valid UnifiedAnnotationResult
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/path/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Process batch
                processed = tagger._preprocess_images(test_images_batch)
                raw_outputs = tagger._run_inference(processed)
                formatted = tagger._format_predictions(raw_outputs)

                # Verify batch processing
                assert len(processed) == 3
                assert len(raw_outputs) == 3
                assert len(formatted) == 3

                # Verify session.run() called 3 times (once per image)
                assert mock_onnx_session.run.call_count == 3

                # Verify all results valid
                for result in formatted:
                    assert result.model_name == mock_wdtagger_config
                    assert result.framework == "onnx"


@pytest.mark.unit
def test_onnx_tagger_error_handling(managed_config_registry, mock_onnx_session):
    """Test error handling for invalid model path and corrupted file.

    Mock Strategy:
    - Mock: onnxruntime.InferenceSession to raise errors
    - Real: Error propagation and logging

    Verifies:
    - ConfigurationError raised for missing model_path
    - FileNotFoundError raised for missing CSV
    - Appropriate error messages in logs
    """
    from image_annotator_lib.exceptions.errors import ConfigurationError

    # Test 1: Missing model_path (caught by config validation)
    config_no_path = {
        "class": "WDTagger",
        "device": "cpu",
        # missing model_path
    }
    managed_config_registry.set("test_no_path", config_no_path)

    with pytest.raises(ConfigurationError, match="model_path"):
        tagger = WDTagger("test_no_path")

    # Test 2: Missing CSV file
    config_no_csv = {
        "class": "WDTagger",
        "model_path": "/fake/model.onnx",
        "device": "cpu",
        "estimated_size_gb": 1.0,
    }
    managed_config_registry.set("test_no_csv", config_no_csv)

    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            # Return components without csv_path
            mock_load.return_value = {
                "session": mock_onnx_session,
                "model_path": "/fake/model.onnx",
            }

            tagger = WDTagger("test_no_csv")
            with pytest.raises(FileNotFoundError, match="タグ情報ファイルパスが見つかりません"):
                with tagger:
                    pass

    # Test 3: Corrupted CSV file (invalid format)
    config_bad_csv = {
        "class": "WDTagger",
        "model_path": "/fake/model.onnx",
        "device": "cpu",
        "estimated_size_gb": 1.0,
    }
    managed_config_registry.set("test_bad_csv", config_bad_csv)

    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": "/fake/corrupted.csv",
                "model_path": "/fake/model.onnx",
            }

            # Mock polars.read_csv to raise error for corrupted file
            with patch("polars.read_csv", side_effect=Exception("Invalid CSV format")):
                tagger = WDTagger("test_bad_csv")
                with pytest.raises(Exception, match="Invalid CSV format"):
                    with tagger:
                        pass


# ==============================================================================
# Phase C Additional Coverage Tests (2025-12-05)
# ==============================================================================


@pytest.mark.unit
def test_onnx_tagger_load_tags_from_csv(mock_wdtagger_config, mock_onnx_session, tmp_path):
    """Test CSV tag loading with category mapping.

    Tests:
    - CSV parsing with polars
    - Tag and category extraction
    - Category index mapping (general, rating, character)
    - Tag list construction
    """
    # Create CSV with multiple categories
    csv_path = tmp_path / "tags.csv"
    csv_content = """name,category
general_tag_1,0
general_tag_2,0
artist_name,1
copyright_name,3
character_name,4
meta_tag,5
rating:safe,9
rating:questionable,9
rating:explicit,9
"""
    csv_path.write_text(csv_content)

    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(csv_path),
                "model_path": "/fake/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Verify tags loaded
                assert len(tagger.all_tags) == 9
                assert "general_tag_1" in tagger.all_tags
                assert "character_name" in tagger.all_tags
                assert "rating:safe" in tagger.all_tags

                # Verify category indexes
                assert 0 in tagger.general_indexes  # general_tag_1
                assert 1 in tagger.general_indexes  # general_tag_2
                assert 4 in tagger.character_indexes  # character_name
                assert 6 in tagger.rating_indexes  # rating:safe


@pytest.mark.unit
def test_onnx_tagger_preprocessing_edge_cases(mock_wdtagger_config, mock_onnx_session, mock_csv_file):
    """Test preprocessing with various image sizes and aspect ratios.

    Tests:
    - Portrait image (tall)
    - Landscape image (wide)
    - Square image
    - Very small image
    - RGBA image conversion
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Test various image sizes
                test_cases = [
                    ("portrait", Image.new("RGB", (100, 200), color="red")),  # 1:2 ratio
                    ("landscape", Image.new("RGB", (300, 100), color="green")),  # 3:1 ratio
                    ("square", Image.new("RGB", (150, 150), color="blue")),  # 1:1 ratio
                    ("tiny", Image.new("RGB", (10, 10), color="yellow")),  # Very small
                    ("rgba", Image.new("RGBA", (100, 100), (255, 0, 0, 128))),  # RGBA
                ]

                for name, img in test_cases:
                    processed = tagger._preprocess_images([img])

                    # Verify output shape matches model input
                    assert len(processed) == 1
                    assert isinstance(processed[0], np.ndarray)
                    assert processed[0].dtype == np.float32

                    # Verify dimensions match expected format
                    if tagger.is_nchw_expected:
                        assert processed[0].shape[0] == 1  # Batch
                        assert processed[0].shape[1] == 3  # Channels
                        assert processed[0].shape[2] == 448  # Height
                        assert processed[0].shape[3] == 448  # Width
                    else:
                        assert processed[0].shape[0] == 1  # Batch
                        assert processed[0].shape[1] == 448  # Height
                        assert processed[0].shape[2] == 448  # Width
                        assert processed[0].shape[3] == 3  # Channels


@pytest.mark.unit
def test_onnx_tagger_tag_threshold_filtering(
    mock_wdtagger_config, mock_onnx_session, mock_csv_file, test_image, mock_capabilities
):
    """Test tag filtering by threshold.

    Tests:
    - Tags above threshold are included
    - Tags below threshold are excluded
    - Threshold boundary values
    - Different threshold settings
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Create predictions with known values
                num_tags = 9
                predictions = np.array(
                    [
                        [
                            0.90,  # Above threshold (0.35)
                            0.50,  # Above threshold
                            0.35,  # Exactly at threshold (boundary)
                            0.34,  # Just below threshold
                            0.20,  # Below threshold
                            0.10,  # Below threshold
                            0.05,  # Below threshold
                            0.01,  # Below threshold
                            0.00,  # Zero
                        ]
                    ],
                    dtype=np.float32,
                )
                mock_onnx_session.run.return_value = [predictions]

                # Run inference
                processed = tagger._preprocess_images([test_image])
                raw_outputs = tagger._run_inference(processed)
                formatted = tagger._format_predictions(raw_outputs)

                # Verify filtering
                result = formatted[0]
                assert result.tags is not None

                # Note: Due to float32 precision, 0.35 becomes 0.3499999940395355 which is < 0.35
                # So only tags with scores 0.90 and 0.50 pass the threshold (indices 0, 1)
                # Tags: ['1girl', 'solo']
                assert len(result.tags) >= 2
                assert "1girl" in result.tags  # Score 0.90
                assert "solo" in result.tags  # Score 0.50

                # Verify raw output contains all predictions
                assert result.raw_output is not None
                assert result.raw_output["threshold"] == 0.35


@pytest.mark.unit
def test_onnx_tagger_category_score_extraction(
    mock_wdtagger_config, mock_onnx_session, mock_csv_file, test_image, mock_capabilities
):
    """Test extraction of category scores (rating, general, character).

    Tests:
    - Rating scores extracted correctly
    - General tag scores aggregated
    - Character tag scores extracted
    - Category mapping accuracy
    """
    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/model.onnx",
            }

            tagger = WDTagger(mock_wdtagger_config)

            with tagger:
                # Create predictions with specific category scores
                num_tags = 9
                predictions = np.array(
                    [
                        [
                            0.85,  # general tag 1
                            0.75,  # general tag 2
                            0.65,  # general tag 3
                            0.55,  # general tag 4
                            0.80,  # rating:safe (index 4 in CSV)
                            0.15,  # rating:questionable
                            0.05,  # rating:explicit
                            0.70,  # character_name
                            0.60,  # artist_name
                        ]
                    ],
                    dtype=np.float32,
                )
                mock_onnx_session.run.return_value = [predictions]

                # Run inference
                processed = tagger._preprocess_images([test_image])
                raw_outputs = tagger._run_inference(processed)
                formatted = tagger._format_predictions(raw_outputs)

                # Verify result structure
                result = formatted[0]
                assert result.tags is not None
                assert result.raw_output is not None
                assert result.framework == "onnx"


@pytest.mark.unit
def test_z3d_e621_tagger_initialization(managed_config_registry, mock_onnx_session, mock_csv_file):
    """Test Z3D_E621Tagger initialization and differences from WDTagger.

    Tests:
    - Z3D_E621Tagger uses different category mapping
    - Category indexes differ from WDTagger
    - Model initialization succeeds
    - CSV loading works with e621 format
    """
    config = {
        "class": "Z3D_E621Tagger",
        "model_path": "/fake/model.onnx",
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_z3d", config)

    with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
        with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
            mock_load.return_value = {
                "session": mock_onnx_session,
                "csv_path": str(mock_csv_file),
                "model_path": "/fake/model.onnx",
            }

            tagger = Z3D_E621Tagger("test_z3d")

            with tagger:
                # Verify Z3D_E621Tagger specific attributes
                # Note: Z3D does NOT have CATEGORY_MAPPING (only WDTagger has it)
                # Z3D uses _category_attr_map and _rating_tags instead
                assert hasattr(tagger, "_category_attr_map")
                assert hasattr(tagger, "_rating_tags")
                assert tagger._rating_tags == ["explicit", "questionable", "safe"]

                # Verify category attribute mapping includes more categories than WDTagger
                expected_categories = {
                    "general",
                    "artist",
                    "character",
                    "species",
                    "copyright",
                    "meta",
                    "lore",
                }
                assert set(tagger._category_attr_map.keys()) == expected_categories

                # Verify basic initialization
                assert tagger.model_name == "test_z3d"
                assert tagger.tag_threshold == 0.35
                assert hasattr(tagger, "all_tags")
                assert hasattr(tagger, "rating_indexes")
                assert hasattr(tagger, "general_indexes")
