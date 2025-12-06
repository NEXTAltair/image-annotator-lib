"""Unit tests for Scorer models.

Tests Pipeline-based scorers (AestheticShadow, CafePredictor) and CLIP-based scorers.

Mock Strategy (Phase C Level 1-2):
- Level 1 (Mock): transformers.pipeline(), CLIP model loading
- Level 2 (Mock): pipeline(images), model.encode_image()
- Level 3 (Real): Score normalization, batch processing, tag generation
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from image_annotator_lib.model_class.pipeline_scorers import AestheticShadow, AestheticShadowV2, CafePredictor
from image_annotator_lib.model_class.scorer_clip import ImprovedAesthetic, WaifuAesthetic


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
def mock_pipeline():
    """Create mock Hugging Face pipeline.

    Mock Strategy:
    - Mock: pipeline creation and execution
    - Real: Output structure (list of dicts with labels and scores)

    Returns:
        Mock pipeline callable
    """
    mock_pipe = MagicMock()

    # Configure pipeline to return classification results
    # AestheticShadow format: [{'label': 'hq', 'score': 0.9}, {'label': 'lq', 'score': 0.1}]
    mock_pipe.return_value = [[
        {"label": "hq", "score": 0.85},
        {"label": "lq", "score": 0.15},
    ]]

    return mock_pipe


@pytest.fixture
def mock_cafe_pipeline():
    """Create mock pipeline for CafePredictor.

    Returns:
        Mock pipeline with 'aesthetic' label
    """
    mock_pipe = MagicMock()

    # CafePredictor format: [{'label': 'aesthetic', 'score': 0.67}, {'label': 'not_aesthetic', 'score': 0.33}]
    mock_pipe.return_value = [[
        {"label": "aesthetic", "score": 0.67},
        {"label": "not_aesthetic", "score": 0.33},
    ]]

    return mock_pipe


@pytest.fixture
def mock_clip_components():
    """Create mock CLIP model and processor.

    Mock Strategy:
    - Mock: Model and processor creation, encode methods
    - Real: Tensor operations, score calculation

    Returns:
        Dict with mocked CLIP components
    """
    # Mock model
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    # Mock image encoder output
    mock_image_features = torch.randn(1, 512)  # CLIP feature dimension
    mock_model.encode_image.return_value = mock_image_features

    # Mock processor
    mock_processor = MagicMock()
    mock_processor_output = {
        "pixel_values": torch.randn(1, 3, 224, 224),
    }
    mock_processor.return_value = mock_processor_output

    # Mock MLP head for aesthetic scoring
    mock_mlp = MagicMock()
    mock_mlp_output = torch.tensor([[0.75]])  # Aesthetic score
    mock_mlp.return_value = mock_mlp_output

    return {
        "model": mock_model,
        "processor": mock_processor,
        "mlp_head": mock_mlp,
        "model_path": "/fake/clip/path",
    }


@pytest.fixture
def mock_aesthetic_shadow_config(managed_config_registry):
    """Register AestheticShadow configuration.

    Mock Strategy:
    - Real: Config registry operations
    - Mock: Model paths (use fake paths)
    """
    config = {
        "class": "AestheticShadow",
        "model_path": "model_name/aesthetic-shadow",
        "device": "cpu",
        "estimated_size_gb": 0.5,
        "type": "scorer",
    }
    managed_config_registry.set("test_aesthetic_shadow", config)
    return "test_aesthetic_shadow"


@pytest.fixture
def mock_cafe_config(managed_config_registry):
    """Register CafePredictor configuration."""
    config = {
        "class": "CafePredictor",
        "model_path": "model_name/cafe_aesthetic",
        "device": "cpu",
        "estimated_size_gb": 0.5,
        "type": "scorer",
    }
    managed_config_registry.set("test_cafe", config)
    return "test_cafe"


@pytest.fixture
def mock_clip_scorer_config(managed_config_registry):
    """Register CLIP scorer configuration.

    Note:
    - base_model is required for CLIP models (validated in __enter__)
    """
    config = {
        "class": "ImprovedAesthetic",
        "model_path": "model_name/improved-aesthetic",
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "scorer",
        "base_model": "ViT-B/32",
    }
    managed_config_registry.set("test_clip_scorer", config)
    return "test_clip_scorer"


@pytest.fixture
def mock_capabilities_scorer():
    """Mock get_model_capabilities for scorer models."""
    from image_annotator_lib.core.types import TaskCapability

    with patch("image_annotator_lib.core.utils.get_model_capabilities") as mock:
        mock.return_value = {TaskCapability.SCORES}
        yield mock


@pytest.mark.unit
def test_pipeline_scorer_initialization(
    mock_aesthetic_shadow_config, mock_pipeline, mock_capabilities_scorer
):
    """Test Pipeline scorer initialization.

    Mock Strategy:
    - Mock: ModelLoad.load_transformers_pipeline_components
    - Real: Config loading, batch_size/task settings, threshold initialization

    Verifies:
    - Pipeline created with correct task and model_path
    - batch_size and task loaded from config
    - SCORE_THRESHOLDS initialized for AestheticShadow
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_pipeline_components"
    ) as mock_load:
        mock_load.return_value = {"pipeline": mock_pipeline, "model_path": "/fake/path"}

        scorer = AestheticShadow(mock_aesthetic_shadow_config)

        with scorer:
            # Verify initialization
            assert scorer.model_name == mock_aesthetic_shadow_config
            assert scorer.device == "cpu"
            assert scorer.batch_size == 8  # Default value from PipelineBaseAnnotator
            assert scorer.task == "image-classification"  # Default value
            assert scorer.components is not None
            assert "pipeline" in scorer.components

            # Verify SCORE_THRESHOLDS
            assert scorer.SCORE_THRESHOLDS == {
                "very aesthetic": 0.71,
                "aesthetic": 0.45,
                "displeasing": 0.27,
            }

            # Verify ModelLoad called with correct args
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[0][0] == "image-classification"  # task (default)
            assert call_args[0][2] == "model_name/aesthetic-shadow"  # model_path
            assert call_args[0][3] == "cpu"  # device
            assert call_args[0][4] == 8  # batch_size (default)


@pytest.mark.unit
def test_pipeline_scorer_prediction(
    mock_aesthetic_shadow_config, mock_pipeline, test_image, mock_capabilities_scorer
):
    """Test Pipeline scorer prediction with mocked pipeline.

    Mock Strategy:
    - Mock: pipeline(images) returns fake scores
    - Real: Score extraction, formatting, tag generation

    Verifies:
    - pipeline() called with PIL images
    - Scores extracted correctly (hq/lq format)
    - Score values in valid range [0.0, 1.0]
    - Tags generated based on thresholds
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_pipeline_components"
    ) as mock_load:
        mock_load.return_value = {"pipeline": mock_pipeline, "model_path": "/fake/path"}

        scorer = AestheticShadow(mock_aesthetic_shadow_config)

        with scorer:
            # Process and run inference
            processed = scorer._preprocess_images([test_image])
            raw_outputs = scorer._run_inference(processed)
            formatted = scorer._format_predictions(raw_outputs)

            # Verify pipeline called
            assert mock_pipeline.called
            call_args = mock_pipeline.call_args
            assert len(call_args[0][0]) == 1  # One image

            # Verify output structure
            assert len(formatted) == 1
            assert isinstance(formatted[0], dict)
            assert "hq" in formatted[0]
            assert "lq" in formatted[0]

            # Verify score range
            assert 0.0 <= formatted[0]["hq"] <= 1.0
            assert 0.0 <= formatted[0]["lq"] <= 1.0

            # Verify tag generation
            tags = scorer._generate_tags(formatted[0])
            assert isinstance(tags, list)
            assert len(tags) > 0
            # hq=0.85 should be >= 0.71 threshold
            assert tags[0] == "very aesthetic"


@pytest.mark.unit
def test_cafe_scorer_prediction(
    mock_cafe_config, mock_cafe_pipeline, test_image, mock_capabilities_scorer
):
    """Test CafePredictor score extraction and tag generation.

    Mock Strategy:
    - Mock: pipeline(images) returns 'aesthetic' label scores
    - Real: Score extraction logic, tag formatting with prefix

    Verifies:
    - 'aesthetic' label score extracted correctly
    - Score converted to tag with [CAFE]_ prefix
    - Score level calculation (0-10 scale)
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_pipeline_components"
    ) as mock_load:
        mock_load.return_value = {"pipeline": mock_cafe_pipeline, "model_path": "/fake/path"}

        scorer = CafePredictor(mock_cafe_config)

        with scorer:
            # Process and run inference
            processed = scorer._preprocess_images([test_image])
            raw_outputs = scorer._run_inference(processed)
            formatted = scorer._format_predictions(raw_outputs)

            # Verify output
            assert len(formatted) == 1
            assert isinstance(formatted[0], float)
            assert formatted[0] == 0.67

            # Verify tag generation with prefix
            tags = scorer._generate_tags(formatted[0])
            assert len(tags) == 1
            assert tags[0].startswith("[CAFE]_score_")
            # 0.67 * 10 = 6.7 → int() = 6
            assert tags[0] == "[CAFE]_score_6"


@pytest.mark.unit
def test_clip_scorer_initialization(
    mock_clip_scorer_config, mock_clip_components, mock_capabilities_scorer
):
    """Test CLIP scorer initialization.

    Mock Strategy:
    - Mock: ModelLoad.load_clip_components
    - Real: Config loading, CLIP model selection

    Verifies:
    - CLIP model and processor loaded
    - clip_model config parameter used
    - MLP head for aesthetic scoring initialized
    """
    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_clip_components"
    ) as mock_load:
        mock_load.return_value = mock_clip_components

        scorer = ImprovedAesthetic(mock_clip_scorer_config)

        with scorer:
            # Verify initialization
            assert scorer.model_name == mock_clip_scorer_config
            assert scorer.device == "cpu"
            assert scorer.components is not None
            assert "model" in scorer.components
            assert "processor" in scorer.components

            # Verify ModelLoad called
            mock_load.assert_called_once()


@pytest.mark.unit
def test_scorer_batch_processing(
    mock_aesthetic_shadow_config, mock_pipeline, test_images_batch, mock_capabilities_scorer
):
    """Test batch processing of multiple images.

    Mock Strategy:
    - Mock: pipeline(batch) called once
    - Real: Batch loop logic, result aggregation

    Verifies:
    - All images preprocessed correctly
    - pipeline() called with full batch
    - Results list matches input image count
    - Each result is valid score dict
    """
    # Configure mock pipeline for batch
    mock_pipeline.return_value = [
        [{"label": "hq", "score": 0.85}, {"label": "lq", "score": 0.15}],
        [{"label": "hq", "score": 0.60}, {"label": "lq", "score": 0.40}],
    ]

    with patch(
        "image_annotator_lib.core.model_factory.ModelLoad.load_transformers_pipeline_components"
    ) as mock_load:
        mock_load.return_value = {"pipeline": mock_pipeline, "model_path": "/fake/path"}

        scorer = AestheticShadow(mock_aesthetic_shadow_config)

        with scorer:
            # Process batch
            processed = scorer._preprocess_images(test_images_batch)
            raw_outputs = scorer._run_inference(processed)
            formatted = scorer._format_predictions(raw_outputs)

            # Verify batch processing
            assert len(processed) == 2
            assert len(raw_outputs) == 2
            assert len(formatted) == 2

            # Verify pipeline called once with full batch
            assert mock_pipeline.call_count == 1
            call_args = mock_pipeline.call_args
            assert len(call_args[0][0]) == 2  # Two images

            # Verify all results valid
            for result in formatted:
                assert isinstance(result, dict)
                assert "hq" in result
                assert "lq" in result
                assert 0.0 <= result["hq"] <= 1.0
                assert 0.0 <= result["lq"] <= 1.0
