"""Unit tests for rating-only ONNX annotators."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.model_class.rating_onnx import AnimeRatingAnnotator


@pytest.fixture
def test_image() -> Image.Image:
    return Image.new("RGB", (96, 128), color="white")


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    mock_input = Mock()
    mock_input.name = "input"
    session.get_inputs.return_value = [mock_input]
    mock_output = Mock()
    mock_output.name = "output"
    session.get_outputs.return_value = [mock_output]
    session.run.return_value = [np.array([[0.05, 0.88, 0.07]], dtype=np.float32)]
    return session


@pytest.fixture
def anime_rating_config(managed_config_registry, tmp_path: Path) -> str:
    model_dir = tmp_path / "anime_rating" / "mobilenetv3_sce_dist"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake onnx")
    (model_dir / "meta.json").write_text(
        '{"labels": ["safe", "r15", "r18"], "name": "mobilenetv3_large_100"}',
        encoding="utf-8",
    )
    managed_config_registry.set(
        "test_anime_rating",
        {
            "class": "AnimeRatingAnnotator",
            "model_path": str(tmp_path / "anime_rating"),
            "model_variant": "mobilenetv3_sce_dist",
            "device": "cpu",
            "estimated_size_gb": 0.047,
            "type": "tagger",
            "capabilities": ["ratings"],
        },
    )
    return "test_anime_rating"


@pytest.mark.unit
def test_anime_rating_outputs_sankaku3_rating(
    anime_rating_config: str, mock_session: MagicMock, test_image: Image.Image
) -> None:
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        annotator = AnimeRatingAnnotator(anime_rating_config)
        with annotator:
            processed = annotator._preprocess_images([test_image])
            raw_outputs = annotator._run_inference(processed)
            formatted = annotator._format_predictions(raw_outputs)

    assert processed.shape == (1, 3, 384, 384)
    assert processed.dtype == np.float32
    assert len(formatted) == 1
    result = formatted[0]
    assert result.capabilities == {TaskCapability.RATINGS}
    assert result.tags is None
    assert result.ratings is not None
    assert result.ratings[0].raw_label == "r15"
    assert result.ratings[0].confidence_score == pytest.approx(0.88)
    assert result.ratings[0].source_scheme == "sankaku3"
    assert result.raw_output == {
        "ratings": {"safe": pytest.approx(0.05), "r15": pytest.approx(0.88), "r18": pytest.approx(0.07)}
    }


@pytest.mark.unit
def test_anime_rating_softmaxes_logits(anime_rating_config: str) -> None:
    annotator = AnimeRatingAnnotator(anime_rating_config)
    probabilities = annotator._ensure_probabilities(np.array([[1.0, 3.0, 2.0]], dtype=np.float32))

    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.argmax(axis=1).tolist() == [1]


@pytest.mark.unit
def test_anime_rating_falls_back_to_default_labels(anime_rating_config: str, tmp_path: Path) -> None:
    annotator = AnimeRatingAnnotator(anime_rating_config)
    labels = annotator._load_labels(tmp_path / "missing_meta.json")

    assert labels == ["safe", "r15", "r18"]


@pytest.mark.unit
def test_anime_rating_resolves_local_variant_directory(tmp_path: Path) -> None:
    variant_dir = tmp_path / "mobilenetv3_sce_dist"
    variant_dir.mkdir()
    model_file = variant_dir / "model.onnx"
    meta_file = variant_dir / "meta.json"
    model_file.write_bytes(b"fake onnx")
    meta_file.write_text('{"labels": ["safe", "r15", "r18"]}', encoding="utf-8")

    resolved_model, resolved_meta = AnimeRatingAnnotator._resolve_model_files(
        str(variant_dir), "mobilenetv3_sce_dist"
    )

    assert resolved_model == model_file
    assert resolved_meta == meta_file
