"""Unit tests for CamieTagger."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_annotator_lib.model_class.tagger_onnx import CamieTagger


@pytest.fixture
def camie_config(managed_config_registry) -> str:
    managed_config_registry.set(
        "test_camie",
        {
            "class": "CamieTagger",
            "model_path": "Camais03/camie-tagger",
            "device": "cpu",
            "estimated_size_gb": 0.856,
            "type": "tagger",
            "capabilities": ["tags", "ratings"],
            "tag_threshold": 0.325,
        },
    )
    return "test_camie"


@pytest.fixture
def camie_metadata_path(tmp_path: Path) -> Path:
    metadata = {
        "idx_to_tag": {
            "0": "year_2024",
            "1": "1girl",
            "2": "hakurei_reimu",
            "3": "rating_general",
            "4": "rating_sensitive",
            "5": "rating_questionable",
            "6": "rating_explicit",
            "7": "artist_name",
        },
        "tag_to_category": {
            "year_2024": "year",
            "1girl": "general",
            "hakurei_reimu": "character",
            "rating_general": "rating",
            "rating_sensitive": "rating",
            "rating_questionable": "rating",
            "rating_explicit": "rating",
            "artist_name": "artist",
        },
    }
    path = tmp_path / "model_initial_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _logits(*probabilities: float) -> np.ndarray:
    probs = np.array([probabilities], dtype=np.float32)
    return np.log(probs / (1.0 - probs)).astype(np.float32)


@pytest.mark.unit
def test_camie_loads_json_metadata(camie_config: str, camie_metadata_path: Path) -> None:
    annotator = CamieTagger(camie_config)
    annotator.components = {"session": object(), "metadata_path": camie_metadata_path}

    annotator._load_tags()

    assert annotator.all_tags == [
        "year_2024",
        "1girl",
        "hakurei_reimu",
        "rating_general",
        "rating_sensitive",
        "rating_questionable",
        "rating_explicit",
        "artist_name",
    ]
    assert annotator.general_indexes == [1]
    assert annotator.character_indexes == [2]
    assert annotator.rating_indexes == [3, 4, 5, 6]
    # ライブラリはカテゴリを削らない: year / artist も保持される
    assert annotator.year_indexes == [0]
    assert annotator.artist_indexes == [7]


@pytest.mark.unit
def test_camie_formats_sigmoid_tags_and_rating(camie_config: str, camie_metadata_path: Path) -> None:
    annotator = CamieTagger(camie_config)
    annotator.components = {"session": object(), "metadata_path": camie_metadata_path}
    annotator._load_tags()

    result = annotator._format_predictions_single(_logits(0.95, 0.80, 0.70, 0.05, 0.85, 0.20, 0.01, 0.99))

    assert result.error is None
    # rating 以外の全カテゴリ (year / artist 含む) のタグを confidence 降順で出力する
    assert result.tags == ["artist_name", "year_2024", "1girl", "hakurei_reimu"]
    assert result.ratings is not None
    assert result.ratings[0].raw_label == "sensitive"
    assert result.ratings[0].confidence_score == pytest.approx(0.85)
    assert result.ratings[0].source_scheme == "danbooru4"
    # 全カテゴリの raw スコアが category_scores に残る (情報を削らない)
    assert set(result.raw_output["category_scores"]) == {
        "year",
        "general",
        "character",
        "artist",
        "ratings",
    }


@pytest.mark.unit
def test_camie_preprocess_uses_black_padding(camie_config: str) -> None:
    annotator = CamieTagger(camie_config)
    annotator.target_size = (512, 512)
    image = Image.new("RGB", (512, 256), color="red")

    [processed] = annotator._preprocess_images([image])

    assert processed.shape == (1, 3, 512, 512)
    assert processed.dtype == np.float32
    assert processed[0, :, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert processed[0, :, 256, 256].tolist() == [1.0, 0.0, 0.0]
