"""Unit tests for generalized ONNX loader file resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.loaders.onnx_loader import ONNXLoader
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.core.utils import download_onnx_model_files, download_onnx_tagger_model


@pytest.mark.unit
def test_download_onnx_model_files_uses_custom_filenames() -> None:
    with (
        patch(
            "image_annotator_lib.core.utils.huggingface_hub.list_repo_files",
            return_value=["model_initial.onnx", "model_initial_metadata.json"],
        ),
        patch(
            "image_annotator_lib.core.utils.huggingface_hub.hf_hub_download",
            side_effect=["/cache/model_initial_metadata.json", "/cache/model_initial.onnx"],
        ) as mock_download,
    ):
        metadata_path, model_path = download_onnx_model_files(
            "Camais03/camie-tagger",
            model_filename="model_initial.onnx",
            metadata_filename="model_initial_metadata.json",
            metadata_extension=".json",
        )

    assert metadata_path == Path("/cache/model_initial_metadata.json")
    assert model_path == Path("/cache/model_initial.onnx")
    assert mock_download.call_args_list[0].args == (
        "Camais03/camie-tagger",
        "model_initial_metadata.json",
    )
    assert mock_download.call_args_list[1].args == ("Camais03/camie-tagger", "model_initial.onnx")


@pytest.mark.unit
def test_download_onnx_tagger_model_keeps_csv_fallback() -> None:
    with (
        patch(
            "image_annotator_lib.core.utils.huggingface_hub.list_repo_files",
            return_value=["model.onnx", "tags.csv"],
        ),
        patch(
            "image_annotator_lib.core.utils.huggingface_hub.hf_hub_download",
            side_effect=["/cache/tags.csv", "/cache/model.onnx"],
        ) as mock_download,
    ):
        csv_path, model_path = download_onnx_tagger_model("SmilingWolf/example")

    assert csv_path == Path("/cache/tags.csv")
    assert model_path == Path("/cache/model.onnx")
    assert mock_download.call_args_list[0].args == ("SmilingWolf/example", "tags.csv")
    assert mock_download.call_args_list[1].args == ("SmilingWolf/example", "model.onnx")


@pytest.mark.unit
def test_onnx_loader_returns_metadata_and_csv_compatibility(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    model_path = tmp_path / "model.onnx"
    metadata_path.write_text("{}", encoding="utf-8")
    model_path.write_bytes(b"fake")
    session = MagicMock()

    with (
        patch(
            "image_annotator_lib.core.loaders.onnx_loader.utils.download_onnx_model_files",
            return_value=(metadata_path, model_path),
        ),
        patch("onnxruntime.InferenceSession", return_value=session),
    ):
        components = ONNXLoader("test-model", "cpu")._load_components_internal(
            "test/repo",
            model_filename="model_initial.onnx",
            metadata_filename="model_initial_metadata.json",
            metadata_extension=".json",
        )

    assert components["session"] is session
    assert components["metadata_path"] == metadata_path
    assert components["csv_path"] == metadata_path


@pytest.mark.unit
def test_model_load_passes_onnx_filename_options() -> None:
    with patch("image_annotator_lib.core.model_factory.ONNXLoader") as mock_loader_cls:
        mock_loader = mock_loader_cls.return_value
        mock_loader.load_components.return_value = {"session": MagicMock()}

        result = ModelLoad.load_onnx_components(
            "camie",
            "Camais03/camie-tagger",
            "cpu",
            model_filename="model_initial.onnx",
            metadata_filename="model_initial_metadata.json",
            metadata_extension=".json",
        )

    assert result == {"session": mock_loader.load_components.return_value["session"]}
    mock_loader.load_components.assert_called_once_with(
        "Camais03/camie-tagger",
        model_filename="model_initial.onnx",
        metadata_filename="model_initial_metadata.json",
        metadata_extension=".json",
    )
