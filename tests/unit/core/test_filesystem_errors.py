"""Unit tests for filesystem error handling.

Tests file system errors (FileNotFoundError, PermissionError, corrupted files).

Mock Strategy (Phase C):
- Real: File operations, error propagation
- Mock: File paths, file existence
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_filesystem_error(tmp_path):
    """Create helper for filesystem error simulation."""

    def _create_error(error_type: type, path: str):
        """Create filesystem error instance.

        Args:
            error_type: Exception class (FileNotFoundError, PermissionError, etc.)
            path: File path that triggered the error

        Returns:
            Exception instance configured with path message
        """
        return error_type(f"Mock error for {path}")

    return _create_error


@pytest.mark.unit
def test_missing_model_file_error(managed_config_registry):
    """Test FileNotFoundError for non-existent model file.

    Mock Strategy:
    - Real: Error propagation through context manager
    - Mock: ModelLoad.load_onnx_components to raise FileNotFoundError

    Verifies:
    - FileNotFoundError propagated correctly from ModelLoad
    - Error message includes file path
    - Context manager handles error appropriately
    """
    from image_annotator_lib.model_class.tagger_onnx import WDTagger

    # Register config with non-existent path
    config = {
        "class": "WDTagger",
        "model_path": "/nonexistent/path/model.onnx",
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_missing_file", config)

    with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
        # Simulate file not found during model loading
        mock_load.side_effect = FileNotFoundError(
            "[Errno 2] No such file or directory: '/nonexistent/path/model.onnx'"
        )

        tagger = WDTagger("test_missing_file")

        # Attempt to enter context (load model)
        with pytest.raises(FileNotFoundError, match="/nonexistent/path/model.onnx"):
            with tagger:
                pass


@pytest.mark.unit
def test_corrupted_model_file_error(managed_config_registry, tmp_path):
    """Test RuntimeError for corrupted/invalid model file.

    Mock Strategy:
    - Real: Error propagation through context manager
    - Mock: ModelLoad.load_onnx_components to raise RuntimeError

    Verifies:
    - RuntimeError or ONNXRuntimeError raised from ModelLoad
    - Error message indicates corruption/invalid format
    - Error logged appropriately
    """
    from image_annotator_lib.model_class.tagger_onnx import WDTagger

    # Create fake corrupted file path
    fake_model = tmp_path / "corrupted.onnx"
    fake_model.write_bytes(b"INVALID ONNX DATA")

    config = {
        "class": "WDTagger",
        "model_path": str(fake_model),
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_corrupted", config)

    with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
        # Simulate ONNX load error for corrupted file
        mock_load.side_effect = RuntimeError("Failed to load model: Invalid ONNX format")

        tagger = WDTagger("test_corrupted")

        with pytest.raises(RuntimeError, match="Invalid ONNX format"):
            with tagger:
                pass


@pytest.mark.unit
def test_missing_csv_file_error(managed_config_registry, tmp_path):
    """Test FileNotFoundError for missing CSV tag file.

    Mock Strategy:
    - Real: Error propagation through _load_tags()
    - Mock: ModelLoad.load_onnx_components to return components with missing csv_path

    Verifies:
    - FileNotFoundError raised when CSV file missing
    - Error occurs during _load_tags() call in __enter__
    - Error message indicates missing tag file
    """
    from image_annotator_lib.model_class.tagger_onnx import WDTagger

    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"fake model")

    config = {
        "class": "WDTagger",
        "model_path": str(fake_model),
        "device": "cpu",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_missing_csv", config)

    # Mock ONNX session
    mock_session = MagicMock()
    mock_input = Mock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 448, 448]
    mock_session.get_inputs.return_value = [mock_input]

    mock_output = Mock()
    mock_output.name = "output"
    mock_session.get_outputs.return_value = [mock_output]

    # Return components without csv_path to trigger error
    with patch("image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components") as mock_load:
        mock_load.return_value = {
            "session": mock_session,
            "csv_path": "",  # Empty csv_path triggers error
            "model_path": str(fake_model),
        }

        tagger = WDTagger("test_missing_csv")

        # _load_tags() checks csv_path and raises FileNotFoundError
        with pytest.raises(FileNotFoundError, match="タグ情報ファイルパスが見つかりません"):
            with tagger:
                pass
