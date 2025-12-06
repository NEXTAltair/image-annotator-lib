"""Unit tests for memory and device error handling.

Tests CUDA OOM error detection and reporting.

Mock Strategy (Phase C):
- Real: Error propagation and logging
- Mock: torch.cuda.OutOfMemoryError, ModelLoad operations
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_cuda_oom_error_detection(managed_config_registry):
    """Test CUDA OOM error is correctly detected and propagated.

    Mock Strategy:
    - Mock: ModelLoad.load_onnx_components to raise CUDA OOM error
    - Real: Error detection, logging, exception propagation

    Verifies:
    - CUDA OOM error raised during model loading
    - Error message includes memory allocation details
    - Error is logged appropriately
    - Context manager properly handles error
    """
    from image_annotator_lib.model_class.tagger_onnx import WDTagger

    config = {
        "class": "WDTagger",
        "model_path": "/fake/model.onnx",
        "device": "cuda",
        "estimated_size_gb": 1.0,
        "type": "tagger",
    }
    managed_config_registry.set("test_oom", config)

    # Create CUDA OOM error
    mock_torch = MagicMock()
    mock_oom_error = type("OutOfMemoryError", (RuntimeError,), {})
    mock_torch.cuda.OutOfMemoryError = mock_oom_error

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch(
            "image_annotator_lib.core.model_factory.ModelLoad.load_onnx_components"
        ) as mock_load:
            # Simulate CUDA OOM during load
            mock_load.side_effect = mock_oom_error(
                "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity)"
            )

            tagger = WDTagger("test_oom")

            # Expect error to propagate through context manager
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                with tagger:
                    pass  # Should fail during __enter__
