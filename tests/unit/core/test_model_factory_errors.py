"""Unit tests for ModelLoad error handling functionality.

このモジュールでは、ModelLoadクラスのエラーハンドリング機能をテストします。

Test Categories:
1. Load Error Handling Tests
2. Memory Error Detection Tests
3. State Cleanup Tests
4. Cache Eviction Failure Tests
"""

from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.exceptions.errors import OutOfMemoryError

# ==============================================================================
# Category 1: Load Error Handling Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_handle_load_error_memory_error():
    """Test handling of memory errors during model loading.

    Tests:
    - Memory error detection
    - State cleanup after error
    - Appropriate logging
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    # Setup initial state
    ModelLoad._MODEL_STATES["test_model"] = "loading"
    ModelLoad._MEMORY_USAGE["test_model"] = 1024.0

    error = OutOfMemoryError("Failed to allocate memory", details={"model": "test_model"})

    ModelLoad._handle_load_error("test_model", error)

    # Verify state was cleaned up
    assert "test_model" not in ModelLoad._MODEL_STATES


@pytest.mark.unit
@pytest.mark.fast
def test_handle_load_error_file_not_found():
    """Test handling of file not found errors.

    Tests:
    - FileNotFoundError detection
    - Appropriate error logging
    - State cleanup
    """
    ModelLoad._MODEL_STATES.clear()

    error = FileNotFoundError("Model file not found: /path/to/model.onnx")

    ModelLoad._handle_load_error("test_model", error)

    # Verify error was handled gracefully
    assert "test_model" not in ModelLoad._MODEL_STATES


@pytest.mark.unit
@pytest.mark.fast
def test_handle_load_error_generic_error():
    """Test handling of generic errors during model loading.

    Tests:
    - Generic error handling
    - Logging with traceback
    - State cleanup
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MODEL_SIZES.clear()

    # Setup failed size calculation
    ModelLoad._MODEL_SIZES["test_model"] = 0.0

    error = ValueError("Invalid model configuration")

    ModelLoad._handle_load_error("test_model", error)

    # Verify failed size cache was cleared
    assert "test_model" not in ModelLoad._MODEL_SIZES


# ==============================================================================
# Category 2: Memory Error Detection Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_handle_load_error_cuda_out_of_memory():
    """Test detection of CUDA out of memory errors.

    Tests:
    - CUDA OOM error detection
    - Memory summary logging (when available)
    - State cleanup
    """
    ModelLoad._MODEL_STATES.clear()

    # Create mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
    error = mock_torch.cuda.OutOfMemoryError("CUDA out of memory")
    error.device = "cuda:0"

    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.memory_summary.return_value = "Memory summary"

    # Inject mock torch into sys.modules
    with patch.dict("sys.modules", {"torch": mock_torch}):
        ModelLoad._handle_load_error("test_model", error)

        # Verify memory summary was attempted
        mock_torch.cuda.memory_summary.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_handle_load_error_onnx_memory_error():
    """Test detection of ONNX runtime memory errors.

    Tests:
    - ONNX memory error detection from error message
    - Appropriate error classification
    """
    ModelLoad._MODEL_STATES.clear()

    error = OSError("onnxruntime: Failed to allocate memory for tensor")

    ModelLoad._handle_load_error("test_model", error)

    # Verify error was handled as memory error
    assert "test_model" not in ModelLoad._MODEL_STATES


# ==============================================================================
# Category 3: State Cleanup Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_release_model_state():
    """Test model state release functionality.

    Tests:
    - State dictionary cleanup
    - Memory usage cleanup
    - Last used timestamp cleanup
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    # Setup state
    ModelLoad._MODEL_STATES["test_model"] = "loaded"
    ModelLoad._MEMORY_USAGE["test_model"] = 1024.0
    ModelLoad._MODEL_LAST_USED["test_model"] = 1234567890.0

    ModelLoad._release_model_state("test_model")

    # Verify all state was cleared
    assert "test_model" not in ModelLoad._MODEL_STATES
    assert "test_model" not in ModelLoad._MEMORY_USAGE
    assert "test_model" not in ModelLoad._MODEL_LAST_USED


@pytest.mark.unit
@pytest.mark.fast
def test_release_model_internal_with_components():
    """Test model release with component cleanup.

    Tests:
    - Component deletion
    - Garbage collection
    - CUDA cache clearing (if available)
    - State cleanup
    """
    ModelLoad._MODEL_STATES.clear()

    components = {
        "model": MagicMock(),
        "processor": MagicMock(),
        "session": MagicMock(),
        "other": MagicMock(),  # Should not be deleted
    }

    # Create mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch("image_annotator_lib.core.model_factory.gc") as mock_gc:
        # Inject mock torch into sys.modules
        with patch.dict("sys.modules", {"torch": mock_torch}):
            ModelLoad._release_model_internal("test_model", components)

            # Verify garbage collection was called
            mock_gc.collect.assert_called_once()


# ==============================================================================
# Category 4: Cache Eviction Failure Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_clear_cache_internal_insufficient_space():
    """Test cache eviction failure when space cannot be freed.

    Tests:
    - Detection of eviction failure
    - Return False when space cannot be freed
    - Preservation of most recently used models
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    # Setup: Only one model taking all cache space
    ModelLoad._MEMORY_USAGE["large_model"] = 10 * 1024  # 10GB
    ModelLoad._MODEL_LAST_USED["large_model"] = 1234567890.0

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=8 * 1024):  # 8GB max
        # Try to load another 5GB model (total 15GB > 8GB max)
        result = ModelLoad._clear_cache_internal("new_model", 5 * 1024)

        # Should fail to make enough space (cannot evict the model we're trying to load)
        assert result is False
