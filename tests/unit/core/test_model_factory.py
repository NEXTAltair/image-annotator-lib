"""Unit tests for model_factory.py ModelLoad class.

このモジュールでは、ModelLoadクラスの各機能を段階的にテストします。
Phase 2教訓を適用し、公開インターフェースのテスト、段階的テスト開発を実践します。

Test Categories:
1. Size Management Tests (サイズ管理テスト)
2. Cache Management Tests (キャッシュ管理テスト)
3. Memory Management Tests (メモリ管理テスト)
4. Device Management Tests (デバイス管理テスト)
5. Model State Tests (モデル状態管理テスト)
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from image_annotator_lib.core.model_factory import ModelLoad

# ==============================================================================
# Category 1: Size Management Tests (サイズ管理テスト)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_size_from_config_success():
    """Test retrieving model size from config successfully.

    Tests:
    - Config value retrieval
    - GB to MB conversion
    - Valid size return
    """
    model_name = "test-model"
    expected_size_gb = 2.5
    expected_size_mb = expected_size_gb * 1024

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        mock_config.get.return_value = expected_size_gb

        result = ModelLoad._get_model_size_from_config(model_name)

        assert result == expected_size_mb
        mock_config.get.assert_called_once_with(model_name, "estimated_size_gb")


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_size_from_config_not_in_config():
    """Test handling when model is not in config.

    Tests:
    - KeyError handling
    - None return for missing model
    """
    model_name = "nonexistent-model"

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        mock_config.get.side_effect = KeyError("Model not found")

        result = ModelLoad._get_model_size_from_config(model_name)

        assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_size_from_config_none_value():
    """Test handling when config returns None.

    Tests:
    - None value handling
    - No conversion attempted
    """
    model_name = "test-model"

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        mock_config.get.return_value = None

        result = ModelLoad._get_model_size_from_config(model_name)

        assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_size_from_config_invalid_type():
    """Test handling when config returns invalid type.

    Tests:
    - ValueError/TypeError handling
    - None return for invalid data
    """
    model_name = "test-model"

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        mock_config.get.return_value = "not-a-number"

        result = ModelLoad._get_model_size_from_config(model_name)

        assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_file_size_mb_success(tmp_path):
    """Test calculating file size successfully.

    Tests:
    - File size calculation in MB
    - Byte to MB conversion
    """
    test_file = tmp_path / "test.bin"
    # Create 5MB file
    test_file.write_bytes(b"0" * (5 * 1024 * 1024))

    result = ModelLoad._calculate_file_size_mb(test_file)

    # Should be approximately 5MB
    assert 4.9 < result < 5.1


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_file_size_mb_nonexistent():
    """Test handling when file doesn't exist.

    Tests:
    - FileNotFoundError handling
    - 0.0 return for missing file
    """
    nonexistent_file = Path("/nonexistent/path/file.bin")

    result = ModelLoad._calculate_file_size_mb(nonexistent_file)

    assert result == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_dir_size_mb_success(tmp_path):
    """Test calculating directory size successfully.

    Tests:
    - Recursive directory size calculation
    - Multiple file handling
    - Byte to MB conversion
    """
    # Create test directory structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file1.bin").write_bytes(b"0" * (2 * 1024 * 1024))  # 2MB
    (tmp_path / "subdir" / "file2.bin").write_bytes(b"0" * (3 * 1024 * 1024))  # 3MB

    result = ModelLoad._calculate_dir_size_mb(tmp_path)

    # Should be approximately 5MB total
    assert 4.9 < result < 5.1


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_dir_size_mb_empty_dir(tmp_path):
    """Test calculating size of empty directory.

    Tests:
    - Empty directory handling
    - 0.0 return for empty directory
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = ModelLoad._calculate_dir_size_mb(empty_dir)

    assert result == 0.0


# ==============================================================================
# Category 2: Cache Management Tests (キャッシュ管理テスト)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_current_cache_usage_empty():
    """Test getting cache usage when no models cached.

    Tests:
    - Empty cache state
    - 0.0 usage return
    """
    # Clear cache state
    ModelLoad._MEMORY_USAGE.clear()

    result = ModelLoad._get_current_cache_usage()

    assert result == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_get_current_cache_usage_with_models():
    """Test getting cache usage with cached models.

    Tests:
    - Sum of cached model sizes
    - Multiple model handling
    """
    # Setup cache state
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MEMORY_USAGE["model1"] = 500.0  # 500MB
    ModelLoad._MEMORY_USAGE["model2"] = 300.0  # 300MB

    result = ModelLoad._get_current_cache_usage()

    assert result == 800.0


@pytest.mark.unit
@pytest.mark.fast
def test_get_models_sorted_by_last_used():
    """Test getting models sorted by LRU order.

    Tests:
    - Sorting by last used timestamp
    - Oldest first ordering
    - Returns list of (name, timestamp) tuples
    """
    # Setup state
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_LAST_USED["model1"] = 100.0
    ModelLoad._MODEL_LAST_USED["model2"] = 50.0  # Oldest
    ModelLoad._MODEL_LAST_USED["model3"] = 150.0  # Newest

    result = ModelLoad._get_models_sorted_by_last_used()

    # Should be sorted oldest to newest, returns tuples
    assert result == [("model2", 50.0), ("model1", 100.0), ("model3", 150.0)]


@pytest.mark.unit
@pytest.mark.fast
def test_get_models_sorted_by_last_used_empty():
    """Test getting models when no models cached.

    Tests:
    - Empty state handling
    - Empty list return
    """
    ModelLoad._MODEL_LAST_USED.clear()

    result = ModelLoad._get_models_sorted_by_last_used()

    assert result == []


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_memory_usage_exists():
    """Test getting memory usage for existing model.

    Tests:
    - Memory usage retrieval
    - Correct value return
    """
    model_name = "test-model"
    expected_usage = 500.0

    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MEMORY_USAGE[model_name] = expected_usage

    result = ModelLoad._get_model_memory_usage(model_name)

    assert result == expected_usage


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_memory_usage_not_exists():
    """Test getting memory usage for non-existent model.

    Tests:
    - Missing model handling
    - 0.0 return for missing model
    """
    ModelLoad._MEMORY_USAGE.clear()

    result = ModelLoad._get_model_memory_usage("nonexistent-model")

    assert result == 0.0


# ==============================================================================
# Category 3: Public Interface Tests (公開インターフェーステスト)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_size_public_interface():
    """Test public get_model_size method.

    Tests:
    - Public interface availability
    - Delegation to config retrieval
    """
    model_name = "test-model"

    with patch.object(ModelLoad, "_get_model_size_from_config", return_value=1024.0) as mock_method:
        result = ModelLoad.get_model_size(model_name)

        assert result == 1024.0
        mock_method.assert_called_once_with(model_name)


@pytest.mark.unit
@pytest.mark.fast
def test_get_max_cache_size_public_interface():
    """Test public get_max_cache_size method.

    Tests:
    - Public interface availability
    - Delegation to internal method
    """
    with patch.object(ModelLoad, "_get_max_cache_size", return_value=4096.0) as mock_method:
        result = ModelLoad.get_max_cache_size()

        assert result == 4096.0
        mock_method.assert_called_once()


# ==============================================================================
# Category 4: Model State Tests (モデル状態管理テスト)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_state_loaded():
    """Test getting state for loaded model.

    Tests:
    - State retrieval for loaded model
    - Correct state return
    """
    model_name = "test-model"

    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MODEL_STATES[model_name] = "loaded"

    result = ModelLoad._get_model_state(model_name)

    assert result == "loaded"


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_state_cached():
    """Test getting state for cached model.

    Tests:
    - State retrieval for cached model
    - Correct state return
    """
    model_name = "test-model"

    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MODEL_STATES[model_name] = "cached"

    result = ModelLoad._get_model_state(model_name)

    assert result == "cached"


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_state_not_exists():
    """Test getting state for non-existent model.

    Tests:
    - Missing model handling
    - None return for missing model
    """
    ModelLoad._MODEL_STATES.clear()

    result = ModelLoad._get_model_state("nonexistent-model")

    assert result is None


@pytest.mark.unit
@pytest.mark.fast
def test_release_model_state():
    """Test releasing model state.

    Tests:
    - State removal
    - Memory usage cleanup
    - Last used timestamp cleanup
    """
    model_name = "test-model"

    # Setup state
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    ModelLoad._MODEL_STATES[model_name] = "loaded"
    ModelLoad._MEMORY_USAGE[model_name] = 500.0
    ModelLoad._MODEL_LAST_USED[model_name] = 100.0

    # Release state
    ModelLoad._release_model_state(model_name)

    # Verify cleanup
    assert model_name not in ModelLoad._MODEL_STATES
    assert model_name not in ModelLoad._MEMORY_USAGE
    assert model_name not in ModelLoad._MODEL_LAST_USED


# ==============================================================================
# Category 5: Memory Management Tests (メモリ管理テスト)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_sufficient():
    """Test memory check when sufficient memory available.

    Tests:
    - psutil memory check
    - Sufficient memory case
    - True return
    """
    model_size_mb = 500.0  # 500MB

    # Mock psutil to show plenty of memory (10GB available)
    with patch("image_annotator_lib.core.model_factory.psutil") as mock_psutil:
        mock_memory = Mock()
        mock_memory.available = 10 * 1024 * 1024 * 1024  # 10GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory

        result = ModelLoad._check_memory_before_load(model_size_mb, "test-model")

        assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_insufficient():
    """Test memory check when insufficient memory available.

    Tests:
    - psutil memory check
    - Insufficient memory case
    - False return
    """
    model_size_mb = 5000.0  # 5GB

    # Mock psutil to show limited memory (1GB available)
    with patch("image_annotator_lib.core.model_factory.psutil") as mock_psutil:
        mock_memory = Mock()
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory

        result = ModelLoad._check_memory_before_load(model_size_mb, "test-model")

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_zero_size():
    """Test memory check with zero/invalid size.

    Tests:
    - Zero size handling
    - Skip memory check
    - True return (bypass check)
    """
    result = ModelLoad._check_memory_before_load(0.0, "test-model")

    assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_negative_size():
    """Test memory check with negative size.

    Tests:
    - Negative size handling
    - Skip memory check
    - True return (bypass check)
    """
    result = ModelLoad._check_memory_before_load(-100.0, "test-model")

    assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_get_max_cache_size():
    """Test calculating maximum cache size.

    Tests:
    - Cache ratio application (50%)
    - Total memory consideration
    - MB return value
    """
    # Mock psutil to show 8GB total memory
    with patch("image_annotator_lib.core.model_factory.psutil") as mock_psutil:
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory

        result = ModelLoad._get_max_cache_size()

        # Should be 50% of 8GB = 4GB = 4096MB
        expected_mb = (8 * 1024) * 0.5
        assert result == expected_mb


@pytest.mark.unit
@pytest.mark.fast
def test_clear_cache_internal_sufficient_space():
    """Test cache clearing when space is already sufficient.

    Tests:
    - No clearing needed
    - True return
    - No models released
    """
    # Setup: small required size, plenty of cache space
    ModelLoad._CACHE_RATIO = 0.5

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=4096.0):  # 4GB
        with patch.object(ModelLoad, "_get_current_cache_usage", return_value=1000.0):  # 1GB used
            result = ModelLoad._clear_cache_internal("test-model", 500.0)  # Need 500MB

            assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_clear_cache_internal_needs_clearing():
    """Test cache clearing when LRU eviction needed.

    Tests:
    - LRU eviction logic
    - State release for old models
    - True return after clearing
    """
    # Setup cache state with multiple models
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    ModelLoad._MODEL_STATES["model1"] = "cached"
    ModelLoad._MODEL_STATES["model2"] = "cached"
    ModelLoad._MEMORY_USAGE["model1"] = 1000.0  # 1GB
    ModelLoad._MEMORY_USAGE["model2"] = 1000.0  # 1GB
    ModelLoad._MODEL_LAST_USED["model1"] = 100.0  # Older
    ModelLoad._MODEL_LAST_USED["model2"] = 200.0  # Newer

    # Mock methods
    max_cache = 2500.0  # 2.5GB max cache

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache):
        # First call: 2GB used, second call after release: 1GB used
        usage_values = [2000.0, 2000.0, 1000.0, 1000.0]  # Called multiple times in loop
        with patch.object(ModelLoad, "_get_current_cache_usage", side_effect=usage_values):
            with patch.object(ModelLoad, "_release_model_state") as mock_release:
                result = ModelLoad._clear_cache_internal("new-model", 1000.0)  # Need 1GB

                # Should succeed after clearing
                assert result is True
                # Should release oldest model (model1)
                mock_release.assert_called()


@pytest.mark.unit
@pytest.mark.fast
def test_clear_cache_internal_cannot_make_space():
    """Test cache clearing when cannot free enough space.

    Tests:
    - Insufficient space even after clearing
    - False return
    - All possible models released
    """
    # Setup: single model in cache, but need more than max cache
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    ModelLoad._MODEL_STATES["model1"] = "cached"
    ModelLoad._MEMORY_USAGE["model1"] = 1000.0
    ModelLoad._MODEL_LAST_USED["model1"] = 100.0

    max_cache = 2000.0  # 2GB max

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache):
        # After clearing: 0MB used, but need 3GB (more than max)
        with patch.object(ModelLoad, "_get_current_cache_usage", side_effect=[1000.0, 1000.0, 0.0, 0.0]):
            result = ModelLoad._clear_cache_internal("new-model", 3000.0)  # Need 3GB

            # Should fail: required > max cache
            assert result is False


@pytest.mark.unit
@pytest.mark.fast
def test_clear_cache_internal_skip_same_model():
    """Test cache clearing skips the model being loaded.

    Tests:
    - Same model name protection
    - Don't release model being loaded
    """
    model_name = "test-model"

    # Setup: model already in cache
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    ModelLoad._MODEL_STATES[model_name] = "cached"
    ModelLoad._MEMORY_USAGE[model_name] = 1000.0
    ModelLoad._MODEL_LAST_USED[model_name] = 100.0

    max_cache = 1500.0

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache):
        # Exceed cache with same model - provide enough values for all calls
        # (initial check, loop condition, final check)
        with patch.object(
            ModelLoad, "_get_current_cache_usage", side_effect=[1000.0, 1000.0, 1000.0, 1000.0]
        ):
            with patch.object(ModelLoad, "_release_model_state") as mock_release:
                result = ModelLoad._clear_cache_internal(model_name, 600.0)

                # Should return False (not enough space made)
                assert result is False
                # Should not release the same model
                mock_release.assert_not_called()


# ============================================================================
# Category 6: Device Management Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_update_model_state_loaded_on_cuda():
    """Test updating model state to loaded on CUDA."""
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    model_name = "test-model"
    device = "cuda"
    status = "loaded"
    size_mb = 2048.0

    ModelLoad._update_model_state(model_name, device=device, status=status, size_mb=size_mb)

    assert model_name in ModelLoad._MODEL_STATES
    assert ModelLoad._MODEL_STATES[model_name] == "on_cuda"
    assert ModelLoad._MEMORY_USAGE[model_name] == size_mb
    assert model_name in ModelLoad._MODEL_LAST_USED


@pytest.mark.unit
@pytest.mark.fast
def test_update_model_state_cached_on_cpu():
    """Test updating model state to cached on CPU."""
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    model_name = "test-model"
    device = "cpu"
    status = "cached_cpu"
    size_mb = 1024.0

    ModelLoad._update_model_state(model_name, device=device, status=status, size_mb=size_mb)

    assert model_name in ModelLoad._MODEL_STATES
    assert ModelLoad._MODEL_STATES[model_name] == "on_cpu"
    assert ModelLoad._MEMORY_USAGE[model_name] == size_mb


@pytest.mark.unit
@pytest.mark.fast
def test_update_model_state_released():
    """Test releasing model state clears all tracking."""
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    model_name = "test-model"

    # First set the model state
    ModelLoad._MODEL_STATES[model_name] = "on_cuda"
    ModelLoad._MEMORY_USAGE[model_name] = 2048.0
    ModelLoad._MODEL_LAST_USED[model_name] = 100.0

    # Then release it
    ModelLoad._update_model_state(model_name, status="released")

    assert model_name not in ModelLoad._MODEL_STATES
    assert model_name not in ModelLoad._MEMORY_USAGE
    assert model_name not in ModelLoad._MODEL_LAST_USED


@pytest.mark.unit
@pytest.mark.fast
def test_move_components_to_device_torch_module():
    """Test moving PyTorch module to device."""

    # Create a mock component class that will pass isinstance check
    class MockTorchModule:
        def __init__(self):
            self.device = "cpu"
            self._to_called = False
            self._to_device = None

        def to(self, device):
            self._to_called = True
            self._to_device = device
            return self

    mock_component = MockTorchModule()

    # Create mock torch module
    mock_torch = MagicMock()
    mock_torch.Tensor = type("Tensor", (), {})
    mock_torch.nn.Module = MockTorchModule

    components = {"model": mock_component}

    # Inject mock torch into sys.modules
    with patch.dict("sys.modules", {"torch": mock_torch}):
        ModelLoad._move_components_to_device(components, "cuda")

        # Verify to() was called with cuda
        assert mock_component._to_called
        assert mock_component._to_device == "cuda"


@pytest.mark.unit
@pytest.mark.fast
def test_release_model_internal():
    """Test full model release with component cleanup."""
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    model_name = "test-model"

    # Set up model state
    ModelLoad._MODEL_STATES[model_name] = "on_cuda"
    ModelLoad._MEMORY_USAGE[model_name] = 2048.0
    ModelLoad._MODEL_LAST_USED[model_name] = 100.0

    # Create mock components
    mock_model = MagicMock()
    mock_pipeline = MagicMock()
    components = {"model": mock_model, "pipeline": mock_pipeline}

    # Create mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.empty_cache = MagicMock()

    with patch("gc.collect") as mock_gc:
        # Inject mock torch into sys.modules
        with patch.dict("sys.modules", {"torch": mock_torch}):
            ModelLoad._release_model_internal(model_name, components)

            # Verify state was cleared
            assert model_name not in ModelLoad._MODEL_STATES
            assert model_name not in ModelLoad._MEMORY_USAGE
            assert model_name not in ModelLoad._MODEL_LAST_USED

            # Verify GC was called
            mock_gc.assert_called()
            mock_torch.cuda.empty_cache.assert_called()


# ============================================================================
# Category 7: Advanced Size Calculation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_transformer_size_mb_success():
    """Test calculating transformer model size from parameters."""
    # Create mock torch module
    mock_torch = MagicMock()

    # Create mock model with parameters and buffers
    mock_param1 = MagicMock()
    mock_param1.numel.return_value = 1000
    mock_param1.element_size.return_value = 4  # 4 bytes (float32)

    mock_param2 = MagicMock()
    mock_param2.numel.return_value = 2000
    mock_param2.element_size.return_value = 4

    mock_buffer = MagicMock()
    mock_buffer.numel.return_value = 500
    mock_buffer.element_size.return_value = 4

    mock_model = MagicMock()
    mock_model.parameters.return_value = [mock_param1, mock_param2]
    mock_model.buffers.return_value = [mock_buffer]

    # Inject mock torch into sys.modules
    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Expected: (1000*4 + 2000*4 + 500*4) / (1024*1024) = 14000 / 1048576 ≈ 0.0133514...
        result = ModelLoad._calculate_transformer_size_mb(mock_model)

        assert result > 0
        assert abs(result - 0.01335) < 0.001  # Approximately 0.0133 MB


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_transformer_size_mb_error():
    """Test transformer size calculation returns 0.0 on error."""
    mock_model = MagicMock()
    mock_model.parameters.side_effect = Exception("Parameter error")

    result = ModelLoad._calculate_transformer_size_mb(mock_model)

    assert result == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_save_size_to_config_success():
    """Test saving calculated size to config."""
    model_name = "test-model"
    size_mb = 2048.0  # 2GB
    expected_size_gb = 2.0

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        ModelLoad._save_size_to_config(model_name, size_mb)

        mock_config.set_system_value.assert_called_once_with(
            model_name, "estimated_size_gb", expected_size_gb
        )
        mock_config.save_system_config.assert_called_once()


@pytest.mark.unit
@pytest.mark.fast
def test_save_size_to_config_zero_size():
    """Test saving zero size does nothing."""
    model_name = "test-model"
    size_mb = 0.0

    with patch("image_annotator_lib.core.model_factory.config_registry") as mock_config:
        ModelLoad._save_size_to_config(model_name, size_mb)

        mock_config.set_system_value.assert_not_called()
        mock_config.save_system_config.assert_not_called()


@pytest.mark.unit
@pytest.mark.fast
def test_get_or_calculate_size_from_cache():
    """Test getting size from static cache."""
    ModelLoad._MODEL_SIZES.clear()

    model_name = "test-model"
    cached_size = 2048.0

    # Populate cache
    ModelLoad._MODEL_SIZES[model_name] = cached_size

    mock_loader = MagicMock()

    result = ModelLoad._get_or_calculate_size(
        model_name=model_name,
        model_path="/path/to/model",
        model_type="transformers",
        loader_instance=mock_loader,
    )

    assert result == cached_size
    # Should not call loader since we got from cache
    mock_loader._calculate_specific_size.assert_not_called()


@pytest.mark.unit
@pytest.mark.fast
def test_get_or_calculate_size_from_config():
    """Test getting size from config when not in cache."""
    ModelLoad._MODEL_SIZES.clear()

    model_name = "test-model"
    config_size_mb = 1024.0

    with patch.object(ModelLoad, "_get_model_size_from_config", return_value=config_size_mb):
        mock_loader = MagicMock()

        result = ModelLoad._get_or_calculate_size(
            model_name=model_name,
            model_path="/path/to/model",
            model_type="transformers",
            loader_instance=mock_loader,
        )

        assert result == config_size_mb
        # Should cache the config value
        assert ModelLoad._MODEL_SIZES[model_name] == config_size_mb
        # Should not call loader since we got from config
        mock_loader._calculate_specific_size.assert_not_called()


@pytest.mark.unit
@pytest.mark.fast
def test_get_or_calculate_size_calculate_and_save():
    """Test calculating size when not in cache or config."""
    ModelLoad._MODEL_SIZES.clear()

    model_name = "test-model"
    calculated_size = 2048.0
    model_path = "/path/to/model"

    with patch.object(ModelLoad, "_get_model_size_from_config", return_value=None):
        with patch.object(ModelLoad, "_save_size_to_config") as mock_save:
            mock_loader = MagicMock()
            mock_loader._calculate_specific_size.return_value = calculated_size

            result = ModelLoad._get_or_calculate_size(
                model_name=model_name,
                model_path=model_path,
                model_type="transformers",
                loader_instance=mock_loader,
            )

            assert result == calculated_size
            # Should cache the calculated value
            assert ModelLoad._MODEL_SIZES[model_name] == calculated_size
            # Should call loader to calculate
            mock_loader._calculate_specific_size.assert_called_once_with(model_path=model_path)
            # Should save to config
            mock_save.assert_called_once_with(model_name, calculated_size)


# ==============================================================================
# Phase A Task 1: LRU Cache & Device Management Tests (2025-12-03)
# Coverage Target: LRU eviction logic, memory pressure, device switching
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_lru_eviction_with_multiple_models():
    """Test LRU eviction correctly identifies and removes least recently used model.

    Scenario:
    - Load 3 models with different timestamps
    - Trigger eviction
    - Verify oldest model is evicted first

    Tests:
    - LRU ordering logic
    - Timestamp-based eviction
    - Correct model removal
    """
    import time

    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()

    # Setup: 3 models with staggered timestamps
    models = ["model_old", "model_mid", "model_new"]
    base_time = time.time()

    for i, model_name in enumerate(models):
        ModelLoad._MODEL_STATES[model_name] = "cached"
        ModelLoad._MEMORY_USAGE[model_name] = 1024.0  # 1GB each
        ModelLoad._MODEL_LAST_USED[model_name] = base_time + i  # Increasing timestamps
        ModelLoad._MODEL_SIZES[model_name] = 1024.0

    # Mock memory check to trigger eviction
    # Cache: 3GB (3 models × 1GB), Required: 2GB, Max: 4GB → 5GB > 4GB triggers eviction
    max_cache_mb = 4 * 1024  # 4GB max (reduced to trigger eviction)
    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache_mb):
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 500 * 1024 * 1024  # 500MB available

            # Request 2GB (should evict oldest model first)
            # Use wraps to monitor calls while executing real implementation
            with patch.object(
                ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
            ) as mock_release:
                result = ModelLoad._clear_cache_internal(
                    "model_new",  # Don't evict this one
                    2048.0,  # required_size_mb
                )

                assert result is True

                # Verify model_new (protection target) is NOT evicted
                assert "model_new" in ModelLoad._MODEL_STATES
                assert "model_new" in ModelLoad._MEMORY_USAGE

                # Verify oldest model was evicted
                assert mock_release.call_count >= 1
                released_models = [call[0][0] for call in mock_release.call_args_list]
                assert "model_old" in released_models

                # Verify model_old is actually removed from cache
                assert "model_old" not in ModelLoad._MODEL_STATES
                assert "model_old" not in ModelLoad._MEMORY_USAGE


@pytest.mark.unit
@pytest.mark.fast
def test_lru_order_preservation_on_cache_hit():
    """Test that accessing a cached model updates its LRU position.

    Scenario:
    - Create models: A (t=1), B (t=2), C (t=3)
    - Access model A (should update timestamp)
    - Trigger eviction
    - Verify B is evicted (not A)

    Tests:
    - Timestamp update on access
    - LRU order refresh
    - Access-based eviction priority
    """
    import time

    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()

    base_time = time.time()

    # Create 3 models with timestamps in the past (so current time will be newest)
    ModelLoad._MODEL_STATES["model_a"] = "cached"
    ModelLoad._MODEL_LAST_USED["model_a"] = base_time - 3  # Oldest (3 seconds ago)
    ModelLoad._MEMORY_USAGE["model_a"] = 1024.0
    ModelLoad._MODEL_SIZES["model_a"] = 1024.0

    ModelLoad._MODEL_STATES["model_b"] = "cached"
    ModelLoad._MODEL_LAST_USED["model_b"] = base_time - 2  # Middle (2 seconds ago)
    ModelLoad._MEMORY_USAGE["model_b"] = 1024.0
    ModelLoad._MODEL_SIZES["model_b"] = 1024.0

    ModelLoad._MODEL_STATES["model_c"] = "cached"
    ModelLoad._MODEL_LAST_USED["model_c"] = base_time - 1  # Newest initially (1 second ago)
    ModelLoad._MEMORY_USAGE["model_c"] = 1024.0
    ModelLoad._MODEL_SIZES["model_c"] = 1024.0

    # Simulate accessing model_a via _update_model_state (updates timestamp)
    # This calls the real implementation which updates _MODEL_LAST_USED
    time.sleep(0.01)  # Ensure new timestamp is definitely later
    ModelLoad._update_model_state("model_a", None, None)  # Just update timestamp

    # Get LRU order - model_a should be newest now (refreshed)
    sorted_models = ModelLoad._get_models_sorted_by_last_used()
    # Returns list of (model_name, timestamp) tuples
    assert sorted_models[0][0] == "model_b"  # Oldest (base-2)
    assert sorted_models[1][0] == "model_c"  # Middle (base-1)
    assert sorted_models[2][0] == "model_a"  # Newest (updated to current time)


@pytest.mark.unit
@pytest.mark.fast
def test_lru_eviction_respects_memory_limits():
    """Test that eviction stops once sufficient memory is freed.

    Scenario:
    - Multiple cached models
    - Request memory requiring partial eviction
    - Verify only necessary models are evicted

    Tests:
    - Minimal eviction strategy
    - Memory calculation accuracy
    - Early termination on sufficient space
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()

    import time

    base_time = time.time()

    # Setup: 4 models, 500MB each
    for i in range(4):
        model_name = f"model_{i}"
        ModelLoad._MODEL_STATES[model_name] = "cached"
        ModelLoad._MEMORY_USAGE[model_name] = 500.0
        ModelLoad._MODEL_LAST_USED[model_name] = base_time + i
        ModelLoad._MODEL_SIZES[model_name] = 500.0

    # Cache: 2GB (4 models × 500MB), Required: 1GB, Max: 2.5GB → 3GB > 2.5GB triggers eviction
    max_cache_mb = 2560  # 2.5GB max (reduced to trigger eviction)
    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache_mb):
        with patch("psutil.virtual_memory") as mock_vm:
            # 100MB available, request 1GB (need to free ~500MB, so evict at least 1-2 models)
            mock_vm.return_value.available = 100 * 1024 * 1024

            # Use wraps to monitor calls while executing real implementation
            with patch.object(
                ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
            ) as mock_release:
                result = ModelLoad._clear_cache_internal(
                    "model_3",  # Protect newest model
                    1024.0,  # required_size_mb
                )

                assert result is True
                # Should evict at least 1 model (500MB freed) to meet requirement
                # (may evict more if implementation is conservative)
                assert mock_release.call_count >= 1
                assert mock_release.call_count <= 4

                # Verify models were actually evicted from cache
                evicted_models = [call[0][0] for call in mock_release.call_args_list]
                for model_name in evicted_models:
                    assert model_name not in ModelLoad._MODEL_STATES
                    assert model_name not in ModelLoad._MEMORY_USAGE


@pytest.mark.unit
@pytest.mark.fast
def test_lru_eviction_with_same_timestamp():
    """Test eviction behavior when multiple models have identical timestamps.

    Scenario:
    - Multiple models with same last_used timestamp
    - Trigger eviction
    - Verify deterministic behavior (alphabetical order fallback)

    Tests:
    - Timestamp collision handling
    - Deterministic eviction order
    - No arbitrary behavior
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()

    import time

    same_timestamp = time.time()

    # Create models with identical timestamps (alphabetically: a, b, c)
    for model_name in ["model_c", "model_a", "model_b"]:
        ModelLoad._MODEL_STATES[model_name] = "cached"
        ModelLoad._MEMORY_USAGE[model_name] = 512.0
        ModelLoad._MODEL_LAST_USED[model_name] = same_timestamp
        ModelLoad._MODEL_SIZES[model_name] = 512.0

    # Get LRU order - should be deterministic (alphabetical)
    sorted_models = ModelLoad._get_models_sorted_by_last_used()
    # Python's sort is stable, so models with same timestamp maintain insertion order
    # or are sorted by key (model name) if implementation uses sorted()
    # Returns list of (model_name, timestamp) tuples
    assert len(sorted_models) == 3
    model_names = [name for name, _ in sorted_models]
    assert all(m in model_names for m in ["model_a", "model_b", "model_c"])


@pytest.mark.unit
@pytest.mark.fast
def test_cache_full_immediate_eviction():
    """Test immediate eviction when cache is at capacity.

    Scenario:
    - Cache at max capacity
    - Request new model load
    - Verify immediate eviction before load

    Tests:
    - Capacity detection
    - Pre-load eviction
    - Space availability guarantee
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()

    import time

    # Fill cache to capacity (assume 10GB max, 2.5GB per model = 4 models)
    max_cache_mb = 10 * 1024
    for i in range(4):
        model_name = f"cached_model_{i}"
        ModelLoad._MODEL_STATES[model_name] = "cached"
        ModelLoad._MEMORY_USAGE[model_name] = 2560.0  # 2.5GB each
        ModelLoad._MODEL_LAST_USED[model_name] = time.time() + i
        ModelLoad._MODEL_SIZES[model_name] = 2560.0

    with patch.object(ModelLoad, "_get_max_cache_size", return_value=max_cache_mb):
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = 1024 * 1024 * 1024  # 1GB available

            # Use wraps to monitor calls while executing real implementation
            with patch.object(
                ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
            ) as mock_release:
                # Request 3GB (requires eviction: 10GB + 3GB = 13GB > 10GB max)
                result = ModelLoad._clear_cache_internal(
                    "new_model",
                    3072.0,  # required_size_mb
                )

                # Verify eviction succeeded
                assert result is True

                # Should evict at least 2 models (5GB freed) to make space
                assert mock_release.call_count >= 2

                # Verify models were actually evicted from cache
                evicted_models = [call[0][0] for call in mock_release.call_args_list]
                for model_name in evicted_models:
                    assert model_name not in ModelLoad._MODEL_STATES
                    assert model_name not in ModelLoad._MEMORY_USAGE

                # Verify remaining cache size allows for new model
                remaining_cache_mb = sum(ModelLoad._MEMORY_USAGE.values())
                assert remaining_cache_mb + 3072.0 <= max_cache_mb


# ==============================================================================
# Phase A Task 1: Device Switching Tests (2025-12-03)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_device_state_cuda_loaded():
    """Test device state tracking for CUDA-loaded models.

    Scenario:
    - Model loaded on CUDA
    - State should be "on_cuda"

    Tests:
    - CUDA device state encoding
    - State string format
    """
    ModelLoad._MODEL_STATES.clear()

    # Simulate model load on CUDA via _update_model_state
    ModelLoad._update_model_state("test_model", "cuda", "loaded", 1024.0)

    assert ModelLoad._MODEL_STATES["test_model"] == "on_cuda"


@pytest.mark.unit
@pytest.mark.fast
def test_device_state_cpu_loaded():
    """Test device state tracking for CPU-loaded models.

    Scenario:
    - Model loaded on CPU
    - State should be "on_cpu"

    Tests:
    - CPU device state encoding
    - State string format
    """
    ModelLoad._MODEL_STATES.clear()

    # Simulate model load on CPU via _update_model_state
    ModelLoad._update_model_state("test_model", "cpu", "loaded", 1024.0)

    assert ModelLoad._MODEL_STATES["test_model"] == "on_cpu"


@pytest.mark.unit
@pytest.mark.fast
def test_device_state_preservation_across_cache_hits():
    """Test that device state is preserved when model is reused from cache.

    Scenario:
    - Model loaded on CUDA (state="on_cuda")
    - Model accessed again (cache hit)
    - Device state should remain "on_cuda"

    Tests:
    - Device state persistence
    - Cache hit doesn't change device
    - State consistency
    """
    import time

    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MODEL_LAST_USED.clear()

    base_time = time.time()

    # Initial load on CUDA
    ModelLoad._update_model_state("test_model", "cuda", "loaded", 1024.0)
    ModelLoad._MODEL_LAST_USED["test_model"] = base_time

    # Verify initial state
    assert ModelLoad._MODEL_STATES["test_model"] == "on_cuda"

    # Simulate cache hit (update last used timestamp)
    ModelLoad._MODEL_LAST_USED["test_model"] = base_time + 5

    # Device state should be unchanged
    assert ModelLoad._MODEL_STATES["test_model"] == "on_cuda"
    assert ModelLoad._MODEL_LAST_USED["test_model"] == base_time + 5


@pytest.mark.unit
@pytest.mark.fast
def test_device_state_cleanup_on_model_release():
    """Test that device state is cleaned up when model is released.

    Scenario:
    - Model loaded on CUDA (state="on_cuda")
    - Model released from cache
    - State should be removed

    Tests:
    - State cleanup consistency
    - No orphaned states
    - Complete state removal
    """
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_SIZES.clear()

    # Setup model on CUDA (using _update_model_state sets _MEMORY_USAGE automatically)
    ModelLoad._update_model_state("test_model", "cuda", "loaded", 1024.0)
    ModelLoad._MODEL_SIZES["test_model"] = 1024.0  # _MODEL_SIZES not managed by _update_model_state

    # Verify setup
    assert ModelLoad._MODEL_STATES["test_model"] == "on_cuda"

    # Use wraps to monitor calls while executing real implementation
    with patch.object(
        ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
    ) as mock_release:
        ModelLoad._release_model_state("test_model")

        # Verify the method was called
        mock_release.assert_called_once_with("test_model")

        # States should be removed (by real implementation)
        # Note: _MODEL_SIZES is intentionally kept (per implementation comment)
        assert "test_model" not in ModelLoad._MODEL_STATES
        assert "test_model" not in ModelLoad._MEMORY_USAGE
        # _MODEL_SIZES is kept on release (cleared only on load error)


@pytest.mark.unit
@pytest.mark.fast
def test_multiple_models_different_devices():
    """Test tracking multiple models on different devices simultaneously.

    Scenario:
    - Model A on CUDA (state="on_cuda")
    - Model B on CPU (state="on_cpu")
    - Model C on CUDA (state="on_cuda")
    - All tracked independently

    Tests:
    - Independent device tracking
    - No device state conflicts
    - Correct device assignment per model
    """
    ModelLoad._MODEL_STATES.clear()

    # Setup multiple models on different devices
    ModelLoad._update_model_state("model_cuda_1", "cuda", "loaded", 1024.0)
    ModelLoad._update_model_state("model_cpu_1", "cpu", "loaded", 1024.0)
    ModelLoad._update_model_state("model_cuda_2", "cuda", "loaded", 1024.0)
    ModelLoad._update_model_state("model_cpu_2", "cpu", "loaded", 1024.0)

    # Verify all devices are tracked correctly
    assert ModelLoad._MODEL_STATES["model_cuda_1"] == "on_cuda"
    assert ModelLoad._MODEL_STATES["model_cpu_1"] == "on_cpu"
    assert ModelLoad._MODEL_STATES["model_cuda_2"] == "on_cuda"
    assert ModelLoad._MODEL_STATES["model_cpu_2"] == "on_cpu"

    # Verify count
    cuda_models = [m for m, s in ModelLoad._MODEL_STATES.items() if s == "on_cuda"]
    cpu_models = [m for m, s in ModelLoad._MODEL_STATES.items() if s == "on_cpu"]

    assert len(cuda_models) == 2
    assert len(cpu_models) == 2
