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
    with patch("image_annotator_lib.core.model_factory.torch") as mock_torch:
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

        # Set up torch mock to make isinstance check work
        mock_torch.Tensor = type("Tensor", (), {})
        mock_torch.nn.Module = MockTorchModule

        components = {"model": mock_component}

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

    with patch("gc.collect") as mock_gc:
        with patch("image_annotator_lib.core.model_factory.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()

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
    with patch("image_annotator_lib.core.model_factory.torch"):
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
