"""Unit tests for ModelLoad memory management functionality.

このモジュールでは、ModelLoadクラスのメモリ管理機能をテストします。

Test Categories:
1. Cache Usage Calculation Tests
2. LRU (Least Recently Used) Management Tests
3. Memory Availability Tests
4. Cache Eviction Tests
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.core.model_factory import ModelLoad

# ==============================================================================
# Category 1: Cache Usage Calculation Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_current_cache_usage_empty():
    """Test cache usage calculation when no models are loaded.

    Tests:
    - Empty cache returns 0
    - Correct usage of _MEMORY_USAGE dictionary
    """
    ModelLoad._MEMORY_USAGE.clear()

    usage = ModelLoad._get_current_cache_usage()

    assert usage == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_get_current_cache_usage_multiple_models():
    """Test cache usage calculation with multiple models loaded.

    Tests:
    - Sum of all model memory usage
    - Correct calculation with multiple entries
    """
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MEMORY_USAGE["model1"] = 1024.0  # 1GB in MB
    ModelLoad._MEMORY_USAGE["model2"] = 2048.0  # 2GB in MB
    ModelLoad._MEMORY_USAGE["model3"] = 512.0   # 0.5GB in MB

    usage = ModelLoad._get_current_cache_usage()

    expected = 1024.0 + 2048.0 + 512.0
    assert usage == expected


# ==============================================================================
# Category 2: LRU (Least Recently Used) Management Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_models_sorted_by_last_used():
    """Test LRU sorting of models by last used timestamp.

    Tests:
    - Models sorted by timestamp (oldest first)
    - Correct ordering of multiple models
    """
    ModelLoad._MODEL_LAST_USED.clear()

    current_time = time.time()
    ModelLoad._MODEL_LAST_USED["model_new"] = current_time
    ModelLoad._MODEL_LAST_USED["model_old"] = current_time - 100
    ModelLoad._MODEL_LAST_USED["model_middle"] = current_time - 50

    sorted_models = ModelLoad._get_models_sorted_by_last_used()

    assert len(sorted_models) == 3
    assert sorted_models[0][0] == "model_old"  # Oldest first
    assert sorted_models[1][0] == "model_middle"
    assert sorted_models[2][0] == "model_new"  # Newest last


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_memory_usage():
    """Test retrieval of specific model memory usage.

    Tests:
    - Existing model returns correct usage
    - Non-existent model returns 0.0
    """
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MEMORY_USAGE["test_model"] = 1536.0  # 1.5GB in MB

    usage = ModelLoad._get_model_memory_usage("test_model")
    assert usage == 1536.0

    usage_nonexistent = ModelLoad._get_model_memory_usage("nonexistent")
    assert usage_nonexistent == 0.0


# ==============================================================================
# Category 3: Memory Availability Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_sufficient():
    """Test memory check when sufficient memory is available.

    Tests:
    - Returns True when enough memory available
    - Correct memory comparison
    """
    model_size_mb = 1024.0  # 1GB

    with patch("image_annotator_lib.core.model_factory.psutil.virtual_memory") as mock_vmem:
        mock_vmem.return_value = MagicMock(available=5 * 1024 * 1024 * 1024)  # 5GB available

        result = ModelLoad._check_memory_before_load(model_size_mb, "test_model")

        assert result is True


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_insufficient():
    """Test memory check when insufficient memory is available.

    Tests:
    - Returns False when not enough memory
    - Correct warning logged
    """
    model_size_mb = 5 * 1024.0  # 5GB

    with patch("image_annotator_lib.core.model_factory.psutil.virtual_memory") as mock_vmem:
        mock_vmem.return_value = MagicMock(available=1 * 1024 * 1024 * 1024)  # 1GB available

        result = ModelLoad._check_memory_before_load(model_size_mb, "test_model")

        assert result is False


@pytest.mark.unit
@pytest.mark.fast
def test_check_memory_before_load_invalid_size():
    """Test memory check with invalid model size.

    Tests:
    - Returns True when size is 0 or negative (skip check)
    - Correct handling of edge case
    """
    result_zero = ModelLoad._check_memory_before_load(0, "test_model")
    assert result_zero is True

    result_negative = ModelLoad._check_memory_before_load(-100, "test_model")
    assert result_negative is True


# ==============================================================================
# Category 4: Cache Eviction Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_max_cache_size():
    """Test maximum cache size calculation.

    Tests:
    - Correct calculation based on total memory and cache ratio
    - Returns value in MB
    """
    with patch("image_annotator_lib.core.model_factory.psutil.virtual_memory") as mock_vmem:
        mock_vmem.return_value = MagicMock(total=16 * 1024 * 1024 * 1024)  # 16GB

        max_cache = ModelLoad._get_max_cache_size()

        expected = (16 * 1024 * 1024 * 1024) / (1024 * 1024) * ModelLoad._CACHE_RATIO
        assert max_cache == pytest.approx(expected)
