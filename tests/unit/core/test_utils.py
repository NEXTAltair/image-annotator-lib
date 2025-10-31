"""Unit tests for core utility functions.

このモジュールでは、image_annotator_lib.core.utilsの各種ユーティリティ関数をテストします。

Test Categories:
1. Image Hash Calculation Tests
2. File Caching Tests
3. File Path Resolution Tests
4. ZIP Extraction Tests
5. Device Determination Tests
6. Timestamp Conversion Tests
"""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core import utils

# ==============================================================================
# Category 1: Image Hash Calculation Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_phash_valid_image():
    """Test pHash calculation for valid image.

    Tests:
    - pHash calculation returns string
    - String has expected length (16 hex characters)
    - Result is consistent for same image
    """
    # Create simple test image
    image = Image.new("RGB", (100, 100), color="red")

    phash1 = utils.calculate_phash(image)
    phash2 = utils.calculate_phash(image)

    assert isinstance(phash1, str)
    assert len(phash1) == 16  # pHash is 16 hex characters
    assert phash1 == phash2  # Consistent results


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_phash_converts_to_rgb():
    """Test pHash calculation with non-RGB image.

    Tests:
    - RGBA image is converted to RGB
    - pHash calculation succeeds
    """
    # Create RGBA image
    image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

    phash = utils.calculate_phash(image)

    assert isinstance(phash, str)
    assert len(phash) == 16


# ==============================================================================
# Category 2: File Caching Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_cache_path_with_valid_filename():
    """Test cache path generation with valid URL filename.

    Tests:
    - Cache path is generated correctly
    - Filename is extracted from URL
    """
    url = "https://example.com/models/model.onnx"
    cache_dir = Path("/tmp/cache")

    cache_path = utils._get_cache_path(url, cache_dir)

    assert cache_path == cache_dir / "model.onnx"


@pytest.mark.unit
@pytest.mark.fast
def test_get_cache_path_with_short_filename():
    """Test cache path generation with short/missing filename.

    Tests:
    - Short filenames trigger MD5 hash generation
    - Extension is preserved if available
    """
    url = "https://example.com/a.b"
    cache_dir = Path("/tmp/cache")

    cache_path = utils._get_cache_path(url, cache_dir)

    # Should use MD5 hash for short filename
    assert cache_path.parent == cache_dir
    assert len(cache_path.stem) == 32  # MD5 hash length


@pytest.mark.unit
@pytest.mark.fast
def test_is_cached_returns_correct_status():
    """Test cache existence check.

    Tests:
    - Returns False for non-existent file
    - Returns correct Path object
    """
    url = "https://example.com/model.onnx"
    cache_dir = Path("/tmp/nonexistent")

    is_cached, cache_path = utils._is_cached(url, cache_dir)

    assert is_cached is False
    assert isinstance(cache_path, Path)


# ==============================================================================
# Category 3: File Path Resolution Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_file_path_local_file(tmp_path):
    """Test local file path resolution.

    Tests:
    - Existing local file returns absolute path
    - Path is resolved correctly
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    result = utils.get_file_path(str(test_file))

    assert result.is_absolute()
    assert result.exists()


@pytest.mark.unit
@pytest.mark.fast
def test_get_file_path_http_url():
    """Test HTTP URL file path resolution.

    Tests:
    - HTTP URL triggers download
    - Returns path to cached file
    """
    url = "https://example.com/model.onnx"

    with patch.object(utils, "_download_from_url") as mock_download:
        mock_download.return_value = Path("/cache/model.onnx")

        result = utils.get_file_path(url, cache_dir=Path("/cache"))

        mock_download.assert_called_once_with(url, Path("/cache"))
        assert result == Path("/cache/model.onnx")


@pytest.mark.unit
@pytest.mark.fast
def test_get_file_path_invalid_local_path():
    """Test local file path with non-existent file.

    Tests:
    - Non-existent local file raises FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        utils.get_file_path("/nonexistent/file.txt")


# ==============================================================================
# Category 4: ZIP Extraction Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_extract_zip_success(tmp_path):
    """Test successful ZIP extraction.

    Tests:
    - ZIP file is extracted correctly
    - Returns extraction directory path
    - Directory contains extracted files
    """
    # Create test ZIP file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "content")

    extract_dir = utils.extract_zip(zip_path)

    assert extract_dir.exists()
    assert extract_dir.is_dir()
    assert (extract_dir / "test.txt").exists()


@pytest.mark.unit
@pytest.mark.fast
def test_extract_zip_invalid_file(tmp_path):
    """Test ZIP extraction with invalid file.

    Tests:
    - Invalid ZIP file raises RuntimeError
    - Error message is descriptive
    """
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("not a zip file")

    with pytest.raises(RuntimeError, match="ZIPファイルの解凍に失敗しました"):
        utils.extract_zip(invalid_zip)


# ==============================================================================
# Category 5: Device Determination Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_cuda_available():
    """Test device determination when CUDA is available.

    Tests:
    - Requested CUDA device is returned when available
    - No fallback to CPU
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = utils.determine_effective_device("cuda", "test_model")

        assert result == "cuda"


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_cuda_unavailable():
    """Test device determination when CUDA is unavailable.

    Tests:
    - Falls back to CPU when CUDA not available
    - Warning is logged
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = utils.determine_effective_device("cuda", "test_model")

        assert result == "cpu"


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_invalid_device_index():
    """Test device determination with invalid CUDA device index.

    Tests:
    - Falls back to default CUDA device (cuda:0) when index invalid
    - Warning is logged
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1  # Only 1 device (cuda:0)

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = utils.determine_effective_device("cuda:5", "test_model")

        assert result == "cuda"  # Falls back to default


# ==============================================================================
# Category 6: Timestamp Conversion Tests
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_convert_unix_to_iso8601_valid_timestamp():
    """Test Unix timestamp to ISO8601 conversion.

    Tests:
    - Valid timestamp converts correctly
    - Format is ISO8601 with Z suffix
    """
    # 2024-01-01 00:00:00 UTC
    timestamp = 1704067200

    result = utils.convert_unix_to_iso8601(timestamp, "test_model")

    assert result == "2024-01-01T00:00:00Z"


@pytest.mark.unit
@pytest.mark.fast
def test_convert_unix_to_iso8601_none_timestamp():
    """Test Unix timestamp conversion with None input.

    Tests:
    - None timestamp returns "Invalid Timestamp"
    - No exception is raised
    """
    result = utils.convert_unix_to_iso8601(None, "test_model")

    assert result == "Invalid Timestamp"
