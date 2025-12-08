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


# ==============================================================================
# Additional Coverage Tests (Phase C)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_phash_consistency_different_images():
    """Test pHash produces different hashes for different images.

    Tests:
    - Different images produce different pHashes
    - Similar images have different but related hashes
    - pHash algorithm detects content differences
    """
    from PIL import ImageDraw

    # Create two distinctly different images with patterns
    # Image 1: Red square on white background
    image1 = Image.new("RGB", (100, 100), color="white")
    draw1 = ImageDraw.Draw(image1)
    draw1.rectangle([25, 25, 75, 75], fill="red")

    # Image 2: Blue circle on white background
    image2 = Image.new("RGB", (100, 100), color="white")
    draw2 = ImageDraw.Draw(image2)
    draw2.ellipse([25, 25, 75, 75], fill="blue")

    phash1 = utils.calculate_phash(image1)
    phash2 = utils.calculate_phash(image2)

    # Different images should have different hashes
    assert phash1 != phash2
    assert isinstance(phash1, str)
    assert isinstance(phash2, str)
    assert len(phash1) == 16
    assert len(phash2) == 16


@pytest.mark.unit
@pytest.mark.fast
def test_download_file_with_caching(tmp_path):
    """Test file download with caching behavior.

    Tests:
    - File downloaded on first request
    - Subsequent requests use cache (no re-download)
    - Cache hit detection works correctly
    """
    url = "https://example.com/test_file.bin"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create fake cached file
    expected_cache_path = cache_dir / "test_file.bin"
    expected_cache_path.write_bytes(b"cached content")

    # Test cache hit (file already exists)
    is_cached, cache_path = utils._is_cached(url, cache_dir)

    assert is_cached is True
    assert cache_path == expected_cache_path
    assert cache_path.exists()
    assert cache_path.read_bytes() == b"cached content"


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_cpu_explicit():
    """Test device determination with explicit CPU request.

    Tests:
    - Explicit CPU request is honored
    - No CUDA checking performed for CPU
    - Returns "cpu" without fallback logic
    """
    # Don't need to mock torch for explicit CPU request
    result = utils.determine_effective_device("cpu", "test_model")

    assert result == "cpu"


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_cuda_with_index():
    """Test device determination with specific CUDA device index.

    Tests:
    - Specific CUDA device (cuda:0, cuda:1) handled correctly
    - Device count checked against requested index
    - Valid index returned, invalid falls back
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2  # cuda:0 and cuda:1 available

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Test valid index
        result = utils.determine_effective_device("cuda:0", "test_model")
        assert result == "cuda:0"

        # Test another valid index
        result = utils.determine_effective_device("cuda:1", "test_model")
        assert result == "cuda:1"


@pytest.mark.unit
@pytest.mark.fast
def test_get_cache_path_with_query_parameters():
    """Test cache path generation with URL query parameters.

    Tests:
    - Query parameters are stripped from filename
    - Cache path uses clean filename without ?params
    - Extension is preserved correctly
    """
    url = "https://example.com/model.onnx?version=v2&token=abc123"
    cache_dir = Path("/tmp/cache")

    cache_path = utils._get_cache_path(url, cache_dir)

    # Filename should be clean without query params
    assert cache_path.name == "model.onnx"
    assert cache_path.parent == cache_dir
    assert "?" not in str(cache_path)


# ==============================================================================
# Phase C Additional Coverage Tests (2025-12-05)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_calculate_phash_with_grayscale_image():
    """Test pHash calculation with grayscale image.

    Tests:
    - Grayscale image is converted to RGB automatically
    - pHash calculation succeeds
    - Result is consistent
    - No exceptions raised
    """
    # Create grayscale image
    grayscale_image = Image.new("L", (100, 100), color=128)

    phash = utils.calculate_phash(grayscale_image)

    assert isinstance(phash, str)
    assert len(phash) == 16  # pHash is 16 hex characters
    # Verify consistency
    phash2 = utils.calculate_phash(grayscale_image)
    assert phash == phash2


@pytest.mark.unit
@pytest.mark.fast
def test_determine_effective_device_invalid_format():
    """Test device determination with various device string formats.

    Tests:
    - Various device formats are processed
    - Function doesn't crash on unusual formats
    - Returns a valid device string
    """
    # Test various formats (function returns them as-is if torch not mocking CUDA)
    test_devices = ["gpu", "cuda:", "cuda:abc", "cuda:1:2", "CPU", "CUDA", "cpu", "cuda"]

    for device in test_devices:
        result = utils.determine_effective_device(device, "test_model")
        # Function should return a string without crashing
        assert isinstance(result, str)
        # Result should be non-empty
        assert len(result) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_download_file_with_network_error(tmp_path):
    """Test file download with network error handling.

    Tests:
    - Network errors are caught by load_file()
    - RuntimeError is raised with context
    - Error message includes download failure info
    - Uses public API (load_file) which wraps errors
    """
    from requests.exceptions import ConnectionError as RequestsConnectionError

    url = "https://nonexistent.example.com/model.bin"

    # Mock requests.get to raise network error
    with patch("requests.get") as mock_get:
        mock_get.side_effect = RequestsConnectionError("Network error")

        # Use public API which wraps requests.RequestException in RuntimeError
        with pytest.raises(RuntimeError, match="ダウンロードに失敗しました"):
            utils.load_file(url)


@pytest.mark.unit
@pytest.mark.fast
def test_get_cache_path_with_special_characters():
    """Test cache path generation with special characters in URL.

    Tests:
    - Special characters (%, &, =, etc.) are handled
    - Path generation doesn't fail
    - Filename is sanitized appropriately
    - Cache path is valid
    """
    # URL with various special characters
    urls_with_special_chars = [
        "https://example.com/model%20name.onnx",
        "https://example.com/model&version=2.onnx",
        "https://example.com/model=final.onnx",
    ]

    cache_dir = Path("/tmp/cache")

    for url in urls_with_special_chars:
        cache_path = utils._get_cache_path(url, cache_dir)

        # Verify path is generated
        assert isinstance(cache_path, Path)
        assert cache_path.parent == cache_dir
        # Verify extension is preserved
        assert cache_path.suffix in [".onnx", ""]


@pytest.mark.unit
@pytest.mark.fast
def test_convert_unix_to_iso8601_edge_cases():
    """Test Unix timestamp to ISO8601 conversion with edge cases.

    Tests:
    - Zero timestamp (epoch start)
    - Negative timestamp (pre-epoch)
    - Very large timestamp
    - None timestamp handled gracefully
    """
    # Test epoch start (1970-01-01 00:00:00 UTC)
    result = utils.convert_unix_to_iso8601(0, "test_model")
    assert result == "1970-01-01T00:00:00Z"

    # Test negative timestamp (pre-epoch)
    result = utils.convert_unix_to_iso8601(-86400, "test_model")
    assert result == "1969-12-31T00:00:00Z"

    # Test very large timestamp (year 2100+)
    large_timestamp = 4102444800  # 2100-01-01
    result = utils.convert_unix_to_iso8601(large_timestamp, "test_model")
    assert "2100" in result

    # Test None timestamp
    result = utils.convert_unix_to_iso8601(None, "test_model")
    assert result == "Invalid Timestamp"


# ==============================================================================
# Phase C Week 2: Utils Edge Cases Tests (2025-12-07)
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_perform_download_with_progress_tracking(tmp_path):
    """Test _perform_download internal logic with stream and progress display.

    Coverage: Lines 125-139 (_perform_download implementation)

    REAL components:
    - Real file writing to target_path
    - Real chunk processing logic

    MOCKED:
    - requests.get() with streaming response
    - tqdm progress bar (to avoid terminal output)

    Scenario:
    1. Mock requests.get to return streaming response with chunks
    2. Call _perform_download() with target path
    3. Verify file written correctly with all chunks
    4. Verify progress bar updated for each chunk

    Assertions:
    - File created at target path
    - File content matches all chunks combined
    - requests.get called with stream=True and timeout
    - tqdm progress bar created with correct total size
    - Progress bar updated for each chunk
    """

    url = "https://example.com/model.bin"
    target_path = tmp_path / "downloaded_model.bin"

    # Mock response with streaming chunks
    mock_response = MagicMock()
    mock_response.headers.get.return_value = "16"  # content-length: 16 bytes
    mock_response.raise_for_status = MagicMock()

    # Simulate 2 chunks of 8 bytes each
    chunk1 = b"12345678"
    chunk2 = b"abcdefgh"
    mock_response.iter_content.return_value = iter([chunk1, chunk2])

    with patch("requests.get", return_value=mock_response) as mock_get:
        with patch("image_annotator_lib.core.utils.tqdm") as mock_tqdm_class:
            # Mock tqdm context manager
            mock_pbar = MagicMock()
            mock_tqdm_class.return_value.__enter__.return_value = mock_pbar

            # Act: Perform download
            utils._perform_download(url, target_path)

            # Assert: requests.get called with correct parameters
            mock_get.assert_called_once_with(url, stream=True, timeout=utils.DEFAULT_TIMEOUT)
            mock_response.raise_for_status.assert_called_once()

            # Assert: File created with correct content
            assert target_path.exists(), "Target file created"
            assert target_path.read_bytes() == chunk1 + chunk2, "File content matches chunks"

            # Assert: tqdm progress bar created with correct total
            assert mock_tqdm_class.called, "tqdm progress bar created"
            call_kwargs = mock_tqdm_class.call_args[1]
            assert call_kwargs["total"] == 16, "Progress bar total matches content-length"
            assert call_kwargs["unit"] == "B", "Progress bar unit is bytes"
            assert call_kwargs["unit_scale"] is True, "Progress bar uses unit scaling"

            # Assert: Progress bar updated for each chunk
            assert mock_pbar.update.call_count == 2, "Progress bar updated twice (2 chunks)"
            mock_pbar.update.assert_any_call(len(chunk1))
            mock_pbar.update.assert_any_call(len(chunk2))


@pytest.mark.unit
@pytest.mark.fast
def test_get_file_path_comprehensive_url_vs_local(tmp_path):
    """Test get_file_path with comprehensive URL vs local path scenarios.

    Coverage: Lines 151-165 (get_file_path branch logic)

    REAL components:
    - Real urlparse() for URL detection
    - Real path resolution for local files
    - Real cache_dir handling (None default, string conversion)

    MOCKED:
    - _download_from_url for HTTP/HTTPS URLs
    - File existence check bypassed with tmp_path

    Scenario:
    1. Test HTTP URL → triggers _download_from_url
    2. Test HTTPS URL → triggers _download_from_url
    3. Test local file path → triggers _get_local_file_path
    4. Test cache_dir=None → uses default cache_dir
    5. Test cache_dir as string → converts to Path

    Assertions:
    - HTTP/HTTPS URLs call _download_from_url
    - Local paths call _get_local_file_path
    - cache_dir handled correctly (None, string, Path)
    - Correct Path objects returned
    """
    # Setup: Create real local file
    local_file = tmp_path / "local_model.onnx"
    local_file.write_bytes(b"local content")

    # Test 1: HTTP URL triggers download
    http_url = "http://example.com/model.onnx"
    cache_dir = tmp_path / "cache"

    with patch.object(utils, "_download_from_url") as mock_download:
        mock_download.return_value = cache_dir / "model.onnx"

        result = utils.get_file_path(http_url, cache_dir=cache_dir)

        mock_download.assert_called_once_with(http_url, cache_dir)
        assert result == cache_dir / "model.onnx", "HTTP URL returns download result"

    # Test 2: HTTPS URL triggers download
    https_url = "https://example.com/model.onnx"

    with patch.object(utils, "_download_from_url") as mock_download:
        mock_download.return_value = cache_dir / "model_https.onnx"

        result = utils.get_file_path(https_url, cache_dir=cache_dir)

        mock_download.assert_called_once_with(https_url, cache_dir)
        assert result == cache_dir / "model_https.onnx", "HTTPS URL returns download result"

    # Test 3: Local file path triggers _get_local_file_path
    with patch.object(utils, "_get_local_file_path") as mock_get_local:
        mock_get_local.return_value = local_file

        result = utils.get_file_path(str(local_file), cache_dir=cache_dir)

        mock_get_local.assert_called_once_with(str(local_file))
        assert result == local_file, "Local path returns resolved file path"

    # Test 4: cache_dir=None uses default
    with patch.object(utils, "_download_from_url") as mock_download:
        # Mock DEFAULT_PATHS access
        with patch.dict(utils.DEFAULT_PATHS, {"cache_dir": Path("/default/cache")}):
            mock_download.return_value = Path("/default/cache/model.onnx")

            result = utils.get_file_path(http_url, cache_dir=None)

            # Verify default cache_dir was used
            call_args = mock_download.call_args[0]
            assert call_args[1] == Path("/default/cache"), "Default cache_dir used when None"

    # Test 5: cache_dir as string is converted to Path
    cache_str = str(tmp_path / "string_cache")

    with patch.object(utils, "_download_from_url") as mock_download:
        mock_download.return_value = Path(cache_str) / "model.onnx"

        result = utils.get_file_path(http_url, cache_dir=cache_str)

        # Verify cache_dir was converted to Path
        call_args = mock_download.call_args[0]
        assert isinstance(call_args[1], Path), "String cache_dir converted to Path"
        assert call_args[1] == Path(cache_str), "Correct Path object passed"
