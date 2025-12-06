"""Shared fixtures for core unit tests.

This module provides reusable fixtures for error testing scenarios across
the image-annotator-lib core components.
"""

import pytest


@pytest.fixture
def mock_filesystem_error():
    """Mock filesystem errors (FileNotFoundError, PermissionError).

    Returns:
        Callable that creates filesystem error instances.

    Usage:
        error = mock_filesystem_error(FileNotFoundError, "/path/to/file")
    """

    def _mock(error_type: type, path: str):
        """Create filesystem error instance.

        Args:
            error_type: Exception class (FileNotFoundError, PermissionError, etc.)
            path: File path that triggered the error

        Returns:
            Exception instance configured with path message
        """
        return error_type(f"Mock error for {path}")

    return _mock


@pytest.fixture
def mock_api_error():
    """Mock API errors with configurable status/exception.

    Returns:
        Callable that returns AsyncMock configured for API errors.

    Usage:
        mock = mock_api_error(status_code=401)
        mock = mock_api_error(exception=asyncio.TimeoutError())
    """

    def _mock(
        status_code: int | None = None,
        exception: Exception | None = None,
        response_data: dict | None = None,
    ):
        """Create mock API response.

        Args:
            status_code: HTTP status code (401, 429, 500, etc.)
            exception: Exception to raise (asyncio.TimeoutError, HTTPError, etc.)
            response_data: Mock response body (for successful responses)

        Returns:
            AsyncMock configured with specified error/response
        """
        from unittest.mock import AsyncMock

        mock_response = AsyncMock()

        if exception:
            # Raise exception on call (e.g., timeout, connection error)
            mock_response.side_effect = exception
        elif status_code:
            # HTTP error response
            from requests.exceptions import HTTPError

            http_error = HTTPError(f"HTTP {status_code}")
            http_error.response = AsyncMock(status_code=status_code)
            mock_response.side_effect = http_error
        else:
            # Successful response
            mock_response.return_value = response_data or {"tags": ["test"], "score": 0.9}

        return mock_response

    return _mock
