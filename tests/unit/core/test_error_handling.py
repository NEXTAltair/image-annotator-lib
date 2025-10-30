"""Tests for unified error handling system."""

from image_annotator_lib.core.error_handling import ErrorHandler, get_error_handler
from image_annotator_lib.exceptions.errors import (
    AnnotatorError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ModelExecutionError,
    ModelLoadError,
    OutOfMemoryError,
)


class TestErrorHandler:
    """Test cases for ErrorHandler class."""

    def test_init_default(self):
        """Test ErrorHandler initialization with defaults."""
        handler = ErrorHandler()
        assert handler.logger_name == "image_annotator_lib"

    def test_init_custom_logger(self):
        """Test ErrorHandler initialization with custom logger name."""
        handler = ErrorHandler(logger_name="custom_logger")
        assert handler.logger_name == "custom_logger"

    def test_handle_exception_annotator_error(self):
        """Test handling AnnotatorError with structured information."""
        handler = ErrorHandler()
        exc = ModelLoadError("Failed to load", model_path="/path/to/model")

        result = handler.handle_exception(exc)

        assert result["error_type"] == "ModelLoadError"
        assert "Failed to load" in result["message"]
        assert result["details"]["model_path"] == "/path/to/model"
        assert "ja_message" in result

    def test_handle_exception_standard_error(self):
        """Test handling standard Python exception."""
        handler = ErrorHandler()
        exc = ValueError("Invalid value")

        result = handler.handle_exception(exc)

        assert result["error_type"] == "ValueError"
        assert result["message"] == "Invalid value"
        assert result["details"] == {}

    def test_handle_exception_with_context(self):
        """Test handling exception with additional context."""
        handler = ErrorHandler()
        exc = ModelExecutionError("Execution failed", model_name="test_model")
        context = {"image_path": "/path/to/image.jpg", "batch_size": 4}

        result = handler.handle_exception(exc, context=context)

        assert result["details"]["model_name"] == "test_model"
        assert result["details"]["image_path"] == "/path/to/image.jpg"
        assert result["details"]["batch_size"] == 4

    def test_log_error_annotator_error(self):
        """Test logging AnnotatorError."""
        handler = ErrorHandler()
        exc = ApiAuthenticationError(provider_name="openai", status_code=401)

        # Should not raise
        handler.log_error(exc)

    def test_log_error_standard_exception(self):
        """Test logging standard exception."""
        handler = ErrorHandler()
        exc = RuntimeError("Runtime error occurred")

        # Should not raise
        handler.log_error(exc)

    def test_log_error_with_precomputed_info(self):
        """Test logging with pre-computed error info."""
        handler = ErrorHandler()
        exc = ValueError("Test error")
        error_info = {
            "error_type": "CustomError",
            "message": "Custom message",
            "details": {"key": "value"},
        }

        # Should not raise
        handler.log_error(exc, error_info=error_info)

    def test_generate_error_report(self):
        """Test error report generation."""
        handler = ErrorHandler()
        exc = ModelLoadError(
            "Model file not found",
            model_path="/models/test.onnx",
            details={"file_size": "1.2GB", "device": "cuda"},
        )

        report = handler.generate_error_report(exc)

        assert "ModelLoadError" in report
        assert "Model file not found" in report
        assert "モデルロードエラー" in report  # Japanese message
        assert "/models/test.onnx" in report
        assert "file_size" in report
        assert "device" in report
        assert "=" in report  # Formatting

    def test_generate_error_report_no_details(self):
        """Test error report generation without details."""
        handler = ErrorHandler()
        exc = AnnotatorError("Simple error")

        report = handler.generate_error_report(exc)

        assert "AnnotatorError" in report
        assert "Simple error" in report
        assert "Details:" not in report  # No details section

    def test_should_retry_rate_limit(self):
        """Test retry decision for rate limit error."""
        handler = ErrorHandler()
        exc = ApiRateLimitError(provider_name="openai", retry_after=60)

        assert handler.should_retry(exc) is True

    def test_should_retry_timeout_error(self):
        """Test retry decision for timeout error."""
        handler = ErrorHandler()
        exc = ApiTimeoutError(provider_name="anthropic")

        assert handler.should_retry(exc) is True

    def test_should_retry_memory_error(self):
        """Test retry decision for out of memory error."""
        handler = ErrorHandler()
        exc = OutOfMemoryError("Out of memory")

        assert handler.should_retry(exc) is True

    def test_should_retry_auth_error(self):
        """Test retry decision for authentication error (non-retryable)."""
        handler = ErrorHandler()
        exc = ApiAuthenticationError(provider_name="google")

        assert handler.should_retry(exc) is False

    def test_should_retry_model_execution_error(self):
        """Test retry decision for model execution error (non-retryable)."""
        handler = ErrorHandler()
        exc = ModelExecutionError("Model crashed", model_name="test")

        assert handler.should_retry(exc) is False

    def test_should_retry_api_server_error(self):
        """Test retry decision for API server error (retryable)."""
        handler = ErrorHandler()
        exc = ApiServerError("Server error", provider_name="openai")

        assert handler.should_retry(exc) is True

    def test_should_retry_api_request_error(self):
        """Test retry decision for API request error (non-retryable)."""
        handler = ErrorHandler()
        exc = ApiRequestError("Bad request", provider_name="openai")

        assert handler.should_retry(exc) is False

    def test_should_retry_unknown_error(self):
        """Test retry decision for unknown error type (default: no retry)."""
        handler = ErrorHandler()
        exc = ValueError("Unknown error")

        assert handler.should_retry(exc) is False

    def test_get_retry_delay_rate_limit_with_retry_after(self):
        """Test retry delay calculation for rate limit with retry_after."""
        handler = ErrorHandler()
        exc = ApiRateLimitError(provider_name="openai", retry_after=120)

        delay = handler.get_retry_delay(exc, attempt=1)

        assert delay == 120.0

    def test_get_retry_delay_exponential_backoff(self):
        """Test retry delay calculation with exponential backoff."""
        handler = ErrorHandler()
        exc = ApiTimeoutError(provider_name="anthropic")

        # Attempt 1: 2^1 = 2 seconds
        assert handler.get_retry_delay(exc, attempt=1) == 2.0

        # Attempt 2: 2^2 = 4 seconds
        assert handler.get_retry_delay(exc, attempt=2) == 4.0

        # Attempt 3: 2^3 = 8 seconds
        assert handler.get_retry_delay(exc, attempt=3) == 8.0

        # Attempt 10: 2^10 = 1024, but max is 60
        assert handler.get_retry_delay(exc, attempt=10) == 60.0

    def test_get_retry_delay_memory_error(self):
        """Test retry delay for memory error."""
        handler = ErrorHandler()
        exc = OutOfMemoryError("Out of memory")

        delay = handler.get_retry_delay(exc, attempt=1)

        assert delay == 2.0  # Exponential backoff

    def test_determine_log_level_rate_limit(self):
        """Test log level determination for rate limit (warning)."""
        handler = ErrorHandler()
        exc = ApiRateLimitError(provider_name="openai")

        level = handler._determine_log_level(exc)

        assert level == "warning"

    def test_determine_log_level_memory(self):
        """Test log level determination for memory error (warning)."""
        handler = ErrorHandler()
        exc = OutOfMemoryError("Out of memory")

        level = handler._determine_log_level(exc)

        assert level == "warning"

    def test_determine_log_level_timeout(self):
        """Test log level determination for timeout error (info)."""
        handler = ErrorHandler()
        exc = ApiTimeoutError(provider_name="anthropic")

        level = handler._determine_log_level(exc)

        assert level == "info"

    def test_determine_log_level_default(self):
        """Test log level determination for other errors (error)."""
        handler = ErrorHandler()
        exc = ModelExecutionError("Execution failed", model_name="test")

        level = handler._determine_log_level(exc)

        assert level == "error"


class TestGlobalErrorHandler:
    """Test cases for global error handler."""

    def test_get_error_handler(self):
        """Test getting global error handler instance."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        # Should return same instance
        assert handler1 is handler2
        assert isinstance(handler1, ErrorHandler)
        assert handler1.logger_name == "image_annotator_lib"
