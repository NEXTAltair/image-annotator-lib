"""Tests for unified error handling system."""

from image_annotator_lib.core.error_handling import ErrorHandler, get_error_handler
from image_annotator_lib.exceptions.errors import (
    AnnotatorError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ConfigurationError,
    InvalidModelConfigError,
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

    # ==============================================================================
    # Phase C Additional Coverage Tests (2025-12-05)
    # ==============================================================================

    def test_model_load_error_with_nested_exception(self):
        """Test ModelLoadError with nested exception (exception chaining).

        Tests:
        - Nested exception is preserved via __cause__
        - Error details include both original and nested exceptions
        - Error handler processes chained exceptions correctly
        - Report generation includes nested exception info
        """
        handler = ErrorHandler()

        # Create nested exception chain
        try:
            try:
                raise FileNotFoundError("model.onnx not found")
            except FileNotFoundError as e:
                raise ModelLoadError("Failed to load model", model_path="/models/model.onnx") from e
        except ModelLoadError as exc:
            result = handler.handle_exception(exc)

            assert result["error_type"] == "ModelLoadError"
            assert "Failed to load model" in result["message"]
            assert result["details"]["model_path"] == "/models/model.onnx"
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, FileNotFoundError)

            # Verify report includes nested exception
            report = handler.generate_error_report(exc)
            assert "ModelLoadError" in report
            assert "/models/model.onnx" in report

    def test_api_error_with_custom_message(self):
        """Test WebAPI error with custom user-facing message.

        Tests:
        - Custom message is preserved in error
        - Japanese message is available
        - Error details include provider info
        - Error handler formats custom message correctly
        """
        handler = ErrorHandler()

        # API error with custom message
        exc = ApiAuthenticationError(
            provider_name="openai",
            status_code=401,
            message="Invalid API key format. Expected: sk-...",
        )

        result = handler.handle_exception(exc)

        assert result["error_type"] == "ApiAuthenticationError"
        assert "Invalid API key format" in result["message"]
        assert result["details"]["provider_name"] == "openai"
        assert result["details"]["status_code"] == 401
        assert "ja_message" in result

        # Verify report includes custom message
        report = handler.generate_error_report(exc)
        assert "Invalid API key format" in report

    def test_configuration_error_validation_details(self):
        """Test ConfigurationError with validation details.

        Tests:
        - Validation errors are captured in details
        - Error message includes configuration context
        - Missing/invalid fields are reported
        - Error handler extracts validation info
        """
        handler = ErrorHandler()

        # Configuration error with validation details
        exc = ConfigurationError(
            "Model configuration validation failed for test_model",
            field="device",
            details={
                "missing_fields": ["model_path", "class"],
                "invalid_fields": {"device": "Invalid device 'gpu', expected 'cpu' or 'cuda'"},
                "config_file": "/config/annotator_config.toml",
            },
        )

        result = handler.handle_exception(exc)

        assert result["error_type"] == "ConfigurationError"
        assert "validation failed" in result["message"]
        assert result["details"]["field"] == "device"
        assert "missing_fields" in result["details"]
        assert "model_path" in str(result["details"]["missing_fields"])
        assert "invalid_fields" in result["details"]

        # Verify report includes validation details
        report = handler.generate_error_report(exc)
        assert "ConfigurationError" in report
        assert "missing_fields" in report

    def test_timeout_error_with_context(self):
        """Test ApiTimeoutError with execution context.

        Tests:
        - Context information is preserved
        - Timeout details (duration, endpoint) are captured
        - Error handler processes context correctly
        - Retry logic considers timeout context
        """
        handler = ErrorHandler()

        # Timeout error with context
        exc = ApiTimeoutError(
            provider_name="anthropic",
            message="Request timed out after 30s",
            details={
                "timeout_seconds": 30,
                "endpoint": "/v1/messages",
                "model": "claude-3-5-sonnet-20241022",
                "retry_attempt": 2,
            },
        )

        # Add execution context
        context = {"image_path": "/images/test.jpg", "image_size": "1024x1024", "batch_size": 1}

        result = handler.handle_exception(exc, context=context)

        assert result["error_type"] == "ApiTimeoutError"
        assert result["details"]["provider_name"] == "anthropic"
        assert result["details"]["timeout_seconds"] == 30
        assert result["details"]["image_path"] == "/images/test.jpg"
        assert result["details"]["retry_attempt"] == 2

        # Verify retry logic
        assert handler.should_retry(exc) is True
        delay = handler.get_retry_delay(exc, attempt=2)
        assert delay > 0  # Should calculate exponential backoff

    def test_error_base_class_common_interface(self):
        """Test AnnotatorError base class common interface.

        Tests:
        - All error types inherit from AnnotatorError
        - Common attributes (ja_message, details) are available
        - Error hierarchy is correct
        - Error handler works with all error types consistently
        """
        handler = ErrorHandler()

        # Test various error types inherit from AnnotatorError
        error_types = [
            ModelLoadError("Load failed", model_path="/path"),
            ModelExecutionError("Execution failed", model_name="test"),
            ConfigurationError("Config invalid", field="test_field"),
            InvalidModelConfigError("Invalid config", field="test_field"),
            ApiAuthenticationError(provider_name="openai"),
            ApiRateLimitError(provider_name="openai"),
            OutOfMemoryError("OOM occurred"),
        ]

        for exc in error_types:
            # Verify inheritance
            assert isinstance(exc, AnnotatorError)

            # Verify common interface
            assert hasattr(exc, "ja_message")
            assert hasattr(exc, "details")

            # Verify error handler works consistently
            result = handler.handle_exception(exc)
            assert "error_type" in result
            assert "message" in result
            assert "details" in result
            assert "ja_message" in result

            # Verify all errors generate reports
            report = handler.generate_error_report(exc)
            assert len(report) > 0
            assert exc.__class__.__name__ in report


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
