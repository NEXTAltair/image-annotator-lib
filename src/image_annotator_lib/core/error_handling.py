"""Unified error handling and logging system for image-annotator-lib.

This module provides centralized error handling, logging, and retry logic
to improve consistency across the library.
"""

from typing import Any

from ..exceptions.errors import (
    AnnotatorError,
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ModelExecutionError,
    OutOfMemoryError,
)
from .error_messages import get_error_code_for_exception
from .utils import logger


class ErrorHandler:
    """Unified error handling and logging for annotator library.

    Provides consistent error logging, structured error reporting,
    and retry decision logic across all annotator operations.
    """

    def __init__(self, logger_name: str = "image_annotator_lib"):
        """Initialize error handler.

        Args:
            logger_name: Name of the logger to use (default: "image_annotator_lib")
        """
        self.logger_name = logger_name

    def handle_exception(self, exc: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle exception with logging and structured output.

        Args:
            exc: Exception to handle
            context: Additional context information (e.g., model_name, image_path)

        Returns:
            Structured error information as dict
        """
        # Build error information
        if isinstance(exc, AnnotatorError):
            error_info = exc.to_dict()
        else:
            # Handle non-AnnotatorError exceptions
            error_info = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "details": {},
                "ja_message": str(exc),
            }

        # Add error code
        error_code = get_error_code_for_exception(error_info["error_type"])
        error_info["error_code"] = error_code

        # Add context if provided
        if context:
            error_info["details"].update(context)

        # Log the error
        self.log_error(exc, error_info)

        return error_info

    def log_error(self, exception: Exception, error_info: dict[str, Any] | None = None) -> None:
        """Log error with structured information.

        Args:
            exception: Exception to log
            error_info: Optional pre-computed error info dict
        """
        import json

        if error_info is None:
            if isinstance(exception, AnnotatorError):
                error_info = exception.to_dict()
            else:
                error_info = {"error_type": type(exception).__name__, "message": str(exception)}

        # Determine log level based on exception type
        level = self._determine_log_level(exception)

        # Format log message
        error_type = error_info.get("error_type", "UnknownError")
        error_code = error_info.get("error_code", "ERR-UNKNOWN")
        message = error_info.get("message", "No message")
        details = error_info.get("details", {})

        # Use JSON for details to avoid format string conflicts
        # Escape braces to prevent loguru from interpreting them as format placeholders
        log_message = f"[{error_code}] {error_type}: {message}"
        if details:
            details_str = json.dumps(details, ensure_ascii=False)
            # Double braces to escape them for loguru
            details_str_escaped = details_str.replace("{", "{{").replace("}", "}}")
            log_message += f" | Details: {details_str_escaped}"

        # Log at appropriate level
        if level == "error":
            logger.error(log_message, exc_info=isinstance(exception, Exception))
        elif level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def generate_error_report(self, exception: AnnotatorError) -> str:
        """Generate human-readable error report.

        Args:
            exception: AnnotatorError to generate report for

        Returns:
            Formatted error report string
        """
        error_dict = exception.to_dict()

        report_lines = [
            "=" * 60,
            f"Error Type: {error_dict['error_type']}",
            f"Message: {error_dict['message']}",
        ]

        if error_dict.get("ja_message") and error_dict["ja_message"] != error_dict["message"]:
            report_lines.append(f"Japanese: {error_dict['ja_message']}")

        if error_dict.get("details"):
            report_lines.append("\nDetails:")
            for key, value in error_dict["details"].items():
                report_lines.append(f"  - {key}: {value}")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def should_retry(self, exception: Exception) -> bool:
        """Determine if operation should be retried based on exception type.

        Args:
            exception: Exception to evaluate

        Returns:
            True if operation should be retried, False otherwise
        """
        # Retryable error types
        retryable_types = (
            ApiRateLimitError,  # Rate limit - should wait and retry
            ApiTimeoutError,  # Temporary network/timeout issue
            OutOfMemoryError,  # May be resolved after cache cleanup
            ApiServerError,  # 5xx server errors are typically retryable
        )

        # Non-retryable error types
        non_retryable_types = (
            ApiAuthenticationError,  # Auth won't succeed without fixing credentials
            ApiRequestError,  # Bad request - won't succeed on retry
            ModelExecutionError,  # Model execution failed (unlikely to succeed on retry)
        )

        if isinstance(exception, retryable_types):
            return True

        if isinstance(exception, non_retryable_types):
            return False

        # Unknown exception types - default to no retry
        return False

    def get_retry_delay(self, exception: Exception, attempt: int) -> float:
        """Calculate retry delay in seconds based on exception and attempt number.

        Args:
            exception: Exception that occurred
            attempt: Retry attempt number (1-indexed)

        Returns:
            Delay in seconds before next retry
        """
        # Check for rate limit with retry_after
        if isinstance(exception, ApiRateLimitError):
            retry_after = getattr(exception, "retry_after", None)
            if retry_after:
                return float(retry_after)

        # Exponential backoff: 2^attempt seconds (max 60s)
        base_delay = min(2**attempt, 60)

        return base_delay

    def _determine_log_level(self, exception: Exception) -> str:
        """Determine appropriate log level for exception.

        Args:
            exception: Exception to evaluate

        Returns:
            Log level string ("error", "warning", or "info")
        """
        # Warnings: rate limits, memory issues (recoverable)
        warning_types = (ApiRateLimitError, OutOfMemoryError)

        if isinstance(exception, warning_types):
            return "warning"

        # Info: timeout/temporary errors
        info_types = (ApiTimeoutError,)

        if isinstance(exception, info_types):
            return "info"

        # Default: error level
        return "error"


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global ErrorHandler instance.

    Returns:
        Global ErrorHandler instance
    """
    return _error_handler
