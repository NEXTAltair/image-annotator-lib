"""Error message templates and error code system for image-annotator-lib.

This module provides standardized error messages and codes for all
exception types in the library.
"""

from typing import Any

# Error code format: ERR-[CATEGORY]-[NUMBER]
# Categories:
#   CFG: Configuration errors
#   MDL: Model-related errors
#   API: Web API errors
#   MEM: Memory/resource errors
#   VAL: Validation errors
#   EXE: Execution errors


class ErrorCode:
    """Error code constants for all exception types."""

    # Configuration errors (CFG)
    CONFIG_MISSING_FIELD = "ERR-CFG-001"
    CONFIG_INVALID_VALUE = "ERR-CFG-002"
    CONFIG_INVALID_TYPE = "ERR-CFG-003"

    # Model errors (MDL)
    MODEL_NOT_FOUND = "ERR-MDL-001"
    MODEL_LOAD_FAILED = "ERR-MDL-002"
    MODEL_UNSUPPORTED = "ERR-MDL-003"
    MODEL_EXECUTION_FAILED = "ERR-MDL-004"

    # API errors (API)
    API_AUTH_FAILED = "ERR-API-001"
    API_RATE_LIMIT = "ERR-API-002"
    API_REQUEST_FAILED = "ERR-API-003"
    API_SERVER_ERROR = "ERR-API-004"
    API_TIMEOUT = "ERR-API-005"
    API_KEY_MISSING = "ERR-API-006"
    API_CREDITS_INSUFFICIENT = "ERR-API-007"

    # Memory/resource errors (MEM)
    MEMORY_INSUFFICIENT = "ERR-MEM-001"

    # Validation errors (VAL)
    INPUT_INVALID = "ERR-VAL-001"
    OUTPUT_INVALID = "ERR-VAL-002"
    PHASH_CALCULATION_FAILED = "ERR-VAL-003"
    CAPABILITY_VALIDATION_FAILED = "ERR-VAL-004"

    # Dependency errors (DEP)
    DEPENDENCY_MISSING = "ERR-DEP-001"


class ErrorMessageTemplate:
    """Error message templates with placeholders for dynamic values.

    Each template provides:
    - English message (en)
    - Japanese message (ja)
    - Suggested action (action_en, action_ja)
    """

    # Configuration errors
    CONFIG_MISSING_FIELD = {
        "en": "Configuration field '{field}' is missing or empty.",
        "ja": "設定項目 '{field}' が見つからないか、空です。",
        "action_en": "Check your configuration file and ensure '{field}' is properly set.",
        "action_ja": "設定ファイルを確認し、'{field}' が正しく設定されているか確認してください。",
    }

    CONFIG_INVALID_VALUE = {
        "en": "Configuration field '{field}' has invalid value: {value}",
        "ja": "設定項目 '{field}' の値が不正です: {value}",
        "action_en": "Update '{field}' to a valid value. Expected: {expected}",
        "action_ja": "'{field}' を有効な値に更新してください。期待値: {expected}",
    }

    # Model errors
    MODEL_NOT_FOUND = {
        "en": "Model '{model_name}' not found in registry.",
        "ja": "モデル '{model_name}' がレジストリに見つかりません。",
        "action_en": "Verify model name or check available models with list_available_annotators().",
        "action_ja": "モデル名を確認するか、list_available_annotators() で利用可能なモデルを確認してください。",
    }

    MODEL_LOAD_FAILED = {
        "en": "Failed to load model from '{model_path}': {reason}",
        "ja": "モデルのロードに失敗しました '{model_path}': {reason}",
        "action_en": "Verify model path and ensure the file exists and is accessible.",
        "action_ja": "モデルパスを確認し、ファイルが存在しアクセス可能であることを確認してください。",
    }

    MODEL_UNSUPPORTED = {
        "en": "Model '{model_name}' is not supported by this annotator class.",
        "ja": "モデル '{model_name}' はこのアノテータークラスでサポートされていません。",
        "action_en": "Use a compatible annotator class or check model configuration.",
        "action_ja": "互換性のあるアノテータークラスを使用するか、モデル設定を確認してください。",
    }

    MODEL_EXECUTION_FAILED = {
        "en": "Model '{model_name}' execution failed: {reason}",
        "ja": "モデル '{model_name}' の実行に失敗しました: {reason}",
        "action_en": "Check input data format and model configuration.",
        "action_ja": "入力データ形式とモデル設定を確認してください。",
    }

    # API errors
    API_AUTH_FAILED = {
        "en": "{provider} API authentication failed. Status code: {status_code}",
        "ja": "{provider} API認証に失敗しました。ステータスコード: {status_code}",
        "action_en": "Verify API key is valid and has necessary permissions.",
        "action_ja": "APIキーが有効で必要な権限があることを確認してください。",
    }

    API_RATE_LIMIT = {
        "en": "{provider} API rate limit exceeded. Retry after: {retry_after}s",
        "ja": "{provider} APIレート制限に達しました。再試行まで: {retry_after}秒",
        "action_en": "Wait {retry_after} seconds before retrying or reduce request frequency.",
        "action_ja": "{retry_after}秒待ってから再試行するか、リクエスト頻度を減らしてください。",
    }

    API_REQUEST_FAILED = {
        "en": "{provider} API request failed: {reason}",
        "ja": "{provider} APIリクエストに失敗しました: {reason}",
        "action_en": "Check request parameters and API documentation.",
        "action_ja": "リクエストパラメータとAPIドキュメントを確認してください。",
    }

    API_SERVER_ERROR = {
        "en": "{provider} API server error. The service may be temporarily unavailable.",
        "ja": "{provider} APIサーバーエラー。サービスが一時的に利用できない可能性があります。",
        "action_en": "Wait a few minutes and retry. If issue persists, check API status page.",
        "action_ja": "数分待って再試行してください。問題が続く場合は、APIステータスページを確認してください。",
    }

    API_TIMEOUT = {
        "en": "{provider} API request timed out after {timeout}s.",
        "ja": "{provider} APIリクエストがタイムアウトしました（{timeout}秒）。",
        "action_en": "Increase timeout value or check network connectivity.",
        "action_ja": "タイムアウト値を増やすか、ネットワーク接続を確認してください。",
    }

    API_KEY_MISSING = {
        "en": "{provider} API key is not configured.",
        "ja": "{provider} APIキーが設定されていません。",
        "action_en": "Set API key in configuration file or environment variable.",
        "action_ja": "設定ファイルまたは環境変数でAPIキーを設定してください。",
    }

    # Memory errors
    MEMORY_INSUFFICIENT = {
        "en": "Insufficient memory: required {required_gb}GB, available {available_gb}GB",
        "ja": "メモリ不足: 必要 {required_gb}GB、利用可能 {available_gb}GB",
        "action_en": "Free up memory or reduce model size/batch size.",
        "action_ja": "メモリを解放するか、モデルサイズ/バッチサイズを削減してください。",
    }

    # Validation errors
    INPUT_INVALID = {
        "en": "Invalid input: {reason}",
        "ja": "無効な入力: {reason}",
        "action_en": "Verify input format matches expected schema.",
        "action_ja": "入力形式が期待されるスキーマと一致するか確認してください。",
    }

    OUTPUT_INVALID = {
        "en": "Invalid output from model: {reason}",
        "ja": "モデルからの無効な出力: {reason}",
        "action_en": "Check model configuration and output parsing logic.",
        "action_ja": "モデル設定と出力解析ロジックを確認してください。",
    }

    # Dependency errors
    DEPENDENCY_MISSING = {
        "en": "Required dependency '{dependency}' is not installed.",
        "ja": "必要な依存関係 '{dependency}' がインストールされていません。",
        "action_en": "Install '{dependency}' using: pip install {dependency}",
        "action_ja": "'{dependency}' をインストールしてください: pip install {dependency}",
    }


def format_error_message(template: dict[str, str], **kwargs: Any) -> dict[str, str]:
    """Format error message from template with provided values.

    Args:
        template: Error message template dict
        **kwargs: Values to format into template

    Returns:
        Dictionary with formatted messages:
        - message_en: English formatted message
        - message_ja: Japanese formatted message
        - action_en: English suggested action
        - action_ja: Japanese suggested action
    """
    return {
        "message_en": template["en"].format(**kwargs),
        "message_ja": template["ja"].format(**kwargs),
        "action_en": template.get("action_en", "").format(**kwargs),
        "action_ja": template.get("action_ja", "").format(**kwargs),
    }


def get_error_code_for_exception(exception_class_name: str) -> str:
    """Get error code for exception class name.

    Args:
        exception_class_name: Name of exception class

    Returns:
        Error code string (e.g., "ERR-MDL-001")
    """
    error_code_map = {
        "ConfigurationError": ErrorCode.CONFIG_INVALID_VALUE,
        "ModelNotFoundError": ErrorCode.MODEL_NOT_FOUND,
        "ModelLoadError": ErrorCode.MODEL_LOAD_FAILED,
        "UnsupportedModelError": ErrorCode.MODEL_UNSUPPORTED,
        "ModelExecutionError": ErrorCode.MODEL_EXECUTION_FAILED,
        "ApiAuthenticationError": ErrorCode.API_AUTH_FAILED,
        "ApiRateLimitError": ErrorCode.API_RATE_LIMIT,
        "ApiRequestError": ErrorCode.API_REQUEST_FAILED,
        "ApiServerError": ErrorCode.API_SERVER_ERROR,
        "ApiTimeoutError": ErrorCode.API_TIMEOUT,
        "ApiKeyMissingError": ErrorCode.API_KEY_MISSING,
        "InsufficientCreditsError": ErrorCode.API_CREDITS_INSUFFICIENT,
        "OutOfMemoryError": ErrorCode.MEMORY_INSUFFICIENT,
        "InvalidInputError": ErrorCode.INPUT_INVALID,
        "InvalidOutputError": ErrorCode.OUTPUT_INVALID,
        "PHashCalculationError": ErrorCode.PHASH_CALCULATION_FAILED,
        "MissingDependencyError": ErrorCode.DEPENDENCY_MISSING,
    }

    return error_code_map.get(exception_class_name, "ERR-UNKNOWN")
