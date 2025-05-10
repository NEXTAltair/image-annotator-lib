# Stub file for image_annotator_lib.exceptions.errors

class AnnotatorError(Exception):
    """Base exception for the image-annotator-lib. / image-annotator-lib の基底例外クラス。"""
    ...

class ModelLoadError(AnnotatorError):
    """Exception raised for errors during model loading. / モデルのロード中に発生したエラー。

    Attributes:
        message (str): Human readable string describing the error. / エラー内容を示す文字列。
    """
    message: str
    def __init__(self, message: str) -> None: ...

class ModelNotFoundError(AnnotatorError):
    """Exception raised when a requested model is not found. / 要求されたモデルが見つからない場合に発生するエラー。

    Attributes:
        model_name (str): The name of the model that was not found. / 見つからなかったモデルの名前。
    """
    model_name: str
    def __init__(self, model_name: str) -> None: ...

class OutOfMemoryError(AnnotatorError):
    """Exception raised when CUDA runs out of memory. / CUDA メモリが不足した場合に発生するエラー。

    Attributes:
        message (str): Human readable string describing the error. / エラー内容を示す文字列。
    """
    message: str
    def __init__(self, message: str) -> None: ...

class PHashCalculationError(Exception):
    """Exception for errors during pHash calculation. / pHash計算時のエラーを表す例外"""
    ...

class InvalidInputError(AnnotatorError):
    """Exception raised for invalid input data. / 無効な入力データが提供された場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

class InvalidModelConfigError(AnnotatorError):
    """Exception raised for invalid model configuration. / モデル設定が無効な場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

class InvalidOutputError(AnnotatorError):
    """Exception raised for invalid model output. / モデルの出力が無効な場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

class ModelExecutionError(AnnotatorError):
    """Exception raised for errors during model execution. / モデルの実行中にエラーが発生した場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

class UnsupportedModelError(AnnotatorError):
    """Exception raised when an unsupported model is requested. / サポートされていないモデルが要求された場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

class ConfigurationError(AnnotatorError):
    """Exception raised for configuration-related errors. / 設定に関連するエラーが発生した場合の例外。"""
    message: str
    def __init__(self, message: str) -> None: ...

# Web API related exceptions
class WebApiError(AnnotatorError):
    """Base exception for Web API related errors. / Web API 関連の基底例外クラス。"""
    message: str
    provider_name: str
    def __init__(self, message: str, provider_name: str = "") -> None: ...

class ApiAuthenticationError(WebApiError):
    """Exception raised for API authentication failures. / API 認証に失敗した場合の例外。"""
    status_code: int
    def __init__(self, message: str = ..., provider_name: str = "", status_code: int = 401) -> None: ...

class ApiRateLimitError(WebApiError):
    """Exception raised when API rate limits are exceeded. / API レート制限に達した場合の例外。"""
    status_code: int
    retry_after: int
    def __init__(self, message: str = ..., provider_name: str = "", status_code: int = 429, retry_after: int = 60) -> None: ...

class ApiRequestError(WebApiError):
    """Exception raised for errors in the API request format or content. / API リクエストの形式または内容に問題があった場合の例外。"""
    status_code: int
    def __init__(self, message: str, provider_name: str = "", status_code: int = 400) -> None: ...

class ApiServerError(WebApiError):
    """Exception raised for errors on the API server side. / API サーバー側でエラーが発生した場合の例外。"""
    status_code: int
    def __init__(self, message: str, provider_name: str = "", status_code: int = 500) -> None: ...

class ApiTimeoutError(WebApiError):
    """Exception raised when an API request times out. / API リクエストがタイムアウトした場合の例外。"""
    def __init__(self, message: str = ..., provider_name: str = "") -> None: ...

class ApiKeyMissingError(WebApiError):
    """Exception raised when an API key is missing from environment variables. / API キーが環境変数に設定されていない場合の例外。"""
    env_var: str
    def __init__(self, env_var: str, provider_name: str = "") -> None: ...

class InsufficientCreditsError(WebApiError):
    """Exception raised when API credits are insufficient. / APIのクレジット残高が不足している場合の例外。"""
    status_code: int
    def __init__(self, message: str = ..., provider_name: str = "", status_code: int = 402) -> None: ...
