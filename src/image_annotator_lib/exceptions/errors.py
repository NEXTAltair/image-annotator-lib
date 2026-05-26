"""ライブラリ固有のカスタム例外クラス。

Phase 2.1: 構造化エラー情報とプログラム可能なエラー処理を追加
"""

from typing import Any


class AnnotatorError(Exception):
    """image-annotator-lib の基底例外クラス。

    ライブラリ内で発生する特定の運用エラーを示すために使用されます。

    Attributes:
        message: エラーの詳細メッセージ（英語）
        details: 構造化されたエラー情報（辞書形式）
        ja_message: 日本語エラーメッセージ（後方互換性のため）
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None, ja_message: str | None = None):
        """AnnotatorError を初期化します。

        Args:
            message: エラーの詳細メッセージ（英語推奨）
            details: 構造化されたエラー情報（オプション）
            ja_message: 日本語メッセージ（後方互換性、オプション）
        """
        self.message = message
        self.details = details or {}
        self.ja_message = ja_message or message
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """構造化されたエラー情報を辞書として返す。

        プログラム的なエラー処理、ロギング、API レスポンスに使用可能。

        Returns:
            dict: エラー情報を含む辞書
                - error_type: 例外クラス名
                - message: 英語エラーメッセージ
                - details: 追加のコンテキスト情報
                - ja_message: 日本語メッセージ（後方互換性）
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "ja_message": self.ja_message,
        }

    def __str__(self) -> str:
        """エラーメッセージを文字列として返す（英語）。"""
        return self.message

    def __repr__(self) -> str:
        """デバッグ用の詳細表現。"""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class ModelLoadError(AnnotatorError):
    """モデルのロード中にエラーが発生した場合の例外。

    モデルファイルの欠損、フォーマット不正、依存関係の問題などが原因で発生します。

    Attributes:
        message: エラーの詳細メッセージ（英語）
        model_path: ロード試行されたモデルのパス
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, model_path: str | None = None, details: dict[str, Any] | None = None):
        """ModelLoadError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            model_path: ロード試行されたモデルのパス（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if model_path:
            error_details["model_path"] = model_path

        self.model_path = model_path
        super().__init__(
            message=f"Model load error: {message}",
            details=error_details,
            ja_message=f"モデルロードエラー: {message}",
        )


class ModelNotFoundError(AnnotatorError):
    """要求されたモデルがレジストリまたは設定に見つからない場合の例外。

    Attributes:
        model_name: 見つからなかったモデルの名前
        details: 追加のエラーコンテキスト
    """

    def __init__(self, model_name: str, details: dict[str, Any] | None = None):
        """ModelNotFoundError を初期化します。

        Args:
            model_name: 見つからなかったモデルの名前
            details: 追加のエラーコンテキスト（オプション）
        """
        self.model_name = model_name
        error_details = details or {}
        error_details["model_name"] = model_name

        super().__init__(
            message=f"Model '{model_name}' not found in registry or configuration",
            details=error_details,
            ja_message=f"モデル未検出エラー: {model_name}",
        )


class OutOfMemoryError(AnnotatorError):
    """主に CUDA デバイスのメモリが不足した場合の例外。

    モデルのロード時または推論実行中に発生する可能性があります。

    Attributes:
        message: エラーの詳細メッセージ
        device: メモリ不足が発生したデバイス（cuda, cpu等）
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, device: str | None = None, details: dict[str, Any] | None = None):
        """OutOfMemoryError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            device: メモリ不足が発生したデバイス（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if device:
            error_details["device"] = device

        self.device = device
        super().__init__(
            message=f"Out of memory error: {message}",
            details=error_details,
            ja_message=f"メモリ不足エラー: {message}",
        )


class PHashCalculationError(AnnotatorError):
    """pHash計算時のエラーを表す例外

    Attributes:
        image_info: pHash計算に失敗した画像の情報
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, image_info: str | None = None, details: dict[str, Any] | None = None):
        """PHashCalculationError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            image_info: pHash計算に失敗した画像の情報（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if image_info:
            error_details["image_info"] = image_info

        self.image_info = image_info
        super().__init__(
            message=f"pHash calculation error: {message}",
            details=error_details,
            ja_message=f"pHash計算エラー: {message}",
        )


class InvalidInputError(AnnotatorError):
    """無効な入力データが提供された場合の例外。

    Attributes:
        message: エラーの詳細メッセージ
        input_type: 無効だった入力の型
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, input_type: str | None = None, details: dict[str, Any] | None = None):
        """InvalidInputError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            input_type: 無効だった入力の型（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if input_type:
            error_details["input_type"] = input_type

        self.input_type = input_type
        super().__init__(
            message=f"Invalid input error: {message}",
            details=error_details,
            ja_message=f"無効な入力エラー: {message}",
        )


class InvalidModelConfigError(AnnotatorError):
    """モデル設定が無効な場合の例外。

    Attributes:
        message: エラーの詳細メッセージ
        field: 無効だった設定フィールド名
        invalid_value: 無効だった値
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        invalid_value: Any = None,
        details: dict[str, Any] | None = None,
    ):
        """InvalidModelConfigError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            field: 無効だった設定フィールド名（オプション）
            invalid_value: 無効だった値（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        if invalid_value is not None:
            error_details["invalid_value"] = str(invalid_value)

        self.field = field
        self.invalid_value = invalid_value
        super().__init__(
            message=f"Invalid model configuration: {message}",
            details=error_details,
            ja_message=f"無効なモデル設定エラー: {message}",
        )


class InvalidOutputError(AnnotatorError):
    """モデルの出力が無効な場合の例外。

    Attributes:
        message: エラーの詳細メッセージ
        output_info: 無効だった出力の情報
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, output_info: str | None = None, details: dict[str, Any] | None = None):
        """InvalidOutputError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            output_info: 無効だった出力の情報（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if output_info:
            error_details["output_info"] = output_info

        self.output_info = output_info
        super().__init__(
            message=f"Invalid output error: {message}",
            details=error_details,
            ja_message=f"無効な出力エラー: {message}",
        )


class ModelExecutionError(AnnotatorError):
    """モデルの実行中にエラーが発生した場合の例外。

    Attributes:
        message: エラーの詳細メッセージ
        model_name: エラーが発生したモデル名
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, model_name: str | None = None, details: dict[str, Any] | None = None):
        """ModelExecutionError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            model_name: エラーが発生したモデル名（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name

        self.model_name = model_name
        super().__init__(
            message=f"Model execution error: {message}",
            details=error_details,
            ja_message=f"モデル実行エラー: {message}",
        )


class UnsupportedModelError(AnnotatorError):
    """サポートされていないモデルが要求された場合の例外。

    Attributes:
        message: エラーの詳細メッセージ
        model_name: サポートされていないモデル名
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, model_name: str | None = None, details: dict[str, Any] | None = None):
        """UnsupportedModelError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            model_name: サポートされていないモデル名（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name

        self.model_name = model_name
        super().__init__(
            message=f"Unsupported model error: {message}",
            details=error_details,
            ja_message=f"サポートされていないモデルエラー: {message}",
        )


class ConfigurationError(AnnotatorError):
    """設定に関連するエラーが発生した場合の例外。

    設定ファイルの構文エラー、必須キーの欠損、不正な値などが原因で発生します。

    Attributes:
        message: エラーの詳細メッセージ
        field: 問題のある設定フィールド名
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, field: str | None = None, details: dict[str, Any] | None = None):
        """ConfigurationError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            field: 問題のある設定フィールド名（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if field:
            error_details["field"] = field

        self.field = field
        super().__init__(
            message=f"Configuration error: {message}",
            details=error_details,
            ja_message=f"設定エラー: {message}",
        )


# 必要に応じて他の特定の例外をここに追加
# 例:
# class PreprocessingError(AnnotatorError):
#     """画像の前処理中にエラーが発生した場合の例外。"""
#     pass
#
# class InferenceError(AnnotatorError):
#     """モデルの推論実行中にエラーが発生した場合の例外。"""
#     pass


# Web API 関連の例外クラス
class WebApiError(AnnotatorError):
    """Web API 関連の基底例外クラス。

    Attributes:
        message: エラーの詳細メッセージ
        provider_name: APIプロバイダー名
        details: 追加のエラーコンテキスト
    """

    def __init__(self, message: str, provider_name: str = "", details: dict[str, Any] | None = None):
        """WebApiError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名(例: "Google", "OpenAI")
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if provider_name:
            error_details["provider_name"] = provider_name

        self.provider_name = provider_name
        provider_prefix = f"{provider_name} " if provider_name else ""
        en_message = f"{provider_prefix}API error: {message}"
        ja_message = f"{provider_prefix}API エラー: {message}"

        super().__init__(message=en_message, details=error_details, ja_message=ja_message)


class ApiAuthenticationError(WebApiError):
    """API 認証に失敗した場合の例外。

    通常はAPIキーが無効、期限切れ、または権限不足の場合に発生します。

    Attributes:
        status_code: HTTP ステータスコード
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str = "API authentication failed. Check API key validity.",
        provider_name: str = "",
        status_code: int = 401,
        details: dict[str, Any] | None = None,
    ):
        """ApiAuthenticationError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            status_code: HTTPステータスコード
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details["status_code"] = status_code

        self.status_code = status_code
        super().__init__(message, provider_name, error_details)


class ApiRateLimitError(WebApiError):
    """API レート制限に達した場合の例外。

    短時間に多くのリクエストを送信した場合に発生します。

    Attributes:
        status_code: HTTP ステータスコード
        retry_after: 再試行までの推奨待機時間(秒)
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str = "API rate limit reached. Wait before retrying.",
        provider_name: str = "",
        status_code: int = 429,
        retry_after: int = 60,
        details: dict[str, Any] | None = None,
    ):
        """ApiRateLimitError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            status_code: HTTPステータスコード
            retry_after: 再試行までの推奨待機時間(秒)
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details.update({"status_code": status_code, "retry_after": retry_after})

        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message, provider_name, error_details)


class ApiRequestError(WebApiError):
    """API リクエストの形式または内容に問題があった場合の例外。

    通常はクライアント側のエラーです。

    Attributes:
        status_code: HTTP ステータスコード
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str,
        provider_name: str = "",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        """ApiRequestError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            status_code: HTTPステータスコード
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details["status_code"] = status_code

        self.status_code = status_code
        super().__init__(f"Request format or content problem: {message}", provider_name, error_details)


class ApiServerError(WebApiError):
    """API サーバー側でエラーが発生した場合の例外。

    サーバーの問題やメンテナンスなどで発生します。

    Attributes:
        status_code: HTTP ステータスコード
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str,
        provider_name: str = "",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        """ApiServerError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            status_code: HTTPステータスコード
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details["status_code"] = status_code

        self.status_code = status_code
        super().__init__(f"API server error occurred: {message}", provider_name, error_details)


class ApiTimeoutError(WebApiError):
    """API リクエストがタイムアウトした場合の例外。

    ネットワーク問題や負荷の高いリクエストで発生します。

    Attributes:
        timeout_seconds: タイムアウトまでの秒数
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str = "API request timed out. Check network or API status.",
        provider_name: str = "",
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """ApiTimeoutError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            timeout_seconds: タイムアウトまでの秒数（オプション）
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds

        self.timeout_seconds = timeout_seconds
        super().__init__(message, provider_name, error_details)


class ApiKeyMissingError(WebApiError):
    """API キーが環境変数に設定されていない場合の例外。

    Attributes:
        env_var: APIキーが格納されるべき環境変数名
        details: 追加のエラーコンテキスト
    """

    def __init__(self, env_var: str, provider_name: str = "", details: dict[str, Any] | None = None):
        """ApiKeyMissingError を初期化します。

        Args:
            env_var: APIキーが格納されるべき環境変数名
            provider_name: APIプロバイダー名
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details["env_var"] = env_var

        self.env_var = env_var
        super().__init__(f"API key not set in environment variable {env_var}", provider_name, error_details)


class InsufficientCreditsError(WebApiError):
    """APIのクレジット残高が不足している場合の例外。

    Attributes:
        status_code: HTTP ステータスコード
        details: 追加のエラーコンテキスト
    """

    def __init__(
        self,
        message: str = "Insufficient API credits. Check account balance.",
        provider_name: str = "",
        status_code: int = 402,
        details: dict[str, Any] | None = None,
    ):
        """InsufficientCreditsError を初期化します。

        Args:
            message: エラーの詳細メッセージ
            provider_name: APIプロバイダー名
            status_code: HTTPステータスコード
            details: 追加のエラーコンテキスト（オプション）
        """
        error_details = details or {}
        error_details["status_code"] = status_code

        self.status_code = status_code
        super().__init__(message, provider_name, error_details)


# ADR 0023 Phase 1: PydanticAI / LiteLLM WebAPI Inference Boundary
# https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md


class IdMappingError(WebApiError):
    """litellm_model_id の解析に失敗した場合の例外。

    `webapi/model_id.resolve_model_ref()` が prefix 解析や builder dispatch に失敗した
    ときに raise する。空文字 / `provider/model` 形式違反などが該当する。

    Attributes:
        litellm_model_id: 解析を試みた ID
        reason: 解析失敗の理由
    """

    def __init__(
        self,
        litellm_model_id: str,
        reason: str = "invalid format",
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        error_details["litellm_model_id"] = litellm_model_id
        error_details["reason"] = reason

        self.litellm_model_id = litellm_model_id
        self.reason = reason
        super().__init__(
            f"Failed to parse litellm_model_id '{litellm_model_id}': {reason}",
            provider_name="",
            details=error_details,
        )


class UnknownProviderError(WebApiError):
    """SUPPORTED_PROVIDERS に含まれない provider が指定された場合の例外。

    Phase 1 では OpenAI / Anthropic / Google / OpenRouter のみ対応。Vertex AI / xAI
    などは UnknownProviderError として弾く。

    Attributes:
        provider: 解析された provider 名
        litellm_model_id: 元の ID
    """

    def __init__(
        self,
        provider: str,
        litellm_model_id: str,
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        error_details["provider"] = provider
        error_details["litellm_model_id"] = litellm_model_id

        self.provider = provider
        self.litellm_model_id = litellm_model_id
        super().__init__(
            f"Provider '{provider}' is not supported (litellm_model_id='{litellm_model_id}')",
            provider_name=provider,
            details=error_details,
        )


class MissingApiKeyError(WebApiError):
    """api_keys dict に該当 provider のキーが含まれない場合の例外。

    `ApiKeyMissingError` が環境変数ベースの欠落を示すのに対し、本例外は
    ADR 0023 で定めた「明示注入のみ」契約上の欠落を示す。

    Attributes:
        provider: API key が必要な provider
        litellm_model_id: 推論対象モデル
    """

    def __init__(
        self,
        provider: str,
        litellm_model_id: str = "",
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        error_details["provider"] = provider
        if litellm_model_id:
            error_details["litellm_model_id"] = litellm_model_id

        self.provider = provider
        self.litellm_model_id = litellm_model_id
        super().__init__(
            f"API key for provider '{provider}' is missing in api_keys",
            provider_name=provider,
            details=error_details,
        )


class InferenceError(WebApiError):
    """PydanticAI 実行時に発生したエラーをラップする例外。

    HTTP error / validation failure / timeout などを `WebApiError` 階層に正規化する
    ためのカテゴリ。`__cause__` に原因例外が設定される。

    Attributes:
        litellm_model_id: 推論対象モデル
        cause: ラップされた原因例外 (オプション)
    """

    def __init__(
        self,
        message: str,
        litellm_model_id: str = "",
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        if litellm_model_id:
            error_details["litellm_model_id"] = litellm_model_id
        if cause is not None:
            error_details["cause_type"] = type(cause).__name__

        self.litellm_model_id = litellm_model_id
        self.cause = cause
        super().__init__(message, provider_name="", details=error_details)


class SafetyRefusalError(WebApiError):
    """Provider が safety refusal を返した場合の例外。

    OpenAI: finish_reason="content_filter"
    Anthropic: stop_reason="refusal"
    Google Gemini: finishReason="SAFETY" / BlockedReason
    OpenRouter: 各 upstream provider の refusal を踏襲

    ADR 0023 Phase 1.5 (Issue #42): retry しない / Rating テーブルは使わない /
    LoRAIro 側 error_records に記録して以後の WebAPI annotation 対象から除外。

    Attributes:
        litellm_model_id: 拒否したモデル ID
        image_phash: 拒否された画像の pHash (空文字許可)
        provider_refusal_reason: provider 固有の refusal 理由文字列 (空文字許可)
    """

    def __init__(
        self,
        litellm_model_id: str,
        image_phash: str = "",
        provider_refusal_reason: str = "",
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        error_details["litellm_model_id"] = litellm_model_id
        if image_phash:
            error_details["image_phash"] = image_phash
        if provider_refusal_reason:
            error_details["provider_refusal_reason"] = provider_refusal_reason

        self.litellm_model_id = litellm_model_id
        self.image_phash = image_phash
        self.provider_refusal_reason = provider_refusal_reason
        super().__init__(
            f"Safety refusal from '{litellm_model_id}': {provider_refusal_reason or 'no reason'}",
            provider_name="",
            details=error_details,
        )


class ContentPolicyRefusalError(WebApiError):
    """Provider が content policy refusal を返した場合の例外。

    SafetyRefusalError と同じ attribute 構造。主に OpenAI の content_filter
    finish_reason を分類する用途。その他 provider の generic refusal は
    SafetyRefusalError に寄せる。

    ADR 0023 Phase 1.5 (Issue #42): SafetyRefusalError と同じ contract で
    retry せず error_records に記録 → 送信前 filter で除外。

    Attributes:
        litellm_model_id: 拒否したモデル ID
        image_phash: 拒否された画像の pHash (空文字許可)
        provider_refusal_reason: provider 固有の refusal 理由文字列 (空文字許可)
    """

    def __init__(
        self,
        litellm_model_id: str,
        image_phash: str = "",
        provider_refusal_reason: str = "",
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        error_details["litellm_model_id"] = litellm_model_id
        if image_phash:
            error_details["image_phash"] = image_phash
        if provider_refusal_reason:
            error_details["provider_refusal_reason"] = provider_refusal_reason

        self.litellm_model_id = litellm_model_id
        self.image_phash = image_phash
        self.provider_refusal_reason = provider_refusal_reason
        super().__init__(
            f"Content policy refusal from '{litellm_model_id}': {provider_refusal_reason or 'no reason'}",
            provider_name="",
            details=error_details,
        )
