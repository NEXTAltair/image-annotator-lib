"""
Web API component preparation helper functions.

各プロバイダー向けのWeb APIコンポーネント準備を担当するヘルパー関数群。
"""

import os
from collections.abc import Callable
from typing import Any

import dotenv
from google import genai
from pydantic import SecretStr

from ...exceptions.errors import ApiAuthenticationError, ConfigurationError
from ...model_class.annotator_webapi import webapi_shared
from .. import config
from ..types import WebApiComponents
from ..utils import logger
from .adapters import AnthropicAdapter, GoogleClientAdapter, OpenAIAdapter


def _find_model_entry_by_name(
    model_name_short: str, available_models: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    """available_models 辞書から model_name_short に一致するエントリを探す。

    Args:
        model_name_short: 検索する短いモデル名。
        available_models: `available_api_models.toml` からロードされたモデル情報。

    Returns:
        (model_id_on_provider, model_data) のタプル、または見つからない場合は None。
    """
    for model_id, data in available_models.items():
        if data.get("model_name_short") == model_name_short:
            return model_id, data
    return None


def _get_api_key(provider_name: str, api_model_id: str) -> str:
    """プロバイダー名に基づいて環境変数からAPIキーを取得する。

    .env ファイルのロードを試みる。

    Args:
        provider_name: プロバイダー名 (e.g., "Google", "OpenAI", "Anthropic", "OpenRouter").
        api_model_id: モデルID (e.g., "gemini-1.5-pro", "gemma-3-27b-it:free").
    Returns:
        APIキー文字列。

    Raises:
        ApiAuthenticationError: 対応する環境変数が見つからない場合。
        ConfigurationError: サポートされていないプロバイダー名の場合。
    """
    # .env ファイルから直接読み込み(環境変数に設定しない)

    env_values = dotenv.dotenv_values(".env")

    env_var_map = {
        "Google": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
    }
    env_var_name = env_var_map.get(provider_name)

    if not env_var_name or ":" in api_model_id:
        logger.debug(
            f"プロバイダー '{provider_name}' はマッピングされていません。OpenRouterのAPIキーを試みます。"
        )
        env_var_name = "OPENROUTER_API_KEY"

    # まず.envファイルから取得を試行、次に環境変数
    api_key = env_values.get(env_var_name) or os.getenv(env_var_name)
    if not api_key:
        raise ApiAuthenticationError(
            provider_name=provider_name,
            message=f"APIキー '{env_var_name}' が.envファイルまたは環境変数に設定されていません。 (プロバイダー: {provider_name})",
        )

    return api_key


def _process_model_id(model_id_on_provider: str, provider_name: str) -> str:
    """プロバイダーに応じてモデルIDのプレフィックスを除去する。

    Args:
        model_id_on_provider: TOMLから取得した元のモデルID。
        provider_name: プロバイダー名。

    Returns:
        加工済みのモデルID。
    """
    processed_id = model_id_on_provider
    prefix_to_remove = ""

    if ":" in model_id_on_provider or provider_name == "OpenRouter":
        return model_id_on_provider

    if provider_name == "OpenAI" and model_id_on_provider.startswith("openai/"):
        prefix_to_remove = "openai/"
    elif provider_name == "Google" and model_id_on_provider.startswith("google/"):
        prefix_to_remove = "google/"
    elif provider_name == "Anthropic" and model_id_on_provider.startswith("anthropic/"):
        prefix_to_remove = "anthropic/"

    if prefix_to_remove:
        processed_id = model_id_on_provider.removeprefix(prefix_to_remove)
        logger.debug(
            f"{provider_name} モデル ID のプレフィックスを除去: '{model_id_on_provider}' -> '{processed_id}'"
        )

    return processed_id


def _resolve_api_key_from_config(model_config: dict[str, Any]) -> str | None:
    """model_configからAPIキー文字列を解決する。

    SecretStr/str両方に対応。見つからない場合はNoneを返す。

    Args:
        model_config: モデル設定辞書。

    Returns:
        APIキー文字列、またはNone。
    """
    api_key_from_config = model_config.get("api_key")
    if not api_key_from_config:
        return None
    if isinstance(api_key_from_config, SecretStr):
        return api_key_from_config.get_secret_value()
    if isinstance(api_key_from_config, str):
        return api_key_from_config
    return None


def _resolve_api_key(
    model_config: dict[str, Any], env_var_names: list[str], provider_display: str, model_name: str
) -> str:
    """configと環境変数からAPIキーを解決する。見つからなければ例外。

    Args:
        model_config: モデル設定辞書。
        env_var_names: フォールバック用の環境変数名リスト(優先順)。
        provider_display: エラーメッセージ用のプロバイダー表示名。
        model_name: エラーメッセージ用のモデル名。

    Returns:
        APIキー文字列。

    Raises:
        ApiAuthenticationError: APIキーが見つからない場合。
    """
    api_key = _resolve_api_key_from_config(model_config)
    if api_key:
        return api_key

    # .envファイルから直接読み込み(環境変数に設定しない)
    env_values = dotenv.dotenv_values(".env")
    for env_var in env_var_names:
        value = env_values.get(env_var) or os.getenv(env_var)
        if value:
            return value

    raise ApiAuthenticationError(
        provider_name=provider_display,
        message=f"API key for model '{model_name}' not found.",
    )


def _init_openai(
    model_config: dict[str, Any], model_name: str
) -> OpenAIAdapter:
    """OpenAIプロバイダー用クライアントを初期化する。

    Args:
        model_config: モデル設定辞書。
        model_name: ログ出力用のモデル名。

    Returns:
        初期化されたOpenAIAdapter。
    """
    from openai import OpenAI

    api_key = _resolve_api_key(model_config, ["OPENAI_API_KEY"], "OpenAI", model_name)
    system_prompt = model_config.get("system_prompt", webapi_shared.SYSTEM_PROMPT)
    base_prompt = model_config.get("base_prompt", webapi_shared.BASE_PROMPT)
    raw_client = OpenAI(api_key=api_key)
    return OpenAIAdapter(raw_client, system_prompt=system_prompt, base_prompt=base_prompt)


def _init_google(
    model_config: dict[str, Any], model_name: str
) -> GoogleClientAdapter:
    """Googleプロバイダー用クライアントを初期化する。

    Args:
        model_config: モデル設定辞書。
        model_name: ログ出力用のモデル名。

    Returns:
        初期化されたGoogleClientAdapter。

    Raises:
        ApiAuthenticationError: 初期化に失敗した場合。
    """
    api_key = _resolve_api_key(
        model_config, ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "Google", model_name
    )
    system_prompt = model_config.get("system_prompt", webapi_shared.SYSTEM_PROMPT)
    base_prompt = model_config.get("base_prompt", webapi_shared.BASE_PROMPT)

    try:
        client = genai.Client(api_key=api_key)
        logger.info(f"Google GenAI Client initialized successfully for model '{model_name}'.")
        return GoogleClientAdapter(client=client, system_prompt=system_prompt, base_prompt=base_prompt)
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI Client for model '{model_name}': {e}")
        raise ApiAuthenticationError(
            provider_name="Google",
            message=f"Failed to initialize Google GenAI Client for '{model_name}': {e}",
        ) from e


def _init_anthropic(
    model_config: dict[str, Any], model_name: str
) -> AnthropicAdapter:
    """Anthropicプロバイダー用クライアントを初期化する。

    Args:
        model_config: モデル設定辞書。
        model_name: ログ出力用のモデル名。

    Returns:
        初期化されたAnthropicAdapter。
    """
    from anthropic import Anthropic

    api_key = _resolve_api_key(model_config, ["ANTHROPIC_API_KEY"], "Anthropic", model_name)
    raw_client = Anthropic(api_key=api_key)
    return AnthropicAdapter(raw_client)


def _init_openrouter(
    model_config: dict[str, Any], model_name: str
) -> OpenAIAdapter:
    """OpenRouterプロバイダー用クライアントを初期化する。

    Args:
        model_config: モデル設定辞書。
        model_name: ログ出力用のモデル名。

    Returns:
        初期化されたOpenAIAdapter（OpenRouter設定）。
    """
    from openai import OpenAI

    api_key = _resolve_api_key(model_config, ["OPENROUTER_API_KEY"], "OpenRouter", model_name)
    base_url = model_config.get("openrouter_base_url", "https://openrouter.ai/api/v1")
    site_url = model_config.get("openrouter_site_url", "http://localhost:3000")
    app_name = model_config.get("openrouter_app_name", "image-annotator-lib")
    system_prompt = model_config.get("system_prompt", webapi_shared.SYSTEM_PROMPT)
    base_prompt = model_config.get("base_prompt", webapi_shared.BASE_PROMPT)

    raw_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"HTTP-Referer": site_url, "X-Title": app_name},
    )
    return OpenAIAdapter(raw_client, system_prompt=system_prompt, base_prompt=base_prompt)


# プロバイダー名 → 初期化関数のディスパッチテーブル
_PROVIDER_INITIALIZERS: dict[
    str,
    Callable[[dict[str, Any], str], GoogleClientAdapter | OpenAIAdapter | AnthropicAdapter],
] = {
    "openai": _init_openai,
    "google": _init_google,
    "anthropic": _init_anthropic,
    "openrouter": _init_openrouter,
}


def _initialize_api_client(
    provider: str, model_config: dict[str, Any], model_name_for_logging: str
) -> GoogleClientAdapter | OpenAIAdapter | AnthropicAdapter:
    """プロバイダーに応じたAPIクライアントを初期化する。

    Args:
        provider: プロバイダー名 (e.g., "openai", "google", "anthropic").
        model_config: モデル設定辞書（api_key, system_prompt, base_promptなどを含む）。
        model_name_for_logging: ログ出力用のモデル名。

    Returns:
        初期化されたAPIクライアントアダプター。

    Raises:
        ApiAuthenticationError: APIキーが見つからない場合。
        ConfigurationError: サポートされていないプロバイダーの場合。
    """
    logger.debug(f"Initializing API client for provider: {provider}, model: {model_name_for_logging}")
    provider_lower = provider.lower()

    # OpenRouterはモデルIDに":"が含まれる場合も検出
    actual_model_id = model_config.get("api_model_id", model_config.get("model_path"))
    if provider_lower not in _PROVIDER_INITIALIZERS and actual_model_id and ":" in actual_model_id:
        provider_lower = "openrouter"

    initializer = _PROVIDER_INITIALIZERS.get(provider_lower)
    if not initializer:
        raise ConfigurationError(
            f"Unsupported Web API provider: {provider} for model '{model_name_for_logging}'"
        )

    return initializer(model_config, model_name_for_logging)


def prepare_web_api_components(model_name: str) -> WebApiComponents:
    """指定されたモデル名に基づいてWeb APIコンポーネントを準備する。

    Args:
        model_name: モデル名 (設定ファイル内のmodel_name_short)。

    Returns:
        WebApiComponents: 初期化されたAPIクライアント、モデルID、プロバイダー名を含む辞書。

    Raises:
        ConfigurationError: モデルが見つからない、または設定エラーの場合。
        ApiAuthenticationError: APIキーの取得に失敗した場合。
    """
    logger.debug(f"Web API コンポーネント準備開始: model_name='{model_name}'")

    available_models_data = config.load_available_api_models()
    logger.debug(f"available_models_data: {available_models_data}")
    if not available_models_data:
        raise ConfigurationError(
            "利用可能なAPIモデル情報 (available_api_models.toml) がロードされていません。"
        )

    model_entry = _find_model_entry_by_name(model_name, available_models_data)
    logger.debug(f"model_entry: {model_entry}")
    if not model_entry:
        raise ConfigurationError(
            f"モデル名 '{model_name}' に対応するエントリが available_api_models.toml に見つかりません。"
        )
    model_id_on_provider, model_config_from_toml = (
        model_entry  # model_data を model_config_from_toml にリネーム
    )
    provider_name = model_config_from_toml.get("provider")
    if not provider_name:
        raise ConfigurationError(f"モデル '{model_name}' のエントリに 'provider' が含まれていません。")

    logger.debug(f"モデル情報発見: id='{model_id_on_provider}', provider='{provider_name}'")
    # _process_model_id に渡すのは TOML からの生のモデルID
    processed_api_model_id = _process_model_id(model_id_on_provider, provider_name)

    # _initialize_api_client に渡す model_config を準備
    # TOMLからの設定と、加工済みのAPIモデルIDをマージ
    final_model_config_for_init = model_config_from_toml.copy()
    final_model_config_for_init["api_model_id"] = (
        processed_api_model_id  # api_model_id を加工済みのものに上書き
    )
    # final_model_config_for_init["provider"] は既に model_config_from_toml に含まれているはず

    try:
        # provider_name, final_model_config_for_init, model_name を渡す
        initialized_client = _initialize_api_client(
            provider=provider_name,
            model_config=final_model_config_for_init,
            model_name_for_logging=model_name,
        )
    except ApiAuthenticationError as e:  # より具体的なエラーハンドリング
        raise ApiAuthenticationError(
            provider_name=e.provider_name or provider_name,  # e.provider_name があればそれを使う
            message=f"{e.message} (モデル: {model_name})",
        ) from e
    except ConfigurationError as e:
        raise ConfigurationError(f"APIクライアント初期化設定エラー ({model_name}): {e}") from e
    except Exception as e:  # 予期せぬエラー
        logger.error(f"APIクライアント初期化中に予期せぬエラー ({model_name}): {e}", exc_info=True)
        raise ConfigurationError(f"APIクライアント初期化中に予期せぬエラー ({model_name}): {e}") from e

    logger.info(
        f"Web API コンポーネント準備完了: provider='{provider_name}', api_model_id='{processed_api_model_id}'"
    )

    components: WebApiComponents = {
        "client": initialized_client,
        "api_model_id": processed_api_model_id,
        "provider_name": provider_name,
    }
    return components
