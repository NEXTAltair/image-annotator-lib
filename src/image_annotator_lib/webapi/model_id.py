"""LiteLLM ID と PydanticAI 実行 descriptor の境界 (ADR 0023 Phase 1)。

LiteLLM 形式の `provider/model` ID を受け取り、PydanticAI の native provider /
model object を構築するための情報を一箇所に集約する。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..exceptions.errors import IdMappingError, MissingApiKeyError, UnknownProviderError

if TYPE_CHECKING:
    import httpx
    from pydantic_ai.models import Model


@dataclass(frozen=True)
class PydanticAIModelRef:
    """LiteLLM ID から導出される PydanticAI 実行 descriptor。

    Attributes:
        provider: 正規化済 provider 名 (`SUPPORTED_PROVIDERS` のいずれか)。
        litellm_model_id: 入力された LiteLLM 形式 ID (`openai/gpt-4o` 等)。
        provider_model_id: provider 上のモデル名 (`gpt-4o` 等)。
        pydantic_model_id: PydanticAI model string が利用可能な場合の文字列形式。
            provider object 経由のみで構築する場合は None。
        endpoint: 構築すべき OpenAI 系 endpoint 種別。`"chat"` (既定) は
            `OpenAIChatModel`、`"responses"` は `OpenAIResponsesModel` を構築する。
            responses は OpenAI provider 限定 (anthropic/google/openrouter は常に chat)。
    """

    provider: str
    litellm_model_id: str
    provider_model_id: str
    pydantic_model_id: str | None = None
    endpoint: str = "chat"


def _split_litellm_id(litellm_model_id: str) -> tuple[str, str]:
    """`provider/model` 形式の ID を分割する。

    Returns:
        (provider 小文字, provider_model_id) のタプル。
    """
    if not litellm_model_id:
        raise IdMappingError(litellm_model_id, reason="empty model id")
    if "/" not in litellm_model_id:
        raise IdMappingError(litellm_model_id, reason="missing 'provider/' prefix")
    provider, _, provider_model_id = litellm_model_id.partition("/")
    provider = provider.strip().lower()
    provider_model_id = provider_model_id.strip()
    if not provider or not provider_model_id:
        raise IdMappingError(litellm_model_id, reason="empty provider or model after split")
    return provider, provider_model_id


def _build_openai_ref(litellm_model_id: str, provider_model_id: str) -> PydanticAIModelRef:
    return PydanticAIModelRef(
        provider="openai",
        litellm_model_id=litellm_model_id,
        provider_model_id=provider_model_id,
        pydantic_model_id=f"openai:{provider_model_id}",
    )


def _build_anthropic_ref(litellm_model_id: str, provider_model_id: str) -> PydanticAIModelRef:
    return PydanticAIModelRef(
        provider="anthropic",
        litellm_model_id=litellm_model_id,
        provider_model_id=provider_model_id,
        pydantic_model_id=f"anthropic:{provider_model_id}",
    )


def _build_google_ref(litellm_model_id: str, provider_model_id: str) -> PydanticAIModelRef:
    # LiteLLM の `gemini/` と `google/` は同じ Google プロバイダーを指すため
    # provider 名は "google" に正規化する。
    return PydanticAIModelRef(
        provider="google",
        litellm_model_id=litellm_model_id,
        provider_model_id=provider_model_id,
        pydantic_model_id=f"google-gla:{provider_model_id}",
    )


def _build_openrouter_ref(litellm_model_id: str, provider_model_id: str) -> PydanticAIModelRef:
    # OpenRouter は base_url override が必要なため provider object 経由で構築する。
    # `pydantic_model_id` は使用しない (None)。
    return PydanticAIModelRef(
        provider="openrouter",
        litellm_model_id=litellm_model_id,
        provider_model_id=provider_model_id,
        pydantic_model_id=None,
    )


_BUILDER_DISPATCH: dict[str, Callable[[str, str], PydanticAIModelRef]] = {
    "openai": _build_openai_ref,
    "anthropic": _build_anthropic_ref,
    "google": _build_google_ref,
    "gemini": _build_google_ref,  # LiteLLM ID prefix の別名 (Google プロバイダー)
    "openrouter": _build_openrouter_ref,
}
"""provider 名 -> builder 関数の dispatch テーブル。

Phase 1 の `SUPPORTED_PROVIDERS` の真の source。allowlist は keys から導出されるため、
新 provider 追加は本テーブルへの 1 entry 追加で完結する (allowlist 編集不要)。
"""

SUPPORTED_PROVIDERS: frozenset[str] = frozenset(_BUILDER_DISPATCH.keys())


def resolve_model_ref(
    litellm_model_id: str,
    config: dict[str, Any] | None = None,
    *,
    mode: str = "chat",
) -> PydanticAIModelRef:
    """LiteLLM model ID から `PydanticAIModelRef` を生成する。

    Args:
        litellm_model_id: `openai/gpt-4o` のような LiteLLM 形式 ID。
        config: 将来の per-model override 用。Phase 1 では未使用。
        mode: registry metadata 由来の推論 endpoint 種別 (`"chat"` 既定 / `"responses"`)。
            `provider == "openai"` かつ `mode == "responses"` の場合のみ ref の
            `endpoint` を `"responses"` に設定する。それ以外の provider は常に chat。

    Returns:
        provider 別 builder で構築された descriptor。

    Raises:
        IdMappingError: ID 形式が不正 (空文字 / `/` 区切りなし / 片方が空)。
        UnknownProviderError: provider 名が `SUPPORTED_PROVIDERS` 外。
    """
    provider, provider_model_id = _split_litellm_id(litellm_model_id)
    builder = _BUILDER_DISPATCH.get(provider)
    if builder is None:
        raise UnknownProviderError(provider=provider, litellm_model_id=litellm_model_id)
    ref = builder(litellm_model_id, provider_model_id)
    # responses endpoint は OpenAI provider 限定。他 provider は builder の chat 既定のまま。
    if provider == "openai" and mode == "responses":
        ref = replace(ref, endpoint="responses")
    return ref


def _build_openai_model(
    ref: PydanticAIModelRef,
    api_key: str,
    http_client: httpx.AsyncClient | None,
) -> Model:
    """OpenAI provider の Chat / Responses endpoint を `ref.endpoint` で選択して構築する。

    `mode=responses` 専用モデル (pro ティア等) は `OpenAIResponsesModel`、それ以外は
    従来どおり `OpenAIChatModel` を使う (iam-lib #131)。provider object は共通。
    """
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key=api_key, http_client=http_client)
    if ref.endpoint == "responses":
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name=ref.provider_model_id, provider=provider)
    from pydantic_ai.models.openai import OpenAIChatModel

    return OpenAIChatModel(model_name=ref.provider_model_id, provider=provider)


def build_pydantic_model(
    ref: PydanticAIModelRef,
    api_key: str,
    config: dict[str, Any] | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> Model:
    """PydanticAI Model object を api_key 明示注入で構築する。

    `os.environ` mutate は行わない。すべての API key は provider object に直接注入する。

    Args:
        ref: `resolve_model_ref()` の戻り値。
        api_key: provider 用 API key。空文字や None は `MissingApiKeyError`。
        config: provider 固有の追加設定 (OpenRouter の referer / app_name 等)。
        http_client: provider object に注入する `httpx.AsyncClient`。
            ADR 0023 Phase 1.8 (Issue #46) で transport retry 付き client を渡す経路。
            `None` の場合は PydanticAI が default client (`create_async_http_client`) を生成する。

    Returns:
        Agent に渡せる PydanticAI Model object。

    Raises:
        MissingApiKeyError: api_key が空。
        UnknownProviderError: ref.provider が `SUPPORTED_PROVIDERS` 外
            (通常は `resolve_model_ref()` 段階で弾かれる)。
    """
    if not api_key:
        raise MissingApiKeyError(provider=ref.provider, litellm_model_id=ref.litellm_model_id)

    if ref.provider == "openai":
        return _build_openai_model(ref, api_key, http_client)

    if ref.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        anthropic_provider = AnthropicProvider(api_key=api_key, http_client=http_client)
        return AnthropicModel(model_name=ref.provider_model_id, provider=anthropic_provider)

    if ref.provider == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        google_provider = GoogleProvider(api_key=api_key, http_client=http_client)
        return GoogleModel(model_name=ref.provider_model_id, provider=google_provider)

    if ref.provider == "openrouter":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "http_client": http_client,
        }
        if config:
            headers: dict[str, str] = {}
            referer = config.get("referer")
            if referer:
                headers["HTTP-Referer"] = str(referer)
            app_name = config.get("app_name")
            if app_name:
                headers["X-Title"] = str(app_name)
            if headers:
                provider_kwargs["default_headers"] = headers
        openrouter_provider = OpenAIProvider(**provider_kwargs)
        return OpenAIChatModel(model_name=ref.provider_model_id, provider=openrouter_provider)

    raise UnknownProviderError(provider=ref.provider, litellm_model_id=ref.litellm_model_id)


__all__ = [
    "SUPPORTED_PROVIDERS",
    "PydanticAIModelRef",
    "build_pydantic_model",
    "resolve_model_ref",
]
