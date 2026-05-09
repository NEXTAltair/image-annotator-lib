"""HTTP transport retry policy (ADR 0023 Phase 1.8 / Issue #46).

ADR 0023 line 304-329 で確定済の HTTP/API transient failure retry policy を
PydanticAI provider 用 ``httpx.AsyncClient`` 向けにコード化する。

実装する内容:

- 408/409/429/500/502/503/504 を retry 対象 status code とし、
  ``validate_response`` callback 内で ``httpx.HTTPStatusError`` 化する。
- ``httpx.ConnectError`` / ``ReadTimeout`` / ``WriteTimeout`` /
  ``RemoteProtocolError`` といった主要 transient network 例外も retry する。
- ``initial + 2 retries`` (max 3 attempts) で停止する。
- ``Retry-After`` ヘッダを尊重し、``max_wait = HTTP_RETRY_MAX_WAIT_SECONDS`` で cap する。
- ``RetryConfig(reraise=True)`` で原因例外をそのまま raise し、
  ``provider_manager._classify_refusal()`` の判定ロジックに干渉しない。

実装しないもの:

- model fallback / LiteLLM Router retry (ADR 0023 line 313-314)
- ``Retry-After > max_wait`` の halt-on-exceed (Phase 2 へ繰り延べ)
- output validation retry (これは PydanticAI ``output_retries=1`` で別経路、
  ``provider_manager._OUTPUT_RETRIES`` を参照)

関連ドキュメント: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

import httpx
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt

# ADR 0023 line 309 の retry 対象 HTTP status code allowlist。
# 401/403 (auth) や 400 (bad request) はここに含めず、SDK 例外層で terminal
# failure として扱う。422 (validation) も意味的に retry しても結果が変わらない。
RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

# ADR 0023 line 308 の attempts 数。initial + 2 retries = 最大 3 回呼び出し。
HTTP_RETRY_MAX_ATTEMPTS = 3

# ADR 0023 line 308 の Retry-After cap (Phase 1.8 補遺で根拠を明記)。
# OpenAI / Anthropic の token bucket は分単位で補充され、典型 Retry-After は
# 60 秒以内に収まる。worst case 3 attempts x 60s = 120s 待機後に error 伝播 ->

# Qt batch worker のキャンセル応答性を確保する。pydantic-ai default 300s は
# CLI 想定で Qt UI には過大。
HTTP_RETRY_MAX_WAIT_SECONDS = 60.0

# Tenacity ``retry_if_exception_type`` に渡す例外タプル。
# - ``httpx.HTTPStatusError``: ``_validate_response_for_retry`` 経由で
#   retryable status code に対してのみ raise される。
# - ``httpx.TimeoutException``: ``ConnectTimeout`` / ``ReadTimeout`` /
#   ``WriteTimeout`` / ``PoolTimeout`` の base class。connect 段階の TCP/TLS
#   timeout (`ConnectTimeout`) と pool 枯渇 (`PoolTimeout`) も含めて全 timeout を retry 対象にする
#   (Codex P1 r3214045319: 旧コードは `ReadTimeout` / `WriteTimeout` のみ列挙していたため
#   `ConnectTimeout` が漏れていた)。
# - ``httpx.ConnectError`` (TCP 接続失敗 / DNS 等) と ``httpx.RemoteProtocolError``
#   (server-side connection close) は ``NetworkError`` / ``ProtocolError`` の代表。
# - ``HTTPError`` など上位を捕まえると ``InvalidURL`` 等の terminal error も
#   retry してしまうので、目的別の sub class のみ列挙する。
_RETRYABLE_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.HTTPStatusError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def _validate_response_for_retry(response: httpx.Response) -> None:
    """Retryable な status code に対してのみ ``HTTPStatusError`` を raise する。

    ``AsyncTenacityTransport`` は ``validate_response`` で発生した例外を
    tenacity controller に投げ、retry policy に従って再試行する。
    Auth (401/403) や bad request (400) は raise せず通過させ、SDK 例外層で
    terminal failure として扱わせる (`InferenceError` 経路へ流れる)。
    """
    if response.status_code in RETRYABLE_HTTP_STATUSES:
        response.raise_for_status()


def build_retry_transport() -> AsyncTenacityTransport:
    """ADR 0023 Phase 1.8 の retry policy を反映した async transport を返す。

    PydanticAI の ``wait_retry_after`` は ``Retry-After`` ヘッダ
    (秒数 / HTTP date) を尊重し、未指定時は ``wait_exponential(max=60)``
    fallback を使う仕様 (`pydantic_ai/retries.py` 参照)。
    """
    return AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(_RETRYABLE_NETWORK_EXCEPTIONS),
            wait=wait_retry_after(max_wait=HTTP_RETRY_MAX_WAIT_SECONDS),
            stop=stop_after_attempt(HTTP_RETRY_MAX_ATTEMPTS),
            reraise=True,
        ),
        validate_response=_validate_response_for_retry,
    )


def build_retry_http_client() -> httpx.AsyncClient:
    """Retry transport を組み込んだ ``httpx.AsyncClient`` を新規生成する。

    Caller (``provider_manager.run_inference_with_model_async``) は
    ``try/finally`` で必ず ``await client.aclose()`` を呼ぶ責務を持つ。
    Agent / Provider / Model キャッシュなし方針 (ADR 0023 Agent ライフサイクル)
    に合わせ、AsyncClient も推論呼び出しごとに新規生成・破棄する。
    """
    return httpx.AsyncClient(transport=build_retry_transport())


__all__ = [
    "HTTP_RETRY_MAX_ATTEMPTS",
    "HTTP_RETRY_MAX_WAIT_SECONDS",
    "RETRYABLE_HTTP_STATUSES",
    "build_retry_http_client",
    "build_retry_transport",
]
