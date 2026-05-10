"""ADR 0023 Phase 1.8 (Issue #46): http_retry モジュールの policy 定数 + factory unit test."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic_ai.retries import AsyncTenacityTransport
from tenacity.stop import stop_after_attempt

from image_annotator_lib.core.http_retry import (
    HTTP_RETRY_MAX_ATTEMPTS,
    HTTP_RETRY_MAX_WAIT_SECONDS,
    RETRYABLE_HTTP_STATUSES,
    _validate_response_for_retry,
    build_retry_http_client,
    build_retry_transport,
)


def _make_response(status_code: int) -> httpx.Response:
    """Validate response 用に最小限の `httpx.Response` を生成する。"""
    request = httpx.Request("GET", "https://example.invalid/")
    return httpx.Response(status_code=status_code, request=request)


class TestPolicyConstants:
    """ADR 0023 line 304-329 で確定済の数値が constant に反映されているか確認する。"""

    def test_retryable_statuses_match_adr(self) -> None:
        assert RETRYABLE_HTTP_STATUSES == frozenset({408, 409, 429, 500, 502, 503, 504})

    def test_max_attempts_is_3(self) -> None:
        # initial + 2 retries = 3 attempts
        assert HTTP_RETRY_MAX_ATTEMPTS == 3

    def test_max_wait_seconds_is_60(self) -> None:
        # ADR 0023 Phase 1.8: provider token-bucket window
        assert HTTP_RETRY_MAX_WAIT_SECONDS == 60.0


class TestValidateResponseForRetry:
    """`_validate_response_for_retry` は retryable な status code に対してのみ raise する。"""

    @pytest.mark.parametrize("status", sorted(RETRYABLE_HTTP_STATUSES))
    def test_retryable_statuses_raise_http_status_error(self, status: int) -> None:
        response = _make_response(status)
        with pytest.raises(httpx.HTTPStatusError):
            _validate_response_for_retry(response)

    @pytest.mark.parametrize("status", [200, 201, 204, 301, 304])
    def test_success_and_redirect_statuses_pass_through(self, status: int) -> None:
        response = _make_response(status)
        # raise なし = pass through
        _validate_response_for_retry(response)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 410, 422])
    def test_non_retryable_4xx_pass_through(self, status: int) -> None:
        # auth/permission/bad request は SDK 例外層で terminal failure として扱う設計。
        # validate_response は raise しないでそのまま返す。
        response = _make_response(status)
        _validate_response_for_retry(response)


class TestBuildRetryTransport:
    """`build_retry_transport()` の構築結果が ADR の policy を反映しているか。"""

    def test_returns_async_tenacity_transport(self) -> None:
        transport = build_retry_transport()
        assert isinstance(transport, AsyncTenacityTransport)

    def test_validate_response_callback_is_set(self) -> None:
        transport = build_retry_transport()
        assert transport.validate_response is _validate_response_for_retry

    def test_retry_config_uses_stop_after_attempt(self) -> None:
        transport = build_retry_transport()
        stop = transport.config["stop"]
        # stop_after_attempt instance であること
        assert isinstance(stop, stop_after_attempt)

    def test_retry_config_reraise_is_true(self) -> None:
        # `reraise=True` で原因例外をそのまま伝搬させ、`_classify_refusal()` が
        # PydanticAI/SDK 例外型 / body 構造を直接判定できるようにする。
        transport = build_retry_transport()
        assert transport.config["reraise"] is True


class TestBuildRetryHttpClient:
    """`build_retry_http_client()` は AsyncClient を返し、retry transport が組み込まれている。"""

    def test_returns_async_client_with_retry_transport(self) -> None:
        async def _check() -> None:
            client = build_retry_http_client()
            try:
                assert isinstance(client, httpx.AsyncClient)
                # AsyncClient の transport は private 属性 _transport で保持される
                transport = client._transport
                assert isinstance(transport, AsyncTenacityTransport)
            finally:
                await client.aclose()

        asyncio.run(_check())
