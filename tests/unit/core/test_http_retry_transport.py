"""ADR 0023 Phase 1.8 (Issue #46): AsyncTenacityTransport の振る舞い integration test.

mock `AsyncBaseTransport` (スクリプト化されたレスポンス列) を `build_retry_transport()`
の返す `AsyncTenacityTransport` に差し込み、httpx.AsyncClient 経由で実 HTTP request を
発行して retry policy を検証する。実 wait は `wait_retry_after(max_wait=60)` だが、
unit test で 60 秒待つわけにいかないので test では `wait` を no-op に override する。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic_ai.retries import AsyncTenacityTransport
from tenacity import wait_none

from image_annotator_lib.core.http_retry import (
    HTTP_RETRY_MAX_ATTEMPTS,
    HTTP_RETRY_MAX_WAIT_SECONDS,
    build_retry_transport,
)


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """事前に並べた callable を順番に呼び出すモックトランスポート。

    各エントリは ``Callable[[httpx.Request], httpx.Response]``。例外を返したい場合は
    例外を raise する callable を登録する。リクエスト数が script を超えた場合は
    AssertionError で test を fail させる (想定外の余分な retry を検知する)。
    """

    def __init__(
        self,
        responses: list[Callable[[httpx.Request], httpx.Response]],
    ) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        if self.call_count > len(self._responses):
            raise AssertionError(
                f"Unexpected extra request #{self.call_count}; only {len(self._responses)} responses scripted"
            )
        producer = self._responses[self.call_count - 1]
        return producer(request)


def _ok(body: bytes = b"OK") -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status_code=200, content=body, request=request)


def _status(
    status_code: int, headers: dict[str, str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def _producer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=status_code, headers=headers or {}, request=request)

    return _producer


def _raises(exc: BaseException) -> Callable[[httpx.Request], httpx.Response]:
    def _producer(_request: httpx.Request) -> httpx.Response:
        raise exc

    return _producer


def _make_transport_with_mock(scripted: _ScriptedTransport) -> AsyncTenacityTransport:
    """production の `build_retry_transport()` を使い、wrapped を mock に + wait を no-op にする。"""
    transport = build_retry_transport()
    transport.wrapped = scripted
    transport.config["wait"] = wait_none()
    return transport


async def _send_get(transport: AsyncTenacityTransport) -> httpx.Response:
    async with httpx.AsyncClient(transport=transport) as client:
        return await client.get("https://example.invalid/test")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestRetryableStatusCodes:
    """408/409/429/500/502/503/504 は `HTTP_RETRY_MAX_ATTEMPTS` まで retry される。"""

    def test_503_then_200_succeeds_on_second_attempt(self) -> None:
        scripted = _ScriptedTransport([_status(503), _ok()])
        transport = _make_transport_with_mock(scripted)

        response = _run(_send_get(transport))

        assert response.status_code == 200
        assert scripted.call_count == 2

    def test_503_persistent_exhausts_attempts_and_reraises(self) -> None:
        scripted = _ScriptedTransport([_status(503)] * HTTP_RETRY_MAX_ATTEMPTS)
        transport = _make_transport_with_mock(scripted)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _run(_send_get(transport))

        assert exc_info.value.response.status_code == 503
        # initial + 2 retries = 3 attempts ちょうどで止まる
        assert scripted.call_count == HTTP_RETRY_MAX_ATTEMPTS

    @pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504])
    def test_each_retryable_status_triggers_retry(self, status_code: int) -> None:
        scripted = _ScriptedTransport([_status(status_code), _ok()])
        transport = _make_transport_with_mock(scripted)

        response = _run(_send_get(transport))

        assert response.status_code == 200
        assert scripted.call_count == 2


class TestNonRetryableStatusCodes:
    """auth / bad request / not found は retry されず、SDK 例外層で扱う設計。"""

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_non_retryable_4xx_returns_immediately(self, status_code: int) -> None:
        scripted = _ScriptedTransport([_status(status_code)])
        transport = _make_transport_with_mock(scripted)

        # validate_response が raise しないので tenacity は retry しない。
        # response はそのまま httpx.Client に返り、caller (= SDK 層) で
        # raise_for_status() を呼ぶか response の status_code をチェックする想定。
        response = _run(_send_get(transport))

        assert response.status_code == status_code
        assert scripted.call_count == 1


class TestNetworkExceptionRetry:
    """`httpx.ConnectError` / `ReadTimeout` 等の transient network 例外も retry される。"""

    def test_connect_error_then_success(self) -> None:
        scripted = _ScriptedTransport([_raises(httpx.ConnectError("DNS resolution failed")), _ok()])
        transport = _make_transport_with_mock(scripted)

        response = _run(_send_get(transport))

        assert response.status_code == 200
        assert scripted.call_count == 2

    def test_read_timeout_exhausts_attempts(self) -> None:
        scripted = _ScriptedTransport(
            [_raises(httpx.ReadTimeout("read timeout")) for _ in range(HTTP_RETRY_MAX_ATTEMPTS)]
        )
        transport = _make_transport_with_mock(scripted)

        with pytest.raises(httpx.ReadTimeout):
            _run(_send_get(transport))

        assert scripted.call_count == HTTP_RETRY_MAX_ATTEMPTS

    def test_remote_protocol_error_then_success(self) -> None:
        scripted = _ScriptedTransport(
            [_raises(httpx.RemoteProtocolError("server closed connection")), _ok()]
        )
        transport = _make_transport_with_mock(scripted)

        response = _run(_send_get(transport))

        assert response.status_code == 200
        assert scripted.call_count == 2

    def test_terminal_exception_does_not_retry(self) -> None:
        # `httpx.HTTPError` 階層外の `RuntimeError` は retryable に含まれないので
        # initial 1 回で raise する。
        scripted = _ScriptedTransport([_raises(RuntimeError("unexpected"))])
        transport = _make_transport_with_mock(scripted)

        with pytest.raises(RuntimeError):
            _run(_send_get(transport))

        assert scripted.call_count == 1


class TestPolicyEvidence:
    """policy 定数値の根拠 regression test。値が変わったら test も意識的に更新する責務を強制する。"""

    def test_max_wait_seconds_is_60(self) -> None:
        # ADR 0023 Phase 1.8 補遺: provider token-bucket window 想定 (Anthropic continuously
        # replenished bucket / OpenAI RPM/ITPM)。pydantic-ai default 300s は CLI 想定で
        # Qt UI 用途には過大、変更時は ADR 補遺の再評価が必要。
        assert HTTP_RETRY_MAX_WAIT_SECONDS == 60.0

    def test_max_attempts_is_3(self) -> None:
        # ADR 0023 line 308: initial + 2 retries
        assert HTTP_RETRY_MAX_ATTEMPTS == 3
