"""LoRAIro #302: sync wrapper の event loop probe が診断 chain を汚さないことの検証。

``ProviderManager.run_inference_with_model`` は running loop 検出に
``asyncio.get_running_loop()`` を使う。旧実装はこの probe の ``RuntimeError`` を
``except`` ブロックで受け、その **中** で ``asyncio.run()`` を呼んでいた。すると
coroutine 内で raise される素の例外 (`__cause__` 無し) の ``__context__`` に probe の
``RuntimeError("no running event loop")`` が連鎖し、``_format_exception_chain`` が
生成する診断文字列に偽の根本原因として混入していた (LoRAIro #274 の実機検証で発覚)。

修正後は probe を ``_running_loop_active()`` helper に分離し ``RuntimeError`` を
完全に消費してから ``asyncio.run()`` を ``except`` ブロック外で実行する。本モジュールは
その回帰防止と probe helper / running-loop ガードの挙動を検証する。
"""

from __future__ import annotations

import asyncio

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.exceptions.errors import InferenceError


class _FakeRateLimitError(Exception):
    """429 retry 枯渇後に transport が reraise する ``httpx.HTTPStatusError`` 相当。

    ``raise_for_status()`` の ``from`` 無し raise と同じく ``__cause__`` を持たない素の
    例外。診断 chain walker (`_format_exception_chain`) が ``__cause__`` 経路で止まれず
    ``__context__`` 経路へ落ちる条件を再現する。
    """


class _FailingAgent:
    """``run()`` が ``__cause__`` 無しの素の例外を fresh raise する最小 Agent スタブ。

    ``AsyncMock(side_effect=instance)`` ではなく実 ``async def`` 内の fresh ``raise`` を
    使うことで、``__context__`` 連鎖を確実に再現する。
    """

    async def run(self, *_args: object, **_kwargs: object) -> object:
        raise _FakeRateLimitError("Client error '429 Too Many Requests'")


@pytest.mark.unit
class TestSyncWrapperEventLoopContext:
    """``run_inference_with_model`` の sync wrapper が偽の ``__context__`` を残さない。"""

    def test_sync_wrapper_error_does_not_leak_no_running_event_loop(self) -> None:
        """sync wrapper 経由の error 文字列に probe の RuntimeError が混入しない (#302)。"""
        image = Image.new("RGB", (8, 8), color="white")

        results = ProviderManager.run_inference_with_model(
            model_name="test-webapi",
            images_list=[image],
            litellm_model_id="openai/gpt-4o-mini",
            api_keys={"openai": "fake-key"},
            _test_agent=_FailingAgent(),
        )

        assert len(results) == 1
        error_msg = results[next(iter(results))]["error"]
        assert error_msg is not None
        # 真の原因 (429) は診断 chain に保持される
        assert "429" in error_msg
        # probe RuntimeError の __context__ 漏れが混入しない
        assert "no running event loop" not in error_msg

    def test_sync_wrapper_from_running_loop_raises_inference_error(self) -> None:
        """running loop 内からの呼び出しは従来通り ``InferenceError`` を raise する。"""

        async def _call_from_loop() -> None:
            ProviderManager.run_inference_with_model(
                model_name="test-webapi",
                images_list=[Image.new("RGB", (8, 8), color="white")],
                litellm_model_id="openai/gpt-4o-mini",
                api_keys={"openai": "fake-key"},
                _test_agent=_FailingAgent(),
            )

        with pytest.raises(InferenceError):
            asyncio.run(_call_from_loop())


@pytest.mark.unit
class TestRunningLoopActive:
    """``_running_loop_active()`` probe helper の挙動。"""

    def test_running_loop_active_false_outside_loop(self) -> None:
        """event loop が無いスレッドでは ``False`` を返す。"""
        assert ProviderManager._running_loop_active() is False

    def test_running_loop_active_true_inside_asyncio_run(self) -> None:
        """``asyncio.run`` の coroutine 内では ``True`` を返す。"""

        async def _check() -> bool:
            return ProviderManager._running_loop_active()

        assert asyncio.run(_check()) is True
