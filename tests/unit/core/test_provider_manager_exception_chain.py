"""LoRAIro #274: WebAPI 推論失敗時の診断情報強化。

``openai.APIConnectionError`` 等の SDK ラッパー例外は ``str(exc)`` が
``"Connection error."`` に潰れ、根本原因 (httpx の ``ConnectTimeout`` /
``ConnectError`` 等) が ``AnnotationResult.error`` から失われる。

``provider_manager._format_exception_chain`` が cause/context chain を辿って
原因特定に足る診断文字列を生成し、``run_inference_with_model`` の generic error
経路が log・result.error の双方にそれを伝搬することを検証する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from image_annotator_lib.core.provider_manager import (
    ProviderManager,
    _format_exception_chain,
)


@pytest.mark.unit
class TestFormatExceptionChain:
    """``_format_exception_chain`` の chain 整形ロジック。"""

    def test_single_exception_no_cause(self) -> None:
        """cause/context が無い例外は単層の '型名: message' になる。"""
        result = _format_exception_chain(ValueError("bad value"))
        # builtins は module prefix を付けない
        assert result == "ValueError: bad value"

    def test_non_builtin_module_prefixed(self) -> None:
        """builtins 以外の例外は module 修飾名が付く。"""

        class _CustomError(Exception):
            pass

        result = _format_exception_chain(_CustomError("boom"))
        assert result.endswith("_CustomError: boom")
        # module prefix (テストモジュール名) が付与される
        assert "." in result.split(":")[0]

    def test_empty_message_placeholder(self) -> None:
        """message が空の例外は '(no message)' に置換される。"""
        result = _format_exception_chain(ValueError(""))
        assert result == "ValueError: (no message)"

    def test_explicit_cause_chain(self) -> None:
        """raise X from Y は 'X <- caused by Y' で連結される。"""
        try:
            try:
                raise ConnectionResetError("tcp reset")
            except ConnectionResetError as inner:
                raise RuntimeError("Connection error.") from inner
        except RuntimeError as exc:
            result = _format_exception_chain(exc)

        assert result == "RuntimeError: Connection error. <- caused by ConnectionResetError: tcp reset"

    def test_implicit_context_chain(self) -> None:
        """except 中の暗黙 raise は __context__ 経由で辿られる。"""
        try:
            try:
                raise ConnectionResetError("tcp reset")
            except ConnectionResetError:
                raise RuntimeError("Connection error.")  # noqa: B904
        except RuntimeError as exc:
            result = _format_exception_chain(exc)

        assert "RuntimeError: Connection error." in result
        assert "caused by ConnectionResetError: tcp reset" in result

    def test_suppressed_context_not_walked(self) -> None:
        """raise X from None は context を辿らない。"""
        try:
            try:
                raise ConnectionResetError("tcp reset")
            except ConnectionResetError:
                raise RuntimeError("Connection error.") from None
        except RuntimeError as exc:
            result = _format_exception_chain(exc)

        assert result == "RuntimeError: Connection error."
        assert "tcp reset" not in result

    def test_max_depth_caps_chain(self) -> None:
        """max_depth で chain の段数が制限される。"""
        exc = ValueError("level-0")
        exc.__cause__ = ValueError("level-1")
        exc.__cause__.__cause__ = ValueError("level-2")

        result = _format_exception_chain(exc, max_depth=2)

        assert "level-0" in result
        assert "level-1" in result
        assert "level-2" not in result

    def test_cycle_is_safe(self) -> None:
        """cause が循環していても無限ループしない。"""
        exc_a = ValueError("a")
        exc_b = ValueError("b")
        exc_a.__cause__ = exc_b
        exc_b.__cause__ = exc_a

        result = _format_exception_chain(exc_a)

        # 各例外は 1 回ずつだけ現れる
        assert result.count("ValueError: a") == 1
        assert result.count("ValueError: b") == 1


@pytest.mark.unit
def test_run_inference_error_propagates_exception_chain() -> None:
    """LoRAIro #274 regression: generic error 経路で result.error に cause chain が乗る。

    ``agent.run()`` が ``APIConnectionError`` 相当 (str() は "Connection error." に
    潰れるが ``__cause__`` に httpx 例外相当を持つ) を raise した場合、
    ``result["error"]`` が単なる "Connection error." ではなく根本原因を含む。
    """

    class _FakeAPIConnectionError(Exception):
        """openai.APIConnectionError 相当 (str() が定型文に潰れる)。"""

        def __str__(self) -> str:
            return "Connection error."

    def _raise_chained() -> None:
        try:
            raise ConnectionError("All connection attempts failed")
        except ConnectionError as inner:
            raise _FakeAPIConnectionError() from inner

    try:
        _raise_chained()
    except _FakeAPIConnectionError as built:
        chained_exc = built

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(side_effect=chained_exc)

    image = Image.new("RGB", (8, 8), color="white")
    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=[image],
        litellm_model_id="openai/gpt-4o-mini",
        api_keys={"openai": "fake-key"},
        _test_agent=fake_agent,
    )

    assert len(results) == 1
    error_msg = results[next(iter(results))]["error"]
    assert error_msg is not None
    # 定型文だけでなく根本原因 (inner ConnectionError) が含まれる
    assert "Connection error." in error_msg
    assert "caused by" in error_msg
    assert "All connection attempts failed" in error_msg
    # 単純な str(exc) ("Connection error.") ではない
    assert error_msg != "Connection error."
