"""Issue #69 / LoRAIro #275: logger.error f-string + exc_info=True で str(exc) に
dict 表現が含まれると loguru が format placeholder と解釈し KeyError を leak する。

Regression test for the loguru f-string formatting bug discovered in
`provider_manager.py:run_inference_with_model_async`.

問題:
    `logger.error(f"WebAPI 推論失敗: ...error={exc}", exc_info=True)` で `str(exc)` が
    Python dict 表現 (`{'type': 'error', ...}`) を含むと、loguru の内部 record 構築で
    message を再フォーマットし、`{'type'}` を positional/keyword placeholder と解釈して
    `KeyError("'type'")` を raise する。

検証経路:
    `ProviderManager.run_inference_with_model` 経由で `_test_agent` 注入し、agent.run()
    で ModelHTTPError 相当の例外を raise させる。修正前は KeyError がそのまま leak、
    修正後は例外を swallow して result.error にオリジナル `str(exc)` を格納する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from image_annotator_lib.webapi.provider_manager import ProviderManager


class _FakeModelHTTPError(Exception):
    """`pydantic_ai.exceptions.ModelHTTPError` 相当の str() を返す例外。

    body 属性は省略 (logger.error 経路の再現に str() のみが必要)。
    実 ModelHTTPError は body を Python dict として保持し、`__str__` で `f"...body: {body}"`
    を返すため、最終的に str(exc) に `{'type': 'error', ...}` が含まれる。本テストでは
    その str() 結果を直接模倣する。
    """

    def __str__(self) -> str:
        return (
            "status_code: 400, model_name: claude-haiku-4-5, "
            "body: {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'foo'}, 'request_id': 'req_xyz'}"
        )


@pytest.mark.unit
def test_logger_error_does_not_leak_keyerror_when_exception_str_contains_dict() -> None:
    """Regression test for iam-lib #69 / LoRAIro #275.

    `logger.error(f"...{exc}", exc_info=True)` で str(exc) に dict 表現が含まれても
    `KeyError("'type'")` が leak せず、result.error に str(exc) がそのまま格納される
    ことを確認する。
    """
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(side_effect=_FakeModelHTTPError())

    image = Image.new("RGB", (8, 8), color="white")

    # 修正前: provider_manager.py:132 の logger.error で KeyError("'type'") が raise され、
    # この呼び出し全体が KeyError で例外伝搬する (本 assert 行までも到達しない)。
    # 修正後: 例外は内部で swallow され、result に error メッセージが入る。
    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=[image],
        litellm_model_id="anthropic/claude-haiku-4-5",
        api_keys={"anthropic": "fake-key"},
        _test_agent=fake_agent,
    )

    assert len(results) == 1
    result = next(iter(results.values()))
    error_msg = result["error"]
    # str(exc) がそのまま error に流れること (KeyError の "'type'" ではない)
    assert error_msg is not None
    assert "status_code: 400" in error_msg
    assert "claude-haiku-4-5" in error_msg
    assert error_msg != "'type'"
