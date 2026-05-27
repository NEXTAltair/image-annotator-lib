"""OpenAI Chat Completions Batch API runtime smoke (Issue #518, ADR 0001 amended).

LoRAIro #518 / iam-lib annotation Batch: PydanticAI 同期経路と同じ structured-output
(function_calling) contract が `/v1/chat/completions` Batch JSONL + output JSONL parse
で動作するかを開発者ローカルで検証する on-demand validation。

Runtime cost:
    1 batch submit + 即時 cancel ≒ $0 (batch 課金は completed 時のみ。cancel された
    batch は課金されない)。input file upload は 1 image / 数 KB。

Skip 仕様:
    `OPENAI_API_KEY` が未設定なら skip。resource image が存在しない環境でも skip。
    billing hard limit / quota 等の environment 起因失敗も explicit skip。

Marker:
    `@pytest.mark.calls_real_webapi` (CI 不経由、ローカル only)。

Coverage:
    Full lifecycle (poll until completed → fetch_batch_results → tool_call parse) は
    OpenAI Batch SLA が最大 24h のため smoke 内では実施しない。tool_calls JSONL parse
    は `tests/unit/webapi/batch/test_openai_adapter_chat_completions.py` の mocked
    fixture で別途検証する。

Related:
    iam-lib #118 / #119 / #120 / #122 / #123 (rating preflight infrastructure)
    LoRAIro #507 / #518 (Provider Batch annotation pipeline)
    ADR 0001 amended (runtime_validation lane)
"""

import os
from pathlib import Path

import pytest

from image_annotator_lib.webapi.batch.adapters.openai import OpenAIBatchAdapter
from image_annotator_lib.webapi.batch.types import (
    BatchJobError,
    BatchJobHandle,
    BatchStatus,
    BatchSubmitItem,
    BatchSubmitRequest,
)

_RESOURCE_IMG = Path(__file__).parent.parent / "resources" / "img" / "1_img" / "file07.webp"
_MODEL_ID = "openai/gpt-4o-mini"
_PROVIDER = "openai"


def _api_key_or_skip() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set (LoRAIro runtime validation lane)")
    return api_key


def _build_request(image_path: Path, api_key: str) -> BatchSubmitRequest:
    return BatchSubmitRequest(
        provider=_PROVIDER,
        endpoint="/v1/chat/completions",
        litellm_model_id=_MODEL_ID,
        prompt_profile="default",
        description="iam-lib #518 chat completions runtime smoke",
        api_keys={_PROVIDER: api_key},
        items=[
            BatchSubmitItem(
                custom_id="img-annot-smoke-1",
                image_id=1,
                image_path=image_path,
            ),
        ],
    )


@pytest.mark.calls_real_webapi
def test_openai_chat_completions_batch_submit_retrieve_cancel() -> None:
    """submit → retrieve → cancel の最小 lifecycle で実 OpenAI Batch contract を検証。

    確認項目:
        1. `submit_batch()` が input JSONL upload + batch create を完走し
           `provider_job_id` を返す (annotation tool 込みの request payload)
        2. `retrieve_batch()` で初期 status (`submitted` / `running` / `completed` /
           `unknown`) を取得できる
        3. `cancel_batch()` で課金回避 (smoke は full lifecycle を待たない)

    Full lifecycle (poll until completed → `fetch_batch_results` で tool_call parse)
    は OpenAI Batch SLA (最大 24h) のため smoke では実施しない。
    """
    api_key = _api_key_or_skip()
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    adapter = OpenAIBatchAdapter()
    try:
        submission = adapter.submit_batch(_build_request(_RESOURCE_IMG, api_key))
    except BatchJobError as exc:
        # billing / quota / file upload 等の environment 起因失敗は smoke の対象外
        msg = str(exc).lower()
        if any(keyword in msg for keyword in ("billing", "quota", "limit", "insufficient")):
            pytest.skip(f"OpenAI environment limit reached, skipping smoke: {exc}")
        raise

    assert submission.provider_job_id, "submit_batch did not return provider_job_id"
    assert submission.provider == _PROVIDER
    assert submission.request_count >= 1

    handle = BatchJobHandle(
        provider=_PROVIDER,
        provider_job_id=submission.provider_job_id,
        api_keys={_PROVIDER: api_key},
    )

    try:
        status = adapter.retrieve_batch(handle)
        assert status.provider_job_id == submission.provider_job_id
        assert status.status in {
            BatchStatus.SUBMITTED,
            BatchStatus.RUNNING,
            BatchStatus.COMPLETED,
            BatchStatus.UNKNOWN,
        }, f"unexpected initial status: {status.status}"
    finally:
        # 課金回避のため cancel を確実に呼ぶ
        canceled = adapter.cancel_batch(handle)
        assert canceled.provider_job_id == submission.provider_job_id
        assert canceled.status in {
            BatchStatus.CANCELED,
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.RUNNING,
        }, f"unexpected cancel status: {canceled.status}"
