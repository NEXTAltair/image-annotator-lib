"""OpenAI Moderations Batch API runtime smoke (ADR 0001 amended / LoRAIro ADR 0026).

LoRAIro #507 / iam-lib #120: provider artifact JSONL parse + custom_id 経由の
`RatingPrediction` 構築が実 OpenAI Batch contract と整合することを開発者ローカルで
検証する on-demand validation。

Runtime cost:
    1 batch submit + 即時 cancel ≒ $0 (batch 課金は completed 時のみ。cancel された
    batch は課金されない)。input file upload は最小 1 image / 数 KB。

Skip 仕様:
    `OPENAI_API_KEY` が未設定なら skip。
    resource image が存在しない環境でも skip。

Marker:
    `@pytest.mark.calls_real_webapi` (CI 不経由、ローカル only)。

Coverage:
    Full lifecycle (submit → poll → completed → fetch_batch_results) は OpenAI Batch
    SLA が最大 24h のため smoke 内では実施しない。fetch の JSONL parse は
    `tests/unit/webapi/batch/test_openai_adapter.py` の mocked fixture で検証する。

Related:
    iam-lib #119 / #120 / #121 / #122 (T1+T2 既 merged)
    LoRAIro #505 / #506 / #509 / #510 / #511 (既 merged)
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
_MODEL_ID = "openai/omni-moderation-latest"
_PROVIDER = "openai"


def _api_key_or_skip() -> str:
    """Return OPENAI_API_KEY or skip the test."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set (LoRAIro runtime validation lane)")
    return api_key


def _build_request(image_path: Path, api_key: str) -> BatchSubmitRequest:
    return BatchSubmitRequest(
        provider=_PROVIDER,
        endpoint="/v1/moderations",
        litellm_model_id=_MODEL_ID,
        prompt_profile="default",
        description="iam-lib #120 runtime smoke",
        api_keys={_PROVIDER: api_key},
        items=[
            BatchSubmitItem(
                custom_id="img-smoke-1",
                image_id=1,
                image_path=image_path,
            ),
        ],
    )


@pytest.mark.calls_real_webapi
def test_openai_moderations_batch_submit_retrieve_cancel() -> None:
    """submit → retrieve → cancel の最小 lifecycle で実 OpenAI Batch contract を検証。

    確認項目:
        1. `submit_batch()` が `provider_job_id` を返す (input file upload + batch create)
        2. `retrieve_batch()` が初期 status (submitted / running / validating) を返す
        3. `cancel_batch()` で課金回避 (smoke は full lifecycle を待たない)

    Full lifecycle (poll until completed → fetch_batch_results) は OpenAI Batch SLA
    (最大 24h) のため smoke では実施しない。output JSONL parse は unit test (mocked)
    で別途検証する。
    """
    api_key = _api_key_or_skip()
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    adapter = OpenAIBatchAdapter()
    try:
        submission = adapter.submit_batch(_build_request(_RESOURCE_IMG, api_key))
    except BatchJobError as exc:
        # billing / quota / file upload size などの environment 起因失敗は smoke の
        # 検証対象外。LoRAIro 側 wiring と adapter 自体は別途 unit test (mocked) で
        # 担保されており、ここでは provider 側 contract のみを確認する。
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
        # cancel を確実に呼ぶ (課金 / quota / job queue cleanup)
        canceled = adapter.cancel_batch(handle)
        assert canceled.provider_job_id == submission.provider_job_id
        assert canceled.status in {
            BatchStatus.CANCELED,
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.RUNNING,
        }, f"unexpected cancel status: {canceled.status}"
