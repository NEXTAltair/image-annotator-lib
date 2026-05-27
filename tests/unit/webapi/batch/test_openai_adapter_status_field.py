"""Tests for OpenAI Batch adapter status mapping against real API response shape.

OpenAI Batch API は実際には `status` field と `request_counts: {completed, failed,
total}` を返す。古い実装は `processing_status` と `request_counts: {succeeded,
errored}` を仮定していたが #518 runtime smoke で `BatchStatus.UNKNOWN` が出て発覚。
本 file は実 API shape を unit test 側で固定する regression guard。
"""

from __future__ import annotations

from image_annotator_lib.webapi.batch.adapters import openai as openai_adapter_module
from image_annotator_lib.webapi.batch.types import BatchStatus

# Real OpenAI Batch object shape (確認: 2026-05-27 / openai SDK 経由)。
_REAL_CANCELLING_BATCH = {
    "id": "batch_abc",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "status": "cancelling",
    "processing_status": None,  # 実 API では None
    "request_counts": {"completed": 0, "failed": 0, "total": 0},
}

_REAL_CANCELLED_BATCH = {
    "id": "batch_abc",
    "object": "batch",
    "status": "cancelled",
    "processing_status": None,
    "request_counts": {"completed": 0, "failed": 0, "total": 0},
}

_REAL_COMPLETED_BATCH = {
    "id": "batch_abc",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "status": "completed",
    "processing_status": None,
    "request_counts": {"completed": 3, "failed": 0, "total": 3},
}

_REAL_FAILED_BATCH = {
    "id": "batch_abc",
    "status": "failed",
    "processing_status": None,
    "request_counts": {"completed": 0, "failed": 2, "total": 2},
}


def test_status_from_batch_reads_status_field_for_cancelling() -> None:
    """OpenAI 実 response の `status: cancelling` を RUNNING に map する。"""
    assert openai_adapter_module._status_from_batch(_REAL_CANCELLING_BATCH) is BatchStatus.RUNNING


def test_status_from_batch_reads_status_field_for_cancelled() -> None:
    assert openai_adapter_module._status_from_batch(_REAL_CANCELLED_BATCH) is BatchStatus.CANCELED


def test_status_from_batch_reads_status_field_for_completed() -> None:
    assert openai_adapter_module._status_from_batch(_REAL_COMPLETED_BATCH) is BatchStatus.COMPLETED


def test_status_from_batch_reads_status_field_for_failed() -> None:
    assert openai_adapter_module._status_from_batch(_REAL_FAILED_BATCH) is BatchStatus.FAILED


def test_request_count_reads_total_from_real_response() -> None:
    """OpenAI 実 response の `request_counts.total` を request_count として返す。"""
    assert openai_adapter_module._request_count(_REAL_COMPLETED_BATCH) == 3


def test_count_succeeded_reads_completed_key_from_real_response() -> None:
    """`succeeded` (legacy 名) は実 response の `completed` key にマップされる。"""
    assert openai_adapter_module._count(_REAL_COMPLETED_BATCH, "succeeded") == 3


def test_count_errored_reads_failed_key_from_real_response() -> None:
    """`errored` (legacy 名) は実 response の `failed` key にマップされる。"""
    assert openai_adapter_module._count(_REAL_FAILED_BATCH, "errored") == 2


def test_status_from_batch_keeps_processing_status_backward_compat() -> None:
    """既存 mock test との backward compat: `processing_status` だけ持つ payload も読める。"""
    legacy = {"id": "batch_legacy", "processing_status": "in_progress", "request_counts": {}}
    assert openai_adapter_module._status_from_batch(legacy) is BatchStatus.RUNNING
