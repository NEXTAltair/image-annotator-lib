from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_annotator_lib.webapi.batch import (
    BatchErrorPhase,
    BatchItemStatus,
    BatchJobError,
    BatchJobHandle,
    BatchProviderItemStatus,
    BatchStatus,
    BatchSubmitItem,
    BatchSubmitRequest,
)
from image_annotator_lib.webapi.batch.adapters import anthropic as anthropic_adapter_module
from image_annotator_lib.webapi.batch.adapters.anthropic import AnthropicBatchAdapter


@dataclass
class FakeBatches:
    created_requests: list[dict] | None = None
    results_stream: list[dict] | None = None

    def create(self, *, requests: list[dict]):
        self.created_requests = requests
        return {
            "id": "msgbatch_123",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": len(requests),
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }

    def retrieve(self, batch_id: str):
        return {
            "id": batch_id,
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 1,
                "canceled": 1,
                "expired": 1,
            },
            "created_at": "2026-05-25T00:00:00Z",
            "ended_at": "2026-05-25T01:00:00Z",
            "expires_at": "2026-05-26T00:00:00Z",
            "results_url": f"https://example.test/{batch_id}/results",
        }

    def cancel(self, batch_id: str):
        return {
            "id": batch_id,
            "processing_status": "canceling",
            "request_counts": {
                "processing": 2,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }

    def results(self, batch_id: str):
        return iter(self.results_stream or [])


class FakeAnthropic:
    batches = FakeBatches()

    def __init__(self, *, api_key: str):
        self.api_key = api_key
        self.messages = SimpleNamespace(batches=self.batches)


def install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, fake_batches: FakeBatches) -> None:
    class FakeAnthropicClient:
        def __init__(self, *, api_key: str):
            self.api_key = api_key
            self.messages = SimpleNamespace(batches=fake_batches)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropicClient))


def make_request(tmp_path: Path, image_name: str = "image.webp") -> BatchSubmitRequest:
    image_path = tmp_path / image_name
    image_path.write_bytes(b"fake-image-bytes")
    return BatchSubmitRequest(
        provider="anthropic",
        endpoint="messages",
        litellm_model_id="anthropic/claude-3-5-haiku-latest",
        prompt_profile="default",
        description=None,
        api_keys={"anthropic": "test-key"},
        items=[BatchSubmitItem(custom_id="img-1", image_id=1, image_path=image_path)],
    )


def test_submit_builds_anthropic_request_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_batches = FakeBatches()
    install_fake_anthropic(monkeypatch, fake_batches)

    result = AnthropicBatchAdapter().submit_batch(make_request(tmp_path))

    assert result.provider_job_id == "msgbatch_123"
    assert result.status is BatchStatus.RUNNING
    assert fake_batches.created_requests is not None
    provider_request = fake_batches.created_requests[0]
    assert provider_request["custom_id"] == "img-1"
    assert provider_request["params"]["model"] == "claude-3-5-haiku-latest"
    content = provider_request["params"]["messages"][0]["content"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/webp"
    assert content[1]["source"]["type"] == "base64"


def test_submit_rejects_invalid_custom_id(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    request.items[0] = BatchSubmitItem(
        custom_id="invalid custom id",
        image_id=1,
        image_path=request.items[0].image_path,
    )

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.phase is BatchErrorPhase.PREPARE
    assert exc_info.value.code == "invalid_custom_id"


def test_submit_rejects_unsupported_image_format(tmp_path: Path) -> None:
    request = make_request(tmp_path, image_name="image.gif")

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.code == "unsupported_image_format"


def test_submit_rejects_missing_api_key(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    request.api_keys.clear()

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.code == "missing_api_key"


def test_submit_rejects_payload_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = make_request(tmp_path)
    monkeypatch.setattr(anthropic_adapter_module, "_MAX_ANTHROPIC_BODY_BYTES", 1)

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.code == "payload_too_large"


def test_submit_rejects_unsupported_prompt_profile(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    request = BatchSubmitRequest(
        provider=request.provider,
        endpoint=request.endpoint,
        litellm_model_id=request.litellm_model_id,
        prompt_profile="alternate",
        description=request.description,
        api_keys=request.api_keys,
        items=request.items,
    )

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.code == "unsupported_prompt_profile"


def test_submit_wraps_invalid_model_id(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    request = BatchSubmitRequest(
        provider=request.provider,
        endpoint=request.endpoint,
        litellm_model_id="not-a-litellm-id",
        prompt_profile=request.prompt_profile,
        description=request.description,
        api_keys=request.api_keys,
        items=request.items,
    )

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.phase is BatchErrorPhase.PREPARE
    assert exc_info.value.code == "invalid_model_id"


def test_submit_wraps_image_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = make_request(tmp_path)

    def fail_read_bytes(self: Path) -> bytes:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().submit_batch(request)

    assert exc_info.value.phase is BatchErrorPhase.PREPARE
    assert exc_info.value.code == "image_read_failed"


def test_retrieve_and_cancel_normalize_status(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_batches = FakeBatches()
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )
    adapter = AnthropicBatchAdapter()

    retrieved = adapter.retrieve_batch(handle)
    canceled = adapter.cancel_batch(handle)

    assert retrieved.status is BatchStatus.COMPLETED
    assert retrieved.request_count == 4
    assert retrieved.failed_count == 1
    assert retrieved.submitted_at is not None
    assert canceled.status is BatchStatus.RUNNING


def test_fetch_results_stream_normalizes_success_and_terminal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_batches = FakeBatches(
        results_stream=[
            {
                "custom_id": "img-2",
                "result": {
                    "type": "errored",
                    "error": {"message": "invalid request"},
                },
            },
            {
                "custom_id": "img-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"tags":["tag a"],"captions":["caption"],"score":7.5}',
                            }
                        ],
                        "stop_reason": "end_turn",
                    },
                },
            },
            {"custom_id": "img-3", "result": {"type": "canceled"}},
            {"custom_id": "img-4", "result": {"type": "expired"}},
        ]
    )
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )

    result = AnthropicBatchAdapter().fetch_batch_results(handle)

    assert [item.custom_id for item in result.items] == ["img-2", "img-1", "img-3", "img-4"]
    success = result.items[1]
    assert success.status is BatchItemStatus.SUCCEEDED
    assert success.provider_status is BatchProviderItemStatus.SUCCEEDED
    assert success.annotation is not None
    assert success.annotation.tags == ["tag a"]
    assert result.items[0].error is not None
    assert result.items[0].error.code == "provider_item_error"
    assert result.items[0].error.retryable is False
    assert result.items[2].provider_status is BatchProviderItemStatus.CANCELED
    assert result.items[3].provider_status is BatchProviderItemStatus.EXPIRED


def test_fetch_results_marks_transient_provider_item_error_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_batches = FakeBatches(
        results_stream=[
            {
                "custom_id": "img-1",
                "result": {
                    "type": "errored",
                    "error": {"type": "server_error", "message": "try later"},
                },
            }
        ]
    )
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )

    result = AnthropicBatchAdapter().fetch_batch_results(handle)

    assert result.items[0].error is not None
    assert result.items[0].error.retryable is True


def test_fetch_results_all_errored_terminal_batch_is_fetchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AllErroredBatches(FakeBatches):
        def retrieve(self, batch_id: str):
            return {
                "id": batch_id,
                "processing_status": "ended",
                "request_counts": {
                    "processing": 0,
                    "succeeded": 0,
                    "errored": 1,
                    "canceled": 0,
                    "expired": 0,
                },
            }

    fake_batches = AllErroredBatches(
        results_stream=[
            {
                "custom_id": "img-1",
                "result": {"type": "errored", "error": {"message": "bad request"}},
            }
        ]
    )
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )

    result = AnthropicBatchAdapter().fetch_batch_results(handle)

    assert result.status is BatchStatus.FAILED
    assert result.items[0].error is not None
    assert result.items[0].error.code == "provider_item_error"


def test_fetch_results_maps_refusal_to_item_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_batches = FakeBatches(
        results_stream=[
            {
                "custom_id": "img-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "I cannot help with that."}],
                        "stop_reason": "refusal",
                    },
                },
            },
        ]
    )
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )

    result = AnthropicBatchAdapter().fetch_batch_results(handle)

    item = result.items[0]
    assert item.status is BatchItemStatus.FAILED
    assert item.error is not None
    assert item.error.code == "safety_refusal"


def test_fetch_results_invalid_json_line_raises_batch_job_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_batches = FakeBatches(results_stream=["not-json"])
    install_fake_anthropic(monkeypatch, fake_batches)
    handle = BatchJobHandle(
        provider="anthropic", provider_job_id="msgbatch_123", api_keys={"anthropic": "k"}
    )

    with pytest.raises(BatchJobError) as exc_info:
        AnthropicBatchAdapter().fetch_batch_results(handle)

    assert exc_info.value.phase is BatchErrorPhase.PARSE
    assert exc_info.value.code == "result_stream_parse_failed"
