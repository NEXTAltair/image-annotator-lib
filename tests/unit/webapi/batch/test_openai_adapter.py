"""Tests for OpenAI Batch adapter."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_annotator_lib.core.types import RatingPrediction
from image_annotator_lib.webapi.batch import (
    BatchItemStatus,
    BatchProviderItemStatus,
    BatchStatus,
    BatchSubmitItem,
    BatchSubmitRequest,
    BatchJobHandle,
)
from image_annotator_lib.webapi.batch.adapters import openai as openai_adapter_module
from image_annotator_lib.webapi.batch.adapters.openai import OpenAIBatchAdapter


def _jsonl(lines: list[object]) -> str:
    return "\n".join(json.dumps(line, separators=(",", ":")) for line in lines)


@dataclass
class FakeOpenAIFiles:
    uploaded_file_content: str | None = None
    contents: dict[str, str] | None = None

    def create(self, *, file: tuple[str, str], purpose: str) -> dict[str, str]:
        self.uploaded_file_content = file[1]
        return {"id": "file_001"}

    def content(self, file_id: str) -> str:
        if not self.contents or file_id not in self.contents:
            raise RuntimeError(f"missing content for {file_id}")
        return self.contents[file_id]


@dataclass
class FakeOpenAIBatches:
    created_input_file_id: str | None = None
    created_endpoint: str | None = None
    output_payload: dict[str, object] | None = None
    output_file_id: str = "file_out"
    error_file_id: str = "file_err"

    def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> dict[str, object]:
        self.created_input_file_id = input_file_id
        self.created_endpoint = endpoint
        return {
            "id": "batch_001",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 2,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }

    def retrieve(self, batch_id: str) -> dict[str, object]:
        if self.output_payload is None:
            return {
                "id": batch_id,
                "processing_status": "completed",
                "request_counts": {
                    "processing": 0,
                    "succeeded": 2,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                },
                "output_file_id": self.output_file_id,
                "error_file_id": self.error_file_id,
            }
        return self.output_payload

    def cancel(self, batch_id: str) -> dict[str, object]:
        return {"id": batch_id, "processing_status": "cancelled", "request_counts": {"canceled": 1}}


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, files: FakeOpenAIFiles, batches: FakeOpenAIBatches) -> None:
    class FakeOpenAIFactory:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.files = files
            self.batches = batches

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIFactory))


def make_request(tmp_path: Path, model_id: str = "openai/omni-moderation-latest") -> BatchSubmitRequest:
    image_path = tmp_path / "img-1.png"
    image_path.write_bytes(b"fake-image-bytes")
    return BatchSubmitRequest(
        provider="openai",
        endpoint="/v1/moderations",
        litellm_model_id=model_id,
        prompt_profile="default",
        description=None,
        api_keys={"openai": "test-key"},
        items=[BatchSubmitItem(custom_id="img-1", image_id=1, image_path=image_path)],
    )


def test_submit_batch_builds_openai_jsonl_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FakeOpenAIFiles()
    batches = FakeOpenAIBatches()
    install_fake_openai(monkeypatch, files=files, batches=batches)
    request = make_request(tmp_path)

    result = OpenAIBatchAdapter().submit_batch(request)

    assert result.provider_job_id == "batch_001"
    assert result.status is BatchStatus.RUNNING
    assert files.uploaded_file_content is not None
    assert batches.created_endpoint == "/v1/moderations"
    payload = files.uploaded_file_content.splitlines()[0]
    assert '"custom_id":"img-1"' in payload
    assert '"model":"omni-moderation-latest"' in payload


def test_fetch_batch_results_normalizes_openai_output_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FakeOpenAIFiles(
        contents={
            "file_out": _jsonl(
                [
                    {
                        "custom_id": "img-1",
                        "response": {"status_code": 200, "body": {"results": [{"category_scores": {"sexual": 0.88}}]}},
                    },
                    {
                        "custom_id": "img-2",
                        "response": {"status_code": 200, "body": {"results": [{"category_scores": {"violence": 0.25}}]}},
                    },
                ]
            ),
            "file_err": _jsonl(
                [
                    {"custom_id": "img-3", "error": {"type": "server_error", "message": "provider transient"}},
                ]
            ),
        }
    )
    batches = FakeOpenAIBatches()
    install_fake_openai(monkeypatch, files=files, batches=batches)
    monkeypatch.setattr(
        openai_adapter_module,
        "_CATEGORY_SCORES_TO_RATING_PREDICTION",
        lambda _: RatingPrediction(raw_label="r", source_scheme="openai_moderation_v1", confidence_score=0.88),
    )

    handle = BatchJobHandle(provider="openai", provider_job_id="batch_001", api_keys={"openai": "test-key"})
    result = OpenAIBatchAdapter().fetch_batch_results(handle)

    assert result.status is BatchStatus.COMPLETED
    assert len(result.items) == 3
    ids = [item.custom_id for item in result.items]
    assert ids == ["img-1", "img-2", "img-3"]
    assert result.items[0].status is BatchItemStatus.SUCCEEDED
    assert result.items[0].annotation is not None
    assert result.items[0].annotation.ratings[0].raw_label == "r"
    assert result.items[2].status is BatchItemStatus.FAILED
    assert result.items[2].provider_status is BatchProviderItemStatus.FAILED


def test_fetch_batch_results_marks_output_item_error_and_ignores_extra_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FakeOpenAIFiles(
        contents={
            "file_out": _jsonl(
                [
                    {
                        "custom_id": "img-1",
                        "response": {"status_code": 400, "error": {"message": "model cannot parse"}},
                    },
                    {"note": "meta"},
                    {
                        "custom_id": "img-2",
                        "response": {"status_code": 200, "body": {"results": [{"category_scores": {"sexual": 0.5}}]}},
                    },
                ]
            ),
            "file_err": "",
        }
    )
    batches = FakeOpenAIBatches(
        output_payload={
            "id": "batch_001",
            "processing_status": "completed",
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 1,
                "canceled": 0,
                "expired": 0,
            },
            "output_file_id": "file_out",
            "error_file_id": "file_err",
        },
    )
    install_fake_openai(monkeypatch, files=files, batches=batches)
    monkeypatch.setattr(
        openai_adapter_module,
        "_CATEGORY_SCORES_TO_RATING_PREDICTION",
        lambda _: RatingPrediction(raw_label="pg", source_scheme="openai_moderation_v1", confidence_score=None),
    )

    handle = BatchJobHandle(provider="openai", provider_job_id="batch_001", api_keys={"openai": "test-key"})
    result = OpenAIBatchAdapter().fetch_batch_results(handle)

    assert [item.custom_id for item in result.items] == ["img-1", "img-2"]
    assert result.items[0].status is BatchItemStatus.FAILED
    assert result.items[1].status is BatchItemStatus.SUCCEEDED


def test_fetch_batch_results_preserves_custom_id_mapping_for_success_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FakeOpenAIFiles(
        contents={
            "file_out": _jsonl(
                [
                    {
                        "custom_id": "custom-image-42",
                        "response": {"status_code": 200, "body": {"results": [{"category_scores": {"sexual": 0.2}}]}},
                    },
                ]
            ),
            "file_err": "",
        }
    )
    batches = FakeOpenAIBatches()
    install_fake_openai(monkeypatch, files=files, batches=batches)
    monkeypatch.setattr(
        openai_adapter_module,
        "_CATEGORY_SCORES_TO_RATING_PREDICTION",
        lambda _: RatingPrediction(raw_label="pg", source_scheme="openai_moderation_v1", confidence_score=None),
    )

    handle = BatchJobHandle(provider="openai", provider_job_id="batch_001", api_keys={"openai": "test-key"})
    result = OpenAIBatchAdapter().fetch_batch_results(handle)

    assert result.items[0].custom_id == "custom-image-42"
    assert result.items[0].annotation is not None
