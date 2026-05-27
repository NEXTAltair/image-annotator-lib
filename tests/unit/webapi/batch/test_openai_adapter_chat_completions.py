"""Tests for OpenAI Batch adapter `/v1/chat/completions` annotation path (Issue #518)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.webapi.batch import (
    BatchItemStatus,
    BatchProviderItemStatus,
    BatchStatus,
    BatchSubmitItem,
    BatchSubmitRequest,
    BatchJobHandle,
)
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
    output_file_id: str = "file_out"

    def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> dict[str, object]:
        self.created_input_file_id = input_file_id
        self.created_endpoint = endpoint
        return {
            "id": "batch_001",
            "processing_status": "in_progress",
            "endpoint": endpoint,
            "request_counts": {
                "processing": 1,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }

    def retrieve(self, batch_id: str) -> dict[str, object]:
        return {
            "id": batch_id,
            "processing_status": "completed",
            "endpoint": "/v1/chat/completions",
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "output_file_id": self.output_file_id,
        }


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch, files: FakeOpenAIFiles, batches: FakeOpenAIBatches
) -> None:
    class FakeOpenAIFactory:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.files = files
            self.batches = batches

    from image_annotator_lib.webapi.batch.adapters import openai as openai_adapter_module

    monkeypatch.setattr(
        openai_adapter_module.OpenAIBatchAdapter,
        "_client",
        lambda self, api_key: FakeOpenAIFactory(api_key),
    )


def _make_request(tmp_path: Path) -> BatchSubmitRequest:
    image_path = tmp_path / "annot_input.png"
    # minimal PNG header (1x1)
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f"
        b"\x15\xc4\x89\x00\x00\x00\rIDAT\x78\x9c\x62\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return BatchSubmitRequest(
        provider="openai",
        endpoint="/v1/chat/completions",
        litellm_model_id="openai/gpt-4o-mini",
        prompt_profile="default",
        description="annotation batch smoke",
        api_keys={"openai": "sk-test"},
        items=[
            BatchSubmitItem(custom_id="img-1", image_id=1, image_path=image_path),
        ],
    )


def _success_output_line() -> dict[str, object]:
    arguments = {
        "tags": ["1girl", "blue_eyes", "school_uniform"],
        "captions": ["A girl with blue eyes wearing a school uniform."],
        "score": 7.5,
    }
    return {
        "custom_id": "img-1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "normalize_annotation_output",
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        },
    }


def test_submit_batch_uploads_chat_completions_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FakeOpenAIFiles()
    batches = FakeOpenAIBatches()
    _install_fake_openai(monkeypatch, files, batches)
    adapter = OpenAIBatchAdapter()

    result = adapter.submit_batch(_make_request(tmp_path))

    assert result.provider_job_id == "batch_001"
    assert batches.created_endpoint == "/v1/chat/completions"
    assert files.uploaded_file_content is not None
    payload = json.loads(files.uploaded_file_content.splitlines()[0])
    assert payload["url"] == "/v1/chat/completions"
    body = payload["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "normalize_annotation_output"},
    }
    assert body["tools"][0]["function"]["parameters"]["required"] == [
        "captions",
        "score",
        "tags",
    ]


def test_fetch_batch_results_parses_tool_call_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FakeOpenAIFiles(contents={"file_out": _jsonl([_success_output_line()])})
    batches = FakeOpenAIBatches()
    _install_fake_openai(monkeypatch, files, batches)
    adapter = OpenAIBatchAdapter()
    adapter.submit_batch(_make_request(tmp_path))

    fetch = adapter.fetch_batch_results(
        BatchJobHandle(
            provider="openai",
            provider_job_id="batch_001",
            api_keys={"openai": "sk-test"},
        )
    )

    assert fetch.status is BatchStatus.COMPLETED
    assert len(fetch.items) == 1
    item = fetch.items[0]
    assert item.custom_id == "img-1"
    assert item.status is BatchItemStatus.SUCCEEDED
    annotation = item.annotation
    assert annotation is not None
    assert annotation.tags == ["1girl", "blue_eyes", "school_uniform"]
    assert annotation.captions == ["A girl with blue eyes wearing a school uniform."]
    assert annotation.scores == {"score": 7.5}
    assert TaskCapability.TAGS in annotation.capabilities
    assert TaskCapability.SCORES in annotation.capabilities


def test_fetch_batch_results_normalizes_content_filter_as_safety_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal_line = {
        "custom_id": "img-1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": None,
                        },
                    }
                ]
            },
        },
    }
    files = FakeOpenAIFiles(contents={"file_out": _jsonl([refusal_line])})
    batches = FakeOpenAIBatches()
    _install_fake_openai(monkeypatch, files, batches)
    adapter = OpenAIBatchAdapter()
    adapter.submit_batch(_make_request(tmp_path))

    fetch = adapter.fetch_batch_results(
        BatchJobHandle(
            provider="openai",
            provider_job_id="batch_001",
            api_keys={"openai": "sk-test"},
        )
    )

    assert len(fetch.items) == 1
    item = fetch.items[0]
    assert item.status is BatchItemStatus.FAILED
    assert item.provider_status is BatchProviderItemStatus.FAILED
    assert item.error is not None
    assert item.error.code == "provider_safety_refusal"


def test_fetch_batch_results_handles_missing_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_line = {
        "custom_id": "img-1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "no tool used"},
                    }
                ]
            },
        },
    }
    files = FakeOpenAIFiles(contents={"file_out": _jsonl([missing_line])})
    batches = FakeOpenAIBatches()
    _install_fake_openai(monkeypatch, files, batches)
    adapter = OpenAIBatchAdapter()
    adapter.submit_batch(_make_request(tmp_path))

    fetch = adapter.fetch_batch_results(
        BatchJobHandle(
            provider="openai",
            provider_job_id="batch_001",
            api_keys={"openai": "sk-test"},
        )
    )

    item = fetch.items[0]
    assert item.status is BatchItemStatus.FAILED
    assert item.error is not None
    assert item.error.code == "result_tool_call_missing"


def test_fetch_batch_results_handles_invalid_json_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken_line = {
        "custom_id": "img-1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "normalize_annotation_output",
                                        "arguments": "not-a-json",
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        },
    }
    files = FakeOpenAIFiles(contents={"file_out": _jsonl([broken_line])})
    batches = FakeOpenAIBatches()
    _install_fake_openai(monkeypatch, files, batches)
    adapter = OpenAIBatchAdapter()
    adapter.submit_batch(_make_request(tmp_path))

    fetch = adapter.fetch_batch_results(
        BatchJobHandle(
            provider="openai",
            provider_job_id="batch_001",
            api_keys={"openai": "sk-test"},
        )
    )

    item = fetch.items[0]
    assert item.status is BatchItemStatus.FAILED
    assert item.error is not None
    assert item.error.code == "result_tool_arguments_invalid_json"
