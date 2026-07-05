"""Tests for Google Gemini Developer API Batch adapter (#154)."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from image_annotator_lib.webapi.batch import (
    BatchItemStatus,
    BatchJobHandle,
    BatchStatus,
    BatchSubmitItem,
    BatchSubmitRequest,
)
from image_annotator_lib.webapi.batch.adapters.google import GoogleBatchAdapter
from image_annotator_lib.webapi.batch.types import BatchErrorPhase, BatchJobError


def _jsonl(lines: list[object]) -> bytes:
    return "\n".join(json.dumps(line, separators=(",", ":")) for line in lines).encode("utf-8")


@dataclass
class FakeGoogleFiles:
    uploaded_payload: str | None = None
    uploaded_config: dict[str, Any] | None = None
    download_payloads: dict[str, bytes] = field(default_factory=dict)
    downloaded_names: list[str] = field(default_factory=list)

    def upload(self, *, file: io.BytesIO, config: dict[str, Any]) -> SimpleNamespace:
        self.uploaded_payload = file.read().decode("utf-8")
        self.uploaded_config = dict(config)
        return SimpleNamespace(name="files/input-001")

    def download(self, *, file: str) -> bytes:
        self.downloaded_names.append(file)
        if file not in self.download_payloads:
            raise RuntimeError(f"missing download payload for {file}")
        return self.download_payloads[file]


@dataclass
class FakeGoogleBatches:
    created_model: str | None = None
    created_src: str | None = None
    created_config: dict[str, Any] | None = None
    job: dict[str, Any] = field(default_factory=dict)
    canceled_names: list[str] = field(default_factory=list)

    def create(self, *, model: str, src: str, config: dict[str, Any]) -> dict[str, Any]:
        self.created_model = model
        self.created_src = src
        self.created_config = dict(config)
        return {"name": "batches/job-001", "state": "JOB_STATE_PENDING"}

    def get(self, *, name: str) -> dict[str, Any]:
        return {"name": name, **self.job}

    def cancel(self, *, name: str) -> None:
        self.canceled_names.append(name)


def install_fake_google(
    monkeypatch: pytest.MonkeyPatch, files: FakeGoogleFiles, batches: FakeGoogleBatches
) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.files = files
            self.batches = batches

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=FakeClient)))


def make_request(tmp_path: Path, model_id: str = "gemini/gemini-2.5-flash") -> BatchSubmitRequest:
    image_path = tmp_path / "img-1.png"
    image_path.write_bytes(b"fake-image-bytes")
    return BatchSubmitRequest(
        provider="google",
        endpoint="",
        litellm_model_id=model_id,
        prompt_profile="default",
        description=None,
        api_keys={"google": "test-key"},
        items=[BatchSubmitItem(custom_id="img-1", image_id=1, image_path=image_path)],
    )


def make_handle() -> BatchJobHandle:
    return BatchJobHandle(
        provider="google", provider_job_id="batches/job-001", api_keys={"google": "test-key"}
    )


def _success_line(key: str = "img-1", *, camel: bool = True) -> dict[str, Any]:
    function_call_key = "functionCall" if camel else "function_call"
    return {
        "key": key,
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                function_call_key: {
                                    "name": "normalize_annotation_output",
                                    "args": {"tags": ["1girl"], "captions": ["a girl"], "score": 0.8},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    }


def test_submit_batch_builds_gemini_jsonl_and_creates_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = FakeGoogleFiles()
    batches = FakeGoogleBatches()
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().submit_batch(make_request(tmp_path))

    assert result.provider == "google"
    assert result.provider_job_id == "batches/job-001"
    assert result.status is BatchStatus.RUNNING
    assert result.request_count == 1
    # File API へ JSONL を upload し、その file 名で job を作る
    assert files.uploaded_config == {"display_name": "batch-requests", "mime_type": "jsonl"}
    assert batches.created_src == "files/input-001"
    assert batches.created_model == "gemini-2.5-flash"
    assert files.uploaded_payload is not None
    line = json.loads(files.uploaded_payload.splitlines()[0])
    assert line["key"] == "img-1"
    request = line["request"]
    assert request["system_instruction"]["parts"][0]["text"]
    parts = request["contents"][0]["parts"]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    declarations = request["tools"][0]["function_declarations"]
    assert declarations[0]["name"] == "normalize_annotation_output"
    # Gemini は $ref / $defs を受けないため flat schema であること
    assert "$defs" not in json.dumps(declarations)
    config = request["tool_config"]["function_calling_config"]
    assert config["mode"] == "ANY"
    assert config["allowed_function_names"] == ["normalize_annotation_output"]


def test_submit_batch_rejects_non_google_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=FakeGoogleBatches())
    request = make_request(tmp_path, model_id="openai/gpt-4o")

    with pytest.raises(BatchJobError) as excinfo:
        GoogleBatchAdapter().submit_batch(request)

    assert excinfo.value.phase is BatchErrorPhase.PREPARE
    assert excinfo.value.code == "unsupported_model_provider"


def test_submit_batch_requires_google_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=FakeGoogleBatches())
    request = make_request(tmp_path)
    request = BatchSubmitRequest(
        provider=request.provider,
        endpoint=request.endpoint,
        litellm_model_id=request.litellm_model_id,
        prompt_profile=request.prompt_profile,
        description=None,
        api_keys={},
        items=request.items,
    )

    with pytest.raises(BatchJobError) as excinfo:
        GoogleBatchAdapter().submit_batch(request)

    assert excinfo.value.code == "missing_api_key"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("JOB_STATE_PENDING", BatchStatus.RUNNING),
        ("JOB_STATE_RUNNING", BatchStatus.RUNNING),
        ("JOB_STATE_SUCCEEDED", BatchStatus.COMPLETED),
        ("JOB_STATE_FAILED", BatchStatus.FAILED),
        ("JOB_STATE_CANCELLED", BatchStatus.CANCELED),
        ("JOB_STATE_EXPIRED", BatchStatus.EXPIRED),
        ("SOMETHING_ELSE", BatchStatus.UNKNOWN),
    ],
)
def test_retrieve_batch_maps_job_states(
    monkeypatch: pytest.MonkeyPatch, state: str, expected: BatchStatus
) -> None:
    batches = FakeGoogleBatches(job={"state": state})
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    result = GoogleBatchAdapter().retrieve_batch(make_handle())

    assert result.status is expected
    assert result.provider_job_id == "batches/job-001"


def test_retrieve_batch_maps_enum_like_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK が enum を返しても (str 化に enum class 名が乗っても) 状態を解決できる。"""
    batches = FakeGoogleBatches(job={"state": SimpleNamespace(name="JOB_STATE_SUCCEEDED")})
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    result = GoogleBatchAdapter().retrieve_batch(make_handle())

    assert result.status is BatchStatus.COMPLETED


def test_cancel_batch_cancels_then_returns_latest_status(monkeypatch: pytest.MonkeyPatch) -> None:
    batches = FakeGoogleBatches(job={"state": "JOB_STATE_CANCELLED"})
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    result = GoogleBatchAdapter().cancel_batch(make_handle())

    assert batches.canceled_names == ["batches/job-001"]
    assert result.status is BatchStatus.CANCELED


def test_fetch_batch_results_requires_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    batches = FakeGoogleBatches(job={"state": "JOB_STATE_RUNNING"})
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    with pytest.raises(BatchJobError) as excinfo:
        GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert excinfo.value.code == "job_not_completed"
    assert excinfo.value.retryable is True


def test_fetch_batch_results_raises_when_result_file_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = FakeGoogleBatches(job={"state": "JOB_STATE_FAILED", "error": {"message": "quota exceeded"}})
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    with pytest.raises(BatchJobError) as excinfo:
        GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert excinfo.value.code == "result_file_missing"
    assert "quota exceeded" in excinfo.value.message


def test_fetch_batch_results_normalizes_output_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    files = FakeGoogleFiles(
        download_payloads={
            "files/output-001": _jsonl(
                [
                    _success_line("img-1", camel=True),
                    _success_line("img-2", camel=False),
                    {"key": "img-3", "error": {"code": 500, "message": "internal error"}},
                    {
                        "key": "img-4",
                        "response": {"candidates": [{"finishReason": "SAFETY"}]},
                    },
                    {
                        "key": "img-5",
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "cannot help"}]}, "finishReason": "STOP"}
                            ]
                        },
                    },
                ]
            )
        }
    )
    batches = FakeGoogleBatches(
        job={"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/output-001"}}
    )
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert result.status is BatchStatus.COMPLETED
    assert files.downloaded_names == ["files/output-001"]
    by_id = {item.custom_id: item for item in result.items}
    assert set(by_id) == {"img-1", "img-2", "img-3", "img-4", "img-5"}

    # camelCase / snake_case どちらの functionCall も success として parse できる
    for key in ("img-1", "img-2"):
        item = by_id[key]
        assert item.status is BatchItemStatus.SUCCEEDED
        assert item.annotation is not None
        assert item.annotation.tags == ["1girl"]
        assert item.annotation.scores == {"score": 0.8}
        assert item.annotation.provider_name == "google"

    # provider error line
    error_item = by_id["img-3"]
    assert error_item.status is BatchItemStatus.FAILED
    assert error_item.error is not None
    assert error_item.error.code == "provider_item_error"
    assert error_item.error.retryable is True

    # native safety signal (ADR 0005)
    safety_item = by_id["img-4"]
    assert safety_item.error is not None
    assert safety_item.error.code == "safety_refusal"
    assert safety_item.error.retryable is False

    # function call が無い自由文応答は annotation_output_unparseable
    unparseable = by_id["img-5"]
    assert unparseable.error is not None
    assert unparseable.error.code == "annotation_output_unparseable"


def test_fetch_batch_results_dedupes_keys_and_skips_keyless_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = FakeGoogleFiles(
        download_payloads={
            "files/output-001": _jsonl(
                [
                    _success_line("img-1"),
                    _success_line("img-1"),
                    {"response": {"candidates": []}},
                ]
            )
        }
    )
    batches = FakeGoogleBatches(
        job={"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/output-001"}}
    )
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert [item.custom_id for item in result.items] == ["img-1"]


def test_fetch_batch_results_reads_metadata_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    line = _success_line("ignored")
    del line["key"]
    line["metadata"] = {"key": "img-9"}
    files = FakeGoogleFiles(download_payloads={"files/output-001": _jsonl([line])})
    batches = FakeGoogleBatches(
        job={"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/output-001"}}
    )
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert [item.custom_id for item in result.items] == ["img-9"]


def test_prompt_feedback_block_is_content_policy_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    files = FakeGoogleFiles(
        download_payloads={
            "files/output-001": _jsonl(
                [
                    {
                        "key": "img-1",
                        "response": {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}},
                    }
                ]
            )
        }
    )
    batches = FakeGoogleBatches(
        job={"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/output-001"}}
    )
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().fetch_batch_results(make_handle())

    assert result.items[0].error is not None
    assert result.items[0].error.code == "content_policy_refusal"


def test_google_adapter_is_registered_in_batch_dispatch() -> None:
    from image_annotator_lib.webapi.batch import service

    assert service._BATCH_ADAPTERS["google"] is GoogleBatchAdapter
    metadata = GoogleBatchAdapter.batch_metadata()
    assert metadata["provider_batch_api"] == "gemini_developer_batch"
    assert metadata["library_max_items"] == 500


def test_provider_item_error_with_canonical_status_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """google.rpc.Status の canonical code / status 名を retryable と判定する (Codex P2)。"""
    files = FakeGoogleFiles(
        download_payloads={
            "files/output-001": _jsonl(
                [
                    {"key": "img-1", "error": {"code": 8, "message": "quota"}},
                    {"key": "img-2", "error": {"status": "UNAVAILABLE", "message": "down"}},
                    {"key": "img-3", "error": {"code": 3, "message": "invalid argument"}},
                ]
            )
        }
    )
    batches = FakeGoogleBatches(
        job={"state": "JOB_STATE_SUCCEEDED", "dest": {"file_name": "files/output-001"}}
    )
    install_fake_google(monkeypatch, files=files, batches=batches)

    result = GoogleBatchAdapter().fetch_batch_results(make_handle())

    by_id = {item.custom_id: item for item in result.items}
    assert by_id["img-1"].error is not None and by_id["img-1"].error.retryable is True
    assert by_id["img-2"].error is not None and by_id["img-2"].error.retryable is True
    assert by_id["img-3"].error is not None and by_id["img-3"].error.retryable is False


def test_retrieve_batch_populates_counts_from_batch_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """batchStats の requestCount / successful / failed を status へ反映する (Codex P2)。"""
    batches = FakeGoogleBatches(
        job={
            "state": "JOB_STATE_SUCCEEDED",
            "batchStats": {
                "requestCount": "3",
                "successfulRequestCount": 2,
                "failedRequestCount": 1,
            },
        }
    )
    install_fake_google(monkeypatch, files=FakeGoogleFiles(), batches=batches)

    result = GoogleBatchAdapter().retrieve_batch(make_handle())

    assert result.request_count == 3
    assert result.succeeded_count == 2
    assert result.failed_count == 1


def test_gemini_schema_uses_shared_prompt_score_scale() -> None:
    """score スケールは共有 prompt (1.00-10.00) と揃える (Codex P2)。"""
    from image_annotator_lib.webapi.batch.preparation import (
        build_google_annotation_function_declaration,
    )

    declaration = build_google_annotation_function_declaration()

    assert "1.00 and 10.00" in declaration["parameters"]["properties"]["score"]["description"]


def test_list_batch_capable_models_filters_google_capabilities_to_supported(monkeypatch) -> None:
    """registry が RATINGS を持っていても Google は生成可能な capability に絞る (Codex P2)。"""
    from image_annotator_lib.core.types import TaskCapability
    from image_annotator_lib.webapi.batch import service

    monkeypatch.setattr(service, "list_available_annotators", lambda: ["gemini"])
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "gemini": {
                "provider": "google",
                "litellm_model_id": "gemini/gemini-2.5-flash",
                "capabilities": ["tags", "captions", "scores", "ratings"],
            },
        }.get(name),
    )

    models = service.list_batch_capable_models()

    assert len(models) == 1
    assert models[0].provider == "google"
    assert TaskCapability.RATINGS not in models[0].capabilities
    assert TaskCapability.TAGS in models[0].capabilities
