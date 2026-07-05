"""Google Gemini Developer API Batch adapter (ADR 0005 Phase 3 / #154).

ADR 0005 "Provider submit forms" の通り、input file (JSONL + File API) 方式のみを
実装する。inline requests / Vertex AI BatchPredictionJob route はスコープ外。
transport は ``google-genai`` SDK。
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from image_annotator_lib.core.types import (
    AnnotationSchema,
    TaskCapability,
    UnifiedAnnotationResult,
)
from image_annotator_lib.webapi.model_id import resolve_model_ref
from image_annotator_lib.webapi.output_normalization import normalize_annotation_output

from ..preparation import (
    build_google_generate_content_annotation_jsonl,
    prepare_items,
)
from ..types import (
    BatchErrorPhase,
    BatchFetchResult,
    BatchItemError,
    BatchItemStatus,
    BatchJobError,
    BatchJobHandle,
    BatchProviderItemStatus,
    BatchResultItem,
    BatchStatus,
    BatchStatusResult,
    BatchSubmitRequest,
    BatchSubmitResult,
)

_PROVIDER = "google"
_MAX_LIBRARY_ITEMS = 500
_SUPPORTED_PROMPT_PROFILES = frozenset({"default"})
# Gemini Developer API Batch の input file 上限 (2 GB)。File API 自体の制約で、
# library_max_items=500 の resized image 前提では実質届かない safety 情報。
_MAX_GOOGLE_INPUT_FILE_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_ANNOTATION_CAPABILITIES = frozenset(
    {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
)
_ANNOTATION_TOOL_NAME = "normalize_annotation_output"
_ANNOTATION_BASE_PROMPT = (
    "You analyze images and return structured annotations via the "
    "`normalize_annotation_output` tool. Respect required fields per capability."
)

# Gemini finishReason → item error code (ADR 0005 "Error handling")。
# native signal がある場合のみ refusal 系に正規化する。
_FINISH_REASON_ERROR_CODES: dict[str, tuple[str, bool]] = {
    "SAFETY": ("safety_refusal", False),
    "IMAGE_SAFETY": ("safety_refusal", False),
    "PROHIBITED_CONTENT": ("content_policy_refusal", False),
    "BLOCKLIST": ("content_policy_refusal", False),
    "SPII": ("content_policy_refusal", False),
    "RECITATION": ("content_policy_refusal", False),
    "MAX_TOKENS": ("max_tokens", False),
}

# job state (JOB_STATE_*) → BatchStatus。SDK は enum を返すが、str 化した値の末尾
# サフィックスで判定して SDK 内部表現の変化に依存しない。
_JOB_STATE_TO_STATUS: dict[str, BatchStatus] = {
    "JOB_STATE_PENDING": BatchStatus.RUNNING,
    "JOB_STATE_QUEUED": BatchStatus.RUNNING,
    "JOB_STATE_RUNNING": BatchStatus.RUNNING,
    "JOB_STATE_CANCELLING": BatchStatus.RUNNING,
    "JOB_STATE_PAUSED": BatchStatus.RUNNING,
    "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_FAILED": BatchStatus.FAILED,
    "JOB_STATE_CANCELLED": BatchStatus.CANCELED,
    "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
}

_TERMINAL_STATUSES = frozenset(
    {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELED, BatchStatus.EXPIRED}
)


class GoogleBatchAdapter:
    """Adapter for Gemini Developer API Batch (input file 方式)。"""

    @staticmethod
    def batch_metadata() -> dict[str, Any]:
        return {
            "provider_batch_api": "gemini_developer_batch",
            # File API の artifact は 48 時間で自動削除される (input / result とも)
            "result_retention_days": 2,
            "zero_data_retention_eligible": False,
            "provider_max_requests": None,
            "provider_max_body_bytes": _MAX_GOOGLE_INPUT_FILE_BYTES,
            "library_max_items": _MAX_LIBRARY_ITEMS,
        }

    def submit_batch(self, request: BatchSubmitRequest) -> BatchSubmitResult:
        if request.provider.lower() != _PROVIDER:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_provider",
                f"Google adapter cannot submit provider: {request.provider}",
                retryable=False,
            )
        if len(request.items) > _MAX_LIBRARY_ITEMS:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "too_many_items",
                f"Batch contains {len(request.items)} items; maximum is {_MAX_LIBRARY_ITEMS}",
                retryable=False,
            )
        if request.prompt_profile not in _SUPPORTED_PROMPT_PROFILES:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_prompt_profile",
                f"Unsupported Google batch prompt_profile: {request.prompt_profile}",
                retryable=False,
            )
        api_key = self._api_key(request.api_keys, provider_job_id=None, phase=BatchErrorPhase.PREPARE)

        model_ref = self._resolve_model_ref(request.litellm_model_id)
        prepared = prepare_items(request.items, provider=_PROVIDER)
        request_payload = build_google_generate_content_annotation_jsonl(
            prepared,
            system_prompt=_ANNOTATION_BASE_PROMPT,
            capabilities=_DEFAULT_ANNOTATION_CAPABILITIES,
        )

        client = self._client(api_key)
        try:
            input_file = client.files.upload(
                file=io.BytesIO(request_payload.encode("utf-8")),
                config={"display_name": "batch-requests", "mime_type": "jsonl"},
            )
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.UPLOAD,
                None,
                "upload_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        input_file_name = str(_get(input_file, "name") or "")
        if not input_file_name:
            raise self._job_error(
                BatchErrorPhase.UPLOAD,
                None,
                "missing_input_file_name",
                "Google batch input upload response did not include a file name",
                retryable=False,
            )

        try:
            job = client.batches.create(
                model=model_ref.provider_model_id,
                src=input_file_name,
                config={"display_name": request.description or "image-annotator-lib batch"},
            )
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.SUBMIT,
                None,
                "submit_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        provider_job_id = str(_get(job, "name") or "")
        if not provider_job_id:
            raise self._job_error(
                BatchErrorPhase.SUBMIT,
                None,
                "missing_provider_job_id",
                "Google batch create response did not include a job name",
                retryable=False,
            )
        return BatchSubmitResult(
            provider=_PROVIDER,
            provider_job_id=provider_job_id,
            status=_status_from_job(job),
            request_count=len(request.items),
        )

    def retrieve_batch(self, handle: BatchJobHandle) -> BatchStatusResult:
        job = self._get_job(handle, phase=BatchErrorPhase.POLL)
        return _to_status_result(job, provider_job_id=handle.provider_job_id)

    def cancel_batch(self, handle: BatchJobHandle) -> BatchStatusResult:
        api_key = self._api_key(
            handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.CANCEL
        )
        client = self._client(api_key)
        try:
            client.batches.cancel(name=handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.CANCEL,
                handle.provider_job_id,
                "cancel_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc
        # cancel は job を返さないため、最新状態を取得し直して返す
        job = self._get_job(handle, phase=BatchErrorPhase.CANCEL)
        return _to_status_result(job, provider_job_id=handle.provider_job_id)

    def fetch_batch_results(self, handle: BatchJobHandle) -> BatchFetchResult:
        job = self._get_job(handle, phase=BatchErrorPhase.DOWNLOAD)
        status = _to_status_result(job, provider_job_id=handle.provider_job_id)
        if status.status not in _TERMINAL_STATUSES:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "job_not_completed",
                "Google batch results are not available until processing has ended",
                retryable=True,
            )

        result_file_name = _result_file_name(job)
        if not result_file_name:
            job_error = _get(job, "error")
            message = (
                str(_get(job_error, "message") or job_error)
                if job_error is not None
                else "Google batch job ended without a result file"
            )
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "result_file_missing",
                message,
                retryable=False,
            )

        api_key = self._api_key(
            handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.DOWNLOAD
        )
        client = self._client(api_key)
        try:
            payload = client.files.download(file=result_file_name)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "result_file_download_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        lines = self._decode_jsonl(payload)
        seen: set[str] = set()
        items: list[BatchResultItem] = []
        for raw in lines:
            item = self._normalize_result_line(raw)
            if item is None or item.custom_id in seen:
                continue
            seen.add(item.custom_id)
            items.append(item)

        return BatchFetchResult(
            provider=_PROVIDER,
            provider_job_id=handle.provider_job_id,
            status=status.status,
            items=items,
        )

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _get_job(self, handle: BatchJobHandle, *, phase: BatchErrorPhase) -> Any:
        api_key = self._api_key(handle.api_keys, provider_job_id=handle.provider_job_id, phase=phase)
        client = self._client(api_key)
        try:
            return client.batches.get(name=handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                phase,
                handle.provider_job_id,
                "retrieve_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

    def _decode_jsonl(self, payload: Any) -> list[str]:
        if isinstance(payload, bytes):
            return payload.decode("utf-8").splitlines()
        if isinstance(payload, str):
            return payload.splitlines()
        raise self._job_error(
            BatchErrorPhase.PARSE,
            None,
            "result_payload_invalid",
            f"Google batch result download returned unsupported type: {type(payload).__name__}",
            retryable=False,
        )

    def _normalize_result_line(self, raw_line: str) -> BatchResultItem | None:
        if not raw_line.strip():
            return None
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise self._job_error(
                BatchErrorPhase.PARSE,
                None,
                "result_line_parse_failed",
                self._format_exception(exc),
                retryable=False,
            ) from exc
        if not isinstance(obj, dict):
            return None

        custom_id = _extract_custom_id(obj)
        if not custom_id:
            return None

        error = obj.get("error") or obj.get("status")
        if error:
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.FAILED,
                BatchErrorPhase.NORMALIZE,
                "provider_item_error",
                str(_get(error, "message") or error),
                retryable=_provider_item_retryable(error),
            )

        response = obj.get("response")
        if not isinstance(response, Mapping):
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.UNKNOWN,
                BatchErrorPhase.PARSE,
                "missing_result",
                "Google batch output line did not include a response payload",
                retryable=False,
            )
        return self._build_annotation_item(custom_id, response)

    def _build_annotation_item(self, custom_id: str, response: Mapping[str, Any]) -> BatchResultItem:
        """GenerateContentResponse から BatchResultItem を組み立てる。

        Gemini の native signal (finishReason / promptFeedback.blockReason) がある場合
        のみ refusal 系へ正規化し (ADR 0005)、それ以外で function call が取れない場合は
        ``annotation_output_unparseable`` とする。JSON key は REST (camelCase) と
        SDK dump (snake_case) の両方を受ける。
        """
        prompt_feedback = _get_either(response, "promptFeedback", "prompt_feedback")
        block_reason = str(_get_either(prompt_feedback, "blockReason", "block_reason") or "").upper()
        if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.FAILED,
                BatchErrorPhase.NORMALIZE,
                "content_policy_refusal",
                f"Google prompt was blocked: {block_reason}",
                retryable=False,
            )

        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.UNKNOWN,
                BatchErrorPhase.PARSE,
                "missing_result",
                "Google batch response had no candidates",
                retryable=False,
            )
        candidate = candidates[0]
        finish_reason = (
            str(_get_either(candidate, "finishReason", "finish_reason") or "")
            .upper()
            .removeprefix("FINISH_REASON_")
        )
        if finish_reason in _FINISH_REASON_ERROR_CODES:
            code, retryable = _FINISH_REASON_ERROR_CODES[finish_reason]
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.FAILED,
                BatchErrorPhase.NORMALIZE,
                code,
                f"Google batch item ended with finishReason={finish_reason}",
                retryable=retryable,
            )

        arguments = _extract_function_call_args(candidate)
        if arguments is None:
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.UNKNOWN,
                BatchErrorPhase.PARSE,
                "annotation_output_unparseable",
                "Google batch response had no `normalize_annotation_output` function call; "
                f"finishReason={finish_reason or 'unknown'}",
                retryable=False,
            )

        try:
            schema = normalize_annotation_output(**arguments)
        except Exception as exc:
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.FAILED,
                BatchErrorPhase.NORMALIZE,
                "annotation_schema_invalid",
                f"normalize_annotation_output failed: {self._format_exception(exc)}",
                retryable=False,
            )

        return BatchResultItem(
            custom_id=custom_id,
            status=BatchItemStatus.SUCCEEDED,
            provider_status=BatchProviderItemStatus.SUCCEEDED,
            annotation=_to_unified_annotation(schema),
            error=None,
        )

    def _api_key(
        self, api_keys: dict[str, str], *, provider_job_id: str | None, phase: BatchErrorPhase
    ) -> str:
        api_key = api_keys.get(_PROVIDER)
        if not api_key:
            raise self._job_error(
                phase,
                provider_job_id,
                "missing_api_key",
                "Missing Google API key in api_keys['google']",
                retryable=False,
            )
        return api_key

    def _resolve_model_ref(self, litellm_model_id: str):
        try:
            ref = resolve_model_ref(litellm_model_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "invalid_model_id",
                self._format_exception(exc),
                retryable=False,
            ) from exc
        if ref.provider != _PROVIDER:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_model_provider",
                f"Google batch requires a google/gemini model, got: {litellm_model_id}",
                retryable=False,
            )
        return ref

    def _client(self, api_key: str) -> Any:
        from google import genai

        return genai.Client(api_key=api_key)

    def _job_error(
        self,
        phase: BatchErrorPhase,
        provider_job_id: str | None,
        code: str,
        message: str,
        *,
        retryable: bool,
    ) -> BatchJobError:
        return BatchJobError(
            phase=phase,
            provider=_PROVIDER,
            provider_job_id=provider_job_id,
            code=code,
            message=message,
            retryable=retryable,
        )

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        # google-genai の APIError は HTTP status を `code` に持つ
        for attribute in ("code", "status_code"):
            status_code = getattr(exc, attribute, None)
            if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
                return True
        return False


def _extract_custom_id(obj: Mapping[str, Any]) -> str:
    key = obj.get("key")
    if isinstance(key, str) and key:
        return key
    metadata = obj.get("metadata")
    if isinstance(metadata, Mapping):
        meta_key = metadata.get("key")
        if isinstance(meta_key, str) and meta_key:
            return meta_key
    return ""


def _extract_function_call_args(candidate: Any) -> dict[str, Any] | None:
    """candidate.content.parts から annotation tool の function call args を取り出す。"""
    content = _get_either(candidate, "content", "content")
    parts = _get(content, "parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        function_call = _get_either(part, "functionCall", "function_call")
        if not isinstance(function_call, Mapping):
            continue
        if str(function_call.get("name") or "") != _ANNOTATION_TOOL_NAME:
            continue
        args = function_call.get("args")
        if isinstance(args, Mapping):
            return dict(args)
    return None


def _provider_item_retryable(error: Any) -> bool:
    code = _get(error, "code")
    return isinstance(code, int) and (code == 429 or code >= 500)


def _failed_result_item(
    custom_id: str,
    provider_status: BatchProviderItemStatus,
    phase: BatchErrorPhase,
    code: str,
    message: str,
    *,
    retryable: bool,
) -> BatchResultItem:
    return BatchResultItem(
        custom_id=custom_id,
        status=BatchItemStatus.FAILED,
        provider_status=provider_status,
        annotation=None,
        error=BatchItemError(
            phase=phase,
            code=code,
            message=message,
            retryable=retryable,
        ),
    )


def _to_unified_annotation(schema: AnnotationSchema) -> UnifiedAnnotationResult:
    """検証済み `AnnotationSchema` を `UnifiedAnnotationResult` に包む (openai adapter と同型)。"""
    capabilities: set[TaskCapability] = set()
    if schema.tags:
        capabilities.add(TaskCapability.TAGS)
    if schema.captions:
        capabilities.add(TaskCapability.CAPTIONS)
    if schema.score is not None:
        capabilities.add(TaskCapability.SCORES)
    if schema.ratings:
        capabilities.add(TaskCapability.RATINGS)
    return UnifiedAnnotationResult(
        model_name="google_batch",
        capabilities=frozenset(capabilities) if capabilities else _DEFAULT_ANNOTATION_CAPABILITIES,
        tags=list(schema.tags) if schema.tags else None,
        captions=list(schema.captions) if schema.captions else None,
        scores={"score": float(schema.score)} if schema.score is not None else None,
        ratings=list(schema.ratings) if schema.ratings else None,
        provider_name=_PROVIDER,
        framework="google_batch",
        raw_output=None,
    )


def _job_state_text(job: Any) -> str:
    state = _get(job, "state")
    if state is None:
        return ""
    name = getattr(state, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(state).upper()


def _status_from_job(job: Any) -> BatchStatus:
    state_text = _job_state_text(job)
    for state_name, status in _JOB_STATE_TO_STATUS.items():
        if state_text.endswith(state_name):
            return status
    return BatchStatus.UNKNOWN


def _result_file_name(job: Any) -> str:
    dest = _get(job, "dest")
    file_name = _get_either(dest, "fileName", "file_name")
    return str(file_name) if file_name else ""


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _to_status_result(job: Any, *, provider_job_id: str) -> BatchStatusResult:
    return BatchStatusResult(
        provider=_PROVIDER,
        provider_job_id=str(_get(job, "name") or provider_job_id),
        status=_status_from_job(job),
        # Gemini Developer API Batch は request 単位の集計 count を返さない
        request_count=None,
        succeeded_count=None,
        failed_count=None,
        canceled_count=None,
        expired_count=None,
        submitted_at=_parse_datetime(_get_either(job, "createTime", "create_time")),
        completed_at=_parse_datetime(_get_either(job, "endTime", "end_time")),
        expires_at=None,
    )


def _get(obj: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_either(obj: Any, camel: str, snake: str, default: Any | None = None) -> Any:
    value = _get(obj, camel)
    if value is not None:
        return value
    return _get(obj, snake, default)
