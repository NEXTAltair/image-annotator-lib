"""OpenAI Batch API adapter."""

from __future__ import annotations

from collections.abc import Mapping
import json
from datetime import datetime
from importlib import import_module
from typing import Any

from image_annotator_lib.core.types import RatingPrediction, TaskCapability, UnifiedAnnotationResult
from image_annotator_lib.webapi.model_id import resolve_model_ref

from ..preparation import build_openai_moderations_jsonl, prepare_items
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


_PROVIDER = "openai"
_MAX_LIBRARY_ITEMS = 500
_SUPPORTED_PROMPT_PROFILES = frozenset({"default"})
_SUPPORTED_ENDPOINTS = frozenset({"/v1/moderations", "/v1/moderations/"})
_DEFAULT_COMPLETION_WINDOW = "24h"
_SUPPORTED_CAPABILITIES = frozenset({TaskCapability.RATINGS})


def _load_openai_converter() -> Any:
    try:
        module = import_module("image_annotator_lib.webapi.openai_moderations")
        return module.category_scores_to_rating_prediction
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "OpenAI batch adapter requires image_annotator_lib.webapi.openai_moderations. "
            "Merge feat/issue-119-openai-moderations before using /v1/moderations batch."
        ) from exc


_CATEGORY_SCORES_TO_RATING_PREDICTION = _load_openai_converter()


class OpenAIBatchAdapter:
    """Adapter for OpenAI Batch API."""

    @staticmethod
    def batch_metadata() -> dict[str, Any]:
        return {
            "provider_batch_api": "openai_batch",
            "result_retention_days": 29,
            "zero_data_retention_eligible": True,
            "provider_max_requests": 100_000,
            "provider_max_body_bytes": 4 * 1024 * 1024,
            "library_max_items": _MAX_LIBRARY_ITEMS,
        }

    def submit_batch(self, request: BatchSubmitRequest) -> BatchSubmitResult:
        if request.provider.lower() != _PROVIDER:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_provider",
                f"OpenAI adapter cannot submit provider: {request.provider}",
                retryable=False,
            )
        endpoint = request.endpoint or ""
        normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        if normalized_endpoint not in _SUPPORTED_ENDPOINTS:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_endpoint",
                f"OpenAI batch adapter only supports {sorted(_SUPPORTED_ENDPOINTS)}, got: {endpoint}",
                retryable=False,
            )
        api_key = self._api_key(request.api_keys, provider_job_id=None, phase=BatchErrorPhase.PREPARE)
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
                f"Unsupported OpenAI batch prompt_profile: {request.prompt_profile}",
                retryable=False,
            )

        model_ref = self._resolve_model_ref(request.litellm_model_id)
        prepared = prepare_items(request.items, provider=_PROVIDER)
        request_payload = build_openai_moderations_jsonl(
            prepared,
            endpoint=normalized_endpoint,
            litellm_model_id=model_ref.provider_model_id,
        )

        try:
            input_file = self._client(api_key).files.create(
                file=("requests.jsonl", request_payload),
                purpose="batch",
            )
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.UPLOAD,
                None,
                "upload_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        input_file_id = str(_get(input_file, "id") or "")
        if not input_file_id:
            raise self._job_error(
                BatchErrorPhase.UPLOAD,
                None,
                "missing_input_file_id",
                "OpenAI batch input upload response did not include an id",
                retryable=False,
            )

        try:
            batch = self._client(api_key).batches.create(
                input_file_id=input_file_id,
                endpoint=normalized_endpoint,
                completion_window=_DEFAULT_COMPLETION_WINDOW,
            )
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.SUBMIT,
                None,
                "submit_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        provider_job_id = str(_get(batch, "id") or "")
        if not provider_job_id:
            raise self._job_error(
                BatchErrorPhase.SUBMIT,
                None,
                "missing_provider_job_id",
                "OpenAI batch create response did not include an id",
                retryable=False,
            )
        return BatchSubmitResult(
            provider=_PROVIDER,
            provider_job_id=provider_job_id,
            status=_status_from_batch(batch),
            request_count=_request_count(batch) or len(request.items),
        )

    def retrieve_batch(self, handle: BatchJobHandle) -> BatchStatusResult:
        return self._retrieve_status(handle, phase=BatchErrorPhase.POLL)

    def cancel_batch(self, handle: BatchJobHandle) -> BatchStatusResult:
        api_key = self._api_key(handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.CANCEL)
        try:
            batch = self._client(api_key).batches.cancel(handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.CANCEL,
                handle.provider_job_id,
                "cancel_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc
        return _to_status_result(batch, provider_job_id=handle.provider_job_id)

    def fetch_batch_results(self, handle: BatchJobHandle) -> BatchFetchResult:
        status = self._retrieve_status(handle, phase=BatchErrorPhase.DOWNLOAD)
        if status.status not in {
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELED,
            BatchStatus.EXPIRED,
        }:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "job_not_completed",
                "OpenAI batch results are not available until processing has ended",
                retryable=True,
            )

        api_key = self._api_key(handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.DOWNLOAD)
        try:
            batch = self._client(api_key).batches.retrieve(handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "download_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        status = _to_status_result(batch, provider_job_id=handle.provider_job_id)
        output_file_id = _get(batch, "output_file_id")
        error_file_id = _get(batch, "error_file_id")

        if not output_file_id and not error_file_id:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "result_file_missing",
                "OpenAI batch completed but did not provide output/error file IDs",
                retryable=False,
            )

        seen: dict[str, int] = {}
        items: list[BatchResultItem] = []
        if output_file_id:
            output_lines = self._read_jsonl_lines(api_key, str(output_file_id))
            items.extend(self._parse_output_lines(output_lines, seen))
        if error_file_id:
            error_lines = self._read_jsonl_lines(api_key, str(error_file_id))
            items.extend(self._parse_error_lines(error_lines, seen))

        return BatchFetchResult(
            provider=_PROVIDER,
            provider_job_id=handle.provider_job_id,
            status=status.status,
            items=items,
        )

    def _retrieve_status(self, handle: BatchJobHandle, *, phase: BatchErrorPhase) -> BatchStatusResult:
        api_key = self._api_key(handle.api_keys, provider_job_id=handle.provider_job_id, phase=phase)
        try:
            batch = self._client(api_key).batches.retrieve(handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                phase,
                handle.provider_job_id,
                "retrieve_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc
        return _to_status_result(batch, provider_job_id=handle.provider_job_id)

    def _parse_output_lines(self, lines: list[str], seen: dict[str, int]) -> list[BatchResultItem]:
        parsed: list[BatchResultItem] = []
        for raw in lines:
            parsed_item = self._normalize_result_item(raw, line_type="output")
            if parsed_item is None:
                continue
            if parsed_item.custom_id in seen:
                continue
            seen[parsed_item.custom_id] = len(parsed) + 1
            parsed.append(parsed_item)
        return parsed

    def _parse_error_lines(self, lines: list[str], seen: dict[str, int]) -> list[BatchResultItem]:
        parsed: list[BatchResultItem] = []
        for raw in lines:
            parsed_item = self._normalize_result_item(raw, line_type="error")
            if parsed_item is None:
                continue
            if parsed_item.custom_id in seen:
                continue
            seen[parsed_item.custom_id] = len(parsed) + 1
            parsed.append(parsed_item)
        return parsed

    def _normalize_result_item(self, raw_line: str, *, line_type: str) -> BatchResultItem | None:
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

        custom_id = str(obj.get("custom_id") or "")
        if not custom_id:
            return None

        if line_type == "error":
            error = _get(obj, "error")
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.FAILED,
                BatchErrorPhase.NORMALIZE,
                "provider_batch_error",
                str(_get(error, "message") or _get(obj, "message") or "OpenAI batch item error"),
                retryable=_provider_item_retryable(error),
            )

        response = _get(obj, "response")
        if not isinstance(response, dict):
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.UNKNOWN,
                BatchErrorPhase.PARSE,
                "result_response_missing",
                "OpenAI output line did not include response payload",
                retryable=False,
            )

        provider_status = _provider_item_status(response)
        if provider_status is not BatchProviderItemStatus.SUCCEEDED:
            error = _get(response, "error")
            return _failed_result_item(
                custom_id,
                provider_status,
                BatchErrorPhase.NORMALIZE,
                "provider_item_error",
                str(_get(error, "message") or _get(response, "error") or "OpenAI batch item failed"),
                retryable=_provider_item_retryable(error),
            )

        moderation_response = _extract_moderation_response(response)
        if not isinstance(moderation_response, Mapping):
            return _failed_result_item(
                custom_id,
                BatchProviderItemStatus.UNKNOWN,
                BatchErrorPhase.PARSE,
                "result_response_invalid",
                "OpenAI output response payload was not a mapping",
                retryable=False,
            )
        rating = _CATEGORY_SCORES_TO_RATING_PREDICTION(moderation_response)
        return BatchResultItem(
            custom_id=custom_id,
            status=BatchItemStatus.SUCCEEDED,
            provider_status=BatchProviderItemStatus.SUCCEEDED,
            annotation=_to_unified(rating),
            error=None,
        )

    def _read_jsonl_lines(self, api_key: str, file_id: str) -> list[str]:
        try:
            content = self._client(api_key).files.content(file_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                None,
                "result_file_download_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        if isinstance(content, str):
            return content.splitlines()
        if isinstance(content, bytes):
            return content.decode("utf-8").splitlines()
        if isinstance(content, dict) and "text" in content:
            return str(content["text"]).splitlines()
        if isinstance(getattr(content, "text", None), str):
            return str(content.text).splitlines()
        if hasattr(content, "read"):
            payload = content.read()
            if isinstance(payload, bytes):
                return payload.decode("utf-8").splitlines()
            if isinstance(payload, str):
                return payload.splitlines()
            if payload is not None:
                return str(payload).splitlines()
        return []

    def _api_key(self, api_keys: dict[str, str], *, provider_job_id: str | None, phase: BatchErrorPhase) -> str:
        api_key = api_keys.get(_PROVIDER)
        if not api_key:
            raise self._job_error(
                phase,
                provider_job_id,
                "missing_api_key",
                "Missing OpenAI API key in api_keys['openai']",
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
                f"OpenAI batch requires an openai model, got: {litellm_model_id}",
                retryable=False,
            )
        return ref

    def _client(self, api_key: str) -> Any:
        import openai

        return openai.OpenAI(api_key=api_key)

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
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        error_type = getattr(exc, "type", "")
        return str(error_type).lower() in {"api_error", "server_error", "overloaded_error", "rate_limit_error"}


def _to_unified(prediction: RatingPrediction) -> UnifiedAnnotationResult:
    return UnifiedAnnotationResult(
        model_name="openai_batch",
        capabilities=_SUPPORTED_CAPABILITIES,
        ratings=[prediction],
        provider_name=_PROVIDER,
        framework="openai_batch",
        raw_output=None,
    )


def _extract_moderation_response(response: Mapping[str, Any] | Any) -> Any:
    body = _get(response, "body")
    if isinstance(body, dict):
        return body
    return response


def _provider_item_status(response: Mapping[str, Any] | Any) -> BatchProviderItemStatus:
    if not isinstance(response, Mapping):
        return BatchProviderItemStatus.UNKNOWN
    status_code = _get(response, "status_code")
    if isinstance(status_code, int):
        if 200 <= status_code < 300:
            return BatchProviderItemStatus.SUCCEEDED
        return BatchProviderItemStatus.FAILED
    return BatchProviderItemStatus.UNKNOWN


def _provider_item_retryable(error: Any) -> bool:
    if not isinstance(error, Mapping):
        return False
    if str(_get(error, "type")).lower() in {"server_error", "overloaded_error", "rate_limit_error", "api_error"}:
        return True
    status_code = _get(error, "status_code")
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)


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


def _request_counts(batch: Any) -> Any:
    return _get(batch, "request_counts") or {}


def _count(batch: Any, name: str) -> int | None:
    value = _get(_request_counts(batch), name)
    return int(value) if isinstance(value, (int, float)) else None


def _request_count(batch: Any) -> int | None:
    counts = [_count(batch, key) for key in ("processing", "succeeded", "errored", "canceled", "expired")]
    known_counts = [count for count in counts if count is not None]
    return sum(known_counts) if known_counts else None


def _status_from_batch(batch: Any) -> BatchStatus:
    processing_status = str(_get(batch, "processing_status") or "").lower()
    if processing_status in {"in_progress", "validating", "cancelling", "canceling", "finalizing", "queued"}:
        return BatchStatus.RUNNING
    if processing_status == "cancelled":
        return BatchStatus.CANCELED
    if processing_status == "expired":
        return BatchStatus.EXPIRED
    if processing_status == "failed":
        return BatchStatus.FAILED
    if processing_status in {"completed", "ended"}:
        total = _request_count(batch) or 0
        succeeded = _count(batch, "succeeded") or 0
        errored = _count(batch, "errored") or 0
        canceled = _count(batch, "canceled") or 0
        expired = _count(batch, "expired") or 0
        if total and canceled == total:
            return BatchStatus.CANCELED
        if total and expired == total:
            return BatchStatus.EXPIRED
        if total and succeeded == 0 and errored == total:
            return BatchStatus.FAILED
        return BatchStatus.COMPLETED
    return BatchStatus.UNKNOWN


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _to_status_result(batch: Any, *, provider_job_id: str) -> BatchStatusResult:
    return BatchStatusResult(
        provider=_PROVIDER,
        provider_job_id=str(_get(batch, "id") or provider_job_id),
        status=_status_from_batch(batch),
        request_count=_request_count(batch),
        succeeded_count=_count(batch, "succeeded"),
        failed_count=_count(batch, "errored"),
        canceled_count=_count(batch, "canceled"),
        expired_count=_count(batch, "expired"),
        submitted_at=_parse_datetime(_get(batch, "created_at")),
        completed_at=_parse_datetime(_get(batch, "completed_at") or _get(batch, "ended_at")),
        expires_at=_parse_datetime(_get(batch, "expires_at")),
    )


def _get(obj: Any, name: str, default: Any | None = None) -> Any:
    dumped = _as_dict(obj)
    if isinstance(dumped, dict):
        return dumped.get(name, default)
    return getattr(obj, name, default)


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__") and isinstance(value.__dict__, dict):
        return dict(value.__dict__)
    return None
