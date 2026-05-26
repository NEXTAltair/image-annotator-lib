"""Anthropic Message Batches adapter."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from typing import Any

from pydantic_ai import ModelRetry

from image_annotator_lib.core.model_id import resolve_model_ref
from image_annotator_lib.core.output_normalization import build_annotation_output_normalizer
from image_annotator_lib.core.types import AnnotationSchema, TaskCapability, UnifiedAnnotationResult
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT

from ..preparation import PreparedBatchItem, prepare_items
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

_PROVIDER = "anthropic"
_MAX_LIBRARY_ITEMS = 500
_MAX_ANTHROPIC_BODY_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_TOKENS = 1800
_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SUPPORTED_PROMPT_PROFILES = frozenset({"default"})
_CAPABILITIES = frozenset(
    {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES, TaskCapability.RATINGS}
)
_USER_PROMPT_TEXT = "Analyze this image and return only the requested JSON object."
_SYSTEM_PROMPT = (
    BASE_PROMPT + "\n\nReturn only valid JSON with keys `tags`, `captions`, `score`, and optionally "
    "`ratings`. Do not include markdown fences or any explanatory text."
)


class AnthropicBatchAdapter:
    """Adapter for Anthropic Message Batches."""

    @staticmethod
    def batch_metadata() -> dict[str, Any]:
        return {
            "provider_batch_api": "anthropic_message_batches",
            "result_retention_days": 29,
            "zero_data_retention_eligible": False,
            "provider_max_requests": 100_000,
            "provider_max_body_bytes": _MAX_ANTHROPIC_BODY_BYTES,
            "library_max_items": _MAX_LIBRARY_ITEMS,
        }

    def submit_batch(self, request: BatchSubmitRequest) -> BatchSubmitResult:
        if request.provider.lower() != _PROVIDER:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "unsupported_provider",
                f"Anthropic adapter cannot submit provider: {request.provider}",
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
                f"Unsupported Anthropic batch prompt_profile: {request.prompt_profile}",
                retryable=False,
            )

        self._validate_custom_ids([item.custom_id for item in request.items])
        prepared = prepare_items(request.items, provider=_PROVIDER)
        requests = self._build_requests(request.litellm_model_id, prepared)
        self._validate_payload_size(requests)

        try:
            batch = self._client(api_key).messages.batches.create(requests=requests)
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
                "Anthropic batch create response did not include an id",
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
        api_key = self._api_key(
            handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.CANCEL
        )
        try:
            batch = self._client(api_key).messages.batches.cancel(handle.provider_job_id)
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
                "Anthropic batch results are not available until processing has ended",
                retryable=True,
            )

        api_key = self._api_key(
            handle.api_keys, provider_job_id=handle.provider_job_id, phase=BatchErrorPhase.DOWNLOAD
        )
        try:
            stream = self._client(api_key).messages.batches.results(handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                BatchErrorPhase.DOWNLOAD,
                handle.provider_job_id,
                "download_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc

        items: list[BatchResultItem] = []
        try:
            for raw_item in stream:
                items.append(self._normalize_result_item(raw_item))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise self._job_error(
                BatchErrorPhase.PARSE,
                handle.provider_job_id,
                "result_stream_parse_failed",
                self._format_exception(exc),
                retryable=False,
            ) from exc
        return BatchFetchResult(
            provider=_PROVIDER,
            provider_job_id=handle.provider_job_id,
            status=status.status,
            items=items,
        )

    def _retrieve_status(self, handle: BatchJobHandle, *, phase: BatchErrorPhase) -> BatchStatusResult:
        api_key = self._api_key(handle.api_keys, provider_job_id=handle.provider_job_id, phase=phase)
        try:
            batch = self._client(api_key).messages.batches.retrieve(handle.provider_job_id)
        except Exception as exc:
            raise self._job_error(
                phase,
                handle.provider_job_id,
                "retrieve_failed",
                self._format_exception(exc),
                retryable=self._is_retryable_exception(exc),
            ) from exc
        return _to_status_result(batch, provider_job_id=handle.provider_job_id)

    def _build_requests(
        self, litellm_model_id: str, items: list[PreparedBatchItem]
    ) -> list[dict[str, Any]]:
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
                f"Anthropic batch requires an anthropic model, got: {litellm_model_id}",
                retryable=False,
            )

        requests: list[dict[str, Any]] = []
        for item in items:
            try:
                image_bytes = item.image_path.read_bytes()
            except OSError as exc:
                raise self._job_error(
                    BatchErrorPhase.PREPARE,
                    None,
                    "image_read_failed",
                    self._format_exception(exc),
                    retryable=False,
                ) from exc
            encoded = base64.b64encode(image_bytes).decode("ascii")
            requests.append(
                {
                    "custom_id": item.custom_id,
                    "params": {
                        "model": ref.provider_model_id,
                        "max_tokens": _DEFAULT_MAX_TOKENS,
                        "system": _SYSTEM_PROMPT,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": _USER_PROMPT_TEXT},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": item.image_mime_type,
                                            "data": encoded,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                }
            )
        return requests

    def _validate_payload_size(self, requests: list[dict[str, Any]]) -> None:
        size = len(json.dumps({"requests": requests}, separators=(",", ":")).encode("utf-8"))
        if size > _MAX_ANTHROPIC_BODY_BYTES:
            raise self._job_error(
                BatchErrorPhase.PREPARE,
                None,
                "payload_too_large",
                f"Serialized Anthropic batch request is {size} bytes; maximum is {_MAX_ANTHROPIC_BODY_BYTES}",
                retryable=False,
            )

    def _validate_custom_ids(self, custom_ids: list[str]) -> None:
        seen: set[str] = set()
        for custom_id in custom_ids:
            if not _CUSTOM_ID_PATTERN.fullmatch(custom_id):
                raise self._job_error(
                    BatchErrorPhase.PREPARE,
                    None,
                    "invalid_custom_id",
                    "Anthropic custom_id must be 1-64 chars and contain only letters, numbers, hyphens, or underscores",
                    retryable=False,
                )
            if custom_id in seen:
                raise self._job_error(
                    BatchErrorPhase.PREPARE,
                    None,
                    "duplicate_custom_id",
                    f"Duplicate custom_id in batch request: {custom_id}",
                    retryable=False,
                )
            seen.add(custom_id)

    def _normalize_result_item(self, raw_item: Any) -> BatchResultItem:
        item = _decode_json_line(raw_item)
        custom_id = str(_get(item, "custom_id") or "")
        result = _get(item, "result") or {}
        result_type = str(_get(result, "type") or "unknown")
        provider_status = _provider_item_status(result_type)

        if result_type != "succeeded":
            return BatchResultItem(
                custom_id=custom_id,
                status=BatchItemStatus.FAILED,
                provider_status=provider_status,
                annotation=None,
                error=_provider_item_error(result_type, result),
            )

        message = _get(result, "message") or {}
        stop_reason = _get(message, "stop_reason")
        if stop_reason == "refusal":
            return _failed_result_item(
                custom_id,
                provider_status,
                BatchErrorPhase.NORMALIZE,
                "safety_refusal",
                "Anthropic response stop_reason was refusal",
                retryable=False,
            )
        text = _extract_text(message)
        if not text:
            return _failed_result_item(
                custom_id,
                provider_status,
                BatchErrorPhase.PARSE,
                "annotation_output_unparseable",
                "Anthropic response did not contain text content",
                retryable=False,
            )
        try:
            schema = _parse_annotation_schema(text)
            annotation = _schema_to_unified(schema)
        except (ValueError, ModelRetry) as exc:
            return _failed_result_item(
                custom_id,
                provider_status,
                BatchErrorPhase.NORMALIZE,
                "annotation_output_unparseable",
                str(exc),
                retryable=False,
            )
        return BatchResultItem(
            custom_id=custom_id,
            status=BatchItemStatus.SUCCEEDED,
            provider_status=provider_status,
            annotation=annotation,
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
                "Missing Anthropic API key in api_keys['anthropic']",
                retryable=False,
            )
        return api_key

    def _client(self, api_key: str) -> Any:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

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
        return type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }


def _decode_json_line(raw_item: Any) -> Any:
    if isinstance(raw_item, bytes):
        raw_item = raw_item.decode("utf-8")
    if isinstance(raw_item, str):
        return json.loads(raw_item)
    return raw_item


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped.get(name, default)
    return getattr(value, name, default)


def _request_counts(batch: Any) -> Any:
    return _get(batch, "request_counts") or {}


def _count(batch: Any, name: str) -> int | None:
    value = _get(_request_counts(batch), name)
    return int(value) if value is not None else None


def _request_count(batch: Any) -> int | None:
    counts = [_count(batch, key) for key in ("processing", "succeeded", "errored", "canceled", "expired")]
    known_counts = [count for count in counts if count is not None]
    return sum(known_counts) if known_counts else None


def _status_from_batch(batch: Any) -> BatchStatus:
    processing_status = str(_get(batch, "processing_status") or "").lower()
    if processing_status in {"in_progress", "canceling"}:
        return BatchStatus.RUNNING
    if processing_status != "ended":
        return BatchStatus.UNKNOWN

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
        completed_at=_parse_datetime(_get(batch, "ended_at")),
        expires_at=_parse_datetime(_get(batch, "expires_at")),
    )


def _provider_item_status(result_type: str) -> BatchProviderItemStatus:
    try:
        return BatchProviderItemStatus(result_type)
    except ValueError:
        return BatchProviderItemStatus.UNKNOWN


def _provider_item_error(result_type: str, result: Any) -> BatchItemError:
    error = _get(result, "error") or {}
    message = _get(error, "message") or f"Anthropic batch item result type was {result_type}"
    code = {
        "errored": "provider_item_error",
        "canceled": "provider_item_canceled",
        "expired": "provider_item_expired",
    }.get(result_type, "unknown_item_error")
    return BatchItemError(
        phase=BatchErrorPhase.NORMALIZE,
        code=code,
        message=str(message),
        retryable=_provider_item_retryable(result_type, error),
    )


def _provider_item_retryable(result_type: str, error: Any) -> bool:
    if result_type == "expired":
        return True
    if result_type != "errored":
        return False

    error_type = str(_get(error, "type") or _get(error, "code") or "").lower()
    if error_type in {"api_error", "server_error", "overloaded_error", "rate_limit_error"}:
        return True
    status_code = _get(error, "status_code")
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)


def _extract_text(message: Any) -> str:
    content = _get(message, "content") or []
    parts: list[str] = []
    for block in content:
        if _get(block, "type") == "text":
            text = _get(block, "text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _parse_annotation_schema(text: str) -> AnnotationSchema:
    normalizer = build_annotation_output_normalizer(_CAPABILITIES)
    payload = _parse_json_payload(text) or _parse_legacy_text_payload(text)
    if payload is None:
        raise ValueError("Could not parse annotation output as JSON or legacy tags/caption/score text")
    return normalizer(
        tags=payload.get("tags"),
        captions=payload.get("captions", payload.get("caption")),
        score=payload.get("score"),
        ratings=payload.get("ratings"),
        rating=payload.get("rating"),
        rating_confidence=payload.get("rating_confidence"),
    )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_legacy_text_payload(text: str) -> dict[str, Any] | None:
    tags_match = re.search(r"(?im)^tags:\s*(.+)$", text)
    caption_match = re.search(r"(?im)^captions?:\s*(.+)$", text)
    score_match = re.search(r"(?im)^score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not tags_match or not caption_match or not score_match:
        return None
    return {
        "tags": tags_match.group(1),
        "captions": caption_match.group(1),
        "score": score_match.group(1),
    }


def _schema_to_unified(schema: AnnotationSchema) -> UnifiedAnnotationResult:
    return UnifiedAnnotationResult(
        model_name="anthropic_batch",
        capabilities=set(_CAPABILITIES),
        tags=schema.tags,
        captions=schema.captions,
        scores={"overall": float(schema.score)} if schema.score is not None else None,
        ratings=schema.ratings or None,
        provider_name=_PROVIDER,
        framework="anthropic_batch",
        raw_output=None,
    )


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
        error=BatchItemError(phase=phase, code=code, message=message, retryable=retryable),
    )
