"""Provider Batch API dispatch layer."""

from __future__ import annotations

from collections.abc import Iterable

from image_annotator_lib.core.registry import get_webapi_metadata, list_available_annotators
from image_annotator_lib.core.types import TaskCapability

from .adapters.anthropic import AnthropicBatchAdapter
from .types import (
    BatchErrorPhase,
    BatchFetchResult,
    BatchJobError,
    BatchJobHandle,
    BatchModelInfo,
    BatchStatusResult,
    BatchSubmitRequest,
    BatchSubmitResult,
)


def _adapter_for_provider(provider: str) -> AnthropicBatchAdapter:
    normalized = provider.lower()
    if normalized == "anthropic":
        return AnthropicBatchAdapter()
    raise BatchJobError(
        phase=BatchErrorPhase.PREPARE,
        provider=normalized,
        provider_job_id=None,
        code="unsupported_provider",
        message=f"Provider Batch API is not implemented for provider: {provider}",
        retryable=False,
    )


def _capabilities_from_metadata(value: object) -> frozenset[TaskCapability]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return frozenset({TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES})
    capabilities: set[TaskCapability] = set()
    for item in value:
        try:
            capabilities.add(item if isinstance(item, TaskCapability) else TaskCapability(str(item)))
        except ValueError:
            continue
    return frozenset(capabilities or {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES})


def list_batch_capable_models() -> list[BatchModelInfo]:
    """Return direct Anthropic models usable with the provider batch route."""
    models: list[BatchModelInfo] = []
    for model_name in list_available_annotators():
        metadata = get_webapi_metadata(model_name)
        if not metadata or str(metadata.get("provider", "")).lower() != "anthropic":
            continue
        litellm_model_id = metadata.get("litellm_model_id")
        if not isinstance(litellm_model_id, str) or not litellm_model_id:
            continue
        models.append(
            BatchModelInfo(
                provider="anthropic",
                litellm_model_id=litellm_model_id,
                display_name=model_name,
                capabilities=_capabilities_from_metadata(metadata.get("capabilities")),
                metadata=AnthropicBatchAdapter.batch_metadata(),
            )
        )
    return models


def submit_batch(request: BatchSubmitRequest) -> BatchSubmitResult:
    return _adapter_for_provider(request.provider).submit_batch(request)


def retrieve_batch(handle: BatchJobHandle) -> BatchStatusResult:
    return _adapter_for_provider(handle.provider).retrieve_batch(handle)


def cancel_batch(handle: BatchJobHandle) -> BatchStatusResult:
    return _adapter_for_provider(handle.provider).cancel_batch(handle)


def fetch_batch_results(handle: BatchJobHandle) -> BatchFetchResult:
    return _adapter_for_provider(handle.provider).fetch_batch_results(handle)
