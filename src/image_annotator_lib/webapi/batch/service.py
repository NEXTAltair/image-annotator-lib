"""Provider Batch API dispatch layer."""

from __future__ import annotations

from collections.abc import Iterable

from image_annotator_lib.core.registry import get_webapi_metadata, list_available_annotators
from image_annotator_lib.core.types import TaskCapability

from .adapters.anthropic import AnthropicBatchAdapter
from .adapters.openai import OpenAIBatchAdapter
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

BatchAdapter = AnthropicBatchAdapter | OpenAIBatchAdapter

# adapter 実装済み provider の単一情報源 (SSoT)。eligibility gate と dispatch の両方が
# この registry を参照するため、provider 名のハードコード重複が生じない。
# ADR 0005 "Model eligibility": batch eligibility は LiteLLM batch pricing field ではなく
# adapter 実装の有無を gate とする (LiteLLM 同梱 DB が direct anthropic route の
# batch pricing field を持たないため、pricing-field を一律ゲートにすると Anthropic を
# 誤って除外してしまう。実 dispatch 可否 = adapter 実装の有無)。
_BATCH_ADAPTERS: dict[str, type[BatchAdapter]] = {
    "anthropic": AnthropicBatchAdapter,
    "openai": OpenAIBatchAdapter,
}

# OpenAI `gpt-5.5-pro` family cost-safety denylist (ADR 0005)。
# `gpt-5.5-pro` / `gpt-5.5-pro-2026-04-23` 等を除外する。非 pro の `gpt-5.5` は対象外。
_DENYLISTED_MODEL_SUBSTRINGS: frozenset[str] = frozenset({"gpt-5.5-pro"})


def _adapter_for_provider(provider: str) -> BatchAdapter:
    normalized = provider.lower()
    adapter_cls = _BATCH_ADAPTERS.get(normalized)
    if adapter_cls is None:
        raise BatchJobError(
            phase=BatchErrorPhase.PREPARE,
            provider=normalized,
            provider_job_id=None,
            code="unsupported_provider",
            message=f"Provider Batch API is not implemented for provider: {provider}",
            retryable=False,
        )
    return adapter_cls()


def _is_denylisted(litellm_model_id: str) -> bool:
    """ADR 0005 cost-safety denylist (`gpt-5.5-pro` family) に該当するか判定する。"""
    lowered = litellm_model_id.lower()
    return any(token in lowered for token in _DENYLISTED_MODEL_SUBSTRINGS)


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
    """Provider batch route で使える direct provider モデルを返す。

    eligibility = adapter 実装済み provider か (`_BATCH_ADAPTERS`)。ADR 0005 の通り
    LiteLLM batch pricing field は判定に使わない (direct anthropic route に同 field が
    無く false negative を招くため)。`gpt-5.5-pro` family は cost-safety denylist で除外する。
    """
    models: list[BatchModelInfo] = []
    for model_name in list_available_annotators():
        metadata = get_webapi_metadata(model_name)
        if metadata is None:
            continue
        provider = str(metadata.get("provider", "")).lower()
        adapter_cls = _BATCH_ADAPTERS.get(provider)
        if adapter_cls is None:
            continue
        litellm_model_id = metadata.get("litellm_model_id")
        if not isinstance(litellm_model_id, str) or not litellm_model_id:
            continue
        if _is_denylisted(litellm_model_id):
            continue
        capabilities = _capabilities_from_metadata(metadata.get("capabilities"))
        models.append(
            BatchModelInfo(
                provider=provider,
                litellm_model_id=litellm_model_id,
                display_name=model_name,
                capabilities=capabilities,
                metadata=adapter_cls.batch_metadata(),
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
