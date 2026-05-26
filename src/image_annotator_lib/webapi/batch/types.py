"""DTOs for provider batch annotation APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult


@dataclass(frozen=True)
class BatchModelInfo:
    provider: str
    litellm_model_id: str
    display_name: str
    capabilities: frozenset[TaskCapability]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchSubmitItem:
    custom_id: str
    image_id: int
    image_path: Path


@dataclass(frozen=True)
class BatchSubmitRequest:
    provider: str
    endpoint: str
    litellm_model_id: str
    prompt_profile: str
    description: str | None
    api_keys: dict[str, str]
    items: list[BatchSubmitItem]


class BatchStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BatchSubmitResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    request_count: int


@dataclass(frozen=True)
class BatchJobHandle:
    provider: str
    provider_job_id: str
    api_keys: dict[str, str]


@dataclass(frozen=True)
class BatchStatusResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    request_count: int | None
    succeeded_count: int | None
    failed_count: int | None
    canceled_count: int | None
    expired_count: int | None
    submitted_at: datetime | None
    completed_at: datetime | None
    expires_at: datetime | None


class BatchItemStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BatchProviderItemStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"
    FAILED = "failed"
    UNKNOWN = "unknown"


class BatchErrorPhase(StrEnum):
    PREPARE = "prepare"
    UPLOAD = "upload"
    SUBMIT = "submit"
    POLL = "poll"
    CANCEL = "cancel"
    DOWNLOAD = "download"
    PARSE = "parse"
    NORMALIZE = "normalize"
    VALIDATE = "validate"


@dataclass(frozen=True)
class BatchItemError:
    phase: BatchErrorPhase
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True)
class BatchResultItem:
    custom_id: str
    status: BatchItemStatus
    provider_status: BatchProviderItemStatus
    annotation: UnifiedAnnotationResult | None
    error: BatchItemError | None


@dataclass(frozen=True)
class BatchFetchResult:
    provider: str
    provider_job_id: str
    status: BatchStatus
    items: list[BatchResultItem]


class BatchJobError(Exception):
    """Job-level Provider Batch API failure."""

    def __init__(
        self,
        *,
        phase: BatchErrorPhase,
        provider: str,
        provider_job_id: str | None,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.provider = provider
        self.provider_job_id = provider_job_id
        self.code = code
        self.message = message
        self.retryable = retryable
