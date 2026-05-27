"""Shared preparation helpers for provider batch adapters."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from .types import BatchErrorPhase, BatchJobError, BatchSubmitItem

SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


@dataclass(frozen=True)
class PreparedBatchItem:
    custom_id: str
    image_id: int
    image_path: Path
    image_mime_type: str


def prepare_items(items: list[BatchSubmitItem], *, provider: str) -> list[PreparedBatchItem]:
    prepared: list[PreparedBatchItem] = []
    for item in items:
        image_path = Path(item.image_path)
        if not image_path.is_file():
            raise BatchJobError(
                phase=BatchErrorPhase.PREPARE,
                provider=provider,
                provider_job_id=None,
                code="image_not_found",
                message=f"Image file does not exist: {image_path}",
                retryable=False,
            )
        mime_type = mimetypes.guess_type(image_path.name)[0]
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise BatchJobError(
                phase=BatchErrorPhase.PREPARE,
                provider=provider,
                provider_job_id=None,
                code="unsupported_image_format",
                message=f"Unsupported image MIME type for {image_path}: {mime_type or 'unknown'}",
                retryable=False,
            )
        prepared.append(
            PreparedBatchItem(
                custom_id=item.custom_id,
                image_id=item.image_id,
                image_path=image_path,
                image_mime_type=mime_type,
            )
        )
    return prepared


def build_openai_moderations_jsonl(
    items: list[PreparedBatchItem], *, endpoint: str, litellm_model_id: str
) -> str:
    """Build OpenAI batch input JSONL for /v1/moderations requests."""

    lines: list[str] = []
    for item in items:
        try:
            payload_bytes = item.image_path.read_bytes()
        except OSError as exc:
            raise BatchJobError(
                phase=BatchErrorPhase.PREPARE,
                provider="openai",
                provider_job_id=None,
                code="image_read_failed",
                message=f"Failed to read image for OpenAI batch input: {item.image_path}: {exc}",
                retryable=False,
            ) from exc
        encoded = base64.b64encode(payload_bytes).decode("ascii")
        lines.append(
            json.dumps(
                {
                    "custom_id": item.custom_id,
                    "method": "POST",
                    "url": endpoint,
                    "body": {
                        "model": litellm_model_id,
                        "input": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.image_mime_type};base64,{encoded}",
                                },
                            }
                        ],
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    return "\n".join(lines)
