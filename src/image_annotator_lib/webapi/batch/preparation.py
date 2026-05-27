"""Shared preparation helpers for provider batch adapters."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_annotator_lib.core.types import AnnotationSchema, TaskCapability
from image_annotator_lib.webapi.image_payload import build_base64_data_url

from .types import BatchErrorPhase, BatchJobError, BatchSubmitItem

SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

# /v1/chat/completions Batch で扱う標準 annotation capability。Ratings は
# /v1/moderations 経路でカバーするため本 builder のデフォルトには含めない。
_DEFAULT_ANNOTATION_CAPABILITIES = frozenset(
    {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
)

# AnnotationSchema property → TaskCapability への対応 (output_normalization 側と整合)。
_PROPERTY_TO_CAPABILITY: dict[str, TaskCapability] = {
    "tags": TaskCapability.TAGS,
    "captions": TaskCapability.CAPTIONS,
    "score": TaskCapability.SCORES,
}

_ANNOTATION_TOOL_NAME = "normalize_annotation_output"
_ANNOTATION_TOOL_DESCRIPTION = (
    "Provide image annotation results as structured data. Required fields depend on "
    "the requested capabilities (tags / captions / score). Ratings are optional."
)

# Agent 同期経路と揃えるための短い user instruction。
_USER_PROMPT_TEXT = "Analyze this image and provide annotations as specified."


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
                                    "url": build_base64_data_url(payload_bytes, item.image_mime_type),
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


def build_annotation_tool_schema(
    capabilities: frozenset[TaskCapability] = _DEFAULT_ANNOTATION_CAPABILITIES,
) -> dict[str, Any]:
    """Build an OpenAI function-tool schema for AnnotationSchema (Issue #518).

    Pydantic の ``AnnotationSchema.model_json_schema()`` を SSoT として、capability に
    応じて required fields を filter する。

    - ``TAGS`` / ``CAPTIONS`` / ``SCORES`` が required かは引数 ``capabilities`` で決まる
    - ``ratings`` は best-effort 出力のため常に optional
    - ``$defs`` (RatingPrediction 等の nested schema) はそのまま保持

    Args:
        capabilities: 出力 schema 上で required にする core capability の集合。
            空集合の場合は all-optional schema (LLM 側でゼロ field 出力も valid)。

    Returns:
        OpenAI function-tool ``parameters`` に渡せる JSON schema 相当の dict。
    """

    schema = AnnotationSchema.model_json_schema()
    required = sorted(
        prop
        for prop, capability in _PROPERTY_TO_CAPABILITY.items()
        if capability in capabilities
    )
    if required:
        schema["required"] = required
    else:
        schema.pop("required", None)
    return schema


def build_openai_chat_completions_annotation_jsonl(
    items: list[PreparedBatchItem],
    *,
    endpoint: str,
    litellm_model_id: str,
    system_prompt: str,
    capabilities: frozenset[TaskCapability] = _DEFAULT_ANNOTATION_CAPABILITIES,
) -> str:
    """Build OpenAI batch input JSONL for /v1/chat/completions annotation requests.

    structured output schema は ``AnnotationSchema`` (Pydantic) を SSoT とし、
    function tool として LLM に publish する (Issue #518 / ADR 0038)。同期側
    PydanticAI 経路が生成する Chat Completions request と同等の shape を作る。

    Args:
        items: ``prepare_items()`` の出力。
        endpoint: ``"/v1/chat/completions"`` (検証は adapter 側で実施)。
        litellm_model_id: ``"openai/gpt-4o-mini"`` 等の registry model id。
        system_prompt: capability に応じた system prompt (同期側 ``_build_system_prompt`` 由来)。
        capabilities: 出力 schema 上で required にする core capability の集合。

    Returns:
        OpenAI Batch input JSONL (各 line = 1 request)。

    Raises:
        BatchJobError: 画像読み込み失敗時。
    """

    tool = {
        "type": "function",
        "function": {
            "name": _ANNOTATION_TOOL_NAME,
            "description": _ANNOTATION_TOOL_DESCRIPTION,
            "parameters": build_annotation_tool_schema(capabilities),
        },
    }
    tool_choice = {"type": "function", "function": {"name": _ANNOTATION_TOOL_NAME}}

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
        data_url = build_base64_data_url(payload_bytes, item.image_mime_type)
        body = {
            "model": litellm_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _USER_PROMPT_TEXT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "tools": [tool],
            "tool_choice": tool_choice,
        }
        lines.append(
            json.dumps(
                {
                    "custom_id": item.custom_id,
                    "method": "POST",
                    "url": endpoint,
                    "body": body,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    return "\n".join(lines)
