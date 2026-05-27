"""Unit tests for the /v1/chat/completions annotation Batch JSONL builder (Issue #518)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_annotator_lib.core.types import AnnotationSchema, TaskCapability
from image_annotator_lib.webapi.batch.preparation import (
    PreparedBatchItem,
    build_annotation_tool_schema,
    build_openai_chat_completions_annotation_jsonl,
)
from image_annotator_lib.webapi.batch.types import BatchErrorPhase, BatchJobError


def _make_item(image_path: Path, custom_id: str = "img-1", image_id: int = 1) -> PreparedBatchItem:
    return PreparedBatchItem(
        custom_id=custom_id,
        image_id=image_id,
        image_path=image_path,
        image_mime_type="image/png",
    )


@pytest.fixture
def png_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    # 1x1 PNG (minimal valid PNG header + IDAT)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f"
        b"\x15\xc4\x89\x00\x00\x00\rIDAT\x78\x9c\x62\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return path


class TestBuildAnnotationToolSchema:
    def test_default_capabilities_make_tags_captions_score_required(self) -> None:
        schema = build_annotation_tool_schema()
        assert schema["required"] == ["captions", "score", "tags"]

    def test_only_tags_capability(self) -> None:
        schema = build_annotation_tool_schema(frozenset({TaskCapability.TAGS}))
        assert schema["required"] == ["tags"]
        # ratings property は ratings capability に関係なく schema 自体には残る (optional)
        assert "ratings" in schema["properties"]

    def test_empty_capabilities_drops_required(self) -> None:
        schema = build_annotation_tool_schema(frozenset())
        assert "required" not in schema
        # AnnotationSchema 本体の properties はすべて維持
        assert set(schema["properties"]) == {"tags", "captions", "score", "ratings"}

    def test_schema_carries_pydantic_defs_for_rating_prediction(self) -> None:
        schema = build_annotation_tool_schema()
        assert "$defs" in schema
        assert "RatingPrediction" in schema["$defs"]

    def test_schema_is_consistent_with_annotation_schema_properties(self) -> None:
        schema = build_annotation_tool_schema()
        # `AnnotationSchema.model_json_schema()` の properties をそのまま運ぶことを保証
        canonical = AnnotationSchema.model_json_schema()
        assert schema["properties"] == canonical["properties"]


class TestBuildOpenAIChatCompletionsAnnotationJsonl:
    def test_single_item_produces_chat_completions_request(self, png_image: Path) -> None:
        items = [_make_item(png_image)]
        jsonl = build_openai_chat_completions_annotation_jsonl(
            items,
            endpoint="/v1/chat/completions",
            litellm_model_id="openai/gpt-4o-mini",
            system_prompt="System prompt body",
        )
        lines = jsonl.splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["custom_id"] == "img-1"
        assert payload["method"] == "POST"
        assert payload["url"] == "/v1/chat/completions"

        body = payload["body"]
        assert body["model"] == "openai/gpt-4o-mini"
        assert body["messages"][0] == {"role": "system", "content": "System prompt body"}
        user = body["messages"][1]
        assert user["role"] == "user"
        text_part, image_part = user["content"]
        assert text_part["type"] == "text"
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_tool_choice_pins_normalize_annotation_output(self, png_image: Path) -> None:
        items = [_make_item(png_image)]
        jsonl = build_openai_chat_completions_annotation_jsonl(
            items,
            endpoint="/v1/chat/completions",
            litellm_model_id="openai/gpt-4o-mini",
            system_prompt="System prompt body",
        )
        body = json.loads(jsonl)["body"]
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "normalize_annotation_output"},
        }
        tools = body["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "normalize_annotation_output"
        assert tools[0]["function"]["parameters"]["required"] == ["captions", "score", "tags"]

    def test_capabilities_override_filters_required_fields(self, png_image: Path) -> None:
        items = [_make_item(png_image)]
        jsonl = build_openai_chat_completions_annotation_jsonl(
            items,
            endpoint="/v1/chat/completions",
            litellm_model_id="openai/gpt-4o-mini",
            system_prompt="System prompt body",
            capabilities=frozenset({TaskCapability.TAGS}),
        )
        body = json.loads(jsonl)["body"]
        assert body["tools"][0]["function"]["parameters"]["required"] == ["tags"]

    def test_multiple_items_emit_one_line_per_request(self, tmp_path: Path, png_image: Path) -> None:
        second = tmp_path / "second.png"
        second.write_bytes(png_image.read_bytes())
        items = [
            _make_item(png_image, custom_id="img-1", image_id=1),
            _make_item(second, custom_id="img-2", image_id=2),
        ]
        jsonl = build_openai_chat_completions_annotation_jsonl(
            items,
            endpoint="/v1/chat/completions",
            litellm_model_id="openai/gpt-4o-mini",
            system_prompt="System prompt body",
        )
        lines = jsonl.splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["custom_id"] for line in lines] == ["img-1", "img-2"]

    def test_missing_image_raises_batch_job_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.png"
        items = [_make_item(missing)]
        with pytest.raises(BatchJobError) as excinfo:
            build_openai_chat_completions_annotation_jsonl(
                items,
                endpoint="/v1/chat/completions",
                litellm_model_id="openai/gpt-4o-mini",
                system_prompt="System prompt body",
            )
        assert excinfo.value.phase is BatchErrorPhase.PREPARE
        assert excinfo.value.code == "image_read_failed"
