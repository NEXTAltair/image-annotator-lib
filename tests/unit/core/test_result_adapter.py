"""ADR 0023 Phase 1: core/result_adapter.py の unit test."""

from __future__ import annotations

from image_annotator_lib.core.result_adapter import to_annotation_result
from image_annotator_lib.core.types import AnnotationSchema


class TestToAnnotationResult:
    """`to_annotation_result()` の AnnotationSchema → AnnotationResult 変換を確認する。"""

    def test_normal_schema_output(self) -> None:
        schema = AnnotationSchema(tags=["1girl", "blue eyes"], captions=["a girl"], score=8.5)
        result = to_annotation_result(schema, phash="abc123")

        assert result["phash"] == "abc123"
        assert result["error"] is None
        assert result["tags"] == ["1girl", "blue eyes"]
        formatted = result["formatted_output"]
        assert isinstance(formatted, dict)
        assert formatted["tags"] == ["1girl", "blue eyes"]
        assert formatted["captions"] == ["a girl"]
        assert formatted["score"] == 8.5

    def test_error_path(self) -> None:
        result = to_annotation_result(None, phash="abc123", error="auth failure")
        assert result["phash"] == "abc123"
        assert result["error"] == "auth failure"
        assert result["tags"] == []
        assert result["formatted_output"] is None

    def test_none_schema_without_error(self) -> None:
        # schema_output=None かつ error=None でも error 経路になる
        result = to_annotation_result(None, phash="abc123")
        assert result["phash"] == "abc123"
        assert result["tags"] == []
        assert result["formatted_output"] is None

    def test_strips_whitespace(self) -> None:
        # Issue #47: pre-validation normalization lives in output_normalization.
        # result_adapter keeps only final-defense cleanup for validated schema values.
        schema = AnnotationSchema(tags=["  1girl  ", "blue eyes  "], captions=["  a girl"], score=8.0)
        result = to_annotation_result(schema, phash="abc")
        assert result["tags"] == ["1girl", "blue eyes"]
        formatted = result["formatted_output"]
        assert isinstance(formatted, dict)
        assert formatted["captions"] == ["a girl"]

    def test_excludes_empty_strings(self) -> None:
        # Issue #47: this is final-defense cleanup, not raw output repair.
        schema = AnnotationSchema(tags=["1girl", "", "  ", "blue eyes"], captions=[], score=5.0)
        result = to_annotation_result(schema, phash="abc")
        assert result["tags"] == ["1girl", "blue eyes"]

    def test_score_passed_through_as_float(self) -> None:
        schema = AnnotationSchema(tags=[], captions=[], score=7.25)
        result = to_annotation_result(schema, phash="abc")
        formatted = result["formatted_output"]
        assert isinstance(formatted, dict)
        assert formatted["score"] == 7.25
