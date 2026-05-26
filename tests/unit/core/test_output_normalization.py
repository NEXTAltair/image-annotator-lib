"""Issue #47: PydanticAI output normalization tests."""

from __future__ import annotations

import inspect

import pytest
from pydantic_ai import ModelRetry

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.webapi.output_normalization import (
    build_annotation_output_normalizer,
    normalize_annotation_output,
)


class TestNormalizeAnnotationOutput:
    """Minor output drift is normalized before `AnnotationSchema` validation."""

    def test_normalizes_string_tags_caption_and_score(self) -> None:
        result = normalize_annotation_output(
            tags="cat, smile, outdoors",
            captions="a cat sitting outside",
            score="0.82",
        )

        assert result.tags == ["cat", "smile", "outdoors"]
        assert result.captions == ["a cat sitting outside"]
        assert result.score == 0.82

    def test_single_string_tag_becomes_single_item_list(self) -> None:
        result = normalize_annotation_output(tags="cat", captions=["caption"], score=1)

        assert result.tags == ["cat"]
        assert result.captions == ["caption"]
        assert result.score == 1.0

    def test_trims_and_drops_empty_items(self) -> None:
        result = normalize_annotation_output(
            tags=[" cat ", "", "  ", "dog"],
            captions=[" caption ", ""],
            score=0.5,
        )

        assert result.tags == ["cat", "dog"]
        assert result.captions == ["caption"]

    def test_normalizes_single_rating_with_optional_confidence(self) -> None:
        normalizer = build_annotation_output_normalizer({TaskCapability.RATINGS})

        result = normalizer(
            rating="  questionable  ",
            rating_confidence="0.82",
        )

        assert result.ratings[0].raw_label == "questionable"
        assert result.ratings[0].confidence_score == 0.82
        assert result.ratings[0].source_scheme == "prompt_defined"

    def test_normalizes_rating_objects_without_canonical_mapping(self) -> None:
        normalizer = build_annotation_output_normalizer({TaskCapability.RATINGS})

        result = normalizer(
            ratings=[
                {"label": "PG-13", "confidence": 0.7},
                {"raw_label": "mature"},
            ]
        )

        assert [rating.raw_label for rating in result.ratings] == ["PG-13", "mature"]
        assert result.ratings[0].confidence_score == 0.7
        assert result.ratings[1].confidence_score is None

    @pytest.mark.parametrize(
        ("tags", "captions", "score"),
        [
            ({"tag": "cat"}, ["caption"], 0.5),
            (["cat", 123], ["caption"], 0.5),
            (["cat"], {"caption": "bad"}, 0.5),
            (["cat"], ["caption", None], 0.5),
            (["cat"], ["caption"], "not-a-number"),
            (["cat"], ["caption"], True),
        ],
    )
    def test_unsupported_values_raise_model_retry(
        self, tags: object, captions: object, score: object
    ) -> None:
        with pytest.raises(ModelRetry):
            normalize_annotation_output(tags=tags, captions=captions, score=score)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"captions": ["caption"], "score": 0.5},
            {"tags": ["cat"], "score": 0.5},
            {"tags": ["cat"], "captions": ["caption"]},
        ],
    )
    def test_default_normalizer_retries_when_core_fields_are_missing(
        self, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(ModelRetry):
            normalize_annotation_output(**kwargs)

    def test_capability_normalizer_allows_unrequested_fields_to_be_missing(self) -> None:
        normalizer = build_annotation_output_normalizer({TaskCapability.TAGS})

        result = normalizer(tags="cat")

        assert result.tags == ["cat"]
        assert result.captions == []
        assert result.score is None

    def test_capability_normalizer_allows_rating_to_be_missing(self) -> None:
        """RATINGS 要求時でも rating 欠落は best-effort 扱い: retry せず空リスト (Codex P1)。

        rating はモデルの prompt 由来 best-effort 出力。欠落で tags/captions/score を
        含む正常な annotation 全体を retry/失敗へ巻き込まない。
        """
        normalizer = build_annotation_output_normalizer({TaskCapability.RATINGS})

        result = normalizer()

        assert result.ratings == []

    def test_rating_object_falls_back_when_primary_alias_is_null(self) -> None:
        """raw_label / confidence_score が present かつ null でも alias を拾う (Codex P2)。"""
        normalizer = build_annotation_output_normalizer({TaskCapability.RATINGS})

        result = normalizer(
            ratings=[
                {
                    "raw_label": None,
                    "label": "PG-13",
                    "confidence_score": None,
                    "confidence": 0.7,
                }
            ]
        )

        assert result.ratings[0].raw_label == "PG-13"
        assert result.ratings[0].confidence_score == 0.7

    def test_builder_marks_requested_core_fields_required_in_signature(self) -> None:
        """build_annotation_output_normalizer は要求 core field を schema 上 required にする (Codex P2)。"""
        normalizer = build_annotation_output_normalizer(
            {
                TaskCapability.TAGS,
                TaskCapability.CAPTIONS,
                TaskCapability.SCORES,
                TaskCapability.RATINGS,
            }
        )

        params = inspect.signature(normalizer).parameters

        assert params["tags"].default is inspect.Parameter.empty
        assert params["captions"].default is inspect.Parameter.empty
        assert params["score"].default is inspect.Parameter.empty
        # rating 系は best-effort のため常に optional。
        assert params["rating"].default is None
        assert params["rating_confidence"].default is None
        assert params["ratings"].default is None

    def test_builder_keeps_unrequested_core_fields_optional_in_signature(self) -> None:
        """非要求 core field は schema 上 optional のまま (Codex P2)。"""
        normalizer = build_annotation_output_normalizer({TaskCapability.TAGS})

        params = inspect.signature(normalizer).parameters

        assert params["tags"].default is inspect.Parameter.empty
        assert params["captions"].default is None
        assert params["score"].default is None

    def test_capability_normalizer_ignores_malformed_unrequested_core_fields(self) -> None:
        """RATINGS のみ要求時、付随的な不正形 core field は retry を誘発しない (Codex P2)。"""
        normalizer = build_annotation_output_normalizer({TaskCapability.RATINGS})

        result = normalizer(rating="explicit", tags={"unexpected": 1}, score="not-a-number")

        assert result.ratings[0].raw_label == "explicit"
        assert result.tags == []
        assert result.captions == []
        assert result.score is None

    def test_capability_normalizer_ignores_malformed_unrequested_rating_field(self) -> None:
        """core capability のみ要求時、付随的な不正形 rating field は retry を誘発しない (Codex P1)。"""
        normalizer = build_annotation_output_normalizer(
            {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
        )

        result = normalizer(tags=["cat"], captions=["a cat"], score=0.8, ratings="explicit")

        assert result.tags == ["cat"]
        assert result.captions == ["a cat"]
        assert result.score == 0.8
        assert result.ratings == []
