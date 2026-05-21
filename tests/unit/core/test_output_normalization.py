"""Issue #47: PydanticAI output normalization tests."""

from __future__ import annotations

import pytest
from pydantic_ai import ModelRetry

from image_annotator_lib.core.output_normalization import normalize_annotation_output


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
        result = normalize_annotation_output(
            tags=[],
            captions=[],
            score=None,
            rating="  questionable  ",
            rating_confidence="0.82",
        )

        assert result.ratings[0].raw_label == "questionable"
        assert result.ratings[0].confidence_score == 0.82
        assert result.ratings[0].source_scheme == "prompt_defined"

    def test_normalizes_rating_objects_without_canonical_mapping(self) -> None:
        result = normalize_annotation_output(
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
