"""Tests for OpenAI Moderations -> RatingPrediction conversion."""

from __future__ import annotations

from image_annotator_lib.core.types import RatingPrediction
from image_annotator_lib.webapi.openai_moderations import (
    OPENAI_MODERATION_SOURCE_SCHEME,
    category_scores_to_rating_prediction,
)


def test_category_scores_defaults_to_pg_when_no_match() -> None:
    result = category_scores_to_rating_prediction({"category_scores": {"violence": 0.10}})

    assert result == RatingPrediction(
        raw_label="pg",
        source_scheme=OPENAI_MODERATION_SOURCE_SCHEME,
        confidence_score=None,
    )


def test_ignores_unscoped_categories() -> None:
    result = category_scores_to_rating_prediction(
        {
            "category_scores": {
                "harassment": 0.99,
                "hate": 0.99,
                "illicit/violent": 0.99,
                "sexual/minors": 0.99,
            }
        }
    )

    assert result.raw_label == "pg"
    assert result.source_scheme == OPENAI_MODERATION_SOURCE_SCHEME


def test_sexual_boundary_pg13() -> None:
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.30}}).raw_label == "pg13"
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.29}}).raw_label == "pg"


def test_sexual_boundary_r_and_x() -> None:
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.60}}).raw_label == "r"
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.89}}).raw_label == "r"
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.90}}).raw_label == "x"


def test_sexual_boundary_xxx() -> None:
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.98}}).raw_label == "xxx"
    assert category_scores_to_rating_prediction({"category_scores": {"sexual": 0.99}}).raw_label == "xxx"


def test_violence_graphic_is_capped_at_r() -> None:
    assert category_scores_to_rating_prediction(
        {"category_scores": {"violence/graphic": 0.99}}
    ).raw_label == "r"


def test_violence_graphic_and_violence_choose_same_or_heavier_label() -> None:
    result = category_scores_to_rating_prediction(
        {"category_scores": {"violence/graphic": 0.99, "violence": 0.99}}
    )

    assert result.raw_label == "r"


def test_self_harm_categories_share_r_threshold() -> None:
    result = category_scores_to_rating_prediction(
        {
            "category_scores": {
                "self-harm": 0.70,
                "self-harm/intent": 0.70,
                "self-harm/instructions": 0.70,
            }
        }
    )

    assert result.raw_label == "r"


def test_selects_heaviest_label_across_categories() -> None:
    result = category_scores_to_rating_prediction(
        {
            "category_scores": {
                "sexual": 0.99,
                "violence": 0.99,
            }
        }
    )

    assert result.raw_label == "xxx"


def test_confidence_score_follows_selected_raw_rating_category() -> None:
    result = category_scores_to_rating_prediction(
        {
            "category_scores": {
                "violence": 0.70,
                "sexual": 0.61,
            }
        }
    )

    assert result.confidence_score == 0.61


def test_confidence_score_uses_highest_of_equal_label_weight() -> None:
    result = category_scores_to_rating_prediction(
        {
            "category_scores": {
                "sexual": 0.61,
                "self-harm": 0.70,
            }
        }
    )

    assert result.raw_label == "r"
    assert result.confidence_score == 0.70


def test_accepts_openai_result_payload_shape() -> None:
    result = category_scores_to_rating_prediction(
        {
            "id": "modr-123",
            "results": [
                {
                    "flagged": True,
                    "category_scores": {
                        "sexual": 0.60,
                        "violence": 0.20,
                    },
                }
            ],
        }
    )

    assert result.raw_label == "r"
