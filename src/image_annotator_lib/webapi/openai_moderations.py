"""OpenAI Moderations → LoRAIro rating conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from image_annotator_lib.core.types import RatingPrediction


OPENAI_MODERATION_SOURCE_SCHEME = "openai_moderation_v1"


@dataclass(frozen=True)
class _RatedCategory:
    raw_label: str
    confidence_score: float
    weight: int


_RATING_ORDER = ("pg", "pg13", "r", "x", "xxx")

# Priority from low to high by index; when multiple conditions match, choose the highest index.
_CATEGORY_RULES: dict[str, list[tuple[float, str]]] = {
    "sexual": [(0.30, "pg13"), (0.60, "r"), (0.90, "x"), (0.98, "xxx")],
    "violence": [(0.60, "pg13")],
    "violence/graphic": [(0.50, "r")],
    "self-harm": [(0.70, "r")],
    "self-harm/intent": [(0.70, "r")],
    "self-harm/instructions": [(0.70, "r")],
}

_IGNORED_CATEGORIES = {
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "sexual/minors",
}


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_result_payload(response: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a moderation result object or parent API response."""

    if "results" in response and isinstance(response["results"], list) and response["results"]:
        first = response["results"][0]
        if isinstance(first, dict):
            return first
    return response


def _iter_category_scores(response: Mapping[str, Any]) -> Mapping[str, float]:
    raw_scores = response.get("category_scores")
    if not isinstance(raw_scores, Mapping):
        return {}

    scores: dict[str, float] = {}
    for category, score in raw_scores.items():
        if category in _IGNORED_CATEGORIES:
            continue
        if category not in _CATEGORY_RULES:
            continue
        normalized = _to_float(score)
        if normalized is None:
            continue
        scores[category] = normalized
    return scores


def _select_rating(scores: Mapping[str, float]) -> tuple[str, float] | None:
    best: _RatedCategory | None = None
    for category, score in scores.items():
        rule_list = _CATEGORY_RULES[category]
        candidate: str | None = None
        for threshold, raw_label in rule_list:
            if score >= threshold:
                candidate = raw_label
        if candidate is None:
            continue

        weight = _RATING_ORDER.index(candidate)
        candidate_rating = _RatedCategory(raw_label=candidate, confidence_score=score, weight=weight)

        if (
            best is None
            or candidate_rating.weight > best.weight
            or (
                candidate_rating.weight == best.weight
                and candidate_rating.confidence_score > best.confidence_score
            )
        ):
            best = candidate_rating

    if best is None:
        return None
    return (best.raw_label, best.confidence_score)


def category_scores_to_rating_prediction(response: Mapping[str, Any]) -> RatingPrediction:
    """Convert a single OpenAI moderation result to a `RatingPrediction`."""

    if not isinstance(response, Mapping):
        return RatingPrediction(raw_label="pg", source_scheme=OPENAI_MODERATION_SOURCE_SCHEME, confidence_score=None)

    result = _extract_result_payload(response)
    scores = _iter_category_scores(result)
    selected = _select_rating(scores)

    if selected is None:
        return RatingPrediction(raw_label="pg", source_scheme=OPENAI_MODERATION_SOURCE_SCHEME, confidence_score=None)

    raw_label, confidence = selected
    return RatingPrediction(raw_label=raw_label, source_scheme=OPENAI_MODERATION_SOURCE_SCHEME, confidence_score=confidence)
