"""PydanticAI output normalization for WebAPI annotation results.

ADR 0023 Issue #47: PydanticAI `Agent.output_type` に渡す callable として
`normalize_annotation_output` を提供する。callable は PydanticAI が **tool**
として LLM に公開するため、関数の docstring は LLM 視点の description として
書く (内部実装詳細はこの module docstring に分離)。

軽微な drift 補正は `AnnotationSchema` validation の **前** に行う:
- `tags` の文字列 (カンマ区切りまたは単一) → `list[str]`
- `captions` の文字列 (単一) → `list[str]`
- `score` の数値文字列 → `float`
- `rating` / `ratings` の文字列または dict → `RatingPrediction`
- 各要素の trim、空文字除去

補正不能 (非対応型 / list 内非文字列 / 解釈不能 score 等) は `ModelRetry`
を raise し、PydanticAI `output_retries=1` に従って LLM 再生成へ流す。
壊れた JSON 修復 / free text からの regex 復元 / provider 別 parser は実装しない。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from pydantic_ai import ModelRetry

from .types import AnnotationSchema, RatingPrediction

_WEBAPI_RATING_SOURCE_SCHEME = "prompt_defined"


def _normalize_string_items(value: Any, *, field_name: str, split_commas: bool) -> list[str]:
    """Normalize tag/caption values allowed by ADR 0023."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",") if split_commas and "," in value else [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raise ModelRetry(f"`{field_name}` must be a string or list of strings")

    normalized: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            raise ModelRetry(f"`{field_name}` must contain only strings")
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_score(value: Any) -> float | None:
    """Normalize numeric score values allowed by ADR 0023."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelRetry("`score` must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ModelRetry("`score` must be a numeric string") from exc
    raise ModelRetry("`score` must be a number or numeric string")


def _normalize_confidence(value: Any, *, field_name: str) -> float | None:
    """Normalize optional rating confidence values."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelRetry(f"`{field_name}` must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ModelRetry(f"`{field_name}` must be a numeric string") from exc
    raise ModelRetry(f"`{field_name}` must be a number or numeric string")


def _normalize_rating_item(value: Any, *, confidence: Any = None) -> RatingPrediction | None:
    """Normalize one model-native WebAPI rating prediction."""
    raw_label: Any
    raw_confidence = confidence

    if value is None:
        return None
    if isinstance(value, str):
        raw_label = value
    elif isinstance(value, dict):
        raw_label = value.get("raw_label", value.get("label", value.get("rating")))
        raw_confidence = value.get("confidence_score", value.get("confidence", raw_confidence))
    else:
        raise ModelRetry("`rating` must be a string or object")

    if not isinstance(raw_label, str):
        raise ModelRetry("`rating` label must be a string")

    label = raw_label.strip()
    if not label:
        return None

    return RatingPrediction(
        raw_label=label,
        confidence_score=_normalize_confidence(raw_confidence, field_name="rating_confidence"),
        source_scheme=_WEBAPI_RATING_SOURCE_SCHEME,
    )


def _normalize_ratings(rating: Any, ratings: Any, rating_confidence: Any) -> list[RatingPrediction]:
    """Normalize single or list-shaped rating outputs."""
    normalized: list[RatingPrediction] = []

    single = _normalize_rating_item(rating, confidence=rating_confidence)
    if single is not None:
        normalized.append(single)

    if ratings is None:
        return normalized
    if not isinstance(ratings, list):
        raise ModelRetry("`ratings` must be a list")

    for item in ratings:
        prediction = _normalize_rating_item(item)
        if prediction is not None:
            normalized.append(prediction)
    return normalized


def normalize_annotation_output(
    tags: Any = None,
    captions: Any = None,
    score: Any = None,
    rating: Any = None,
    rating_confidence: Any = None,
    ratings: Any = None,
) -> AnnotationSchema:
    """Provide image annotation results.

    Call this tool to return your image analysis as structured data.
    Each argument may be either the strict shape or a permissive variant:
    obvious string-form drift is normalized automatically before validation.

    Args:
        tags: Tags identifying image elements (characters, composition, lighting, etc).
            Preferred shape is a list of strings (e.g. ``["1girl", "blue eyes",
            "school uniform"]``). A single comma-separated string is also accepted
            and will be split on commas.
        captions: Caption strings describing the image. Preferred shape is a list
            of strings (e.g. ``["A young student sitting at a desk in a sunlit
            classroom."]``). A single string is also accepted and will be wrapped
            in a one-item list (no comma split).
        score: Quality score. Preferred shape is a number (`AnnotationSchema`
            stores it as ``float``; the typical range used by `BASE_PROMPT` is
            1.00 - 10.00). A numeric string such as ``"7.25"`` is also accepted.
        rating: Optional model-native content rating label when a rating task is
            requested. This value is kept separate from tags and score labels.
        rating_confidence: Optional confidence for `rating`. Omit it when the
            model/provider does not provide confidence.
        ratings: Optional list of rating objects. Each object may contain
            ``raw_label`` or ``label`` and optional ``confidence_score`` or
            ``confidence``.

    Returns:
        Validated `AnnotationSchema` with the normalized fields.
    """
    normalized_tags = _normalize_string_items(tags, field_name="tags", split_commas=True)
    normalized_captions = _normalize_string_items(captions, field_name="captions", split_commas=False)
    normalized_score = _normalize_score(score)
    normalized_ratings = _normalize_ratings(rating, ratings, rating_confidence)

    try:
        return AnnotationSchema(
            tags=normalized_tags,
            captions=normalized_captions,
            score=normalized_score,
            ratings=normalized_ratings,
        )
    except ValidationError as exc:
        raise ModelRetry("annotation output does not match AnnotationSchema") from exc


__all__ = ["normalize_annotation_output"]
