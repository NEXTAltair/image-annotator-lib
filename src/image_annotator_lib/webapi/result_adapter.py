"""PydanticAI `AnnotationSchema` → `AnnotationResult` conversion.

ADR 0023 `### Schema → Result 変換` を実装する責務集約モジュール。
`ProviderManager` は inference 実行のみを担当し、結果変換ロジックは持たない。
Validation 前の軽微な schema 揺れは `webapi.output_normalization` が PydanticAI
output 処理内で補正する。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from ..core.types import AnnotationResult, AnnotationSchema


def _clean_string_list(value: list[str]) -> list[str]:
    """Final-defense cleanup for already validated schema lists."""
    return [item.strip() for item in value if item.strip()]


def to_annotation_result(
    schema_output: AnnotationSchema | None,
    phash: str,
    error: str | None = None,
    error_code: str | None = None,
    retryable: bool = False,
) -> AnnotationResult:
    """PydanticAI が返した `AnnotationSchema` を `AnnotationResult` に変換する。

    Args:
        schema_output: PydanticAI `AgentRunResult.output` (`AnnotationSchema` 検証済)。
            error 経路では None を渡してよい。
        phash: 対象画像の pHash。
        error: 推論エラー / outcome 時の message。None なら成功扱い。
        error_code: annotation outcome 分類コード (ADR 0006 amendment: refusal / 空 等)。
            transport の generic error では None のまま (message のみ伝搬)。
        retryable: outcome が caller 再実行で解消し得るか。refusal / 空は False。

    Returns:
        `AnnotationResult` (TypedDict)。error 経路でも `tags=[]` を含めて返す。
    """
    if error is not None or error_code is not None or schema_output is None:
        return AnnotationResult(
            phash=phash,
            tags=[],
            formatted_output=None,
            error=error,
            error_code=error_code,
            retryable=retryable,
        )

    normalized_tags = _clean_string_list(schema_output.tags)
    normalized_captions = _clean_string_list(schema_output.captions)

    formatted_output: dict[str, object] = {
        "tags": normalized_tags,
        "captions": normalized_captions,
        "score": schema_output.score,
        "ratings": [rating.model_dump() for rating in schema_output.ratings],
    }

    return AnnotationResult(
        phash=phash,
        tags=normalized_tags,
        formatted_output=formatted_output,
        error=None,
    )


__all__ = ["to_annotation_result"]
