"""PydanticAI `AnnotationSchema` → `AnnotationResult` 変換 (ADR 0023 Phase 1)。

ADR 0023 `### Schema → Result 変換` を実装する責務集約モジュール。
`ProviderManager` は inference 実行のみを担当し、結果変換ロジックは持たない。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

from typing import Any

from .types import AnnotationResult, AnnotationSchema


def _normalize_string_list(value: Any) -> list[str]:
    """ADR 0023 `Output normalization` の軽微正規化を適用する。

    - `value` が文字列ならカンマ分割または single item list 化
    - 各要素の前後空白を除去
    - 空文字を除外

    壊れた JSON の手修復や regex ベースの強引な復元は行わない。
    """
    if value is None:
        return []
    if isinstance(value, str):
        # 文字列ならカンマ分割を試みる。カンマがなければ single item list。
        items = value.split(",") if "," in value else [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_score(value: Any) -> float | None:
    """`score` フィールドを float に正規化する。失敗時は None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_annotation_result(
    schema_output: AnnotationSchema | None,
    phash: str,
    error: str | None = None,
) -> AnnotationResult:
    """PydanticAI が返した `AnnotationSchema` を `AnnotationResult` に変換する。

    Args:
        schema_output: PydanticAI `AgentRunResult.output` (`AnnotationSchema` 検証済)。
            error 経路では None を渡してよい。
        phash: 対象画像の pHash。
        error: 推論エラー時のメッセージ。None なら成功扱い。

    Returns:
        `AnnotationResult` (TypedDict)。error 経路でも `tags=[]` を含めて返す。
    """
    if error is not None or schema_output is None:
        return AnnotationResult(
            phash=phash,
            tags=[],
            formatted_output=None,
            error=error,
        )

    raw_tags = getattr(schema_output, "tags", None)
    raw_captions = getattr(schema_output, "captions", None)
    raw_score = getattr(schema_output, "score", None)

    normalized_tags = _normalize_string_list(raw_tags)
    normalized_captions = _normalize_string_list(raw_captions)
    normalized_score = _normalize_score(raw_score)

    formatted_output: dict[str, Any] = {
        "tags": normalized_tags,
        "captions": normalized_captions,
        "score": normalized_score,
    }

    return AnnotationResult(
        phash=phash,
        tags=normalized_tags,
        formatted_output=formatted_output,
        error=None,
    )


__all__ = ["to_annotation_result"]
