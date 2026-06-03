"""Issue #134: silent な空アノテーションを EmptyAnnotationError として error 化する。

ADR 0023 Phase 1.5 (Issue #42) の refusal contract の拡張。モデルが output tool を
**空引数で呼んだ** ことで schema-valid な空成功 (`error=None` + 全 capability 空) に
潰れるケースを、`error` 文字列付きの degenerate outcome に変換することを検証する。

#42 の refusal とは粒度を分ける: refusal は「なぜ拒否したか」のシグナルを保持するが、
空成功は理由不明なため別例外 `EmptyAnnotationError` で表現する。
"""

from __future__ import annotations

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.types import AnnotationSchema, TaskCapability
from image_annotator_lib.exceptions.errors import EmptyAnnotationError
from image_annotator_lib.webapi.output_normalization import build_annotation_output_normalizer
from image_annotator_lib.webapi.provider_manager import ProviderManager, _classify_empty_output

_TAGS_CAPTIONS = frozenset({TaskCapability.TAGS, TaskCapability.CAPTIONS})


def _run(custom_output_args: dict[str, object], capabilities: frozenset[TaskCapability]):
    agent = Agent(
        model=TestModel(custom_output_args=custom_output_args),
        output_type=build_annotation_output_normalizer(capabilities),
        retries={"output": 1},
    )
    image = Image.new("RGB", (8, 8), color="white")
    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=[image],
        litellm_model_id="openai/o1",
        capabilities=capabilities,
        _test_agent=agent,
    )
    return next(iter(results.values()))


def test_all_requested_capabilities_empty_becomes_error() -> None:
    """全要求 capability が空の成功 → error_code=EMPTY_ANNOTATION の outcome 化 (本件)。

    S1 (#599 契約): library は refusal/空を error 文字列ではなく structured outcome
    (`error_code` + `retryable`) で返す。`error` は message 専用。
    """
    result = _run({"tags": "", "captions": ""}, _TAGS_CAPTIONS)

    assert result.get("error_code") == "EMPTY_ANNOTATION", "空成功は EMPTY_ANNOTATION に分類されるべき"
    assert result.get("retryable") is False
    assert result["error"], "message は保持される"
    assert result["formatted_output"] is None
    assert result["tags"] == []


def test_non_empty_output_is_not_flagged() -> None:
    """non-empty 出力は誤検出しない (error/error_code 共に None のまま正常)。"""
    result = _run({"tags": "cat, dog", "captions": "a cat outside"}, _TAGS_CAPTIONS)

    assert result["error"] is None
    assert result.get("error_code") is None
    assert result["tags"] == ["cat", "dog"]


def test_partial_empty_is_not_flagged() -> None:
    """一部 capability のみ空 (tags あり / captions 空) は degenerate ではない → 正常。"""
    result = _run({"tags": "cat", "captions": ""}, _TAGS_CAPTIONS)

    assert result["error"] is None
    assert result.get("error_code") is None
    assert result["tags"] == ["cat"]


# ========================================================================
# _classify_empty_output 直接ユニットテスト (edge cases)
# ========================================================================


def _empty_schema() -> AnnotationSchema:
    return AnnotationSchema(tags=[], captions=[], score=None, ratings=[])


def test_classify_empty_output_reports_requested_capabilities() -> None:
    """空成功時、要求 core capability 名 (sorted) を保持する。"""
    exc = _classify_empty_output(_empty_schema(), _TAGS_CAPTIONS, "openai/o1", "phash_x")
    assert isinstance(exc, EmptyAnnotationError)
    assert exc.requested_capabilities == ["captions", "tags"]
    assert exc.litellm_model_id == "openai/o1"
    assert exc.image_phash == "phash_x"


def test_classify_empty_output_none_capabilities_uses_core_default() -> None:
    """capabilities=None は既定 core (tags/captions/scores) 全空として degenerate 扱い。"""
    exc = _classify_empty_output(_empty_schema(), None, "openai/o1", "phash_y")
    assert isinstance(exc, EmptyAnnotationError)
    assert exc.requested_capabilities == ["captions", "scores", "tags"]


def test_classify_empty_output_ratings_only_is_skipped() -> None:
    """RATINGS のみ要求 (core capability 非要求) は空成功判定の対象外 → None。"""
    exc = _classify_empty_output(
        _empty_schema(), frozenset({TaskCapability.RATINGS}), "openai/o1", "phash_z"
    )
    assert exc is None


def test_classify_empty_output_score_present_is_not_empty() -> None:
    """SCORES 要求で score が存在すれば degenerate ではない → None。"""
    schema = AnnotationSchema(tags=[], captions=[], score=1.0, ratings=[])
    exc = _classify_empty_output(
        schema, frozenset({TaskCapability.SCORES}), "openai/o1", "phash_s"
    )
    assert exc is None
