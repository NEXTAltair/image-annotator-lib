"""ADR 0023 Phase 1.5 (Issue #42): refusal 検出ロジックの単体テスト。

`_classify_refusal()` が PydanticAI / provider SDK の例外から
SafetyRefusalError / ContentPolicyRefusalError を正しく分類することを確認する。
"""

from typing import Any

import pytest

from image_annotator_lib.core.provider_manager import _classify_refusal
from image_annotator_lib.exceptions.errors import (
    ContentPolicyRefusalError,
    SafetyRefusalError,
)


@pytest.mark.unit
def test_classify_refusal_openai_content_filter_message():
    """OpenAI の content_filter finish_reason → ContentPolicyRefusalError"""
    exc = Exception("finish_reason=content_filter, content blocked by safety system")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash1",
    )
    assert isinstance(classified, ContentPolicyRefusalError)
    assert classified.litellm_model_id == "openai/gpt-4o"
    assert classified.image_phash == "phash1"
    assert "content_filter" in classified.provider_refusal_reason


@pytest.mark.unit
def test_classify_refusal_openai_content_filter_type_name():
    """exception type 名に "ContentFilter" 含む → ContentPolicyRefusalError"""

    class FakeContentFilterError(Exception):
        pass

    exc = FakeContentFilterError("blocked")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash2",
    )
    assert isinstance(classified, ContentPolicyRefusalError)


@pytest.mark.unit
def test_classify_refusal_anthropic_stop_reason_refusal():
    """Anthropic の stop_reason=refusal → SafetyRefusalError"""
    exc = Exception("stop_reason=refusal: image content violates usage policy")
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash3",
    )
    assert isinstance(classified, SafetyRefusalError)
    assert classified.litellm_model_id == "anthropic/claude-3-5-sonnet-20241022"
    assert classified.image_phash == "phash3"


@pytest.mark.unit
def test_classify_refusal_anthropic_refusal_type_name():
    """exception type 名に "Refusal" 含む → SafetyRefusalError"""

    class APIRefusalError(Exception):
        pass

    exc = APIRefusalError("model declined to process request")
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash4",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_google_safety_finish_reason():
    """Google Gemini の finishReason=SAFETY → SafetyRefusalError"""
    exc = Exception("Response blocked. finishReason: SAFETY, blockedReason=harm_category_dangerous_content")
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash5",
    )
    assert isinstance(classified, SafetyRefusalError)
    assert classified.litellm_model_id == "google/gemini-2.0-flash-lite"


@pytest.mark.unit
def test_classify_refusal_google_blocked_due_to_safety():
    """Google Gemini の "blocked due to safety" メッセージ → SafetyRefusalError"""
    exc = Exception("Generation blocked due to safety filters")
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.5-pro",
        image_phash="phash6",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "Response blocked. finishReason=SAFETY",  # camelCase + `=` (Codex P1: PR #44)
        "Response blocked. finishreason=safety",  # camelCase lowercased + `=`
        "finishReason: SAFETY",  # camelCase + `:`
        "finish_reason: SAFETY",  # snake_case + `:`
        "finish_reason=safety",  # snake_case + `=`
        "Generation halted due to finishReason=SAFETY in candidate",  # in-context
    ],
)
def test_classify_refusal_finish_reason_safety_variants(message: str) -> None:
    """`finishReason=SAFETY` の全 variant (snake_case/camelCase x `:`/`=`) をカバーする。

    Codex P1 review (PR #44): camelCase + `=` の signature が漏れていた回帰防止。
    """
    classified = _classify_refusal(
        Exception(message),
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_finish_reason",
    )
    assert isinstance(classified, SafetyRefusalError), f"variant 取りこぼし: {message!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "stop_reason=refusal",  # snake_case + `=`
        "stop_reason: refusal",  # snake_case + `:`
        "stopReason=refusal",  # camelCase + `=`
        "stopReason: refusal",  # camelCase + `:`
    ],
)
def test_classify_refusal_stop_reason_refusal_variants(message: str) -> None:
    """`stop_reason=refusal` の全 variant (snake_case/camelCase x `:`/`=`) をカバー。"""
    classified = _classify_refusal(
        Exception(message),
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash_stop_reason",
    )
    assert isinstance(classified, SafetyRefusalError), f"variant 取りこぼし: {message!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "finish_reason=content_filter",  # snake_case + `=`
        "finishReason=content_filter",  # camelCase + `=`
        "finishReason: content_filter",  # camelCase + `:`
    ],
)
def test_classify_refusal_finish_reason_content_filter_variants(message: str) -> None:
    """`finish_reason=content_filter` の variant も ContentPolicyRefusalError に分類。"""
    classified = _classify_refusal(
        Exception(message),
        litellm_model_id="openai/gpt-4o",
        image_phash="phash_content_filter",
    )
    assert isinstance(classified, ContentPolicyRefusalError), f"variant 取りこぼし: {message!r}"


@pytest.mark.unit
def test_classify_refusal_non_refusal_returns_none():
    """非 refusal 例外 (TimeoutError 等) → None"""
    exc = TimeoutError("connection timeout after 30s")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash7",
    )
    assert classified is None


@pytest.mark.unit
def test_classify_refusal_generic_runtime_error_returns_none():
    """汎用 RuntimeError は refusal ではない → None"""
    exc = RuntimeError("unexpected JSON parse failure")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash8",
    )
    assert classified is None


@pytest.mark.unit
def test_classify_refusal_content_filter_takes_precedence():
    """exception message に content_filter と refusal 両方含む → ContentPolicyRefusalError 優先"""
    exc = Exception("finish_reason=content_filter and refusal flag set")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash9",
    )
    assert isinstance(classified, ContentPolicyRefusalError)


@pytest.mark.unit
def test_classify_refusal_attributes_in_details():
    """details dict に attribute が反映される (既存 sub-exception パターン踏襲)"""
    exc = Exception("finish_reason=content_filter")
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash10",
    )
    assert classified is not None
    assert classified.details["litellm_model_id"] == "openai/gpt-4o"
    assert classified.details["image_phash"] == "phash10"
    assert classified.details["provider_refusal_reason"] != ""


@pytest.mark.unit
def test_classify_refusal_long_reason_truncated():
    """provider_refusal_reason は 200 文字で truncate される"""
    long_msg = "finish_reason=content_filter " + ("x" * 500)
    exc = Exception(long_msg)
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash11",
    )
    assert classified is not None
    assert len(classified.provider_refusal_reason) == 200


@pytest.mark.unit
def test_classify_refusal_empty_image_phash_allowed():
    """image_phash が空文字でも例外を返せる (details には含まれない)"""
    exc = Exception("stop_reason=refusal")
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="",
    )
    assert isinstance(classified, SafetyRefusalError)
    assert classified.image_phash == ""
    assert "image_phash" not in classified.details


# ========================================================================
# PydanticAI exc.body 構造化ペイロード経路 (Codex P1 r3212094633)
#
# UnexpectedModelBehavior(message, body) / ModelHTTPError(..., body) の body 属性に
# provider response (JSON 文字列 / dict) を保持するパターンを検証。引用符・区切り
# の差異に影響されず、構造化探索で reason を取得することを確認する。
# ========================================================================


class _FakePydanticAIException(Exception):
    """PydanticAI の UnexpectedModelBehavior / ModelHTTPError を模した最小 stub。

    実物の `body` attribute ファースト判定動作を確認するための test 用。
    """

    def __init__(self, message: str, body: Any = None):
        super().__init__(message)
        self.body = body


@pytest.mark.unit
def test_classify_refusal_body_json_finish_reason_safety():
    """Codex P1 (PR #44 r3212094633): exc.body の JSON 文字列で finishReason: SAFETY を検出。

    Google Gemini の典型的な refusal response。引用符付きの quoted JSON を
    body 経由で構造化探索することで、regex の引用符問題を構造的に回避。
    """
    body = '{"candidates":[{"finishReason":"SAFETY","safetyRatings":[]}]}'
    exc = _FakePydanticAIException("model returned no content", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_body_safety",
    )
    assert isinstance(classified, SafetyRefusalError)
    assert classified.litellm_model_id == "google/gemini-2.0-flash-lite"
    assert "safety" in classified.provider_refusal_reason


@pytest.mark.unit
def test_classify_refusal_body_json_stop_reason_refusal():
    """exc.body の JSON で stop_reason: refusal を検出 (Anthropic)。"""
    body = '{"id":"msg_xyz","stop_reason":"refusal","content":[]}'
    exc = _FakePydanticAIException("anthropic refusal", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash_body_anthropic",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_json_finish_reason_content_filter():
    """exc.body の JSON で finish_reason: content_filter を検出 (OpenAI)。"""
    body = '{"choices":[{"finish_reason":"content_filter","message":{}}]}'
    exc = _FakePydanticAIException("openai content filter", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash_body_openai",
    )
    assert isinstance(classified, ContentPolicyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_dict_native():
    """exc.body が既に dict object の場合も正しく扱える (json.loads スキップ)。"""
    body = {"candidates": [{"finishReason": "SAFETY"}]}
    exc = _FakePydanticAIException("gemini dict body", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.5-pro",
        image_phash="phash_body_dict",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_nested_dict():
    """ネストされた dict 構造でも再帰的に reason を見つける。"""
    body = {
        "outer": {"middle": {"candidates": [{"finishReason": "SAFETY"}]}},
    }
    exc = _FakePydanticAIException("nested", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.5-pro",
        image_phash="phash_nested",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_priority_over_string_match():
    """body 構造化判定 (priority 1) が string match (priority 3) より優先される。

    body が SAFETY を示し、message は無関係 noise でも refusal 判定される。
    """
    body = '{"candidates":[{"finishReason":"SAFETY"}]}'
    exc = _FakePydanticAIException(
        "Generic error message without any refusal keywords",
        body=body,
    )
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_priority",
    )
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_invalid_json_falls_back_to_string():
    """body が JSON parse できない文字列の場合、string match fallback (priority 3) で判定。"""
    body = "not a JSON, but contains stop_reason=refusal somewhere"
    exc = _FakePydanticAIException("anthropic non-json", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash_invalid_json",
    )
    # body は parse 失敗で None 返却 → priority 3 (string regex) で match
    assert isinstance(classified, SafetyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_unrelated_finish_reason_returns_none():
    """body の finish_reason が refusal 系以外なら None (e.g. "stop", "length")。"""
    body = '{"candidates":[{"finishReason":"STOP"}]}'
    exc = _FakePydanticAIException("normal completion", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_normal",
    )
    assert classified is None


@pytest.mark.unit
def test_classify_refusal_body_none_falls_back_to_other_priorities():
    """body 属性が None なら priority 2 (type 名) → priority 3 (string) へ fallback。"""

    class FakeContentFilterError(Exception):
        pass

    exc = FakeContentFilterError("blocked")  # body 属性なし
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash_no_body",
    )
    # body 無し → priority 2 で type 名 match
    assert isinstance(classified, ContentPolicyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_quoted_signature_codex_regression():
    """Codex P1 (PR #44 r3212094633) 回帰防止: 引用符付き JSON も struct-aware で取れる。

    旧実装では `"finishReason":"SAFETY"` (key/value 共に引用符付き) が regex に
    マッチせず取りこぼしていた。body 経路では引用符問わず構造化探索で取れる。
    """
    quoted_payload = '{"finishReason":"SAFETY"}'  # 完全 quoted
    exc = _FakePydanticAIException(
        # message には refusal 手がかり無し (旧 string 実装は素抜けする)
        "model returned no content",
        body=quoted_payload,
    )
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_quoted",
    )
    assert isinstance(classified, SafetyRefusalError)


# ========================================================================
# Legacy fallback 経路: regex 拡張 (引用符許容) 単体動作確認
# ========================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        # 引用符付き signature を string match fallback でも吸収できる
        '"finishReason":"SAFETY"',
        '"finishreason": "safety"',
        "finishReason='SAFETY'",
    ],
)
def test_classify_refusal_quoted_signature_in_string_fallback(message: str) -> None:
    """body 属性が無い (旧 SDK 直叩き等) 経路でも、string regex fallback が引用符
    付き signature を拾えること (priority 3 の robustness 確認)。"""
    exc = Exception(message)  # body 属性なし、純粋 string 経路
    classified = _classify_refusal(
        exc,
        litellm_model_id="google/gemini-2.0-flash-lite",
        image_phash="phash_legacy_quoted",
    )
    assert isinstance(classified, SafetyRefusalError), f"variant 取りこぼし: {message!r}"


# ========================================================================
# Codex P2 (r3212139750): body precedence 回帰防止
#
# `_walk_for_reasons` は全 reason を返し、`_classify_reasons` が
# ContentPolicy > Safety の precedence で判定する。両者共存ペイロードでも
# 結果が dict 走査順に依存しないことを確認する。
# ========================================================================


@pytest.mark.unit
def test_classify_refusal_body_content_filter_precedence_when_refusal_first():
    """body に refusal が先、content_filter が後 → ContentPolicyRefusalError を返す。

    Codex P2 r3212139750: 単一 reason 返却の旧設計だと dict 走査順依存で
    SafetyRefusalError を返してしまっていた。`_classify_reasons` の precedence
    で常に ContentPolicy が優先されるようにする。
    """
    body = {
        "stop_reason": "refusal",  # 先に検出される
        "choices": [{"finish_reason": "content_filter"}],  # 後で検出される
    }
    exc = _FakePydanticAIException("mixed-refusal-first", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash_mixed_refusal_first",
    )
    assert isinstance(classified, ContentPolicyRefusalError), (
        "ContentPolicy が precedence で勝つはず (旧実装は dict 順依存で誤判定していた)"
    )


@pytest.mark.unit
def test_classify_refusal_body_content_filter_precedence_when_filter_first():
    """body に content_filter が先、refusal が後 → ContentPolicyRefusalError を返す。

    順序が逆の場合も同じ結果 (順序非依存性の確認)。
    """
    body = {
        "choices": [{"finish_reason": "content_filter"}],  # 先に検出される
        "stop_reason": "refusal",  # 後で検出される
    }
    exc = _FakePydanticAIException("mixed-filter-first", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="openai/gpt-4o",
        image_phash="phash_mixed_filter_first",
    )
    assert isinstance(classified, ContentPolicyRefusalError)


@pytest.mark.unit
def test_classify_refusal_body_walker_collects_all_reasons():
    """`_walk_for_reasons` が複数 reason を全て収集することの直接確認。"""
    from image_annotator_lib.core.provider_manager import _walk_for_reasons

    body = {
        "stop_reason": "refusal",
        "candidates": [{"finishReason": "SAFETY"}],
        "choices": [{"finish_reason": "content_filter"}],
    }
    reasons = _walk_for_reasons(body)
    assert set(reasons) == {"refusal", "safety", "content_filter"}, (
        f"全 reason を収集すべき: 実際 {reasons}"
    )


@pytest.mark.unit
def test_classify_refusal_body_only_safety_reasons():
    """ContentPolicy が body に無く Safety 系のみなら SafetyRefusalError を返す。"""
    body = {
        "stop_reason": "refusal",
        "candidates": [{"finishReason": "SAFETY"}],
    }
    exc = _FakePydanticAIException("only-safety", body=body)
    classified = _classify_refusal(
        exc,
        litellm_model_id="anthropic/claude-3-5-sonnet-20241022",
        image_phash="phash_only_safety",
    )
    assert isinstance(classified, SafetyRefusalError)
