"""ADR 0023 Phase 1.5 (Issue #42): refusal 検出ロジックの単体テスト。

`_classify_refusal()` が PydanticAI / provider SDK の例外から
SafetyRefusalError / ContentPolicyRefusalError を正しく分類することを確認する。
"""

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
