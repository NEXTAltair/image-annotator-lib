"""_build_system_prompt() の単体テスト。"""

import pytest

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT
from image_annotator_lib.webapi.provider_manager import _build_system_prompt


@pytest.mark.unit
def test_build_system_prompt_no_additional_returns_base():
    result = _build_system_prompt(None, None)
    assert result == BASE_PROMPT


@pytest.mark.unit
def test_build_system_prompt_empty_string_returns_base():
    """空文字列は追記しない。"""
    result = _build_system_prompt(None, "")
    assert result == BASE_PROMPT


@pytest.mark.unit
def test_build_system_prompt_whitespace_only_returns_base():
    """空白のみは追記しない。"""
    result = _build_system_prompt(None, "   ")
    assert result == BASE_PROMPT


@pytest.mark.unit
def test_build_system_prompt_with_additional_appends():
    """additional_prompt は BASE_PROMPT の末尾に二重改行で追記される。"""
    result = _build_system_prompt(None, "Focus on red objects.")
    assert result == BASE_PROMPT + "\n\nFocus on red objects."


@pytest.mark.unit
def test_build_system_prompt_strips_additional():
    """additional_prompt の前後の空白は除去される。"""
    result = _build_system_prompt(None, "  extra instruction  ")
    assert result == BASE_PROMPT + "\n\nextra instruction"


@pytest.mark.unit
def test_build_system_prompt_ratings_capability_no_additional():
    """RATINGS capability があれば rating 指示が追記される。"""
    result = _build_system_prompt({TaskCapability.RATINGS}, None)
    assert "Rating task:" in result
    assert result.startswith(BASE_PROMPT)


@pytest.mark.unit
def test_build_system_prompt_ratings_and_additional():
    """RATINGS + additional_prompt の両方が追記される。"""
    result = _build_system_prompt({TaskCapability.RATINGS}, "extra note")
    assert "Rating task:" in result
    assert result.endswith("\n\nextra note")
