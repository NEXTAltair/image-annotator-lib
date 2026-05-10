"""ADR 0023 Phase 1 (Issue #45) — WebAPI モデル絞り込み条件の回帰防止テスト。

`_is_litellm_model_annotation_compatible()` が `supports_response_schema` ではなく
`supports_function_calling` を主条件として動作することを保証する。

受け入れ条件:
- vision=True, function_calling=True, response_schema=False のモデルが登録対象
- vision=True, function_calling=False, response_schema=True のモデルが除外対象
"""

from __future__ import annotations

import pytest

from image_annotator_lib.core.api_model_discovery import (
    _canonicalize_litellm_id,
    _format_litellm_metadata,
    _is_litellm_model_annotation_compatible,
    is_allowed_provider,
)


@pytest.mark.unit
class TestIsLitellmModelAnnotationCompatible:
    """`_is_litellm_model_annotation_compatible()` の判定ロジック検証。"""

    def test_vision_and_function_calling_true_is_compatible(self):
        """vision + function_calling 両方 True なら compatible (response_schema 値は不問)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_response_schema": False,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is True

    def test_function_calling_false_is_excluded(self):
        """vision=True でも function_calling=False なら除外 (response_schema=True でも)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": False,
            "supports_response_schema": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_function_calling_missing_is_excluded(self):
        """function_calling キー欠落でも除外 (`is True` 比較なので False/None/欠落いずれも弾く)。"""
        info = {
            "supports_vision": True,
            "supports_response_schema": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_vision_false_is_excluded(self):
        """vision=False なら function_calling=True でも除外。"""
        info = {
            "supports_vision": False,
            "supports_function_calling": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_responses_mode_is_compatible(self):
        """mode=responses も chat と同様に compatible (Phase 1 仕様)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "responses",
        }
        assert _is_litellm_model_annotation_compatible(info) is True

    def test_completion_mode_is_excluded(self):
        """mode=completion 等は除外 (Phase 1 は chat / responses のみ)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "completion",
        }
        assert _is_litellm_model_annotation_compatible(info) is False


@pytest.mark.unit
class TestFormatLitellmMetadata:
    """`_format_litellm_metadata()` が ADR 0023 Phase 1 の metadata 形式を返すこと。"""

    def test_response_schema_key_is_absent(self):
        """Issue #45: `supports_response_schema` キーは metadata から削除されている。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_response_schema": True,
            "mode": "chat",
        }
        metadata = _format_litellm_metadata("openai/gpt-4o", info)
        assert metadata is not None
        assert "supports_response_schema" not in metadata

    def test_function_calling_key_is_present(self):
        """`supports_function_calling` キーは metadata に保持される。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
        }
        metadata = _format_litellm_metadata("openai/gpt-4o", info)
        assert metadata is not None
        assert metadata["supports_function_calling"] is True

    def test_invalid_model_id_returns_none(self):
        """`provider/model` 形式でない ID は None を返す。"""
        assert _format_litellm_metadata("invalid_id_no_slash", {}) is None

    def test_openrouter_nested_keeps_full_id(self):
        """Issue #51: openrouter/<inner>/<model> は model_name_short が完全 ID と一致する。

        旧実装は `split("/", 1)[1]` で `openrouter/` prefix を剥がし `z-ai/glm-4.7` を
        返していたため、推論時に `_BUILDER_DISPATCH` 未知 prefix で UnknownProviderError が
        発生していた。修正後は LiteLLM オリジナルキーをそのまま保持する。
        """
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        metadata = _format_litellm_metadata("openrouter/z-ai/glm-4.7", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "openrouter/z-ai/glm-4.7"
        assert metadata["display_name"] == "openrouter/z-ai/glm-4.7"
        assert metadata["provider"] == "Openrouter"

    def test_openrouter_openai_keeps_full_id(self):
        """Issue #51: openrouter/openai/<model> も同様に prefix を保持する。"""
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        metadata = _format_litellm_metadata("openrouter/openai/gpt-4.1", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "openrouter/openai/gpt-4.1"
        assert metadata["display_name"] == "openrouter/openai/gpt-4.1"

    def test_direct_provider_id_uses_full_id(self):
        """Issue #51: openai/<model> 直接形式も model_name_short = 完全 ID。"""
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        metadata = _format_litellm_metadata("openai/gpt-4o", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "openai/gpt-4o"
        assert metadata["display_name"] == "openai/gpt-4o"
        assert metadata["provider"] == "OpenAI"

    def test_claude_bare_canonicalized(self):
        """Issue #52: bare `claude-*` は `anthropic/<bare>` に正規化される。"""
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        metadata = _format_litellm_metadata("claude-opus-4-6", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "anthropic/claude-opus-4-6"
        assert metadata["display_name"] == "anthropic/claude-opus-4-6"
        assert metadata["provider"] == "Anthropic"

    def test_claude_bare_with_date_suffix_canonicalized(self):
        """Issue #52: 日付サフィックス付きの bare `claude-*` も同様に正規化される。"""
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        metadata = _format_litellm_metadata("claude-3-5-sonnet-20241022", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "anthropic/claude-3-5-sonnet-20241022"
        assert metadata["provider"] == "Anthropic"

    def test_non_claude_bare_returns_none(self):
        """Issue #52: claude- 以外の bare 名 (gpt-4o 等) は除外維持。"""
        info = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        assert _format_litellm_metadata("gpt-4o", info) is None
        # Bedrock 形式 `anthropic.claude-*` も `claude-` で始まらないので除外
        assert _format_litellm_metadata("anthropic.claude-3-5-sonnet", info) is None


@pytest.mark.unit
class TestCanonicalizeLitellmId:
    """`_canonicalize_litellm_id()` の境界条件 (Issue #52 / ADR 0023 Phase 1.10)。"""

    def test_slash_id_passthrough(self):
        """slash 入り ID はそのまま返される。"""
        assert _canonicalize_litellm_id("openai/gpt-4o") == "openai/gpt-4o"
        assert _canonicalize_litellm_id("anthropic/claude-3-5-sonnet-20241022") == (
            "anthropic/claude-3-5-sonnet-20241022"
        )
        assert _canonicalize_litellm_id("openrouter/z-ai/glm-4.7") == "openrouter/z-ai/glm-4.7"

    def test_claude_bare_normalized(self):
        """bare `claude-*` は `anthropic/<bare>` に補完される。"""
        assert _canonicalize_litellm_id("claude-opus-4-6") == "anthropic/claude-opus-4-6"
        assert _canonicalize_litellm_id("claude-3-5-sonnet-20241022") == (
            "anthropic/claude-3-5-sonnet-20241022"
        )
        assert _canonicalize_litellm_id("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"

    def test_non_claude_bare_returns_none(self):
        """`claude-` 以外の bare 名は対応外として None。"""
        assert _canonicalize_litellm_id("gpt-4o") is None
        # Bedrock 形式: ドット区切りで `claude-` で始まらない
        assert _canonicalize_litellm_id("anthropic.claude-3-5-sonnet") is None
        assert _canonicalize_litellm_id("invalid_id_no_slash") is None

    def test_empty_string_returns_none(self):
        """空文字も None。"""
        assert _canonicalize_litellm_id("") is None


@pytest.mark.unit
class TestIsAllowedProviderBareName:
    """`is_allowed_provider()` の Phase 1.10 拡張 (Issue #52)。"""

    def test_claude_bare_allowed(self):
        """bare `claude-*` は anthropic 経由で SUPPORTED_PROVIDERS 通過。"""
        assert is_allowed_provider("claude-opus-4-6") is True
        assert is_allowed_provider("claude-haiku-4-5") is True

    def test_non_claude_bare_rejected(self):
        """claude- 以外の bare 名は除外維持。"""
        assert is_allowed_provider("gpt-4o") is False
        # Bedrock 形式の bare 名 (claude- で始まらない) は除外
        assert is_allowed_provider("anthropic.claude-3-5-sonnet") is False

    def test_slash_id_unchanged(self):
        """slash 入り ID の判定は Phase 1.9 と変わらず。"""
        assert is_allowed_provider("openai/gpt-4o") is True
        assert is_allowed_provider("anthropic/claude-3-5-sonnet-20241022") is True
        assert is_allowed_provider("openrouter/z-ai/glm-4.7") is True
        # SUPPORTED_PROVIDERS 外はそのまま除外 (vertex_ai は dispatch 未対応)
        assert is_allowed_provider("vertex_ai/claude-opus-4-6") is False
