"""ADR 0023 Phase 1 (Issue #45) — WebAPI モデル絞り込み条件の回帰防止テスト。

`_is_litellm_model_annotation_compatible()` が `supports_response_schema` ではなく
`supports_function_calling` を主条件として動作することを保証する。

受け入れ条件:
- vision=True, function_calling=True, response_schema=False のモデルが登録対象
- vision=True, function_calling=False, response_schema=True のモデルが除外対象
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from image_annotator_lib.webapi.api_model_discovery import (
    _canonicalize_litellm_id,
    _collect_models,
    _format_litellm_metadata,
    _is_litellm_model_annotation_compatible,
    is_allowed_provider,
    is_model_deprecated,
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
        """Issue #52 / #60: bare ``claude-*`` は ``anthropic/<bare>`` に正規化される。

        Issue #60 以降は ``info['litellm_provider']`` を SSoT として参照するため、
        テストでは ``litellm_provider`` を明示して LiteLLM DB 依存を排除する。
        """
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "litellm_provider": "anthropic",
        }
        metadata = _format_litellm_metadata("claude-opus-4-6", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "anthropic/claude-opus-4-6"
        assert metadata["display_name"] == "anthropic/claude-opus-4-6"
        assert metadata["provider"] == "Anthropic"

    def test_claude_bare_with_date_suffix_canonicalized(self):
        """Issue #52 / #60: 日付サフィックス付き bare ``claude-*`` も正規化される。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "litellm_provider": "anthropic",
        }
        metadata = _format_litellm_metadata("claude-3-5-sonnet-20241022", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "anthropic/claude-3-5-sonnet-20241022"
        assert metadata["provider"] == "Anthropic"

    def test_openai_bare_canonicalized(self):
        """Issue #60: bare ``gpt-4o`` 等は ``openai/<bare>`` に正規化される。

        旧実装 (Issue #52) は ``claude-*`` のみ補完していたが、本 PR で LiteLLM
        ``litellm_provider`` field SSoT 方式に切り替え、OpenAI bare 名も対応した。
        """
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "litellm_provider": "openai",
        }
        metadata = _format_litellm_metadata("gpt-4o", info)
        assert metadata is not None
        assert metadata["model_name_short"] == "openai/gpt-4o"
        assert metadata["display_name"] == "openai/gpt-4o"
        assert metadata["provider"] == "OpenAI"

    def test_unsupported_litellm_provider_bare_returns_none(self):
        """Issue #60: ``SUPPORTED_PROVIDERS`` 外の litellm_provider (bedrock 等) は None。

        ``_BUILDER_DISPATCH`` に builder が無い provider は除外する。
        """
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "litellm_provider": "bedrock_converse",
        }
        assert _format_litellm_metadata("anthropic.claude-3-5-sonnet", info) is None
        # litellm_provider field 欠落も None
        info_no_provider = {"supports_vision": True, "supports_function_calling": True, "mode": "chat"}
        assert _format_litellm_metadata("any-bare-id", info_no_provider) is None


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

    def test_anthropic_bare_normalized_via_info(self):
        """Issue #60: bare ``claude-*`` は ``info['litellm_provider']='anthropic'`` で正規化される。

        本テストは LiteLLM upstream DB 状態に依存しないよう info 引数で明示する。
        """
        info = {"litellm_provider": "anthropic"}
        assert _canonicalize_litellm_id("claude-opus-4-6", info=info) == "anthropic/claude-opus-4-6"
        assert _canonicalize_litellm_id("claude-3-5-sonnet-20241022", info=info) == (
            "anthropic/claude-3-5-sonnet-20241022"
        )

    def test_openai_bare_normalized_via_info(self):
        """Issue #60: bare ``gpt-*`` / ``o*`` 系 OpenAI モデルは ``openai/<bare>`` に補完される。"""
        info = {"litellm_provider": "openai"}
        assert _canonicalize_litellm_id("gpt-4o", info=info) == "openai/gpt-4o"
        assert _canonicalize_litellm_id("o3", info=info) == "openai/o3"

    def test_gemini_bare_normalized_via_info(self):
        """Issue #60: bare ``gemini-*`` モデルは ``gemini/<bare>`` に補完される。

        ``_BUILDER_DISPATCH`` には ``gemini`` も ``google`` も登録済 (alias)。
        """
        info = {"litellm_provider": "gemini"}
        assert _canonicalize_litellm_id("gemini-1.5-flash", info=info) == "gemini/gemini-1.5-flash"

    def test_unsupported_litellm_provider_returns_none(self):
        """Issue #60: ``SUPPORTED_PROVIDERS`` 外の litellm_provider は None。

        ``_BUILDER_DISPATCH`` に builder が無い provider (bedrock 系等) は除外。
        """
        info_bedrock = {"litellm_provider": "bedrock_converse"}
        assert _canonicalize_litellm_id("anthropic.claude-3-5-sonnet", info=info_bedrock) is None
        info_vertex = {"litellm_provider": "vertex_ai-language-models"}
        assert _canonicalize_litellm_id("any-bare-id", info=info_vertex) is None

    def test_missing_litellm_provider_returns_none(self):
        """Issue #60: info に ``litellm_provider`` field が無い場合は None。"""
        info_empty = {"supports_vision": True}
        assert _canonicalize_litellm_id("any-bare-id", info=info_empty) is None
        # 空文字や None も None
        info_blank = {"litellm_provider": ""}
        assert _canonicalize_litellm_id("any-bare-id", info=info_blank) is None
        info_none = {"litellm_provider": None}
        assert _canonicalize_litellm_id("any-bare-id", info=info_none) is None

    def test_unknown_bare_id_returns_none_via_db(self):
        """Issue #60: LiteLLM DB に存在しない bare ID は info=None でも None。"""
        # info を渡さなければ litellm.model_cost を lookup、存在しなければ None
        assert _canonicalize_litellm_id("definitely_not_a_real_litellm_model_id_xyz") is None

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

    def test_openai_bare_allowed(self):
        """Issue #60: OpenAI bare 名 (gpt-4o 等) も SUPPORTED_PROVIDERS 経由で通過。

        旧実装 (Issue #52) では ``claude-*`` のみ補完していたため False だった。
        """
        assert is_allowed_provider("gpt-4o") is True

    def test_unsupported_provider_bare_rejected(self):
        """Issue #60: ``SUPPORTED_PROVIDERS`` 外の litellm_provider (bedrock 等) は除外。"""
        # `anthropic.claude-3-5-sonnet` は LiteLLM DB に `bedrock_converse` で存在するが
        # `_BUILDER_DISPATCH` に builder 無し → 除外
        assert is_allowed_provider("anthropic.claude-3-5-sonnet") is False
        # LiteLLM DB に存在しない bare ID も除外
        assert is_allowed_provider("invalid_id_no_slash") is False

    def test_slash_id_unchanged(self):
        """slash 入り ID の判定は Phase 1.9 と変わらず。"""
        assert is_allowed_provider("openai/gpt-4o") is True
        assert is_allowed_provider("anthropic/claude-3-5-sonnet-20241022") is True
        assert is_allowed_provider("openrouter/z-ai/glm-4.7") is True
        # SUPPORTED_PROVIDERS 外はそのまま除外 (vertex_ai は dispatch 未対応)
        assert is_allowed_provider("vertex_ai/claude-opus-4-6") is False


@pytest.mark.unit
class TestCollectModelsPassesProviderToGetModelInfo:
    """Issue #265: _collect_models() が get_model_info() に custom_llm_provider を渡すこと。

    LiteLLM の provider 推論 (get_llm_provider_logic.py:505) が失敗すると
    「Provider List: https://docs.litellm.ai/docs/providers」が print() される。
    litellm.model_cost の各 entry は既に litellm_provider を持つため、
    get_model_info() には custom_llm_provider を明示して provider 推論経路に入らない。
    """

    _COMPAT_INFO: ClassVar[dict] = {
        "litellm_provider": "openai",
        "supports_vision": True,
        "supports_function_calling": True,
        "mode": "chat",
        "max_tokens": 4096,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000015,
    }

    def test_collect_models_passes_custom_llm_provider(self):
        """_collect_models() は get_model_info() に custom_llm_provider=provider を渡す。"""
        mock_cost = {"ft:o4-mini-2025-04-16": self._COMPAT_INFO}
        mock_info = MagicMock(return_value=dict(self._COMPAT_INFO))

        with patch("image_annotator_lib.webapi.api_model_discovery.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_cost
            mock_litellm.get_model_info = mock_info
            _collect_models(require_compatible=False, exclude_deprecated=False)

        mock_info.assert_called_once_with("ft:o4-mini-2025-04-16", custom_llm_provider="openai")

    def test_collect_models_gemini_bare_passes_provider(self):
        """bare gemini-* ID も custom_llm_provider='gemini' を渡す (issue #265 再現モデル)。"""
        info = dict(self._COMPAT_INFO)
        info["litellm_provider"] = "gemini"
        mock_cost = {"gemini-2.0-flash-exp-image-generation": info}
        mock_info = MagicMock(return_value=info)

        with patch("image_annotator_lib.webapi.api_model_discovery.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_cost
            mock_litellm.get_model_info = mock_info
            _collect_models(require_compatible=False, exclude_deprecated=False)

        mock_info.assert_called_once_with(
            "gemini-2.0-flash-exp-image-generation", custom_llm_provider="gemini"
        )


@pytest.mark.unit
class TestIsModelDeprecatedPassesProviderToGetModelInfo:
    """Issue #265: is_model_deprecated() が get_model_info() に custom_llm_provider を渡すこと。"""

    def test_is_model_deprecated_passes_custom_llm_provider(self):
        """is_model_deprecated() は get_model_info() に custom_llm_provider=provider を渡す。"""
        info = {
            "litellm_provider": "openai",
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "deprecation_date": None,
        }
        mock_info = MagicMock(return_value=info)

        with patch("image_annotator_lib.webapi.api_model_discovery.litellm") as mock_litellm:
            mock_litellm.model_cost = {"ft:o4-mini-2025-04-16": info}
            mock_litellm.get_model_info = mock_info
            is_model_deprecated("ft:o4-mini-2025-04-16")

        mock_info.assert_called_once_with("ft:o4-mini-2025-04-16", custom_llm_provider="openai")
