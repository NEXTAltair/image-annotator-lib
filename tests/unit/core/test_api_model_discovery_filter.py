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
    _is_annotation_suitable,
    _is_litellm_model_annotation_compatible,
    discover_available_vision_models,
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

    def test_openai_moderation_model_is_compatible_without_function_calling(self):
        """OpenAI Moderations は PydanticAI tool calling を使わない専用経路として登録対象。"""
        info = {
            "supports_vision": False,
            "supports_function_calling": False,
            "mode": "moderation",
        }
        assert (
            _is_litellm_model_annotation_compatible(info, model_id="openai/omni-moderation-latest") is True
        )

    def test_unrelated_non_tool_model_stays_excluded(self):
        """非 tool モデルの互換条件は OpenAI Moderations 以外へ広げない。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": False,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info, model_id="openai/gpt-4o") is False

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

    def test_responses_mode_is_excluded(self):
        """Issue #130: mode=responses は除外。

        現 runtime (`build_pydantic_model`) は OpenAI を常に OpenAIChatModel
        (`v1/chat/completions`) で構築するため responses 専用モデルは実行不能。
        Responses runtime 対応は #131 で gate 反転予定。
        """
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "responses",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_completion_mode_is_excluded(self):
        """mode=completion 等は除外 (Issue #130 以降 chat のみ)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "completion",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_chat_with_tools_in_supported_openai_params_is_compatible(self):
        """Issue #130: supported_openai_params に tools があれば compatible。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "supported_openai_params": ["max_tokens", "tools", "tool_choice", "response_format"],
        }
        assert _is_litellm_model_annotation_compatible(info) is True

    def test_chat_without_tools_in_supported_openai_params_is_excluded(self):
        """Issue #130: params が populate されているのに tools が無いモデルは除外。

        gpt-5-search-api 族は litellm が supports_function_calling=True と報告するが
        supported_openai_params に tools を持たないため構造化 tool 出力が不可能。
        params (ground truth) を boolean より優先して弾く。
        """
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "supported_openai_params": ["max_tokens", "web_search_options", "response_format"],
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_unpopulated_supported_openai_params_trusts_function_calling(self):
        """Issue #130: params が未populate (Gemini/Anthropic等) なら boolean を信頼する。

        空リスト / キー欠落で tools 必須にすると Gemini/Anthropic を全滅させるため、
        params が無い場合は従来どおり supports_function_calling のみで判定する。
        """
        info_missing = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info_missing) is True
        info_empty = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
            "supported_openai_params": [],
        }
        assert _is_litellm_model_annotation_compatible(info_empty) is True


@pytest.mark.unit
class TestIsAnnotationSuitable:
    """Issue #130 軸B: 用途不適モデルの名前 denylist 検証。"""

    @pytest.mark.parametrize(
        "model_id",
        [
            "gemini/gemini-2.5-pro-preview-tts",
            "gemini/gemini-2.5-computer-use-preview-10-2025",
            "openai/gpt-4o-search-preview",
            "openai/gpt-4o-mini-search-preview-2025-03-11",
            "openai/o3-deep-research",
            "openai/o4-mini-deep-research-2025-06-26",
        ],
    )
    def test_unsuitable_models_excluded(self, model_id):
        """TTS / computer-use / search-preview / deep-research は不適。"""
        assert _is_annotation_suitable(model_id) is False

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai/gpt-4o",
            "openai/gpt-5",
            "anthropic/claude-sonnet-4-5",
            "gemini/gemini-2.5-pro",
            # codex は denylist に入れない: OpenRouter chat codex は動作可能なため残す
            "openrouter/openai/gpt-5.1-codex-max",
        ],
    )
    def test_suitable_models_kept(self, model_id):
        """標準モデルと OpenRouter codex は適格 (除外しない)。"""
        assert _is_annotation_suitable(model_id) is True


@pytest.mark.unit
class TestDiscoverAvailableVisionModelsRegression:
    """Issue #130: 実 litellm 同梱 DB に対する discovery 回帰テスト。

    LITELLM_LOCAL_MODEL_COST_MAP=True (api_model_discovery import 時に設定) のため
    network には出ない。litellm DB 更新で drift し得るが、月次 dependency review で
    確認する前提の固定 (dependency-management.md)。
    """

    def test_responses_and_unsuitable_models_excluded(self):
        models = set(discover_available_vision_models()["models"])
        excluded = [
            # 軸A: mode=responses (endpoint-gate)
            "openai/o3-deep-research",
            "openai/o4-mini-deep-research-2025-06-26",
            "openai/gpt-5-pro",
            "openai/o3-pro",
            # 軸B-1: tools 非対応 (search-api)
            "openai/gpt-5-search-api",
            # 軸B-2: name denylist
            "gemini/gemini-2.5-pro-preview-tts",
            "gemini/gemini-2.5-computer-use-preview-10-2025",
            "openai/gpt-4o-search-preview",
        ]
        for model_id in excluded:
            assert model_id not in models, f"{model_id} は除外されるべき"

    def test_standard_models_retained(self):
        """過剰除外していないこと (標準 chat モデルは残る)。"""
        models = set(discover_available_vision_models()["models"])
        retained = [
            "openai/gpt-4o",
            "openai/gpt-5",
            "anthropic/claude-sonnet-4-5",
            "gemini/gemini-2.5-pro",
        ]
        for model_id in retained:
            assert model_id in models, f"{model_id} は残るべき"


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
        assert "capabilities" not in metadata

    def test_openai_moderation_metadata_declares_ratings_capability(self):
        """OpenAI Moderations must preserve ratings through WebApiAnnotator formatting."""
        info = {
            "supports_vision": False,
            "supports_function_calling": False,
            "mode": "moderation",
        }
        metadata = _format_litellm_metadata("openai/omni-moderation-latest", info)
        assert metadata is not None
        assert metadata["capabilities"] == ["ratings"]

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

    def test_collect_models_registers_openai_moderation_without_function_calling(self):
        """OpenAI moderation allowlist は exact prefix のみに限定される。"""
        moderation_info = {
            "litellm_provider": "openai",
            "supports_vision": False,
            "supports_function_calling": False,
            "mode": "moderation",
        }
        non_tool_info = {
            "litellm_provider": "openai",
            "supports_vision": True,
            "supports_function_calling": False,
            "mode": "chat",
        }
        mock_cost = {
            "omni-moderation-latest": moderation_info,
            "gpt-non-tool": non_tool_info,
        }

        with patch("image_annotator_lib.webapi.api_model_discovery.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_cost
            mock_litellm.get_model_info.side_effect = lambda model_id, custom_llm_provider: mock_cost[
                model_id
            ]
            metadata = _collect_models(require_compatible=True, exclude_deprecated=False)

        assert "openai/omni-moderation-latest" in metadata
        assert metadata["openai/omni-moderation-latest"]["capabilities"] == ["ratings"]
        assert "openai/gpt-non-tool" not in metadata


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
