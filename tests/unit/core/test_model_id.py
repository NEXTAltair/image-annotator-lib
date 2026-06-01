"""ADR 0023 Phase 1: webapi/model_id.py の unit test."""

from __future__ import annotations

from dataclasses import replace

import pytest

from image_annotator_lib.exceptions.errors import (
    IdMappingError,
    MissingApiKeyError,
    UnknownProviderError,
)
from image_annotator_lib.webapi.model_id import (
    SUPPORTED_PROVIDERS,
    PydanticAIModelRef,
    build_pydantic_model,
    resolve_model_ref,
)


class TestSupportedProviders:
    """`SUPPORTED_PROVIDERS` の構造的整合性を確認する。"""

    def test_contains_phase1_providers(self) -> None:
        assert "openai" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS
        assert "google" in SUPPORTED_PROVIDERS
        assert "gemini" in SUPPORTED_PROVIDERS  # google の alias
        assert "openrouter" in SUPPORTED_PROVIDERS

    def test_excludes_out_of_phase1_providers(self) -> None:
        assert "vertex_ai" not in SUPPORTED_PROVIDERS
        assert "xai" not in SUPPORTED_PROVIDERS
        assert "cohere" not in SUPPORTED_PROVIDERS

    def test_is_immutable(self) -> None:
        assert isinstance(SUPPORTED_PROVIDERS, frozenset)


class TestResolveModelRef:
    """`resolve_model_ref()` の prefix 解析と provider dispatch を確認する。"""

    @pytest.mark.parametrize(
        ("litellm_id", "expected_provider", "expected_provider_model"),
        [
            ("openai/gpt-4o", "openai", "gpt-4o"),
            ("anthropic/claude-3-5-sonnet-20241022", "anthropic", "claude-3-5-sonnet-20241022"),
            ("google/gemini-2.5-pro", "google", "gemini-2.5-pro"),
            ("gemini/gemini-2.5-flash", "google", "gemini-2.5-flash"),  # alias
            ("openrouter/google/gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro"),
            # Issue #51: 未知 inner provider の OpenRouter 経路も openrouter builder に dispatch される
            ("openrouter/z-ai/glm-4.7", "openrouter", "z-ai/glm-4.7"),
            ("openrouter/qwen/qwen2-vl-72b-instruct", "openrouter", "qwen/qwen2-vl-72b-instruct"),
        ],
    )
    def test_resolves_supported_provider(
        self, litellm_id: str, expected_provider: str, expected_provider_model: str
    ) -> None:
        ref = resolve_model_ref(litellm_id)
        assert isinstance(ref, PydanticAIModelRef)
        assert ref.provider == expected_provider
        assert ref.litellm_model_id == litellm_id
        assert ref.provider_model_id == expected_provider_model

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(IdMappingError):
            resolve_model_ref("")

    def test_rejects_id_without_slash(self) -> None:
        with pytest.raises(IdMappingError):
            resolve_model_ref("gpt-4o")

    def test_rejects_id_with_empty_provider(self) -> None:
        with pytest.raises(IdMappingError):
            resolve_model_ref("/gpt-4o")

    def test_rejects_id_with_empty_model(self) -> None:
        with pytest.raises(IdMappingError):
            resolve_model_ref("openai/")

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(UnknownProviderError):
            resolve_model_ref("vertex_ai/gemini-pro")

    def test_provider_lowercased(self) -> None:
        # 大文字混在 ID も小文字 provider にマッピングされる
        ref = resolve_model_ref("OpenAI/gpt-4o")
        assert ref.provider == "openai"


class TestPydanticAIModelRef:
    """`PydanticAIModelRef` の immutability と pydantic_model_id 構築を確認する。"""

    def test_is_frozen(self) -> None:
        ref = resolve_model_ref("openai/gpt-4o")
        with pytest.raises((AttributeError, Exception)):
            ref.provider = "anthropic"  # type: ignore[misc]

    def test_openai_pydantic_model_id_format(self) -> None:
        ref = resolve_model_ref("openai/gpt-4o")
        assert ref.pydantic_model_id == "openai:gpt-4o"

    def test_openrouter_uses_provider_object(self) -> None:
        # OpenRouter は base_url override が必要なため pydantic_model_id を使わない
        ref = resolve_model_ref("openrouter/google/gemini-2.5-pro")
        assert ref.pydantic_model_id is None


class TestBuildPydanticModelMissingKey:
    """`build_pydantic_model()` の API key validation を確認する (provider 構築は別の integration test で)。"""

    def test_empty_api_key_raises(self) -> None:
        ref = resolve_model_ref("openai/gpt-4o")
        with pytest.raises(MissingApiKeyError):
            build_pydantic_model(ref, "")


class TestResolveModelRefEndpoint:
    """`mode` -> `PydanticAIModelRef.endpoint` 配線を確認する (Issue #131)。"""

    def test_openai_responses_mode_sets_responses_endpoint(self) -> None:
        ref = resolve_model_ref("openai/gpt-5-pro", mode="responses")
        assert ref.endpoint == "responses"

    def test_openai_default_endpoint_is_chat(self) -> None:
        ref = resolve_model_ref("openai/gpt-4o")
        assert ref.endpoint == "chat"

    def test_openai_explicit_chat_mode_is_chat(self) -> None:
        ref = resolve_model_ref("openai/gpt-4o", mode="chat")
        assert ref.endpoint == "chat"

    def test_anthropic_responses_mode_stays_chat(self) -> None:
        # responses は OpenAI provider 限定。anthropic は常に chat。
        ref = resolve_model_ref("anthropic/claude-3-5-sonnet-20241022", mode="responses")
        assert ref.endpoint == "chat"


class TestBuildPydanticModelEndpoint:
    """endpoint 種別ごとに対応する OpenAI Model class が構築されることを確認する (Issue #131)。

    network は不要 (provider object 構築のみ。実 API call は行わない)。
    """

    def test_responses_endpoint_builds_responses_model(self) -> None:
        from pydantic_ai.models.openai import OpenAIResponsesModel

        ref = replace(resolve_model_ref("openai/gpt-5-pro"), endpoint="responses")
        model = build_pydantic_model(ref, "dummy-key")
        assert isinstance(model, OpenAIResponsesModel)

    def test_chat_endpoint_builds_chat_model(self) -> None:
        from pydantic_ai.models.openai import OpenAIChatModel

        ref = resolve_model_ref("openai/gpt-4o")
        model = build_pydantic_model(ref, "dummy-key")
        assert isinstance(model, OpenAIChatModel)
