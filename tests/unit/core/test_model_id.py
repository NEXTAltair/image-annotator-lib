"""ADR 0023 Phase 1: webapi/model_id.py の unit test."""

from __future__ import annotations

import pytest

from image_annotator_lib.exceptions.errors import (
    IdMappingError,
    MissingApiKeyError,
    UnknownProviderError,
)
from image_annotator_lib.webapi.model_id import (
    SUPPORTED_PROVIDERS,
    PydanticAIModelRef,
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
        from image_annotator_lib.webapi.model_id import build_pydantic_model

        ref = resolve_model_ref("openai/gpt-4o")
        with pytest.raises(MissingApiKeyError):
            build_pydantic_model(ref, "")
