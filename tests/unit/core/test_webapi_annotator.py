"""ADR 0023 Phase 1: core/webapi_annotator.py の unit test."""

from __future__ import annotations

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.core.webapi_annotator import WebApiAnnotator


class TestWebApiAnnotatorBasics:
    """`WebApiAnnotator` の構造的整合性 (provider_manager を呼ばない部分のみ) を確認する。"""

    def test_advertised_capabilities_includes_tags_captions_scores(self) -> None:
        caps = WebApiAnnotator.ADVERTISED_CAPABILITIES
        assert TaskCapability.TAGS in caps
        assert TaskCapability.CAPTIONS in caps
        assert TaskCapability.SCORES in caps

    def test_advertised_capabilities_is_frozenset(self) -> None:
        # registry.AnnotatorInfo.capabilities が frozenset 型を要求するため
        assert isinstance(WebApiAnnotator.ADVERTISED_CAPABILITIES, frozenset)

    def test_init_sets_litellm_model_id_as_default_model_name(self) -> None:
        annotator = WebApiAnnotator(litellm_model_id="openai/gpt-4o")
        assert annotator.model_name == "openai/gpt-4o"
        assert annotator.litellm_model_id == "openai/gpt-4o"

    def test_init_uses_explicit_model_name_when_provided(self) -> None:
        annotator = WebApiAnnotator(
            litellm_model_id="openai/gpt-4o",
            model_name="my-registered-model",
        )
        assert annotator.model_name == "my-registered-model"
        assert annotator.litellm_model_id == "openai/gpt-4o"

    def test_init_does_not_consult_config_registry(self) -> None:
        # BaseAnnotator.__init__ をスキップして config_registry に依存しないことを確認
        # (依存していると direct LiteLLM ID で KeyError になるはず)
        annotator = WebApiAnnotator(
            litellm_model_id="google/gemini-2.5-pro",
            api_keys={"google": "fake-key"},
        )
        assert annotator.api_keys == {"google": "fake-key"}
        assert annotator.device == "api"
        assert annotator.model_path is None
        assert annotator.components is None

    def test_context_manager_is_noop(self) -> None:
        annotator = WebApiAnnotator(litellm_model_id="openai/gpt-4o")
        with annotator as ctx:
            assert ctx is annotator
        # __exit__ で例外なく抜ける
