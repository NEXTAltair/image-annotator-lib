"""ADR 0023 Phase 1: core/annotation_runner._create_annotator_instance の lookup 順序テスト。

Codex review P1 (https://github.com/NEXTAltair/image-annotator-lib/pull/38#discussion_r3203496580)
で指摘された「registry の OpenRouter エントリが direct LiteLLM dispatch に奪われる」問題の
regression test。
"""

from __future__ import annotations

import pytest

from image_annotator_lib.core import annotation_runner
from image_annotator_lib.core.webapi_annotator import WebApiAnnotator


class _StubPydanticAIWebAPIAnnotator:
    """`_is_webapi_annotator_class` がクラス名で判定するための stub。

    Python ではクラス body 内の ``__name__ = ...`` はインスタンス属性扱いになるため、
    クラス自体の ``__name__`` をクラス定義後に明示的に上書きする。
    """


_StubPydanticAIWebAPIAnnotator.__name__ = "PydanticAIWebAPIAnnotator"


class _StubLocalAnnotator:
    """ローカル ML モデルのスタブ (instance 生成可能)。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class TestCreateAnnotatorInstanceLookupOrder:
    """registry-first lookup の検証。"""

    def test_registry_openrouter_entry_takes_precedence_over_direct_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenRouter モデルが registry に slash 形式 (例: ``openai/gpt-4o``) で登録されている場合、
        direct OpenAI ではなく ``openrouter/openai/gpt-4o`` で WebApiAnnotator が構築される。
        """
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (
                ("openai/gpt-4o", _StubPydanticAIWebAPIAnnotator) if name == "openai/gpt-4o" else None
            ),
        )
        monkeypatch.setattr(
            annotation_runner,
            "get_webapi_metadata",
            lambda name: (
                {"litellm_model_id": "openrouter/openai/gpt-4o", "provider": "openrouter"}
                if name == "openai/gpt-4o"
                else None
            ),
        )

        instance = annotation_runner._create_annotator_instance("openai/gpt-4o")

        assert isinstance(instance, WebApiAnnotator)
        assert instance.litellm_model_id == "openrouter/openai/gpt-4o"
        assert instance.model_name == "openai/gpt-4o"

    def test_registry_webapi_with_legacy_api_model_id_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """旧 metadata の ``api_model_id`` キーも `_resolve_litellm_model_id` の
        フォールバックで吸収される。"""
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: ("legacy-name", _StubPydanticAIWebAPIAnnotator) if name == "legacy-name" else None,
        )
        monkeypatch.setattr(
            annotation_runner,
            "get_webapi_metadata",
            lambda name: {"api_model_id": "openai/gpt-4o-mini"} if name == "legacy-name" else None,
        )

        instance = annotation_runner._create_annotator_instance("legacy-name")
        assert isinstance(instance, WebApiAnnotator)
        assert instance.litellm_model_id == "openai/gpt-4o-mini"
        assert instance.model_name == "legacy-name"

    def test_direct_litellm_id_used_when_not_in_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """registry に無い `provider/model` 形式は direct LiteLLM ID として扱う。"""
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: None,
        )
        monkeypatch.setattr(annotation_runner, "get_webapi_metadata", lambda name: None)

        instance = annotation_runner._create_annotator_instance("anthropic/claude-3-5-sonnet-20241022")
        assert isinstance(instance, WebApiAnnotator)
        assert instance.litellm_model_id == "anthropic/claude-3-5-sonnet-20241022"
        assert instance.model_name == "anthropic/claude-3-5-sonnet-20241022"

    def test_local_ml_model_instantiated_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """registry の非 WebAPI クラスは旧来通り直接インスタンス化する。"""
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: ("local-tagger", _StubLocalAnnotator) if name == "local-tagger" else None,
        )
        instance = annotation_runner._create_annotator_instance("local-tagger")
        assert isinstance(instance, _StubLocalAnnotator)
        assert instance.model_name == "local-tagger"

    def test_unknown_model_raises_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """registry にも direct LiteLLM 形式にも該当しない場合 KeyError。"""
        monkeypatch.setattr(annotation_runner, "find_model_class_case_insensitive", lambda name: None)
        monkeypatch.setattr(annotation_runner, "get_cls_obj_registry", lambda: {})
        monkeypatch.setattr(annotation_runner, "get_available_models", lambda: [])

        with pytest.raises(KeyError):
            annotation_runner._create_annotator_instance("just-a-name")
