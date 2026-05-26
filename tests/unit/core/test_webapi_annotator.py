"""ADR 0023 Phase 1: webapi/annotator.py の unit test."""

from __future__ import annotations

import pytest

from image_annotator_lib.core import utils
from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.webapi.annotator import WebApiAnnotator


class TestWebApiAnnotatorBasics:
    """`WebApiAnnotator` の構造的整合性 (provider_manager を呼ばない部分のみ) を確認する。"""

    def test_advertised_capabilities_includes_tags_captions_scores(self) -> None:
        caps = WebApiAnnotator.ADVERTISED_CAPABILITIES
        assert TaskCapability.TAGS in caps
        assert TaskCapability.CAPTIONS in caps
        assert TaskCapability.SCORES in caps
        assert TaskCapability.RATINGS not in caps

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

    def test_init_accepts_explicit_rating_capability(self) -> None:
        annotator = WebApiAnnotator(
            litellm_model_id="openai/gpt-4o",
            capabilities=[TaskCapability.RATINGS.value],
        )

        assert annotator.capabilities == frozenset({TaskCapability.RATINGS})

    def test_init_does_not_consult_config_registry(self) -> None:
        # BaseAnnotator.__init__ をスキップして config_registry に依存しないことを確認
        # (依存していると config_registry に entry の無い model_name で KeyError になるはず)
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


class TestWebApiAnnotatorRatings:
    """Issue #82: WebAPI rating output is separate from tags and score labels."""

    def test_format_predictions_emits_ratings_only_when_capability_declared(self) -> None:
        annotator = WebApiAnnotator(
            litellm_model_id="openai/gpt-4o",
            capabilities={TaskCapability.TAGS, TaskCapability.RATINGS},
        )

        result = annotator._format_predictions(
            [
                {
                    "phash": "abc",
                    "tags": ["solo"],
                    "formatted_output": {
                        "tags": ["solo"],
                        "captions": [],
                        "score": None,
                        "ratings": [
                            {
                                "raw_label": "questionable",
                                "confidence_score": None,
                                "source_scheme": "prompt_defined",
                            }
                        ],
                    },
                    "error": None,
                }
            ]
        )[0]

        assert result.capabilities == {TaskCapability.TAGS, TaskCapability.RATINGS}
        assert result.tags == ["solo"]
        assert result.score_labels is None
        assert result.ratings is not None
        assert result.ratings[0].raw_label == "questionable"
        assert result.ratings[0].confidence_score is None
        assert result.ratings[0].source_scheme == "prompt_defined"
        assert result.raw_output is not None
        assert result.raw_output["formatted_output"]["ratings"][0]["raw_label"] == "questionable"

    def test_format_predictions_ignores_ratings_without_capability(self) -> None:
        annotator = WebApiAnnotator(litellm_model_id="openai/gpt-4o")

        result = annotator._format_predictions(
            [
                {
                    "phash": "abc",
                    "tags": ["solo"],
                    "formatted_output": {
                        "tags": ["solo"],
                        "captions": [],
                        "score": 8.0,
                        "ratings": [
                            {
                                "raw_label": "explicit",
                                "confidence_score": 0.9,
                                "source_scheme": "prompt_defined",
                            }
                        ],
                    },
                    "error": None,
                }
            ]
        )[0]

        assert TaskCapability.RATINGS not in result.capabilities
        assert result.ratings is None
        assert result.tags == ["solo"]
        assert result.score_labels is None


class TestWebApiAnnotatorIssue35Regression:
    """Issue #35 regression: WebAPI 経路で `determine_effective_device` が呼ばれない。"""

    def test_init_does_not_invoke_determine_effective_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`WebApiAnnotator.__init__` 経由で CUDA 判定が走らないこと。

        ADR 0023 Phase 1 (Issue #35) で WebAPI 経路は `BaseAnnotator.__init__` を
        スキップする設計になった。さらに `BaseAnnotator.__init__` 自体からも
        `_validate_device` 呼び出しが除去されたため、ローカル ML 系 base class を
        経由しない `WebApiAnnotator` 構築では `determine_effective_device` 呼び出しは
        構造的に発生しない。
        """
        call_count = {"n": 0}
        original = utils.determine_effective_device

        def spy(*args: object, **kwargs: object) -> str:
            call_count["n"] += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(utils, "determine_effective_device", spy)

        annotator = WebApiAnnotator(
            litellm_model_id="google/gemini-2.5-pro",
            api_keys={"google": "fake-key"},
        )

        assert annotator.device == "api"
        assert call_count["n"] == 0, (
            "WebApi 経路で determine_effective_device が呼ばれてはならない (Issue #35)"
        )

    def test_base_annotator_init_does_not_invoke_determine_effective_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`BaseAnnotator.__init__` 単独でも `determine_effective_device` を呼ばない。

        device 判定はローカル ML 系 base class (`TransformersBaseAnnotator` 等) の
        責務として分離された (Issue #35)。`BaseAnnotator` 直系の test stub では
        device は None のまま残る。
        """
        from image_annotator_lib.core.base.annotator import BaseAnnotator
        from image_annotator_lib.core.config import config_registry

        # 最小限の config を登録 (LocalMLModelConfig は model_path を必須とするため指定する)
        config_registry._merged_config_data["test-base-annotator-issue35"] = {
            "device": "cuda",  # CUDA を要求しても判定が走らないことを確認
            "model_path": "test/path",
            "class": "TestStub",
        }

        call_count = {"n": 0}
        original = utils.determine_effective_device

        def spy(*args: object, **kwargs: object) -> str:
            call_count["n"] += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(utils, "determine_effective_device", spy)

        class _Stub(BaseAnnotator):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def _preprocess_images(self, images):
                return images

            def _run_inference(self, processed):
                return []

            def _format_predictions(self, raw_outputs):
                return []

        try:
            annotator = _Stub("test-base-annotator-issue35")
            # BaseAnnotator は device sentinel として空文字を保持する (subclass 未上書き状態)
            assert annotator.device == ""
            assert call_count["n"] == 0, (
                "BaseAnnotator.__init__ で determine_effective_device が呼ばれてはならない (Issue #35)"
            )
        finally:
            config_registry._merged_config_data.pop("test-base-annotator-issue35", None)
