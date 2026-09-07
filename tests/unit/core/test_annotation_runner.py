"""ADR 0023 Phase 1 / Issue #45: core/annotation_runner._create_annotator_instance の lookup テスト。

Issue #45 で direct LiteLLM ID dispatch 経路が廃止された。本テストでは:
- registry 経由で WebAPI / ローカル ML が正しくインスタンス化されること
- registry 未登録の任意モデル名 (`provider/model` 形式含む) が KeyError で弾かれること
を検証する。
"""

from __future__ import annotations

import pytest

from image_annotator_lib.core import annotation_runner
from image_annotator_lib.webapi.annotator import WebApiAnnotator


class _StubLocalAnnotator:
    """ローカル ML モデルのスタブ (instance 生成可能)。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class TestCreateAnnotatorInstanceLookupOrder:
    """registry-first lookup の検証。"""

    def test_registry_resolves_slash_form_to_litellm_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """registry に slash 形式 (例: ``openai/gpt-4o``) で登録された OpenRouter モデルが、
        metadata の `litellm_model_id` (``openrouter/openai/gpt-4o``) で WebApiAnnotator
        として構築される。Issue #45 で direct dispatch 経路は廃止されたため、
        registry-only の resolution path のみが残る。
        """
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (("openai/gpt-4o", WebApiAnnotator) if name == "openai/gpt-4o" else None),
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

    def test_registry_webapi_injects_responses_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Issue #131: registry metadata の `mode=responses` が WebApiAnnotator に注入される。

        gpt-5-pro / o3-pro 等の responses 系モデルは metadata の `mode` を
        `WebApiAnnotator.mode` まで伝播させ、推論時に `OpenAIResponsesModel` を
        構築できる必要がある。
        """
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (("openai/gpt-5-pro", WebApiAnnotator) if name == "openai/gpt-5-pro" else None),
        )
        monkeypatch.setattr(
            annotation_runner,
            "get_webapi_metadata",
            lambda name: (
                {"litellm_model_id": "openai/gpt-5-pro", "provider": "openai", "mode": "responses"}
                if name == "openai/gpt-5-pro"
                else None
            ),
        )

        instance = annotation_runner._create_annotator_instance("openai/gpt-5-pro")

        assert isinstance(instance, WebApiAnnotator)
        assert instance.mode == "responses"

    def test_registry_webapi_defaults_mode_to_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Issue #131: metadata に `mode` が無い場合は `chat` を既定値とする。"""
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (("openai/gpt-4o", WebApiAnnotator) if name == "openai/gpt-4o" else None),
        )
        monkeypatch.setattr(
            annotation_runner,
            "get_webapi_metadata",
            lambda name: (
                {"litellm_model_id": "openai/gpt-4o", "provider": "openai"}
                if name == "openai/gpt-4o"
                else None
            ),
        )

        instance = annotation_runner._create_annotator_instance("openai/gpt-4o")

        assert isinstance(instance, WebApiAnnotator)
        assert instance.mode == "chat"

    def test_registry_webapi_missing_litellm_model_id_raises_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR 0023 Phase 1 (PR #40): metadata に `litellm_model_id` がない場合は KeyError。

        旧 `api_model_id` キーへのフォールバックは Phase 1.x で廃止された
        (ADR 0023 line 73「互換シムを残さない」)。
        """
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: ("legacy-name", WebApiAnnotator) if name == "legacy-name" else None,
        )
        # 旧 `api_model_id` のみ持つ metadata はもはや解決できず KeyError
        monkeypatch.setattr(
            annotation_runner,
            "get_webapi_metadata",
            lambda name: {"api_model_id": "openai/gpt-4o-mini"} if name == "legacy-name" else None,
        )

        with pytest.raises(KeyError, match="litellm_model_id"):
            annotation_runner._create_annotator_instance("legacy-name")

    def test_unregistered_provider_model_format_raises_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #45: registry に存在しない `provider/model` 形式は direct dispatch されず
        KeyError になる。capability check 抜けで非 vision モデルが API 課金前に弾けない
        問題を回避するため、direct LiteLLM ID 経路は廃止された。
        """
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: None,
        )
        monkeypatch.setattr(annotation_runner, "get_cls_obj_registry", lambda: {})

        with pytest.raises(KeyError, match="not found in registry"):
            annotation_runner._create_annotator_instance("anthropic/claude-3-5-sonnet-20241022")

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
        """registry に該当しないモデル名は KeyError (slash 形式かどうかに関わらず)。"""
        monkeypatch.setattr(annotation_runner, "find_model_class_case_insensitive", lambda name: None)
        monkeypatch.setattr(annotation_runner, "get_cls_obj_registry", lambda: {})

        with pytest.raises(KeyError, match="not found in registry"):
            annotation_runner._create_annotator_instance("just-a-name")


class TestGetAnnotatorInstanceApiKeysCacheBehavior:
    """get_annotator_instance の api_keys={} キャッシュバイパス問題のリグレッションテスト (Issue #146)。"""

    @pytest.mark.unit
    def test_empty_api_keys_uses_instance_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api_keys={} (空 dict) はキャッシュを使い、毎回新インスタンスを生成しない。

        Regression: api_keys={} を渡すと api_keys is not None が True になり、
        ローカル ML モデルで毎回新インスタンスが生成されて 2 回目に ModelLoadError になる
        バグを防ぐ (Issue #146 追加確認)。
        """
        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

        created: list[str] = []

        def _fake_create(
            model_name: str, api_keys: dict | None = None, additional_prompt: str | None = None
        ) -> _StubLocalAnnotator:
            created.append(model_name)
            return _StubLocalAnnotator(model_name)

        monkeypatch.setattr(annotation_runner, "_create_annotator_instance", _fake_create)

        # 1 回目: インスタンスを生成してキャッシュに保存
        inst1 = annotation_runner.get_annotator_instance("my_model", api_keys={})
        # 2 回目: キャッシュから返すはず
        inst2 = annotation_runner.get_annotator_instance("my_model", api_keys={})

        assert inst1 is inst2, "api_keys={} でも同一インスタンスが返されるべき"
        assert len(created) == 1, f"インスタンス生成は 1 回のみのはずだが {len(created)} 回生成された"

        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

    @pytest.mark.unit
    def test_non_empty_api_keys_bypasses_cache_for_webapi_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WebAPI モデルは api_keys 指定時にキャッシュをバイパスする。

        api_keys / additional_prompt は呼び出しごとに変わりうるため、WebAPI モデルは
        毎回新しいインスタンスを作る必要がある。
        """
        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

        created: list[str] = []

        def _fake_create(
            model_name: str, api_keys: dict | None = None, additional_prompt: str | None = None
        ) -> _StubLocalAnnotator:
            created.append(model_name)
            return _StubLocalAnnotator(model_name)

        monkeypatch.setattr(annotation_runner, "_create_annotator_instance", _fake_create)
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (name, WebApiAnnotator),
        )

        api_keys = {"openai": "sk-test"}
        inst1 = annotation_runner.get_annotator_instance("openai/gpt-4o", api_keys=api_keys)
        inst2 = annotation_runner.get_annotator_instance("openai/gpt-4o", api_keys=api_keys)

        assert inst1 is not inst2, "WebAPI モデルは api_keys 指定時に新インスタンスが生成されるべき"
        assert len(created) == 2

        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

    @pytest.mark.unit
    def test_non_empty_api_keys_still_caches_local_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ローカル ML モデルは api_keys が渡されてもキャッシュされる (Issue #162)。

        api_keys は WebAPI 専用の引数であり、ローカル ML モデルは一切使わない。
        旧実装は `if api_keys:` だけで判定していたため、API キーを設定している
        ユーザーだけローカル ONNX/Transformers モデルが毎回作り直され、
        annotate() 呼び出しごとにモデル再ロードが発生していた。
        """
        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

        created: list[str] = []

        def _fake_create(
            model_name: str, api_keys: dict | None = None, additional_prompt: str | None = None
        ) -> _StubLocalAnnotator:
            created.append(model_name)
            return _StubLocalAnnotator(model_name)

        monkeypatch.setattr(annotation_runner, "_create_annotator_instance", _fake_create)
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (name, _StubLocalAnnotator),
        )

        api_keys = {"openai": "sk-test", "anthropic": "sk-ant-test"}
        inst1 = annotation_runner.get_annotator_instance("wd-vit-tagger-v3", api_keys=api_keys)
        inst2 = annotation_runner.get_annotator_instance("wd-vit-tagger-v3", api_keys=api_keys)

        assert inst1 is inst2, "ローカル ML モデルは api_keys 指定時も同一インスタンスを返すべき"
        assert len(created) == 1, f"インスタンス生成は 1 回のみのはずだが {len(created)} 回生成された"

        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

    @pytest.mark.unit
    def test_unregistered_model_is_not_treated_as_webapi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """registry 未登録のモデル名は WebAPI 扱いにしない (キャッシュ経路を通る)。"""
        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()
        monkeypatch.setattr(annotation_runner, "find_model_class_case_insensitive", lambda name: None)

        assert annotation_runner._is_webapi_model("no_such_model") is False

        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()


class _LoadCountingAnnotator:
    """`__enter__` での実ロード回数を数えるローカル ML アノテータのスタブ。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.load_count = 0
        self._loaded = False

    def __enter__(self) -> _LoadCountingAnnotator:
        if not self._loaded:
            self.load_count += 1
            self._loaded = True
        return self

    def __exit__(self, *_args: object) -> None:
        # Issue #162: コンテキストを抜けてもモデルは保持する
        return None

    def predict(self, images: list, phash_list: list) -> list:
        return list(images)


class TestLocalModelIsLoadedOnceAcrossCalls:
    """ローカル ML モデルが annotate() 連続呼び出しで再ロードされないこと (Issue #162)。"""

    @pytest.mark.unit
    def test_local_model_loads_once_across_repeated_calls_with_api_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api_keys を渡してもローカルモデルは 1 回しかロードされない。

        受け入れ条件 (Issue #162): 同一モデルで annotate() を連続呼び出ししても、
        2 回目以降にモデルのロードが発生しない。
        """
        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()

        created: list[str] = []

        def _fake_create(
            model_name: str, api_keys: dict | None = None, additional_prompt: str | None = None
        ) -> _LoadCountingAnnotator:
            created.append(model_name)
            return _LoadCountingAnnotator(model_name)

        monkeypatch.setattr(annotation_runner, "_create_annotator_instance", _fake_create)
        monkeypatch.setattr(
            annotation_runner,
            "find_model_class_case_insensitive",
            lambda name: (name, _LoadCountingAnnotator),
        )

        api_keys = {"openai": "sk-test"}
        # チャンク 3 回ぶんの呼び出しを模す
        for _ in range(3):
            annotator = annotation_runner.get_annotator_instance("wd-vit-tagger-v3", api_keys=api_keys)
            annotation_runner._annotate_model(annotator, ["img"], ["phash"])

        assert len(created) == 1, f"インスタンス生成は 1 回のみのはずが {len(created)} 回"
        cached = annotation_runner._MODEL_INSTANCE_REGISTRY["wd-vit-tagger-v3"]
        assert cached.load_count == 1, f"モデルロードは 1 回のみのはずが {cached.load_count} 回"

        annotation_runner._MODEL_INSTANCE_REGISTRY.clear()
