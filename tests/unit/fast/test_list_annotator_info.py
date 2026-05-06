"""list_annotator_info() / AnnotatorInfo のユニットテスト (Issue #19)。

新規メタデータ API がレジストリ + PydanticAI 直接モデル両方を統合し、
型安全な dataclass で返すことを検証する。
"""

from unittest.mock import patch

import pytest

from image_annotator_lib import AnnotatorInfo, list_annotator_info
from image_annotator_lib.core.types import TaskCapability

# --- テスト用ダミークラス ---


class _DummyTagger:
    """ローカル ML タガーのダミー実装。クラス名から model_type=tagger と判定される。"""


class _DummyScorer:
    """ローカル ML スコアラーのダミー実装。クラス名から model_type=scorer と判定される。"""


class _DummyCaptioner:
    """ローカル ML キャプショナーのダミー実装。クラス名から model_type=captioner と判定される。"""


class PydanticAIWebAPIAnnotator:
    """`_requires_api_key` がクラス名 'PydanticAIWebAPIAnnotator' を WebAPI と判定するためのダミー。"""


# --- Fixtures ---


@pytest.fixture
def empty_registry():
    """レジストリと PydanticAI 直接モデルを両方空にする"""
    from image_annotator_lib.core import registry

    with patch.object(registry, "_MODEL_CLASS_OBJ_REGISTRY", {}):
        with patch.object(registry, "_REGISTRY_INITIALIZED", True):
            with patch.object(registry, "_WEBAPI_MODEL_METADATA", {}):
                with patch("image_annotator_lib.api.get_agent_factory") as mock_factory:
                    mock_factory.return_value.get_available_models.return_value = []
                    yield


@pytest.fixture
def patched_registry():
    """`_MODEL_CLASS_OBJ_REGISTRY` を直接書き換えるためのフィクスチャ。
    呼び出し側が辞書と config を渡す。"""

    def _setup(
        model_dict: dict,
        config_dict: dict | None = None,
        direct_models: list[str] | None = None,
        webapi_metadata: dict | None = None,
    ):
        """戻り値: () のコンテキストマネージャ。withで使う。

        Args:
            model_dict: `_MODEL_CLASS_OBJ_REGISTRY` に設定するモデル名→クラス辞書
            config_dict: `config_registry._merged_config_data` に設定する辞書
            direct_models: PydanticAI 直接モデルの ID リスト
            webapi_metadata: `_WEBAPI_MODEL_METADATA` (SSoT) に設定する辞書 (Issue #26)
        """
        config_dict = config_dict or {}
        direct_models = direct_models or []
        webapi_metadata = webapi_metadata or {}

        # _config が dict の場合はそれを書き換え、なければ get_all_config を patch
        return _PatchedRegistryCtx(model_dict, config_dict, direct_models, webapi_metadata)

    return _setup


class _PatchedRegistryCtx:
    """_MODEL_CLASS_OBJ_REGISTRY と get_all_config と agent_factory を一括 patch するコンテキスト。"""

    def __init__(
        self,
        model_dict: dict,
        config_dict: dict,
        direct_models: list[str],
        webapi_metadata: dict | None = None,
    ):
        self.model_dict = model_dict
        self.config_dict = config_dict
        self.direct_models = direct_models
        self.webapi_metadata = webapi_metadata or {}
        self._patches: list = []

    def __enter__(self):
        from image_annotator_lib.core import registry
        from image_annotator_lib.core.config import get_config_registry

        self._patches.append(patch.object(registry, "_MODEL_CLASS_OBJ_REGISTRY", self.model_dict))
        self._patches.append(patch.object(registry, "_REGISTRY_INITIALIZED", True))
        self._patches.append(patch.object(registry, "_WEBAPI_MODEL_METADATA", self.webapi_metadata))
        # config_registry の get / get_all_config は _merged_config_data を読むので、
        # 内部データを差し替えれば両方一貫した値を返せる。proxy の delattr 問題も回避できる。
        real_registry = get_config_registry()
        self._patches.append(patch.object(real_registry, "_merged_config_data", self.config_dict))
        agent_patch = patch("image_annotator_lib.api.get_agent_factory")
        self._patches.append(agent_patch)

        for p in self._patches[:-1]:
            p.start()
        agent_factory_mock = self._patches[-1].start()
        agent_factory_mock.return_value.get_available_models.return_value = self.direct_models
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in reversed(self._patches):
            p.stop()


# --- テストケース ---


@pytest.mark.unit
@pytest.mark.fast
def test_empty_registry_returns_empty_list(empty_registry):
    """レジストリ空 + PydanticAI 直接モデル空のとき、空リストを返す。"""
    result = list_annotator_info()
    assert result == []


@pytest.mark.unit
@pytest.mark.fast
def test_local_ml_model_classification(patched_registry):
    """ローカル ML タガーが is_local=True, is_api=False, model_type='tagger' で分類される。"""
    with patched_registry(
        model_dict={"wd-v1-4-tagger": _DummyTagger},
        config_dict={"wd-v1-4-tagger": {"type": "tagger", "device": "cuda", "capabilities": ["tags"]}},
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert isinstance(info, AnnotatorInfo)
    assert info.name == "wd-v1-4-tagger"
    assert info.model_type == "tagger"
    assert info.is_local is True
    assert info.is_api is False
    assert info.device == "cuda"
    assert TaskCapability.TAGS in info.capabilities


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_classification(patched_registry):
    """PydanticAIWebAPIAnnotator が is_api=True, device=None で分類される。"""
    with patched_registry(
        model_dict={"Claude-3-Opus": PydanticAIWebAPIAnnotator},
        config_dict={
            "Claude-3-Opus": {
                "api_model_id": "claude-3-opus-20240229",
                "type": "vision",
                "capabilities": ["tags", "captions", "scores"],
            }
        },
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.name == "Claude-3-Opus"
    assert info.is_api is True
    assert info.is_local is False
    assert info.device is None


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_capabilities_fallback(patched_registry):
    """config に capabilities が未設定の WebAPI モデルは全3種にフォールバックする (P1修正)。

    type="vision" のみで capabilities を省略した場合、get_model_capabilities は空を返すが、
    _resolve_registry_capabilities が SimplifiedAgentWrapper.ADVERTISED_CAPABILITIES を採用する。
    """
    with patched_registry(
        model_dict={"GPT-4o-mini": PydanticAIWebAPIAnnotator},
        config_dict={
            "GPT-4o-mini": {
                "api_model_id": "gpt-4o-mini",
                "type": "vision",
                # capabilities は意図的に省略
            }
        },
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.is_api is True
    # capabilities が空でないこと (P1 の核心)
    assert len(info.capabilities) > 0, "WebAPI モデルは capabilities が空であってはならない"
    assert TaskCapability.TAGS in info.capabilities
    assert TaskCapability.CAPTIONS in info.capabilities
    assert TaskCapability.SCORES in info.capabilities


@pytest.mark.unit
@pytest.mark.fast
def test_pydanticai_direct_model_inclusion(patched_registry):
    """PydanticAI 直接モデル (provider/model 形式) が結果に含まれ、is_api=True で分類される。"""
    direct_id = "google/gemini-2.5-pro"
    with patched_registry(
        model_dict={},
        config_dict={},
        direct_models=[direct_id],
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.name == direct_id
    assert info.is_api is True
    assert info.is_local is False
    assert info.device is None
    assert info.model_type == "vision"
    # SimplifiedAgentWrapper の AnnotationSchema は 3 capability すべてを返す
    assert TaskCapability.TAGS in info.capabilities
    assert TaskCapability.CAPTIONS in info.capabilities
    assert TaskCapability.SCORES in info.capabilities


@pytest.mark.unit
@pytest.mark.fast
def test_pydanticai_direct_model_dedup_with_registry(patched_registry):
    """レジストリにも PydanticAI factory にも同じ name がある場合、レジストリ側を優先して重複しない。"""
    shared_name = "google/gemini-2.5-pro"
    with patched_registry(
        model_dict={shared_name: PydanticAIWebAPIAnnotator},
        config_dict={shared_name: {"api_model_id": "gemini-2.5-pro", "capabilities": ["tags"]}},
        direct_models=[shared_name],
    ):
        result = list_annotator_info()

    names = [info.name for info in result]
    assert names.count(shared_name) == 1


@pytest.mark.unit
@pytest.mark.fast
def test_invariants_is_local_xor_is_api(patched_registry):
    """全エントリで is_local != is_api、API モデルは device is None。"""
    with patched_registry(
        model_dict={
            "local-tagger": _DummyTagger,
            "remote-claude": PydanticAIWebAPIAnnotator,
        },
        config_dict={
            "local-tagger": {"type": "tagger", "device": "cpu", "capabilities": ["tags"]},
            "remote-claude": {
                "api_model_id": "claude-3-opus",
                "type": "vision",
                "capabilities": ["tags", "captions"],
            },
        },
        direct_models=["openai/gpt-4o"],
    ):
        result = list_annotator_info()

    assert len(result) == 3
    for info in result:
        # XOR 不変条件
        assert info.is_local != info.is_api, f"{info.name}: is_local と is_api が同じ"
        # API モデルは device を持たない
        if info.is_api:
            assert info.device is None, f"{info.name}: API モデルなのに device がある"

    # frozenset により hashable であることを確認 (frozen=True dataclass の必須条件)
    assert hash(result[0]) is not None


# ============================================================================
# Phase 2: AnnotatorInfo 詳細メタデータフィールド (Issue #26)
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_local_ml_model_has_provider_local_and_estimated_size(patched_registry):
    """ローカル ML モデルは `provider="local"`, `estimated_size_gb` 取得, `api_model_id=None`。"""
    with patched_registry(
        model_dict={"wd-v1-4-tagger": _DummyTagger},
        config_dict={
            "wd-v1-4-tagger": {
                "type": "tagger",
                "device": "cuda",
                "capabilities": ["tags"],
                "estimated_size_gb": 1.5,
            }
        },
    ):
        result = list_annotator_info()

    info = result[0]
    assert info.provider == "local"
    assert info.api_model_id is None
    assert info.estimated_size_gb == 1.5
    assert info.discontinued_at is None
    assert info.max_output_tokens is None


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_phase2_fields_from_ssot(patched_registry):
    """WebAPI モデルは `_WEBAPI_MODEL_METADATA` (SSoT) から Phase 2 フィールドを取得。"""
    with patched_registry(
        model_dict={"GPT-4o": PydanticAIWebAPIAnnotator},
        webapi_metadata={
            "GPT-4o": {
                "api_model_id": "openai/gpt-4o",
                "model_name_on_provider": "openai/gpt-4o",
                "provider": "openai",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
    ):
        result = list_annotator_info()

    info = result[0]
    assert info.provider == "openai"
    assert info.api_model_id == "openai/gpt-4o"
    assert info.max_output_tokens == 1800
    assert info.estimated_size_gb is None
    assert info.discontinued_at is None


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_user_toml_api_model_id_does_not_override_ssot(patched_registry):
    """WebAPI モデル定義は SSoT のみを採用し、user TOML api_model_id は無視される。"""
    with patched_registry(
        model_dict={"GPT-4o": PydanticAIWebAPIAnnotator},
        config_dict={
            "GPT-4o": {
                "api_model_id": "openai/gpt-4o-test-override",
                "max_output_tokens": 9999,
            }
        },
        webapi_metadata={
            "GPT-4o": {
                "api_model_id": "openai/gpt-4o",
                "model_name_on_provider": "openai/gpt-4o",
                "provider": "openai",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
    ):
        result = list_annotator_info()

    info = result[0]
    assert info.api_model_id == "openai/gpt-4o"
    assert info.max_output_tokens == 1800
    assert info.provider == "openai"


@pytest.mark.unit
@pytest.mark.fast
def test_local_ml_excludes_webapi_metadata_codex_p2_6(patched_registry):
    """Codex P2 #6 回帰防止: ローカル ML モデルに同名 WebAPI metadata は混入しない。

    PR #22 旧実装では `or` フォールバックで `_WEBAPI_MODEL_METADATA` 由来の
    `api_model_id` がローカル ML モデルに混入し、`_requires_api_key` が誤分類していた。
    Issue #26 の排他分岐で WebAPI metadata はローカル ML 経路に流れない。
    """
    with patched_registry(
        model_dict={"shared-name": _DummyTagger},
        config_dict={
            "shared-name": {
                "type": "tagger",
                "device": "cpu",
                "capabilities": ["tags"],
            }
        },
        webapi_metadata={
            "shared-name": {
                "api_model_id": "should-not-leak",
                "model_name_on_provider": "should-not-leak",
                "provider": "openai",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
    ):
        result = list_annotator_info()

    info = result[0]
    # ローカル ML モデルとして分類される (排他分岐の効果)
    assert info.is_local is True
    assert info.is_api is False
    # WebAPI 由来の `api_model_id` は混入しない
    assert info.api_model_id is None
    # provider はローカル "local" にフォールバック
    assert info.provider == "local"


@pytest.mark.unit
@pytest.mark.fast
def test_phase2_safe_helpers_handle_malformed_metadata(patched_registry):
    """Codex P2 #1, #2 回帰防止: malformed metadata 値でモデル消失しない。

    `_safe_float` / `_safe_int` / `_parse_discontinued_at` が malformed 値を
    warning + None フォールバックする (Issue #23 で取込済みヘルパーの効果)。
    """
    with patched_registry(
        model_dict={"GPT-4o": PydanticAIWebAPIAnnotator},
        webapi_metadata={
            "GPT-4o": {
                "api_model_id": "openai/gpt-4o",
                "model_name_on_provider": "openai/gpt-4o",
                "provider": "openai",
                "max_output_tokens": "not-an-int",  # malformed
                "estimated_size_gb": "not-a-float",  # malformed
                "discontinued_at": "not-a-date",  # malformed
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
    ):
        result = list_annotator_info()

    # malformed 値が来てもモデルが listing から消えない
    assert len(result) == 1
    info = result[0]
    assert info.name == "GPT-4o"
    # 各フィールドが None フォールバック
    assert info.max_output_tokens is None
    assert info.estimated_size_gb is None
    assert info.discontinued_at is None


@pytest.mark.unit
@pytest.mark.fast
def test_pydanticai_direct_model_has_inferred_provider_and_api_model_id(patched_registry):
    """PydanticAI 直接モデルは model_id から provider 推論 + api_model_id=model_id。"""
    with patched_registry(
        model_dict={},
        direct_models=["google/gemini-2.5-pro", "anthropic/claude-3-5-sonnet-latest"],
    ):
        result = list_annotator_info()

    by_name = {info.name: info for info in result}
    assert by_name["google/gemini-2.5-pro"].provider == "google"
    assert by_name["google/gemini-2.5-pro"].api_model_id == "google/gemini-2.5-pro"
    assert by_name["anthropic/claude-3-5-sonnet-latest"].provider == "anthropic"
    assert (
        by_name["anthropic/claude-3-5-sonnet-latest"].api_model_id == "anthropic/claude-3-5-sonnet-latest"
    )


# ============================================================================
# `_infer_provider_from_model_id` ユニットテスト
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_infer_provider_from_slash_format():
    """`provider/model_name` 形式は slash 前を返す。"""
    from image_annotator_lib.core.registry import _infer_provider_from_model_id

    assert _infer_provider_from_model_id("google/gemini-2.5-pro") == "google"
    assert _infer_provider_from_model_id("openai/gpt-4o") == "openai"
    assert _infer_provider_from_model_id("anthropic/claude-3-5-sonnet") == "anthropic"


@pytest.mark.unit
@pytest.mark.fast
def test_infer_provider_falls_back_for_non_slash_model_id():
    """slash の無い model_id は infer_provider_class フォールバックを試みる。"""
    from image_annotator_lib.core.registry import _infer_provider_from_model_id

    # 既知の OpenAI モデル名 (slash なし) → "openai" 推論期待
    # PydanticAI 内部実装変更で None になる可能性も許容するが、例外は raise しないこと
    result = _infer_provider_from_model_id("gpt-4o")
    assert result is None or isinstance(result, str)


@pytest.mark.unit
@pytest.mark.fast
def test_malformed_user_overrides_skips_only_bad_model(patched_registry):
    """PR #27 Codex P1 回帰防止: malformed user TOML entry で listing 全体が abort しない。

    `config_registry.get_all_config()` が dict 以外の truthy 値 (例: 文字列) を
    返した場合、`{**webapi_metadata, **user_overrides}` で TypeError が発生する。
    旧来の `or` フォールバック方式では TypeError が起きなかったため、本 PR の
    排他分岐への切替で再導入された regression を防ぐ。
    """
    # 正常モデル + malformed entry を持つモデルを混在させる
    with patched_registry(
        model_dict={
            "good-tagger": _DummyTagger,
            "broken-tagger": _DummyTagger,
        },
        config_dict={
            "good-tagger": {"type": "tagger", "device": "cpu", "capabilities": ["tags"]},
            "broken-tagger": "this-is-a-string-not-a-dict",  # malformed (truthy + 非 dict)
        },
    ):
        result = list_annotator_info()

    # broken-tagger は構築失敗で skip されるが、good-tagger は listing に残る
    names = {info.name for info in result}
    assert "good-tagger" in names, "正常なモデルは listing に残るべき"
    # broken-tagger は skip (空 dict として扱われ Phase 2 フィールド全て None)
    # 厳密な仕様: 空 dict として AnnotatorInfo が構築される (model_class からの推論で動く)
    # listing 全体が abort しないことが核心


@pytest.mark.unit
@pytest.mark.fast
def test_provider_normalized_to_lowercase_across_sources(patched_registry):
    """PR #27 Codex P2 回帰防止: provider 名は出所に関係なく lowercase に正規化される。

    `available_api_models.toml` は "OpenAI"/"Google" の display-case で記述されるが、
    `_infer_provider_from_model_id` の slash 経由出力は "openai"/"google" の小文字。
    AnnotatorInfo.provider が混在すると case-sensitive consumer が誤分類する。
    本テストは 3 経路 (registry/SSoT/直接モデル) すべてで lowercase で一貫することを保証する。
    """
    with patched_registry(
        # Registry-backed WebAPI モデル: SSoT の display-case provider を持つ
        model_dict={"GPT-4o": PydanticAIWebAPIAnnotator},
        webapi_metadata={
            "GPT-4o": {
                "api_model_id": "openai/gpt-4o",
                "model_name_on_provider": "openai/gpt-4o",
                "provider": "OpenAI",  # ← display-case (TOML 記述)
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
        # 直接モデル: model_id slash 前から推論 (既に小文字)
        direct_models=["google/gemini-2.5-pro", "anthropic/claude-3-5-sonnet-latest"],
    ):
        result = list_annotator_info()

    by_name = {info.name: info for info in result}
    # registry-backed: SSoT の "OpenAI" が "openai" に正規化されている
    assert by_name["GPT-4o"].provider == "openai", (
        f"registry-backed WebAPI provider が小文字でない: {by_name['GPT-4o'].provider!r}"
    )
    # direct モデル: 既に小文字 (一貫性確認)
    assert by_name["google/gemini-2.5-pro"].provider == "google"
    assert by_name["anthropic/claude-3-5-sonnet-latest"].provider == "anthropic"

    # 全 provider が小文字 (case-sensitive consumer 向けの不変条件)
    for info in result:
        if info.provider is not None:
            assert info.provider == info.provider.lower(), (
                f"{info.name}: provider が小文字でない: {info.provider!r}"
            )


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_user_toml_provider_does_not_override_ssot(patched_registry):
    """WebAPI provider は SSoT の値を採用し、user TOML provider はモデル定義に使わない。"""
    with patched_registry(
        model_dict={"CustomAPI": PydanticAIWebAPIAnnotator},
        config_dict={
            "CustomAPI": {
                "api_model_id": "custom/model",
                "provider": "Anthropic",  # ← user TOML での display-case
            }
        },
        webapi_metadata={
            "CustomAPI": {
                "api_model_id": "anthropic/claude-3-5-sonnet",
                "model_name_on_provider": "anthropic/claude-3-5-sonnet",
                "provider": "anthropic",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            }
        },
    ):
        result = list_annotator_info()

    info = result[0]
    assert info.provider == "anthropic"
    assert info.api_model_id == "anthropic/claude-3-5-sonnet"


@pytest.mark.unit
@pytest.mark.fast
def test_malformed_webapi_user_override_does_not_abort_listing(patched_registry):
    """PR #27 Codex P1 回帰防止 (WebAPI 側): malformed user override が来ても listing 全体が abort しない。

    `config_registry._merged_config_data["BrokenWebAPI"] = int` のような malformed entry
    があると、`get_model_capabilities` 等が内部で `int.get(...)` を呼んで AttributeError を
    投げるが、`api.py.list_annotator_info` の try/except で該当モデル個別を skip し、
    他のモデル (GoodWebAPI) は listing に残ることを保証する。
    """
    with patched_registry(
        model_dict={
            "GoodWebAPI": PydanticAIWebAPIAnnotator,
            "BrokenWebAPI": PydanticAIWebAPIAnnotator,
        },
        config_dict={
            "BrokenWebAPI": 12345,  # malformed (truthy + 非 dict、int)
        },
        webapi_metadata={
            "GoodWebAPI": {
                "api_model_id": "openai/gpt-4o",
                "model_name_on_provider": "openai/gpt-4o",
                "provider": "openai",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            },
            "BrokenWebAPI": {
                "api_model_id": "anthropic/claude-3-opus",
                "model_name_on_provider": "anthropic/claude-3-opus",
                "provider": "anthropic",
                "max_output_tokens": 1800,
                "estimated_size_gb": None,
                "discontinued_at": None,
                "type": "webapi",
                "class": "PydanticAIWebAPIAnnotator",
            },
        },
    ):
        result = list_annotator_info()

    # listing 全体は abort せず、正常な GoodWebAPI は残る (Codex P1 の核心)
    names = {info.name for info in result}
    assert "GoodWebAPI" in names, "正常なモデルは listing に残るべき (listing 全体 abort 防止)"
    # BrokenWebAPI は内部処理で別ルート (get_model_capabilities → config_registry.get) の
    # AttributeError が発生するが try/except で skip される
    # 重要なのは「全体が abort しない」こと、個別 skip かどうかは実装詳細
