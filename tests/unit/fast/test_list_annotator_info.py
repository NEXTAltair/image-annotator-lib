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
            webapi_metadata: ``_WEBAPI_MODEL_METADATA`` を上書きする辞書。
                discovery 経由で登録された WebAPI モデルのメタデータ検証で使う。
        """
        config_dict = config_dict or {}
        direct_models = direct_models or []
        webapi_metadata = webapi_metadata or {}

        # _config が dict の場合はそれを書き換え、なければ get_all_config を patch
        return _PatchedRegistryCtx(model_dict, config_dict, direct_models, webapi_metadata)

    return _setup


class _PatchedRegistryCtx:
    """_MODEL_CLASS_OBJ_REGISTRY と get_all_config と agent_factory を一括 patch するコンテキスト。"""

    def __init__(self, model_dict: dict, config_dict: dict, direct_models: list[str], webapi_metadata: dict):
        self.model_dict = model_dict
        self.config_dict = config_dict
        self.direct_models = direct_models
        self.webapi_metadata = webapi_metadata
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
    # Phase 2: ローカルモデルは provider="local"、API 関連は None
    assert info.provider == "local"
    assert info.api_model_id is None
    assert info.max_output_tokens is None


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
    # Phase 2: config に provider がない WebAPI モデルは provider=None
    assert info.provider is None
    assert info.api_model_id == "claude-3-opus-20240229"


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
    # Phase 2: provider は "google/..." の slash 前から推論
    assert info.provider == "google"
    assert info.api_model_id == direct_id


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


# ---------------------------------------------------------------------------
# Phase 2: 詳細メタデータフィールドのテスト
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.fast
def test_local_model_has_estimated_size_gb(patched_registry):
    """ローカルモデルは estimated_size_gb が config から取れる。"""
    with patched_registry(
        model_dict={"aesthetic-scorer": _DummyScorer},
        config_dict={
            "aesthetic-scorer": {"type": "scorer", "estimated_size_gb": 4.065, "capabilities": ["scores"]}
        },
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.estimated_size_gb == pytest.approx(4.065)
    assert info.provider == "local"
    assert info.api_model_id is None
    assert info.max_output_tokens is None


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_with_provider_and_max_tokens(patched_registry):
    """config に provider / api_model_id / max_output_tokens がある WebAPI モデルは正しく取れる。"""
    with patched_registry(
        model_dict={"GPT-4o": PydanticAIWebAPIAnnotator},
        config_dict={
            "GPT-4o": {
                "provider": "openai",
                "api_model_id": "gpt-4o-2024-11-20",
                "max_output_tokens": 1800,
                "type": "vision",
                "capabilities": ["tags", "captions", "scores"],
            }
        },
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.provider == "openai"
    assert info.api_model_id == "gpt-4o-2024-11-20"
    assert info.max_output_tokens == 1800
    assert info.is_api is True
    assert info.estimated_size_gb is None


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    "model_id,expected_provider",
    [
        ("google/gemini-2.5-pro", "google"),
        ("anthropic/claude-3-7-sonnet", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("openrouter/google/gemini-flash", "openrouter"),
    ],
)
def test_pydanticai_direct_model_provider_inferred(patched_registry, model_id: str, expected_provider: str):
    """PydanticAI 直接モデルは model_id の slash 前から provider を推論する。"""
    with patched_registry(model_dict={}, config_dict={}, direct_models=[model_id]):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    assert info.provider == expected_provider
    assert info.api_model_id == model_id


@pytest.mark.unit
@pytest.mark.fast
def test_discontinued_at_none_for_active_model(patched_registry):
    """現役モデルの discontinued_at は None。"""
    with patched_registry(
        model_dict={"active-tagger": _DummyTagger},
        config_dict={"active-tagger": {"type": "tagger", "capabilities": ["tags"]}},
    ):
        result = list_annotator_info()

    assert result[0].discontinued_at is None


# --- safe 型変換ヘルパー (Codex P2 review for #22) ---


@pytest.mark.unit
@pytest.mark.fast
def test_safe_float_returns_none_for_invalid_value():
    """malformed な estimated_size_gb は None フォールバック (モデル消失防止)。"""
    from image_annotator_lib.core.registry import _safe_float

    assert _safe_float(None, "m", "estimated_size_gb") is None
    assert _safe_float("1.5", "m", "estimated_size_gb") == 1.5
    assert _safe_float(2.0, "m", "estimated_size_gb") == 2.0
    assert _safe_float("not-a-number", "m", "estimated_size_gb") is None
    assert _safe_float([1, 2], "m", "estimated_size_gb") is None


@pytest.mark.unit
@pytest.mark.fast
def test_safe_int_returns_none_for_invalid_value():
    """quoted '1800.0' のような int で parse 不能な値は None フォールバック。"""
    from image_annotator_lib.core.registry import _safe_int

    assert _safe_int(None, "m", "max_output_tokens") is None
    assert _safe_int("1800", "m", "max_output_tokens") == 1800
    assert _safe_int(2048, "m", "max_output_tokens") == 2048
    # int("1800.0") は ValueError → None フォールバック
    assert _safe_int("1800.0", "m", "max_output_tokens") is None
    assert _safe_int("abc", "m", "max_output_tokens") is None


@pytest.mark.unit
@pytest.mark.fast
def test_parse_discontinued_at_normalizes_quoted_string():
    """quoted TOML 文字列 (ISO 8601) は datetime に正規化。"""
    import datetime as _dt

    from image_annotator_lib.core.registry import _parse_discontinued_at

    # ネイティブ datetime はそのまま
    native = _dt.datetime(2025, 12, 31, tzinfo=_dt.UTC)
    assert _parse_discontinued_at(native, "m") is native

    # 普通の ISO 文字列
    parsed = _parse_discontinued_at("2025-12-31T00:00:00", "m")
    assert isinstance(parsed, _dt.datetime)
    assert parsed.year == 2025

    # "Z" suffix → UTC として扱う
    parsed_utc = _parse_discontinued_at("2025-12-31T00:00:00Z", "m")
    assert isinstance(parsed_utc, _dt.datetime)
    assert parsed_utc.tzinfo is not None

    # None はそのまま
    assert _parse_discontinued_at(None, "m") is None

    # malformed 文字列 → None
    assert _parse_discontinued_at("not-a-date", "m") is None

    # 予期しない型 → None
    assert _parse_discontinued_at(123, "m") is None


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_model_merges_discovery_metadata(patched_registry):
    """discovery 経由 (`_WEBAPI_MODEL_METADATA`) と config_registry の値を merge した上で
    AnnotatorInfo を構築する (Codex P2 #22 第 3 指摘)。

    `_register_webapi_models_from_discovery()` は provider/max_output_tokens を
    `_WEBAPI_MODEL_METADATA` に保存するが、`set_system_value()` は class/api_model_id
    のみを config_registry に永続化する。両方を merge しないと Phase 2 メタデータが
    silently dropped される。
    """
    # config_registry は class/api_model_id のみ
    user_config = {
        "openai/gpt-4o": {
            "class": "PydanticAIWebAPIAnnotator",
            "api_model_id": "gpt-4o",
            "type": "vision",
            "capabilities": ["tags", "captions"],
        }
    }
    # discovery は provider/max_output_tokens を含む
    discovery_meta = {
        "openai/gpt-4o": {
            "class": "PydanticAIWebAPIAnnotator",
            "api_model_id": "gpt-4o",
            "provider": "openai",
            "max_output_tokens": 1800,
            "type": "vision",
            "capabilities": ["tags", "captions"],
        }
    }
    with patched_registry(
        model_dict={"openai/gpt-4o": PydanticAIWebAPIAnnotator},
        config_dict=user_config,
        webapi_metadata=discovery_meta,
    ):
        result = list_annotator_info()

    assert len(result) == 1
    info = result[0]
    # discovery のメタデータが反映されている (両方を merge した結果)
    assert info.provider == "openai"
    assert info.max_output_tokens == 1800
    assert info.api_model_id == "gpt-4o"
    assert info.is_api is True


@pytest.mark.unit
@pytest.mark.fast
def test_user_config_overrides_discovery_metadata(patched_registry):
    """config_registry の値は discovery メタデータより優先される (merge 優先順位)。"""
    user_config = {
        "openai/gpt-4o": {
            "class": "PydanticAIWebAPIAnnotator",
            "api_model_id": "gpt-4o",
            "provider": "user-override",  # ユーザー側で provider を上書き
            "type": "vision",
            "capabilities": ["tags"],
        }
    }
    discovery_meta = {
        "openai/gpt-4o": {
            "class": "PydanticAIWebAPIAnnotator",
            "api_model_id": "gpt-4o",
            "provider": "openai",  # discovery 側はデフォルト
            "max_output_tokens": 2048,
            "type": "vision",
            "capabilities": ["tags"],
        }
    }
    with patched_registry(
        model_dict={"openai/gpt-4o": PydanticAIWebAPIAnnotator},
        config_dict=user_config,
        webapi_metadata=discovery_meta,
    ):
        result = list_annotator_info()

    info = result[0]
    # ユーザー設定の provider が discovery を上書き
    assert info.provider == "user-override"
    # ユーザー設定にない max_output_tokens は discovery から取得
    assert info.max_output_tokens == 2048


@pytest.mark.unit
@pytest.mark.fast
def test_list_annotator_info_skips_only_malformed_model_section(patched_registry):
    """1 モデルの config セクションが非マッピング (scalar/list) でも、
    他モデルの listing は継続される (Codex P2 #22 第 4 指摘)。

    malformed: TOML で `[model.section]` の代わりに `model.section = "scalar"` のように
    書かれた場合、`{**user_config}` のスプレッドが TypeError を投げて関数全体が abort
    する問題があった。merge を try ブロックに含めて per-model 失敗にする。
    """
    # 2 モデル: "good" は dict、"malformed" は scalar (TOML の typo を想定)
    user_config = {
        "good-tagger": {"type": "tagger", "capabilities": ["tags"]},
        "malformed-tagger": "this-should-be-a-dict",  # 不正な型
    }
    with patched_registry(
        model_dict={"good-tagger": _DummyTagger, "malformed-tagger": _DummyTagger},
        config_dict=user_config,
    ):
        result = list_annotator_info()

    # malformed-tagger は build 失敗で skip されるが、good-tagger は listing に残る
    names = [info.name for info in result]
    assert "good-tagger" in names
    # malformed の挙動: 非マッピングは warning + 空 dict 扱いされ、build には進む
    # ただし最低限の config (capabilities/type) がないため capabilities フォールバックで
    # 構築されるか、もしくは build 失敗で skip。どちらでも他モデルが残ることが要件。
    # ここでは少なくとも 1 件以上残ることを保証。
    assert len(result) >= 1


@pytest.mark.unit
@pytest.mark.fast
def test_list_annotator_info_keeps_model_with_invalid_metadata(patched_registry):
    """estimated_size_gb / max_output_tokens / discontinued_at が malformed でも
    モデル自体は listing に残り、該当フィールドのみ None になる (Codex P2 #22)。"""
    with patched_registry(
        model_dict={"local-tagger": _DummyTagger},
        config_dict={
            "local-tagger": {
                "type": "tagger",
                "capabilities": ["tags"],
                "estimated_size_gb": "not-a-number",
                "max_output_tokens": "1800.0",  # int() では ValueError
                "discontinued_at": "not-a-date",
            }
        },
    ):
        result = list_annotator_info()

    # モデルは listing から消えていない
    assert len(result) == 1
    info = result[0]
    assert info.name == "local-tagger"
    # 不正値はすべて None フォールバック
    assert info.estimated_size_gb is None
    assert info.max_output_tokens is None
    assert info.discontinued_at is None
