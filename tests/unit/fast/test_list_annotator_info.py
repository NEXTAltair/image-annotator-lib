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

    def _setup(model_dict: dict, config_dict: dict | None = None, direct_models: list[str] | None = None):
        """戻り値: () のコンテキストマネージャ。withで使う。"""
        config_dict = config_dict or {}
        direct_models = direct_models or []

        # _config が dict の場合はそれを書き換え、なければ get_all_config を patch
        return _PatchedRegistryCtx(model_dict, config_dict, direct_models)

    return _setup


class _PatchedRegistryCtx:
    """_MODEL_CLASS_OBJ_REGISTRY と get_all_config と agent_factory を一括 patch するコンテキスト。"""

    def __init__(self, model_dict: dict, config_dict: dict, direct_models: list[str]):
        self.model_dict = model_dict
        self.config_dict = config_dict
        self.direct_models = direct_models
        self._patches: list = []

    def __enter__(self):
        from image_annotator_lib.core import registry
        from image_annotator_lib.core.config import get_config_registry

        self._patches.append(patch.object(registry, "_MODEL_CLASS_OBJ_REGISTRY", self.model_dict))
        self._patches.append(patch.object(registry, "_REGISTRY_INITIALIZED", True))
        self._patches.append(patch.object(registry, "_WEBAPI_MODEL_METADATA", {}))
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
