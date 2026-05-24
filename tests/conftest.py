"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

# Import shared fixtures from fixtures module
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

# ADR 0023 Phase 1: `IMAGE_ANNOTATOR_SKIP_API_DISCOVERY` フラグは廃止された
# (LiteLLM 同梱 DB は network 通信を必要としないため意味が消失)。
# 過去の `os.environ["IMAGE_ANNOTATOR_SKIP_API_DISCOVERY"] = "true"` 設定は削除済。
# テストで WebAPI モデル登録を抑制したい場合は、当該テストの fixture で
# `image_annotator_lib.core.registry._register_webapi_models_from_discovery` を
# monkeypatch すること。

# Add the tests directory to sys.path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Temporarily disable shared fixtures for testing
# from unit.fixtures.mock_libraries import *
# from unit.fixtures.mock_configs import *
# from unit.fixtures.mock_components import *
# from unit.fixtures.shared_fixtures import *


def _remove_test_config_entries(config_registry) -> None:
    """共有 config registry からテスト専用モデルを取り除く。"""
    for config_store_name in (
        "_system_config_data",
        "_user_config_data",
        "_merged_config_data",
    ):
        config_store = getattr(config_registry, config_store_name, {})
        if isinstance(config_store, dict):
            test_models = [k for k in config_store if "test" in k.lower() or k == "dummy-model"]
            for model in test_models:
                config_store.pop(model, None)


@pytest.fixture(autouse=True)
def isolate_system_config(monkeypatch, tmp_path):
    """テスト中の system config 永続化先を一時ファイルに隔離する。"""
    import copy

    from image_annotator_lib.core import config as config_module
    from image_annotator_lib.core.config import _load_config_from_file, config_registry

    original_system_path = config_registry._system_config_path
    original_user_path = config_registry._user_config_path
    original_system_config = copy.deepcopy(config_registry._system_config_data)
    original_user_config = copy.deepcopy(config_registry._user_config_data)
    original_merged_config = copy.deepcopy(config_registry._merged_config_data)

    isolated_system_path = tmp_path / "config" / "annotator_config.toml"
    isolated_user_path = tmp_path / "config" / "user_config.toml"
    isolated_system_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        config_module,
        "DEFAULT_PATHS",
        {
            **config_module.DEFAULT_PATHS,
            "config_toml": isolated_system_path,
            "user_config_toml": isolated_user_path,
        },
    )
    config_registry._system_config_path = isolated_system_path
    config_registry._user_config_path = isolated_user_path
    _load_config_from_file.cache_clear()

    yield

    config_registry._system_config_path = original_system_path
    config_registry._user_config_path = original_user_path
    config_registry._system_config_data = original_system_config
    config_registry._user_config_data = original_user_config
    config_registry._merged_config_data = original_merged_config
    _load_config_from_file.cache_clear()


@pytest.fixture(autouse=True)
def reset_global_state(request):
    """各テスト前後でグローバル状態をリセット"""
    # BDDテストの場合のみグローバル状態のクリアを制限
    # (BDDテストでは各ステップ間でレジストリの状態を保持する必要がある)
    is_bdd_test = (
        hasattr(request, "node")
        and hasattr(request.node, "name")
        and "test_bdd_runner.py" in str(request.node.fspath)
    )

    # テスト前のセットアップ
    yield

    # BDDテスト以外は全てクリーンアップを実行
    if not is_bdd_test:
        # テスト後のクリーンアップ
        # ADR 0023 Phase 1: PydanticAIAgentFactory / ProviderManager の Agent キャッシュは廃止された
        # ため、conftest 側でのキャッシュクリアは不要。registry / ModelLoad / config のみ reset する。
        try:
            from image_annotator_lib.core.config import config_registry
            from image_annotator_lib.core.model_factory import ModelLoad
            from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY

            # レジストリクリア
            _MODEL_CLASS_OBJ_REGISTRY.clear()

            # ModelLoadキャッシュクリア
            if hasattr(ModelLoad, "_instance_cache"):
                ModelLoad._instance_cache.clear()
            if hasattr(ModelLoad, "_model_cache"):
                ModelLoad._model_cache.clear()

            # 設定レジストリクリア
            if hasattr(config_registry, "_config_cache"):
                config_registry._config_cache.clear()
            if hasattr(config_registry, "_config"):
                # テスト用の設定をクリア（システムデフォルトは保持）
                test_models = [k for k in config_registry._config.keys() if "test" in k.lower()]
                for model in test_models:
                    config_registry._config.pop(model, None)
            _remove_test_config_entries(config_registry)

        except ImportError:
            # モジュールがまだロードされていない場合はスキップ
            pass

        # 環境変数のクリーンアップは各テストの@patch.dictに任せる
        # conftest.pyでの環境変数復元は@patch.dictと競合するため削除


resources_dir = Path("tests") / "resources"


@pytest.fixture(scope="session")
def load_image_files() -> Callable[[int], list[Image.Image]]:
    """指定された枚数の画像ファイルをリストとして読み込む関数を返す"""

    def _load_images(count: int = 1) -> list[Image.Image]:
        image_path = resources_dir / "img" / "1_img"
        files = list(image_path.rglob("*.webp"))

        # 指定された枚数だけファイルを取得(ディレクトリ内のファイル数を超えないように)
        count = min(count, len(files))
        files = files[:count]

        # すべての画像をリストに格納して返す
        return [Image.open(file) for file in files if file.exists()]

    return _load_images


# ==============================================================================
# Phase A Task 0: Test Infrastructure Fixtures (2025-12-03)
# ==============================================================================


@pytest.fixture
def managed_config_registry():
    """テスト用設定レジストリfixture

    テスト用のモデル設定を一時的に登録し、テスト終了時に自動クリーンアップする。
    グローバルな config_registry インスタンスを操作するため、並列実行には注意が必要。

    API:
        registry.set(model_name: str, config_dict: dict)
            指定されたモデル名で設定を一括登録
            Example: registry.set("test_model", {"model_path": "path/to/model", "device": "cpu"})

        registry.get(model_name: str, key: str, default: Any) -> Any
            設定値を取得（config_registry.get()の wrapper）

    Usage:
        def test_foo(managed_config_registry):
            managed_config_registry.set("test_model", {
                "model_path": "test/path",
                "device": "cpu",
                "estimated_size_gb": 1.0,
            })
            # Test with registered config
            annotator = ConcreteAnnotator("test_model")
            assert annotator.device == "cpu"
    """
    # テスト開始前のユーザー設定データを保存（Deep Copy）
    import copy

    from image_annotator_lib.core.config import config_registry

    original_user_config = copy.deepcopy(config_registry._user_config_data)
    original_merged_config = copy.deepcopy(config_registry._merged_config_data)

    # Registry wrapper with convenient API
    class ConfigRegistryWrapper:
        """config_registry への便利なラッパー"""

        def set(self, model_name: str, config_dict: dict) -> None:
            """モデル設定を一括登録（内部的には各キーを個別に set）"""
            for key, value in config_dict.items():
                config_registry.set(model_name, key, value)

        def get(self, model_name: str, key: str, default: object = None) -> object:
            """設定値を取得"""
            return config_registry.get(model_name, key, default)

    wrapper = ConfigRegistryWrapper()

    yield wrapper

    # テスト終了後: ユーザー設定データを元に戻す
    config_registry._user_config_data = original_user_config
    config_registry._merged_config_data = original_merged_config


@pytest.fixture
def mock_model_components():
    """モックモデルコンポーネント（Pipeline/Transformers用）

    Returns:
        dict: モックされた model, processor, pipeline コンポーネント
    """
    from unittest.mock import MagicMock

    return {
        "model": MagicMock(),
        "processor": MagicMock(),
        "pipeline": MagicMock(),
    }


@pytest.fixture
def mock_cuda_available(monkeypatch):
    """CUDA利用可能環境のモック

    torch.cuda.is_available() が True を返すようにモンキーパッチ。
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)


@pytest.fixture
def mock_cuda_unavailable(monkeypatch):
    """CUDA利用不可環境のモック

    torch.cuda.is_available() が False を返すようにモンキーパッチ。
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)


@pytest.fixture
def lightweight_test_images():
    """軽量テスト画像セット（ユニット・統合テスト共通）

    各画像は異なる色を持ち、一意の pHash を保証する。

    Returns:
        list[Image.Image]: 3つの RGB テスト画像 (64x64)
    """
    images = []
    for i, color in enumerate(["red", "green", "blue"]):
        img = Image.new("RGB", (64, 64), color)
        # Add a single different pixel to each image to ensure unique phash
        img.putpixel((i, i), (255, 255, 255))
        images.append(img)
    return images
