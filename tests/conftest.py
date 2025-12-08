"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

# Import shared fixtures from fixtures module
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

# テスト環境ではAPI検出を無効化
os.environ["IMAGE_ANNOTATOR_SKIP_API_DISCOVERY"] = "true"

# Add the tests directory to sys.path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Temporarily disable shared fixtures for testing
# from unit.fixtures.mock_libraries import *
# from unit.fixtures.mock_configs import *
# from unit.fixtures.mock_components import *
# from unit.fixtures.shared_fixtures import *


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
        try:
            from image_annotator_lib.core.config import config_registry
            from image_annotator_lib.core.model_factory import ModelLoad
            from image_annotator_lib.core.provider_manager import ProviderManager
            from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
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

            # ProviderManagerキャッシュクリア
            if hasattr(ProviderManager, "_provider_cache"):
                ProviderManager._provider_cache.clear()
            if hasattr(ProviderManager, "_agent_cache"):
                ProviderManager._agent_cache.clear()

            # PydanticAIProviderFactoryキャッシュクリア
            if hasattr(PydanticAIProviderFactory, "_provider_cache"):
                PydanticAIProviderFactory._provider_cache.clear()
            if hasattr(PydanticAIProviderFactory, "_agent_cache"):
                PydanticAIProviderFactory._agent_cache.clear()

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
