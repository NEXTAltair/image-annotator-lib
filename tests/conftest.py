"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

# Import shared fixtures from fixtures module
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

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
    # BDDテストの場合はグローバル状態をクリアしない
    # (BDDテストでは各ステップ間でレジストリの状態を保持する必要がある)
    is_bdd_test = (
        hasattr(request, 'node') and 
        hasattr(request.node, 'name') and 
        'test_bdd_runner.py' in str(request.node.fspath)
    )
    
    # テスト前のセットアップ
    yield
    
    # BDDテストでない場合のみクリーンアップを実行
    if not is_bdd_test:
        # テスト後のクリーンアップ
        # レジストリとキャッシュをクリア
        try:
            from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY
            from image_annotator_lib.core.model_factory import ModelLoad
            from image_annotator_lib.core.config import config_registry
            from image_annotator_lib.core.provider_manager import ProviderManager
            
            # レジストリクリア
            _MODEL_CLASS_OBJ_REGISTRY.clear()
            
            # ModelLoadキャッシュクリア
            if hasattr(ModelLoad, '_instance_cache'):
                ModelLoad._instance_cache.clear()
            if hasattr(ModelLoad, '_model_cache'):
                ModelLoad._model_cache.clear()
                
            # 設定レジストリクリア
            if hasattr(config_registry, '_config_cache'):
                config_registry._config_cache.clear()
                
            # ProviderManagerキャッシュクリア
            if hasattr(ProviderManager, '_provider_cache'):
                ProviderManager._provider_cache.clear()
            if hasattr(ProviderManager, '_agent_cache'):
                ProviderManager._agent_cache.clear()
                
        except ImportError:
            # モジュールがまだロードされていない場合はスキップ
            pass

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
