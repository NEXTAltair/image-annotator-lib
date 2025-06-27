"""
共通フィクスチャ
"""

import time
from unittest.mock import MagicMock

import pytest
from PIL import Image


@pytest.fixture
def test_images():
    """テスト用画像リスト"""
    images = []
    for i in range(3):
        img = MagicMock(spec=Image.Image)
        img.size = (224, 224)
        img.mode = "RGB"
        images.append(img)
    return images


@pytest.fixture
def test_phash_list():
    """テスト用pHashリスト"""
    return ["hash1", "hash2", "hash3"]


@pytest.fixture
def test_annotation_result():
    """標準的なアノテーション結果"""
    return {
        "phash": "test_hash",
        "tags": ["tag1", "tag2"],
        "formatted_output": {"caption": "test caption"},
        "error": None,
    }


@pytest.fixture(autouse=True)
def track_test_performance(request):
    """テスト実行時間を追跡"""
    start = time.time()
    yield
    duration = time.time() - start
    if duration > 1.0:  # 1秒以上のテストを記録
        print(f"SLOW TEST: {request.node.name} ({duration:.2f}s)")


@pytest.fixture
def mock_memory_monitor():
    """メモリ監視のモック"""
    monitor_mock = MagicMock()
    monitor_mock.available = 8 * 1024 * 1024 * 1024  # 8GB
    monitor_mock.total = 16 * 1024 * 1024 * 1024  # 16GB
    monitor_mock.percent = 50.0
    return monitor_mock


@pytest.fixture
def mock_webapi_client():
    """WebAPI クライアントの標準モック"""
    client_mock = MagicMock()

    # 標準的なレスポンス
    response_mock = MagicMock()
    response_mock.data = {"tags": ["test_tag1", "test_tag2"], "caption": "test caption"}

    client_mock.messages.create.return_value = response_mock
    client_mock.run_sync.return_value = response_mock

    return client_mock


def assert_no_real_imports():
    """実際のMLライブラリがインポートされていないことを確認"""
    import sys
    from unittest.mock import MagicMock

    forbidden_modules = ["torch", "transformers", "tensorflow", "onnxruntime"]
    for module in forbidden_modules:
        if module in sys.modules:
            assert isinstance(sys.modules[module], MagicMock), f"Real {module} module detected!"


def create_standard_api_mock():
    """標準的なAPI モックを作成"""
    api_mock = MagicMock()
    api_mock.predict.return_value = [
        {
            "phash": "test_hash",
            "tags": ["api_tag1", "api_tag2"],
            "formatted_output": {"caption": "api caption"},
            "error": None,
        }
    ]
    return api_mock
