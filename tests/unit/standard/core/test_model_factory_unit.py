# tests/unit/test_model_factory_unit.py
import time
from unittest.mock import patch

import pytest

# テスト対象のクラスをインポート
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.exceptions.errors import OutOfMemoryError


@pytest.fixture(autouse=True)
def clear_model_load_state():
    """各テストの前に ModelLoad のクラス変数をクリアするフィクスチャ"""
    ModelLoad._MODEL_STATES.clear()
    ModelLoad._MEMORY_USAGE.clear()
    ModelLoad._MODEL_LAST_USED.clear()
    ModelLoad._MODEL_SIZES.clear()
    yield  # テスト実行


# --- State Management and Helpers ---


@pytest.mark.unit
def test_update_model_state_loaded():
    """_update_model_state が 'loaded' 状態を正しく設定するかのテスト"""
    model_name = "test_model"
    ModelLoad._update_model_state(model_name, device="cuda", status="loaded", size_mb=100.0)

    assert ModelLoad._get_model_state(model_name) == "on_cuda"
    assert ModelLoad._get_model_memory_usage(model_name) == 100.0
    assert model_name in ModelLoad._MODEL_LAST_USED


@pytest.mark.unit
def test_update_model_state_released():
    """_update_model_state が 'released' 状態を正しく処理するかのテスト"""
    model_name = "test_model"
    # 事前に状態を設定
    ModelLoad._update_model_state(model_name, device="cuda", status="loaded", size_mb=100.0)
    assert ModelLoad._get_model_state(model_name) is not None

    # 解放
    ModelLoad._update_model_state(model_name, status="released")

    assert ModelLoad._get_model_state(model_name) is None
    assert ModelLoad._get_model_memory_usage(model_name) == 0.0
    assert model_name not in ModelLoad._MODEL_LAST_USED


@pytest.mark.unit
def test_release_model_state():
    """_release_model_state が状態を完全にクリーンアップするかのテスト"""
    model_name = "test_model"
    ModelLoad._update_model_state(model_name, device="cpu", status="cached_cpu", size_mb=50.0)
    ModelLoad._MODEL_SIZES[model_name] = 50.0  # サイズキャッシュも設定

    ModelLoad._release_model_state(model_name)

    assert model_name not in ModelLoad._MODEL_STATES
    assert model_name not in ModelLoad._MEMORY_USAGE
    assert model_name not in ModelLoad._MODEL_LAST_USED
    # _MODEL_SIZES は解放されないことを確認
    assert model_name in ModelLoad._MODEL_SIZES


@pytest.mark.unit
def test_get_models_sorted_by_last_used():
    """_get_models_sorted_by_last_used が最終使用時刻でソートするかテスト"""
    now = time.time()
    ModelLoad._MODEL_LAST_USED["model_c"] = now
    ModelLoad._MODEL_LAST_USED["model_a"] = now - 200
    ModelLoad._MODEL_LAST_USED["model_b"] = now - 100

    sorted_models = ModelLoad._get_models_sorted_by_last_used()
    assert [name for name, ts in sorted_models] == ["model_a", "model_b", "model_c"]


# --- Config and Size Management ---


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.config_registry")
def test_get_model_size_from_config_success(mock_config_registry):
    """config からモデルサイズを正常に取得するテスト"""
    mock_config_registry.get.return_value = 2.5  # 2.5 GB
    size_mb = ModelLoad._get_model_size_from_config("test_model")
    assert size_mb == 2.5 * 1024
    mock_config_registry.get.assert_called_once_with("test_model", "estimated_size_gb")


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.config_registry")
def test_get_model_size_from_config_not_found(mock_config_registry):
    """config にモデルが存在しない場合のテスト"""
    mock_config_registry.get.side_effect = KeyError
    size_mb = ModelLoad._get_model_size_from_config("unknown_model")
    assert size_mb is None


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.config_registry")
def test_save_size_to_config(mock_config_registry):
    """計算したモデルサイズを config に保存するテスト"""
    ModelLoad._save_size_to_config("test_model", 2048)  # 2GB
    mock_config_registry.set_system_value.assert_called_once_with("test_model", "estimated_size_gb", 2.0)
    mock_config_registry.save_system_config.assert_called_once()


# --- Error Handling ---


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.logger")
def test_handle_load_error_memory_error(mock_logger):
    """メモリ不足エラーを正しく処理するかのテスト"""
    model_name = "oom_model"
    ModelLoad._MODEL_SIZES[model_name] = 0.0  # 計算失敗したと仮定
    error = OutOfMemoryError("Test OOM")

    ModelLoad._handle_load_error(model_name, error)

    mock_logger.error.assert_any_call(f"メモリ不足エラー: モデル '{model_name}' ロード中。詳細: {error}")
    # 計算失敗したサイズキャッシュ(0.0)は削除されるはず
    assert model_name not in ModelLoad._MODEL_SIZES
    # 状態も解放されているはず
    assert ModelLoad._get_model_state(model_name) is None


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.logger")
def test_handle_load_error_file_not_found(mock_logger):
    """ファイル未検出エラーを正しく処理するかのテスト"""
    model_name = "not_found_model"
    error = FileNotFoundError("File not found")

    ModelLoad._handle_load_error(model_name, error)

    mock_logger.error.assert_any_call(
        f"ファイル未検出: モデル '{model_name}' ロード中: {error}", exc_info=False
    )
    assert ModelLoad._get_model_state(model_name) is None


# --- _clear_cache_internal のテスト ---

# psutil.virtual_memory のモック用設定
# total=20GB, available=15GB -> max_cache=10GB (ratio=0.5)
MOCK_VIRTUAL_MEMORY = type(
    "svmem",
    (object,),
    {
        "total": 20 * 1024**3,
        "available": 15 * 1024**3,
        "percent": 25.0,
        "used": 5 * 1024**3,
        "free": 15 * 1024**3,
        "active": 0,
        "inactive": 0,
        "buffers": 0,
        "cached": 0,
        "shared": 0,
    },
)()


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_check_memory_before_load_sufficient(mock_psutil_vm):
    """十分なメモリがある場合の _check_memory_before_load のテスト"""
    # 利用可能メモリは 15GB
    assert ModelLoad._check_memory_before_load(10 * 1024, "model_10gb") is True


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_check_memory_before_load_insufficient(mock_psutil_vm):
    """メモリが不足している場合の _check_memory_before_load のテスト"""
    # 利用可能メモリは 15GB
    assert ModelLoad._check_memory_before_load(16 * 1024, "model_16gb") is False


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_clear_cache_internal_no_release(mock_psutil_vm):
    """キャッシュに十分空きがあり、解放が不要な場合のテスト"""
    # 事前準備: ModelLoad の状態を設定
    ModelLoad._MEMORY_USAGE["model_a"] = 2 * 1024  # 2GB
    ModelLoad._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    # 合計 5GB 使用中、上限は 10GB

    new_model_size_mb = 4 * 1024  # 4GB のモデルをロードしようとする

    # テスト対象メソッド呼び出し (静的メソッドなのでクラスから直接呼び出す)
    result = ModelLoad._clear_cache_internal("new_model", new_model_size_mb)

    # 検証: 何も解放されておらず、成功(True)が返されること
    assert result is True
    assert "model_a" in ModelLoad._MEMORY_USAGE
    assert "model_b" in ModelLoad._MEMORY_USAGE
    assert len(ModelLoad._MEMORY_USAGE) == 2


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_clear_cache_internal_release_oldest(mock_psutil_vm):
    """キャッシュが不足し、最も古いモデルが解放されるテスト"""
    # 事前準備: 状態設定 (model_a が古い)
    ModelLoad._MEMORY_USAGE["model_a"] = 6 * 1024  # 6GB
    ModelLoad._MEMORY_USAGE["model_b"] = 3 * 1024  # 3GB
    ModelLoad._MODEL_LAST_USED["model_a"] = time.time() - 100
    ModelLoad._MODEL_LAST_USED["model_b"] = time.time()
    # model_a と model_b の状態も設定しておく
    ModelLoad._MODEL_STATES["model_a"] = "on_cpu"
    ModelLoad._MODEL_STATES["model_b"] = "on_cpu"
    # 合計 9GB 使用中、上限 10GB

    new_model_size_mb = 2 * 1024  # 2GB のモデルをロード -> 合計 11GB で超過

    # _release_model_state が呼ばれることを確認するためにモック化
    with patch.object(
        ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
    ) as mock_release:
        result = ModelLoad._clear_cache_internal("new_model", new_model_size_mb)

    # 検証: 成功(True)が返されること
    assert result is True
    # 検証: 最も古い model_a が解放されたこと
    mock_release.assert_called_once_with("model_a")
    # 検証: 実際の状態が変更されたこと
    assert "model_a" not in ModelLoad._MEMORY_USAGE
    assert "model_a" not in ModelLoad._MODEL_STATES
    assert "model_b" in ModelLoad._MEMORY_USAGE  # model_b は残っている


@pytest.mark.unit
@patch("image_annotator_lib.core.model_factory.psutil.virtual_memory", return_value=MOCK_VIRTUAL_MEMORY)
def test_clear_cache_internal_insufficient_space(mock_psutil_vm):
    """モデルを解放しても十分なスペースが確保できない場合のテスト"""
    # 事前準備: 状態設定
    ModelLoad._MEMORY_USAGE["model_a"] = 8 * 1024  # 8GB
    ModelLoad._MODEL_LAST_USED["model_a"] = time.time() - 100
    ModelLoad._MODEL_STATES["model_a"] = "on_cpu"
    # 合計 8GB 使用中、上限 10GB

    new_model_size_mb = 12 * 1024  # 12GB の巨大なモデルをロードしようとする
    # model_a(8GB)を解放しても 12GB は入らない

    with patch.object(
        ModelLoad, "_release_model_state", wraps=ModelLoad._release_model_state
    ) as mock_release:
        result = ModelLoad._clear_cache_internal("new_model", new_model_size_mb)

    # 検証: 失敗(False)が返されること
    assert result is False
    # 検証: model_a は解放が試みられること
    mock_release.assert_called_once_with("model_a")
    # 検証: 最終的に model_a の状態は解放されていること
    assert "model_a" not in ModelLoad._MEMORY_USAGE
    assert "model_a" not in ModelLoad._MODEL_STATES
