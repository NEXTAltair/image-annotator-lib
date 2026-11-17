"""core/annotation_runner.py の run_annotation() 直接呼び出しテスト。

api.annotate() 経由のテストは tests/unit/fast/test_api.py が網羅している。
本ファイルは run_annotation() を内部関数として直接呼び出した際の smoke test。
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core import annotation_runner
from image_annotator_lib.core.annotation_runner import run_annotation
from image_annotator_lib.core.types import PHashAnnotationResults, TaskCapability, UnifiedAnnotationResult


@pytest.fixture(autouse=True)
def clear_instance_registry():
    """各テスト前後でインスタンスキャッシュをクリア。"""
    annotation_runner._MODEL_INSTANCE_REGISTRY.clear()
    yield
    annotation_runner._MODEL_INSTANCE_REGISTRY.clear()


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
def test_run_annotation_with_single_model_returns_results(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """run_annotation() が単一モデルで結果を pHash ベース辞書として返す。"""
    mock_get_registry.return_value = {"test-model": MagicMock()}
    mock_calc_phash.return_value = "phash_single"
    mock_get_instance.return_value = MagicMock()
    mock_annotate_model.return_value = [
        UnifiedAnnotationResult(
            model_name="test-model",
            capabilities={TaskCapability.TAGS},
            tags=["tag1"],
        )
    ]

    images = [Image.new("RGB", (100, 100))]
    results = run_annotation(images=images, model_names=["test-model"])

    assert isinstance(results, PHashAnnotationResults)
    assert "phash_single" in results
    assert "test-model" in results["phash_single"]
    assert results["phash_single"]["test-model"].tags == ["tag1"]


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
def test_run_annotation_executes_multiple_models_in_order(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """run_annotation() は model_names の順番通りにモデルを実行する。"""
    mock_get_registry.return_value = {"model_a": MagicMock(), "model_b": MagicMock()}
    mock_calc_phash.return_value = "phash_multi"
    mock_get_instance.side_effect = [MagicMock(), MagicMock()]
    mock_annotate_model.side_effect = [
        [UnifiedAnnotationResult(model_name="model_a", capabilities={TaskCapability.TAGS}, tags=["a"])],
        [UnifiedAnnotationResult(model_name="model_b", capabilities={TaskCapability.TAGS}, tags=["b"])],
    ]

    images = [Image.new("RGB", (100, 100))]
    results = run_annotation(images=images, model_names=["model_a", "model_b"])

    assert "model_a" in results["phash_multi"]
    assert "model_b" in results["phash_multi"]
    # _annotate_model 呼び出しのモデル順序を検証
    assert mock_annotate_model.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
def test_run_annotation_uses_provided_phash_list(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """phash_list が指定された場合、calculate_phash は呼ばれない。"""
    mock_get_registry.return_value = {"test-model": MagicMock()}
    mock_get_instance.return_value = MagicMock()
    mock_annotate_model.return_value = [
        UnifiedAnnotationResult(model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["x"])
    ]

    images = [Image.new("RGB", (100, 100))]
    results = run_annotation(
        images=images, model_names=["test-model"], phash_list=["custom_phash"]
    )

    mock_calc_phash.assert_not_called()
    assert "custom_phash" in results


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
def test_run_annotation_initializes_registry_when_empty(
    mock_init_registry, mock_get_registry, mock_annotate_model, mock_get_instance, mock_calc_phash
):
    """レジストリが空のとき initialize_registry が呼ばれる。"""
    # 1回目: 空, 2回目: 初期化後
    mock_get_registry.side_effect = [{}, {"test-model": MagicMock()}]
    mock_calc_phash.return_value = "phash_init"
    mock_get_instance.return_value = MagicMock()
    mock_annotate_model.return_value = [
        UnifiedAnnotationResult(model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["x"])
    ]

    images = [Image.new("RGB", (100, 100))]
    run_annotation(images=images, model_names=["test-model"])

    mock_init_registry.assert_called_once()
    assert mock_get_registry.call_count == 2


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
def test_run_annotation_records_error_on_model_exception(
    mock_get_capabilities,
    mock_init_registry,
    mock_get_registry,
    mock_annotate_model,
    mock_get_instance,
    mock_calc_phash,
):
    """モデル実行中の例外は error 付き UnifiedAnnotationResult として記録される。"""
    mock_get_registry.return_value = {"test-model": MagicMock()}
    mock_calc_phash.return_value = "phash_err"
    mock_get_instance.return_value = MagicMock()
    mock_annotate_model.side_effect = ValueError("boom")
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    images = [Image.new("RGB", (100, 100))]
    results = run_annotation(images=images, model_names=["test-model"])

    assert "phash_err" in results
    assert "test-model" in results["phash_err"]
    error_result = results["phash_err"]["test-model"]
    assert error_result.error is not None
    assert "ValueError" in error_result.error
    assert "boom" in error_result.error


@pytest.mark.unit
@patch("image_annotator_lib.core.annotation_runner.calculate_phash")
@patch("image_annotator_lib.core.annotation_runner.get_annotator_instance")
@patch("image_annotator_lib.core.annotation_runner._annotate_model")
@patch("image_annotator_lib.core.annotation_runner.get_cls_obj_registry")
@patch("image_annotator_lib.core.annotation_runner.initialize_registry")
@patch("image_annotator_lib.core.utils.get_model_capabilities")
def test_run_annotation_fills_missing_results_with_error(
    mock_get_capabilities,
    mock_init_registry,
    mock_get_registry,
    mock_annotate_model,
    mock_get_instance,
    mock_calc_phash,
):
    """結果数 < 画像数 のとき不足分に error 付きの結果が補完される。"""
    mock_get_registry.return_value = {"test-model": MagicMock()}
    mock_calc_phash.side_effect = ["phash_a", "phash_b"]
    mock_get_instance.return_value = MagicMock()
    # 2 画像に対して 1 結果しか返さない
    mock_annotate_model.return_value = [
        UnifiedAnnotationResult(model_name="test-model", capabilities={TaskCapability.TAGS}, tags=["x"])
    ]
    mock_get_capabilities.return_value = {TaskCapability.TAGS}

    images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    results = run_annotation(images=images, model_names=["test-model"])

    assert "phash_a" in results
    assert "phash_b" in results
    # phash_b は補完された error result
    assert results["phash_b"]["test-model"].error == "処理結果が不足しています"
