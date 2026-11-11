"""TensorFlow 推論の小分け・推論前見積もり・OOM 通知・実デバイス判定 (ADR 0013 / Issue #172)。

実モデルは使わない。fake のモデル呼び出しと空き RAM 取得関数で検証する。
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import image_annotator_lib
from image_annotator_lib.core import tensorflow_runtime
from image_annotator_lib.core.annotation_runner import _record_model_error
from image_annotator_lib.core.base.tensorflow import TensorflowBaseAnnotator
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.inference_runtime import (
    clear_inference_runtime_info,
    get_inference_runtime_info,
)
from image_annotator_lib.core.types import (
    AnnotationErrorCode,
    PHashAnnotationResults,
    TaskCapability,
    UnifiedAnnotationResult,
)
from image_annotator_lib.exceptions.errors import OutOfMemoryError

pytestmark = [pytest.mark.unit, pytest.mark.fast]

MODEL_NAME = "test-tf-chunking"
GB = 1024**3


class FakeChunkedTensorflowAnnotator(TensorflowBaseAnnotator):
    """画像の赤成分を ID として入力し、fake モデルがそのまま返す TF annotator。"""

    __test__ = False

    def _load_tags(self) -> None:
        self.all_tags = ["id"]

    def _input_shape(self) -> tuple[int, int, int]:
        # 1 枚あたりの入力は 100x100x3 float32 = 120,000 bytes
        return (100, 100, 3)

    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([[float(image.getpixel((0, 0))[0])] for image in images], dtype=np.float32)

    def _format_predictions(self, raw_output: Any) -> list[UnifiedAnnotationResult]:
        values = np.asarray(raw_output)
        return [
            UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities={TaskCapability.TAGS},
                tags=[f"id{int(row[0])}"],
                framework="tensorflow",
            )
            for row in values
        ]


class FakeTensorflowModel:
    """呼び出しごとの入力枚数を記録し、指定回で OOM を起こす fake モデル。"""

    def __init__(self, oom_on_call: int | None = None) -> None:
        self.batch_sizes: list[int] = []
        self.oom_on_call = oom_on_call

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        self.batch_sizes.append(int(batch.shape[0]))
        if self.oom_on_call is not None and len(self.batch_sizes) == self.oom_on_call:
            raise tf.errors.ResourceExhaustedError(None, None, "OOM when allocating tensor")
        return batch


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch):
    clear_inference_runtime_info()
    tensorflow_runtime.reset_tensorflow_thread_request()
    # 既定は十分な空き RAM
    monkeypatch.setattr(tensorflow_runtime, "available_memory_bytes", lambda: 64 * GB)
    yield
    clear_inference_runtime_info()
    tensorflow_runtime.reset_tensorflow_thread_request()


def _register(monkeypatch, **extra: Any) -> None:
    config = {"class": "FakeChunkedTensorflowAnnotator", "model_path": "fake/tf", "device": "cpu"}
    config.update(extra)
    monkeypatch.setitem(config_registry._merged_config_data, MODEL_NAME, config)


def _images(count: int) -> list[Image.Image]:
    return [Image.new("RGB", (4, 4), color=(index, 0, 0)) for index in range(count)]


def _annotator(model: FakeTensorflowModel) -> FakeChunkedTensorflowAnnotator:
    annotator = FakeChunkedTensorflowAnnotator(MODEL_NAME)
    annotator.components = {"model": model}
    return annotator


def test_predict_splits_into_configured_chunks_and_keeps_order(monkeypatch):
    _register(monkeypatch, inference_batch_size=3)
    model = FakeTensorflowModel()
    annotator = _annotator(model)

    results = annotator.predict(_images(8))

    assert model.batch_sizes == [3, 3, 2]
    assert [result.tags for result in results] == [[f"id{index}"] for index in range(8)]
    assert all(result.execution_device == "cpu" for result in results)


def test_predict_defaults_to_eight_images_per_inference(monkeypatch):
    _register(monkeypatch)
    model = FakeTensorflowModel()

    _annotator(model).predict(_images(20))

    assert model.batch_sizes == [8, 8, 4]


def test_predict_reduces_chunk_when_estimate_exceeds_available_ram(monkeypatch):
    # 1 枚 = 120,000 bytes x 倍率 10 = 1,200,000 bytes。空き - 余裕 = 2.5 枚分 -> 2 枚ずつ
    _register(
        monkeypatch, inference_batch_size=8, inference_memory_multiplier=10.0, memory_safety_margin_gb=0
    )
    monkeypatch.setattr(tensorflow_runtime, "available_memory_bytes", lambda: 3_000_000)
    model = FakeTensorflowModel()

    results = _annotator(model).predict(_images(5))

    assert model.batch_sizes == [2, 2, 1]
    assert [result.tags for result in results] == [[f"id{index}"] for index in range(5)]
    info = image_annotator_lib.get_inference_runtime_info(MODEL_NAME)
    assert info is not None
    assert info.configured_batch_size == 8
    assert info.effective_batch_size == 1
    assert info.insufficient_memory is False


def test_predict_returns_insufficient_memory_errors_without_inference(monkeypatch):
    _register(monkeypatch, inference_memory_multiplier=10.0, memory_safety_margin_gb=1.0)
    monkeypatch.setattr(tensorflow_runtime, "available_memory_bytes", lambda: GB + 1000)
    model = FakeTensorflowModel()

    results = _annotator(model).predict(_images(3))

    assert model.batch_sizes == []
    assert len(results) == 3
    assert all(result.error_code == AnnotationErrorCode.INSUFFICIENT_MEMORY for result in results)
    info = get_inference_runtime_info(MODEL_NAME)
    assert info is not None
    assert info.insufficient_memory is True
    assert info.processed_images == 0
    assert info.effective_batch_size is None


def test_predict_keeps_completed_chunks_and_marks_rest_out_of_memory(monkeypatch):
    _register(monkeypatch, inference_batch_size=2)
    model = FakeTensorflowModel(oom_on_call=2)
    annotator = _annotator(model)
    discarded = Mock(side_effect=annotator._release_retained_components)
    monkeypatch.setattr(annotator, "_discard_components", discarded)
    cleared = Mock()
    monkeypatch.setattr(annotator, "_clear_tensorflow_session", cleared)

    results = annotator.predict(_images(5))

    # 1 回目の 2 枚は成功、2 回目で OOM。再試行せず残り 3 枚を OOM 結果にする
    assert model.batch_sizes == [2, 2]
    assert [result.tags for result in results[:2]] == [["id0"], ["id1"]]
    assert all(result.error is None for result in results[:2])
    assert [result.error_code for result in results[2:]] == [AnnotationErrorCode.OUT_OF_MEMORY] * 3
    assert len(results) == 5
    discarded.assert_called_once_with()
    cleared.assert_called_once_with()
    assert annotator.components is None
    info = get_inference_runtime_info(MODEL_NAME)
    assert info is not None
    assert info.out_of_memory is True
    assert info.processed_images == 2
    assert info.total_images == 5


def test_runtime_info_reports_requested_and_execution_device(monkeypatch):
    _register(monkeypatch, device="cuda")
    fake_tf = SimpleNamespace(
        config=SimpleNamespace(list_physical_devices=lambda kind: [], threading=FakeThreading())
    )
    monkeypatch.setattr(tensorflow_runtime, "_tensorflow_module", lambda: fake_tf)
    # PyTorch 側が CUDA ありと判定しても TF 判定が優先される
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n=None: d)

    annotator = _annotator(FakeTensorflowModel())
    annotator.predict(_images(1))

    info = get_inference_runtime_info(MODEL_NAME)
    assert info is not None
    assert info.requested_device == "cuda"
    assert info.execution_device == "cpu"
    assert info.framework == "tensorflow"


@pytest.mark.parametrize(
    ("requested", "gpu_count", "expected"),
    [
        ("cpu", 1, "cpu"),
        ("cuda", 0, "cpu"),
        ("cuda", 1, "cuda"),
        ("cuda:1", 2, "cuda:1"),
        ("cuda:3", 1, "cuda"),
    ],
)
def test_determine_tensorflow_device_uses_tensorflow_gpu_list(monkeypatch, requested, gpu_count, expected):
    gpus = [SimpleNamespace(name=f"GPU:{index}") for index in range(gpu_count)]
    fake_tf = SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda kind: gpus))
    monkeypatch.setattr(tensorflow_runtime, "_tensorflow_module", lambda: fake_tf)

    assert tensorflow_runtime.determine_tensorflow_device(requested, "m") == expected


@pytest.mark.parametrize(("device", "name"), [("cpu", "/CPU:0"), ("cuda", "/GPU:0"), ("cuda:1", "/GPU:1")])
def test_tensorflow_device_name(device, name):
    assert tensorflow_runtime.tensorflow_device_name(device) == name


class FakeThreading:
    def __init__(self, initialized: bool = False) -> None:
        self.intra = 0
        self.inter = 0
        self.initialized = initialized

    def get_intra_op_parallelism_threads(self) -> int:
        return self.intra

    def get_inter_op_parallelism_threads(self) -> int:
        return self.inter

    def set_intra_op_parallelism_threads(self, value: int) -> None:
        if self.initialized:
            raise RuntimeError("cannot be modified after initialization")
        self.intra = value

    def set_inter_op_parallelism_threads(self, value: int) -> None:
        if self.initialized:
            raise RuntimeError("cannot be modified after initialization")
        self.inter = value


def test_set_tensorflow_cpu_threads_applies_limits(monkeypatch):
    threading_api = FakeThreading()
    fake_tf = SimpleNamespace(config=SimpleNamespace(threading=threading_api))
    monkeypatch.setattr(tensorflow_runtime, "_tensorflow_module", lambda: fake_tf)

    image_annotator_lib.set_tensorflow_cpu_threads(intra_op_threads=4, inter_op_threads=2)

    assert (threading_api.intra, threading_api.inter) == (4, 2)
    assert tensorflow_runtime.current_tensorflow_threads() == (4, 2)


def test_set_tensorflow_cpu_threads_after_initialization_warns_without_raising(monkeypatch):
    threading_api = FakeThreading(initialized=True)
    fake_tf = SimpleNamespace(config=SimpleNamespace(threading=threading_api))
    monkeypatch.setattr(tensorflow_runtime, "_tensorflow_module", lambda: fake_tf)

    tensorflow_runtime.set_tensorflow_cpu_threads(intra_op_threads=4)

    assert threading_api.intra == 0


@pytest.mark.parametrize("value", [0, -1])
def test_set_tensorflow_cpu_threads_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        tensorflow_runtime.set_tensorflow_cpu_threads(intra_op_threads=value)


def test_annotator_init_applies_pending_thread_limits(monkeypatch):
    _register(monkeypatch)
    applied = Mock()
    monkeypatch.setattr(tensorflow_runtime, "apply_tensorflow_thread_limits", applied)

    FakeChunkedTensorflowAnnotator(MODEL_NAME)

    applied.assert_called_once_with()


@pytest.mark.parametrize(
    ("requested", "remaining", "per_image", "available", "margin", "expected"),
    [
        (8, 100, 0, 0, 0, 8),
        (8, 3, 10, 1000, 0, 3),
        (8, 100, 100, 450, 100, 3),
        (8, 100, 100, 150, 100, 0),
        (8, 100, 100, 50, 100, 0),
    ],
)
def test_plan_inference_chunk_size(requested, remaining, per_image, available, margin, expected):
    assert (
        tensorflow_runtime.plan_inference_chunk_size(requested, remaining, per_image, available, margin)
        == expected
    )


def test_runner_records_out_of_memory_error_code_for_model_failure():
    results = PHashAnnotationResults()

    _record_model_error(
        "test-oom-model", OutOfMemoryError("boom"), ["p0", "p1"], {0: "p0", 1: "p1"}, results
    )

    assert results["p0"]["test-oom-model"].error_code == AnnotationErrorCode.OUT_OF_MEMORY
    assert results["p1"]["test-oom-model"].error_code == AnnotationErrorCode.OUT_OF_MEMORY


def test_runner_leaves_other_model_failures_unclassified():
    results = PHashAnnotationResults()

    _record_model_error("test-other-model", ValueError("bad"), ["p0"], {0: "p0"}, results)

    assert results["p0"]["test-other-model"].error_code is None


def test_public_api_exports():
    assert image_annotator_lib.InferenceRuntimeInfo is not None
    assert "get_inference_runtime_info" in image_annotator_lib.__all__
    assert "set_tensorflow_cpu_threads" in image_annotator_lib.__all__
    assert image_annotator_lib.AnnotationErrorCode.OUT_OF_MEMORY == "OUT_OF_MEMORY"
