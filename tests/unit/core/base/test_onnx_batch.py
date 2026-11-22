"""ONNXBaseAnnotator の複数画像まとめ推論 (Issue #164) のテスト。

小さな実 ONNX モデル (動的 / 静的 batch 軸) と、結果の形を制御できる fake session で検証する。
- 動的 batch のモデルでは、session.run の回数が画像数ではなく「枚数上限で割った回数」になる
- まとめた推論の結果が 1 枚ずつの推論と一致し、順序が保たれる
- 静的 batch、入力 shape の不揃い、batch 軸が文字列 / None、出力が分割できないモデルを壊さない
- 1 回に投げる枚数 (``inference_batch_size``) と前処理済み配列の保持枚数に上限がある
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from unit.fixtures.onnx_models import build_tagger_onnx, reference_probabilities

from image_annotator_lib.core import onnx_runtime
from image_annotator_lib.core.base.onnx import ONNXBaseAnnotator
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.types import AnnotationErrorCode
from image_annotator_lib.core.utils import logger
from image_annotator_lib.exceptions.errors import OutOfMemoryError

MODEL_NAME = "onnx-batch-test-model"
SIZE = 8
TAG_COUNT = 6


class BatchTagger(ONNXBaseAnnotator):
    """テスト用の最小サブクラス。前処理は 8x8 の NCHW (基底の既定実装) を使う。"""

    def _load_tags(self) -> None:
        self.all_tags = [f"tag{i}" for i in range(TAG_COUNT)]

    tag_threshold = 0.5


@pytest.fixture(autouse=True)
def _register_config():
    config_registry._merged_config_data[MODEL_NAME] = {
        "class": "BatchTagger",
        "model_path": "/dummy/model.onnx",
        "device": "cpu",
        "type": "tagger",
    }
    yield
    config_registry._merged_config_data.pop(MODEL_NAME, None)


def _set_batch_size(value: int | None) -> None:
    if value is None:
        config_registry._merged_config_data[MODEL_NAME].pop("inference_batch_size", None)
    else:
        config_registry._merged_config_data[MODEL_NAME]["inference_batch_size"] = value


def _reconfigure(annotator: BatchTagger, value: int) -> None:
    """構築済みの annotator の ``inference_batch_size`` を差し替える (設定オブジェクトは frozen)。"""
    annotator._config = annotator._config.model_copy(update={"inference_batch_size": value})


def _images(count: int, seed: int = 0) -> list[Image.Image]:
    """画像ごとに出力が変わるよう、乱数で塗った画像を返す。"""
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 255, (24 + i, 20, 3), dtype=np.uint8), "RGB") for i in range(count)
    ]


class CountingSession:
    """実 ``InferenceSession`` を包み、``run`` の入力 batch 枚数を記録する。"""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.run_batches: list[int] = []

    def get_inputs(self) -> Any:
        return self._session.get_inputs()

    def get_outputs(self) -> Any:
        return self._session.get_outputs()

    def get_providers(self) -> list[str]:
        return self._session.get_providers()

    def run(self, output_names: Any, feed: dict[str, np.ndarray]) -> Any:
        self.run_batches.append(next(iter(feed.values())).shape[0])
        return self._session.run(output_names, feed)


def _annotator(session: Any) -> BatchTagger:
    annotator = BatchTagger(MODEL_NAME)
    annotator.model_path = "/dummy/model.onnx"
    annotator.components = {"session": session}
    annotator.execution_device = "cpu"
    annotator._prepared = True
    annotator._load_tags()
    annotator._analyze_model_input_format()
    return annotator


def _real_annotator(tmp_path: Path, batch_dim: int | str | None) -> tuple[BatchTagger, CountingSession]:
    path = build_tagger_onnx(tmp_path / "model.onnx", batch_dim=batch_dim, height=SIZE, width=SIZE)
    session = CountingSession(path)
    return _annotator(session), session


def _probabilities(results: list[Any]) -> np.ndarray:
    return np.array([result.raw_output["predictions"] for result in results], dtype=np.float64)


class FakeSession:
    """出力の形と失敗の仕方を制御できる fake。出力は入力の行ごとの平均から作る (行ごとに独立)。"""

    def __init__(
        self,
        input_shape: list[int | str | None],
        *,
        split_output: bool = True,
        fail_on_call: int | None = None,
        failure: Exception | None = None,
        fail_always: Exception | None = None,
        fail_when_batched: Exception | None = None,
    ) -> None:
        self._fail_always = fail_always
        self._fail_when_batched = fail_when_batched
        self._input_shape = input_shape
        self._split_output = split_output
        self._fail_on_call = fail_on_call
        self._failure = failure
        self.run_shapes: list[tuple[int, ...]] = []
        self.providers = ["CPUExecutionProvider"]
        self.providers_after_first_run: list[str] | None = None

    def get_inputs(self) -> list[Any]:
        item = MagicMock()
        item.name = "input"
        item.shape = self._input_shape
        return [item]

    def get_outputs(self) -> list[Any]:
        item = MagicMock()
        item.name = "output"
        return [item]

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = feed["input"]
        self.run_shapes.append(tuple(batch.shape))
        call_number = len(self.run_shapes)
        if self.providers_after_first_run is not None and call_number == 1:
            self.providers = self.providers_after_first_run
        if self._fail_on_call == call_number and self._failure is not None:
            raise self._failure
        if self._fail_always is not None:
            raise self._fail_always
        if self._fail_when_batched is not None and batch.shape[0] > 1:
            raise self._fail_when_batched
        means = batch.reshape(batch.shape[0], -1).mean(axis=1, keepdims=True)
        rows = np.repeat(means, TAG_COUNT, axis=1).astype(np.float32) / 255.0
        if not self._split_output:
            return [rows.mean(axis=0)]  # batch 軸を潰した [K]
        return [rows]


# ==============================================================================
# 受け入れ条件 1: session.run の回数が画像数に比例しない
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize("batch_dim", ["batch", None], ids=["dim_param文字列", "dim未記載None"])
def test_dynamic_batch_runs_once_per_chunk_not_per_image(tmp_path: Path, batch_dim: str | None) -> None:
    _set_batch_size(8)
    annotator, session = _real_annotator(tmp_path, batch_dim)

    results = annotator.predict(_images(20))

    assert len(results) == 20
    assert all(result.error is None for result in results)
    assert session.run_batches == [8, 8, 4]


@pytest.mark.unit
@pytest.mark.fast
def test_run_count_is_ceil_of_images_over_batch_size(tmp_path: Path) -> None:
    _set_batch_size(5)
    annotator, session = _real_annotator(tmp_path, "batch")

    annotator.predict(_images(11))

    assert session.run_batches == [5, 5, 1]


# ==============================================================================
# 受け入れ条件 2: 出力が逐次と一致し、順序が保たれる
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_batched_results_match_sequential_results(tmp_path: Path) -> None:
    images = _images(13)
    annotator, session = _real_annotator(tmp_path, "batch")

    _reconfigure(annotator, 1)
    sequential = annotator.predict(images)
    assert session.run_batches == [1] * 13

    session.run_batches.clear()
    _reconfigure(annotator, 8)
    batched = annotator.predict(images)
    assert session.run_batches == [8, 5]

    assert [r.tags for r in batched] == [r.tags for r in sequential]
    np.testing.assert_allclose(_probabilities(batched), _probabilities(sequential), atol=1e-5)
    # 同一入力で同一タグ集合になるよう、閾値の前後に確率が散らばっていることも確認する。
    assert any(r.tags for r in batched)
    assert any(len(r.tags or []) < TAG_COUNT for r in batched)


@pytest.mark.unit
@pytest.mark.fast
def test_batched_results_keep_input_order(tmp_path: Path) -> None:
    images = _images(10, seed=3)
    _set_batch_size(4)
    annotator, _ = _real_annotator(tmp_path, "batch")

    forward = _probabilities(annotator.predict(images))
    backward = _probabilities(annotator.predict(list(reversed(images))))

    np.testing.assert_allclose(backward, forward[::-1], atol=1e-5)
    # 行ごとの値が入力画像から numpy の参照計算と一致する (他の画像の値が混ざっていない)。
    arrays = np.concatenate(annotator._preprocess_images(images), axis=0)
    np.testing.assert_allclose(forward, reference_probabilities(arrays, TAG_COUNT), atol=1e-5)


# ==============================================================================
# 受け入れ条件 3: 静的 batch / shape 不揃い / 出力が分割できないモデル
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_static_batch_one_stays_sequential(tmp_path: Path) -> None:
    _set_batch_size(8)
    annotator, session = _real_annotator(tmp_path, 1)

    results = annotator.predict(_images(5))

    assert [r.error for r in results] == [None] * 5
    assert session.run_batches == [1] * 5


@pytest.mark.unit
@pytest.mark.fast
def test_static_batch_greater_than_one_is_not_batched_and_does_not_crash(tmp_path: Path) -> None:
    """静的 N>1 のモデルは従来どおり 1 枚ずつ投げる (従来も ORT が shape 不一致で拒否していた)。"""
    _set_batch_size(8)
    annotator, session = _real_annotator(tmp_path, 4)

    results = annotator.predict(_images(3))

    assert len(results) == 3
    assert all(result.error for result in results)
    assert set(session.run_batches) == {1}


@pytest.mark.unit
@pytest.mark.fast
def test_mismatched_input_shapes_fall_back_to_sequential() -> None:
    _set_batch_size(8)
    session = FakeSession(["batch", 3, SIZE, SIZE])
    annotator = _annotator(session)
    small = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
    large = np.zeros((1, 3, SIZE * 2, SIZE * 2), dtype=np.float32)

    outputs = annotator._run_inference([small, large, small])

    assert [shape[0] for shape in session.run_shapes] == [1, 1, 1]
    assert len(outputs) == 3
    assert all(output.shape == (1, TAG_COUNT) for output in outputs)


@pytest.mark.unit
@pytest.mark.fast
def test_inputs_without_leading_batch_axis_fall_back_to_sequential() -> None:
    _set_batch_size(8)
    session = FakeSession(["batch", 3, SIZE, SIZE])
    annotator = _annotator(session)
    items = [np.zeros((2, 3, SIZE, SIZE), dtype=np.float32), np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)]

    annotator._run_inference(items)

    assert [shape[0] for shape in session.run_shapes] == [2, 1]


@pytest.mark.unit
@pytest.mark.fast
def test_output_that_cannot_be_split_falls_back_to_sequential_and_is_remembered() -> None:
    _set_batch_size(4)
    session = FakeSession(["batch", 3, SIZE, SIZE], split_output=False)
    annotator = _annotator(session)
    items = [np.full((1, 3, SIZE, SIZE), value, dtype=np.float32) for value in (10, 20, 30, 40)]

    first = annotator._run_inference(items)

    # 4 枚まとめた 1 回が分割できず、同じ 4 枚を 1 枚ずつ投げ直す。
    assert [shape[0] for shape in session.run_shapes] == [4, 1, 1, 1, 1]
    assert len(first) == 4
    assert [round(float(output.ravel()[0]) * 255) for output in first] == [10, 20, 30, 40]

    session.run_shapes.clear()
    annotator._run_inference(items)
    # 以降はまとめ直さない (毎回失敗してまとめ分を無駄にしない)。
    assert [shape[0] for shape in session.run_shapes] == [1, 1, 1, 1]


@pytest.mark.unit
@pytest.mark.fast
def test_real_model_that_collapses_batch_axis_is_not_batched(tmp_path: Path) -> None:
    """入力は動的でも出力の batch 軸が固定のモデルは、枚数がたまたま出力長と一致しても分割しない。"""
    _set_batch_size(8)
    path = build_tagger_onnx(
        tmp_path / "collapse.onnx", batch_dim="batch", height=SIZE, width=SIZE, collapse_batch=True
    )
    session = CountingSession(path)
    annotator = _annotator(session)

    results = annotator.predict(_images(TAG_COUNT))  # 枚数 == 出力長 (K) で誤分割しやすい条件

    assert len(results) == TAG_COUNT
    assert all(result.error is None for result in results)
    assert session.run_batches == [1] * TAG_COUNT


@pytest.mark.unit
@pytest.mark.fast
def test_single_image_run_is_unchanged() -> None:
    session = FakeSession(["batch", 3, SIZE, SIZE])
    annotator = _annotator(session)

    outputs = annotator._run_inference([np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)])

    assert [shape[0] for shape in session.run_shapes] == [1]
    assert len(outputs) == 1 and outputs[0].shape == (1, TAG_COUNT)


# ==============================================================================
# 受け入れ条件 4: 1 回に投げる枚数の上限と、前処理済み配列の保持上限
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_default_batch_size_is_eight_when_unset(tmp_path: Path) -> None:
    _set_batch_size(None)
    annotator, session = _real_annotator(tmp_path, "batch")

    annotator.predict(_images(20))

    assert onnx_runtime.DEFAULT_INFERENCE_BATCH_SIZE == 8
    assert session.run_batches == [8, 8, 4]


@pytest.mark.unit
@pytest.mark.fast
def test_configured_batch_size_caps_every_run(tmp_path: Path) -> None:
    _set_batch_size(3)
    annotator, session = _real_annotator(tmp_path, "batch")

    annotator.predict(_images(10))

    assert max(session.run_batches) == 3
    assert sum(session.run_batches) == 10


@pytest.mark.unit
@pytest.mark.fast
def test_preprocessed_arrays_are_held_per_chunk_only() -> None:
    """前処理済み配列は 1 回分ずつ作り、全画像分を同時に保持しない。"""
    _set_batch_size(4)
    session = FakeSession(["batch", 3, SIZE, SIZE])
    annotator = _annotator(session)
    preprocess_sizes: list[int] = []
    original = annotator._preprocess_images

    def spy(images: list[Image.Image]) -> Any:
        preprocess_sizes.append(len(images))
        return original(images)

    with patch.object(annotator, "_preprocess_images", side_effect=spy):
        results = annotator.predict(_images(10))

    assert len(results) == 10
    assert preprocess_sizes == [4, 4, 2]
    assert [shape[0] for shape in session.run_shapes] == [4, 4, 2]


# ==============================================================================
# 失敗時の画像数と結果数の対応
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_oom_in_later_chunk_keeps_completed_results_and_marks_rest() -> None:
    _set_batch_size(4)
    session = FakeSession(
        ["batch", 3, SIZE, SIZE],
        fail_on_call=2,
        failure=RuntimeError("Failed to allocate memory for requested buffer"),
    )
    annotator = _annotator(session)

    with patch("image_annotator_lib.core.model_factory.ModelLoad.invalidate_model") as invalidate:
        results = annotator.predict(_images(10))

    assert len(results) == 10
    assert [r.error for r in results[:4]] == [None] * 4
    assert all(r.error_code == AnnotationErrorCode.OUT_OF_MEMORY for r in results[4:])
    assert len(session.run_shapes) == 2  # OOM 後は再試行も後続の推論もしない
    invalidate.assert_called_with(MODEL_NAME, annotator)
    assert annotator.components is None


@pytest.mark.unit
@pytest.mark.fast
def test_non_oom_error_in_sequential_run_returns_one_error_result_per_image() -> None:
    _set_batch_size(4)
    session = FakeSession(["batch", 3, SIZE, SIZE], fail_always=RuntimeError("boom"))
    annotator = _annotator(session)

    results = annotator.predict(_images(10))

    assert len(results) == 10
    assert all(result.error and "boom" in result.error for result in results)


@pytest.mark.unit
@pytest.mark.fast
def test_non_oom_error_in_batched_run_falls_back_to_sequential_and_is_remembered() -> None:
    """入力・出力を動的 batch と宣言しつつ batch>1 でだけ失敗するモデルは、従来どおり 1 枚ずつ成功する。"""
    _set_batch_size(4)
    session = FakeSession(
        ["batch", 3, SIZE, SIZE], fail_when_batched=RuntimeError("Reshape: shape mismatch for batch > 1")
    )
    annotator = _annotator(session)
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        results = annotator.predict(_images(10))
        first_shapes = [shape[0] for shape in session.run_shapes]
        session.run_shapes.clear()
        second = annotator.predict(_images(10))
    finally:
        logger.remove(sink_id)

    assert len(results) == 10
    assert all(result.error is None for result in results)
    # 最初の 4 枚のまとめが失敗し、同じ画像を 1 枚ずつ投げ直す。以降はまとめない。
    assert first_shapes == [4] + [1] * 10
    # 以後の predict も 1 枚ずつで全件成功する (instance が寿命中まとめない)。
    assert all(result.error is None for result in second)
    assert [shape[0] for shape in session.run_shapes] == [1] * 10
    assert annotator._batching_disabled is True
    assert sum("1 枚ずつ推論" in message for message in warnings) == 1


@pytest.mark.unit
@pytest.mark.fast
def test_batched_fallback_results_match_sequential_results() -> None:
    _set_batch_size(4)
    images = _images(6)
    failing = _annotator(
        FakeSession(["batch", 3, SIZE, SIZE], fail_when_batched=RuntimeError("Reshape failed"))
    )
    sequential = _annotator(FakeSession(["batch", 3, SIZE, SIZE]))
    _reconfigure(sequential, 1)

    np.testing.assert_allclose(
        _probabilities(failing.predict(images)), _probabilities(sequential.predict(images)), atol=1e-7
    )


@pytest.mark.unit
@pytest.mark.fast
def test_oom_error_class_is_raised_by_run_inference_for_a_batch() -> None:
    _set_batch_size(4)
    session = FakeSession(
        ["batch", 3, SIZE, SIZE], fail_on_call=1, failure=RuntimeError("CUDA out of memory")
    )
    annotator = _annotator(session)
    items = [np.zeros((1, 3, SIZE, SIZE), dtype=np.float32) for _ in range(4)]

    with patch("image_annotator_lib.core.model_factory.ModelLoad.invalidate_model"):
        with pytest.raises(OutOfMemoryError):
            annotator._run_inference(items)


# ==============================================================================
# #175: session provider の読み直しを維持する
# ==============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_execution_device_follows_session_providers_after_batched_run() -> None:
    _set_batch_size(4)
    session = FakeSession(["batch", 3, SIZE, SIZE])
    session.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session.providers_after_first_run = ["CPUExecutionProvider"]
    annotator = _annotator(session)
    annotator.execution_device = "cuda"

    results = annotator.predict(_images(8))

    assert annotator.execution_device == "cpu"
    assert [r.execution_device for r in results] == ["cpu"] * 8
