"""物体検出 API (`image_annotator_lib.detection`) のユニットテスト。

fake の ONNX セッションで前処理・出力デコード・閾値・NMS・座標逆変換・入力検証を
決定論的に検証する。実モデルの DL と推論は tests/runtime_validation 側 (CI 不経由)。
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import image_annotator_lib
from image_annotator_lib.detection import (
    BUILTIN_DETECTORS,
    Detection,
    DetectorInfo,
    ImageDetections,
    detect_objects,
    list_detectors,
    release_detectors,
)
from image_annotator_lib.detection import runtime as detection_runtime
from image_annotator_lib.detection.processing import (
    LetterboxTransform,
    box_iou,
    decode_detections,
    letterbox,
    non_max_suppression,
    normalize_yolo_output,
)
from image_annotator_lib.exceptions.errors import (
    InvalidInputError,
    InvalidOutputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    OutOfMemoryError,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

PERSON = "person_detect_v1.3_s"


def _identity_transform(width: int = 640, height: int = 640) -> LetterboxTransform:
    return LetterboxTransform(
        scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0, image_width=width, image_height=height
    )


def _yolo_output(rows: list[list[float]], *, channels_first: bool = True) -> npt.NDArray[np.float32]:
    """``[cx, cy, w, h, score...]`` の行から YOLOv8 形式の出力を作る。"""
    array = np.array(rows, dtype=np.float32).reshape(len(rows), -1)
    if channels_first:
        return array.T[np.newaxis, ...]
    return array[np.newaxis, ...]


@dataclass
class _NodeArg:
    name: str


class _FakeSession:
    """onnxruntime.InferenceSession の代役。入力を記録し、固定の出力を返す。"""

    def __init__(self, outputs: Sequence[npt.NDArray[np.float32]]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, npt.NDArray[np.float32]]] = []
        self.output_names: list[list[str] | None] = []

    def get_inputs(self) -> list[_NodeArg]:
        return [_NodeArg("images")]

    def get_outputs(self) -> list[_NodeArg]:
        return [_NodeArg("output0")]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        self.calls.append(input_feed)
        self.output_names.append(output_names)
        return [self._outputs[min(len(self.calls) - 1, len(self._outputs) - 1)]]


class _RaisingSession(_FakeSession):
    def __init__(self, error: BaseException) -> None:
        super().__init__([np.zeros((1, 5, 0), dtype=np.float32)])
        self._error = error

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        raise self._error


@pytest.fixture(autouse=True)
def _clear_session_cache() -> Iterator[None]:
    release_detectors()
    yield
    release_detectors()


@pytest.fixture
def install_session(monkeypatch: pytest.MonkeyPatch):
    """`_create_session` を差し替え、生成回数を数える。"""
    created: list[str] = []

    def _install(session: _FakeSession) -> list[str]:
        def _fake_create(info: DetectorInfo) -> _FakeSession:
            created.append(info.name)
            return session

        monkeypatch.setattr(detection_runtime, "_create_session", _fake_create)
        return created

    return _install


# --- catalog / list_detectors -------------------------------------------------


def test_list_detectors_returns_builtin_catalog_sorted() -> None:
    detectors = list_detectors()
    assert [d.name for d in detectors] == [
        "face_detect_v1.4_s",
        "head_detect_v2.0_s",
        "person_detect_v1.3_s",
    ]
    by_name = {d.name: d for d in detectors}
    assert by_name["person_detect_v1.3_s"] == DetectorInfo(
        name="person_detect_v1.3_s",
        classes=("person",),
        default_min_score=0.324,
        license="MIT",
        repo_id="deepghs/anime_person_detection",
        filename="person_detect_v1.3_s/model.onnx",
        input_size=640,
    )
    assert by_name["face_detect_v1.4_s"].default_min_score == 0.307
    assert by_name["face_detect_v1.4_s"].classes == ("face",)
    assert by_name["head_detect_v2.0_s"].default_min_score == 0.413
    assert by_name["head_detect_v2.0_s"].classes == ("head",)
    assert all(d.license == "MIT" for d in detectors)


# --- letterbox -----------------------------------------------------------------


def test_letterbox_wide_image_pads_vertically_and_normalizes() -> None:
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    tensor, transform = letterbox(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert transform == LetterboxTransform(
        scale_x=640 / 200, scale_y=320 / 100, pad_x=0, pad_y=160, image_width=200, image_height=100
    )
    # 画像領域は赤 (1, 0, 0)、余白は 114/255 の灰色
    assert np.allclose(tensor[0, :, 320, 320], [1.0, 0.0, 0.0])
    assert np.allclose(tensor[0, :, 10, 320], 114 / 255)
    assert np.allclose(tensor[0, :, 630, 320], 114 / 255)
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def test_letterbox_tall_image_pads_horizontally() -> None:
    image = Image.new("RGB", (100, 300), (0, 0, 255))
    _tensor, transform = letterbox(image, 640)
    # 640/300 * 100 = 213.33 -> 213
    assert transform.scale_x == pytest.approx(213 / 100)
    assert transform.scale_y == pytest.approx(640 / 300)
    assert transform.pad_x == (640 - 213) // 2
    assert transform.pad_y == 0


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK"])
def test_letterbox_converts_non_rgb_modes(mode: str) -> None:
    image = Image.new(mode, (64, 32))
    tensor, _ = letterbox(image, 640)
    assert tensor.shape == (1, 3, 640, 640)


# --- output normalization ------------------------------------------------------


def test_normalize_yolo_output_accepts_both_layouts() -> None:
    rows = [[10, 20, 4, 6, 0.9], [30, 40, 8, 2, 0.1]]
    channels_first = normalize_yolo_output(_yolo_output(rows, channels_first=True), 1)
    anchors_first = normalize_yolo_output(_yolo_output(rows, channels_first=False), 1)
    expected = np.array(rows, dtype=np.float32)
    assert np.array_equal(channels_first, expected)
    assert np.array_equal(anchors_first, expected)


@pytest.mark.parametrize(
    "shape",
    [(2, 5, 3), (1, 6, 3), (5,), (1, 1, 5, 3)],
)
def test_normalize_yolo_output_rejects_unexpected_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(InvalidOutputError):
        normalize_yolo_output(np.zeros(shape, dtype=np.float32), 1)


# --- IoU / NMS -------------------------------------------------------------------


def test_box_iou_exact_values() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    others = np.array(
        [[0, 0, 10, 10], [5, 0, 15, 10], [10, 10, 20, 20], [0, 0, 5, 5], [3, 3, 3, 3]],
        dtype=np.float32,
    )
    iou = box_iou(box, others)
    # 同一=1、半分重なり=50/150、接するだけ=0、内包=25/100、面積0=0
    assert iou.tolist() == pytest.approx([1.0, 50 / 150, 0.0, 0.25, 0.0])


def test_non_max_suppression_suppresses_only_above_threshold() -> None:
    boxes = np.array(
        [
            [0, 0, 10, 10],  # score 0.9
            [1, 0, 11, 10],  # IoU 90/110=0.818 -> 抑制
            [5, 0, 15, 10],  # IoU 50/150=0.333 -> 残る
            [50, 50, 60, 60],  # 離れている -> 残る
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert non_max_suppression(boxes, scores, 0.5) == [0, 2, 3]
    # 閾値 1/3 ちょうどは「超える」ではないので残る
    assert non_max_suppression(boxes, scores, 50 / 150) == [0, 2, 3]
    assert non_max_suppression(boxes, scores, 0.3) == [0, 3]


def test_non_max_suppression_orders_by_score_and_is_stable() -> None:
    boxes = np.array([[0, 0, 1, 1], [10, 10, 11, 11], [20, 20, 21, 21]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.5], dtype=np.float32)
    assert non_max_suppression(boxes, scores, 0.5) == [1, 0, 2]


def test_non_max_suppression_empty() -> None:
    assert non_max_suppression(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.5) == []


# --- decode ----------------------------------------------------------------------


def test_decode_detections_inverse_letterbox_exact() -> None:
    # 1280x640 の画像 -> scale 0.5、上下に 160 の余白
    transform = LetterboxTransform(
        scale_x=0.5, scale_y=0.5, pad_x=0, pad_y=160, image_width=1280, image_height=640
    )
    raw = _yolo_output([[320, 320, 100, 50, 0.8]])
    detections = decode_detections(
        raw, ("person",), transform, min_score=0.3, nms_iou=0.5, allowed_classes=frozenset({"person"})
    )
    # モデル座標 (270, 295, 370, 345) -> 元画像 ((270-0)/0.5, (295-160)/0.5, ...)
    assert detections == [
        Detection(class_name="person", score=pytest.approx(0.8), bbox=(540.0, 270.0, 740.0, 370.0))
    ]


def test_decode_detections_matches_letterbox_round_trip() -> None:
    image = Image.new("RGB", (300, 500))
    _tensor, transform = letterbox(image, 640)
    # 元画像 (30, 50)-(150, 250) をモデル座標へ写して出力に置く
    x1 = 30 * transform.scale_x + transform.pad_x
    y1 = 50 * transform.scale_y + transform.pad_y
    x2 = 150 * transform.scale_x + transform.pad_x
    y2 = 250 * transform.scale_y + transform.pad_y
    raw = _yolo_output([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.9]])
    (detection,) = decode_detections(
        raw, ("face",), transform, min_score=0.5, nms_iou=0.5, allowed_classes=frozenset({"face"})
    )
    assert detection.bbox == pytest.approx((30.0, 50.0, 150.0, 250.0), abs=1e-3)


def test_decode_detections_clips_to_image_and_drops_degenerate() -> None:
    transform = _identity_transform(100, 80)
    raw = _yolo_output(
        [
            [0, 0, 40, 40, 0.9],  # 左上にはみ出す -> clip
            [100, 80, 40, 20, 0.8],  # 右下にはみ出す -> clip
            [200, 200, 10, 10, 0.95],  # 完全に画像外 -> clip 後に面積 0 で除外
        ]
    )
    detections = decode_detections(
        raw, ("head",), transform, min_score=0.1, nms_iou=0.5, allowed_classes=frozenset({"head"})
    )
    assert [d.bbox for d in detections] == [(0.0, 0.0, 20.0, 20.0), (80.0, 70.0, 100.0, 80.0)]


def test_decode_detections_threshold_is_inclusive() -> None:
    raw = _yolo_output([[10, 10, 4, 4, 0.5], [50, 50, 4, 4, 0.49]])
    detections = decode_detections(
        raw,
        ("person",),
        _identity_transform(),
        min_score=0.5,
        nms_iou=0.5,
        allowed_classes=frozenset({"person"}),
    )
    assert [d.score for d in detections] == [0.5]


def test_decode_detections_applies_nms_and_sorts_by_score() -> None:
    raw = _yolo_output(
        [
            [100, 100, 20, 20, 0.6],
            [101, 100, 20, 20, 0.9],  # 上と大きく重なる (高 score 側が残る)
            [300, 300, 20, 20, 0.7],
        ]
    )
    detections = decode_detections(
        raw,
        ("person",),
        _identity_transform(),
        min_score=0.3,
        nms_iou=0.5,
        allowed_classes=frozenset({"person"}),
    )
    assert [d.score for d in detections] == pytest.approx([0.9, 0.7])
    assert detections[0].bbox == (91.0, 90.0, 111.0, 110.0)


def test_decode_detections_multi_class_nms_is_per_class_and_filters_classes() -> None:
    # 同じ位置の別クラスは互いに抑制しない
    raw = _yolo_output(
        [
            [100, 100, 20, 20, 0.9, 0.1],  # class a
            [100, 100, 20, 20, 0.2, 0.8],  # class b
        ]
    )
    both = decode_detections(
        raw,
        ("a", "b"),
        _identity_transform(),
        min_score=0.5,
        nms_iou=0.5,
        allowed_classes=frozenset({"a", "b"}),
    )
    assert [d.class_name for d in both] == ["a", "b"]
    only_b = decode_detections(
        raw, ("a", "b"), _identity_transform(), min_score=0.5, nms_iou=0.5, allowed_classes=frozenset({"b"})
    )
    assert [d.class_name for d in only_b] == ["b"]


def test_decode_detections_no_candidates_returns_empty() -> None:
    raw = _yolo_output([[10, 10, 4, 4, 0.1]])
    assert (
        decode_detections(
            raw,
            ("person",),
            _identity_transform(),
            min_score=0.5,
            nms_iou=0.5,
            allowed_classes=frozenset({"person"}),
        )
        == []
    )
    empty = np.zeros((1, 5, 0), dtype=np.float32)
    assert (
        decode_detections(
            empty,
            ("person",),
            _identity_transform(),
            min_score=0.5,
            nms_iou=0.5,
            allowed_classes=frozenset({"person"}),
        )
        == []
    )


# --- detect_objects (fake session) ---------------------------------------------


def test_detect_objects_end_to_end_with_fake_session(install_session) -> None:
    # 1280x640 -> scale 0.5, pad_y 160
    raw = _yolo_output([[320, 320, 100, 50, 0.8], [500, 500, 10, 10, 0.1]])
    session = _FakeSession([raw])
    created = install_session(session)

    image = Image.new("RGB", (1280, 640))
    results = detect_objects([image], detector=PERSON)

    assert created == [PERSON]
    assert results == [
        ImageDetections(
            detector=PERSON,
            image_size=(1280, 640),
            detections=(
                Detection(class_name="person", score=pytest.approx(0.8), bbox=(540.0, 270.0, 740.0, 370.0)),
            ),
        )
    ]
    (feed,) = session.calls
    assert list(feed) == ["images"]
    assert feed["images"].shape == (1, 3, 640, 640)
    assert feed["images"].dtype == np.float32
    assert session.output_names == [["output0"]]


def test_detect_objects_uses_detector_default_threshold_and_override(install_session) -> None:
    raw = _yolo_output([[100, 100, 10, 10, 0.35], [300, 300, 10, 10, 0.40]])
    install_session(_FakeSession([raw]))
    image = Image.new("RGB", (640, 640))

    # person 既定 0.324 -> 両方残る
    (default_result,) = detect_objects([image], detector=PERSON)
    assert [d.score for d in default_result.detections] == pytest.approx([0.40, 0.35])

    (strict_result,) = detect_objects([image], detector=PERSON, min_score=0.38)
    assert [d.score for d in strict_result.detections] == pytest.approx([0.40])


def test_detect_objects_zero_detections_returns_empty_tuple(install_session) -> None:
    install_session(_FakeSession([_yolo_output([[100, 100, 10, 10, 0.01]])]))
    results = detect_objects([Image.new("RGB", (32, 48)), Image.new("RGB", (48, 32))], detector=PERSON)
    assert [r.detections for r in results] == [(), ()]
    assert [r.image_size for r in results] == [(32, 48), (48, 32)]


def test_detect_objects_caches_session_until_released(install_session) -> None:
    created = install_session(_FakeSession([_yolo_output([[1, 1, 1, 1, 0.0]])]))
    image = Image.new("RGB", (16, 16))

    detect_objects([image], detector=PERSON)
    detect_objects([image], detector=PERSON)
    assert created == [PERSON]

    assert release_detectors([PERSON, "not-loaded"]) == 1
    detect_objects([image], detector=PERSON)
    assert created == [PERSON, PERSON]
    assert release_detectors() == 1
    assert release_detectors() == 0


def test_release_detectors_ignores_duplicate_names(install_session) -> None:
    """重複した検出器名を渡しても KeyError にならず、後続の名前も破棄される。"""
    session = _FakeSession([_yolo_output([[1, 1, 1, 1, 0.0]])])
    install_session(session)
    image = Image.new("RGB", (16, 16))
    face = "face_detect_v1.4_s"
    detect_objects([image], detector=PERSON)
    detect_objects([image], detector=face)

    assert release_detectors([PERSON, PERSON, face]) == 2
    assert release_detectors() == 0


def test_detect_objects_classes_filter_accepts_known_class(install_session) -> None:
    install_session(_FakeSession([_yolo_output([[100, 100, 10, 10, 0.9]])]))
    (result,) = detect_objects([Image.new("RGB", (640, 640))], detector=PERSON, classes=["person"])
    assert len(result.detections) == 1


def test_detect_objects_unknown_detector_raises_before_loading(install_session) -> None:
    created = install_session(_FakeSession([]))
    with pytest.raises(ModelNotFoundError):
        detect_objects([Image.new("RGB", (8, 8))], detector="halfbody_detect")
    assert created == []


@pytest.mark.parametrize(
    ("kwargs", "images"),
    [
        ({}, []),
        ({}, ["not an image"]),
        ({"classes": ["face"]}, None),
        ({"classes": []}, None),
        ({"classes": "person"}, None),
        ({"min_score": -0.1}, None),
        ({"min_score": 1.1}, None),
        ({"nms_iou": 0.0}, None),
        ({"nms_iou": 1.5}, None),
    ],
)
def test_detect_objects_invalid_input_raises(
    install_session, kwargs: dict[str, object], images: list[object] | None
) -> None:
    created = install_session(_FakeSession([]))
    target = [Image.new("RGB", (8, 8))] if images is None else images
    with pytest.raises(InvalidInputError):
        detect_objects(target, detector=PERSON, **kwargs)  # 検証は推論前に行う
    assert created == []


def test_detect_objects_maps_runtime_errors(install_session) -> None:
    install_session(_RaisingSession(RuntimeError("Failed to allocate memory for tensor")))
    with pytest.raises(OutOfMemoryError):
        detect_objects([Image.new("RGB", (8, 8))], detector=PERSON)

    release_detectors()
    install_session(_RaisingSession(RuntimeError("broken graph")))
    with pytest.raises(ModelExecutionError):
        detect_objects([Image.new("RGB", (8, 8))], detector=PERSON)


def test_detect_objects_invalid_output_shape_raises(install_session) -> None:
    install_session(_FakeSession([np.zeros((1, 7, 10), dtype=np.float32)]))
    with pytest.raises(InvalidOutputError):
        detect_objects([Image.new("RGB", (8, 8))], detector=PERSON)


def test_download_failure_is_reported_as_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def _offline(*_args: object, **_kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _offline)
    with pytest.raises(ModelLoadError):
        detect_objects([Image.new("RGB", (8, 8))], detector=PERSON)


def test_create_session_uses_cpu_provider(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import onnxruntime

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"")
    calls: list[tuple[str, list[str]]] = []

    def _fake_session(path: str, providers: list[str]) -> _FakeSession:
        calls.append((path, providers))
        return _FakeSession([])

    monkeypatch.setattr(detection_runtime, "_download_model", lambda info: model_file)
    monkeypatch.setattr(onnxruntime, "InferenceSession", _fake_session)
    detection_runtime._create_session(BUILTIN_DETECTORS[PERSON])
    assert calls == [(str(model_file), ["CPUExecutionProvider"])]


def test_create_session_wraps_onnxruntime_errors(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import onnxruntime
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"broken")

    def _broken_session(path: str, providers: list[str]) -> _FakeSession:
        raise InvalidProtobuf("bad model")

    monkeypatch.setattr(detection_runtime, "_download_model", lambda info: model_file)
    monkeypatch.setattr(onnxruntime, "InferenceSession", _broken_session)
    with pytest.raises(ModelLoadError):
        detection_runtime._create_session(BUILTIN_DETECTORS[PERSON])


# --- public API surface ----------------------------------------------------------


def test_top_level_wrappers_match_implementation_signatures() -> None:
    for name in ("detect_objects", "list_detectors", "release_detectors"):
        wrapper = getattr(image_annotator_lib, name)
        impl = getattr(detection_runtime, name)
        assert inspect.signature(wrapper, eval_str=True) == inspect.signature(impl, eval_str=True), name
        assert name in image_annotator_lib.__all__
    for type_name in ("BoundingBox", "Detection", "DetectorInfo", "ImageDetections"):
        assert type_name in image_annotator_lib.__all__


def test_top_level_wrapper_delegates(install_session) -> None:
    install_session(_FakeSession([_yolo_output([[100, 100, 10, 10, 0.9]])]))
    (result,) = image_annotator_lib.detect_objects([Image.new("RGB", (640, 640))], detector=PERSON)
    assert result.detections[0].class_name == "person"
    assert image_annotator_lib.list_detectors() == list_detectors()
    assert image_annotator_lib.release_detectors() == 1


def test_detection_import_does_not_load_heavy_deps() -> None:
    """detection パッケージの import と list_detectors() で重い依存を読み込まない。"""
    sentinel = "IAMLIB_DETECTION_JSON:"
    code = (
        "import sys, json\n"
        "import image_annotator_lib.detection as d\n"
        "d.list_detectors()\n"
        "heavy = {'torch', 'onnxruntime', 'tensorflow', 'transformers'}\n"
        f"print({sentinel!r} + json.dumps(sorted(heavy & set(sys.modules))))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = next(
        line[len(sentinel) :] for line in result.stdout.splitlines() if line.startswith(sentinel)
    )
    assert json.loads(payload) == []
