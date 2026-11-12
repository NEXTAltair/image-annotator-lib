"""検出器の取得・ONNX セッションのキャッシュ・推論実行。

onnxruntime / huggingface_hub は関数内で遅延 import する。セッションはプロセス内で
検出器名ごとに 1 個だけ生成して再利用し、`release_detectors()` で明示的に破棄する。
`ModelLoad` の RAM/VRAM 台帳 (ADR 0012) には参加しない (CPU 実行のみ。ADR 0014)。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

from ..core.utils import logger
from ..exceptions.errors import (
    InvalidInputError,
    ModelExecutionError,
    ModelLoadError,
    ModelNotFoundError,
    OutOfMemoryError,
)
from .catalog import BUILTIN_DETECTORS
from .processing import decode_detections, letterbox
from .types import DetectorInfo, ImageDetections

DEFAULT_NMS_IOU = 0.5
"""NMS の既定 IoU 閾値 (同一画像・同一クラス内)。"""

_CPU_PROVIDERS = ["CPUExecutionProvider"]

# onnxruntime の pybind 例外は RuntimeError を継承しない (Exception 直下) ため名前で集める。
_ORT_ERROR_NAMES = (
    "EPFail",
    "EngineError",
    "Fail",
    "InvalidArgument",
    "InvalidGraph",
    "InvalidProtobuf",
    "ModelLoaded",
    "NoModel",
    "NoSuchFile",
    "NotFound",
    "NotImplemented",
    "RuntimeException",
)


class _OnnxNodeArg(Protocol):
    """onnxruntime の NodeArg のうち本モジュールが使う属性。"""

    @property
    def name(self) -> str: ...


class OnnxSession(Protocol):
    """onnxruntime.InferenceSession のうち本モジュールが使うメソッド。"""

    def get_inputs(self) -> Sequence[_OnnxNodeArg]: ...

    def get_outputs(self) -> Sequence[_OnnxNodeArg]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, npt.NDArray[np.float32]]
    ) -> Sequence[npt.NDArray[np.generic]]: ...


_SESSION_CACHE: dict[str, OnnxSession] = {}
_SESSION_LOCK = threading.Lock()


def list_detectors() -> list[DetectorInfo]:
    """利用可能な検出器のメタデータを名前順で返す。

    Returns:
        DetectorInfo のリスト。モデルの DL やセッション生成は行わない。
    """
    return [BUILTIN_DETECTORS[name] for name in sorted(BUILTIN_DETECTORS)]


def detect_objects(
    images: Sequence[Image.Image],
    detector: str,
    classes: Iterable[str] | None = None,
    min_score: float | None = None,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[ImageDetections]:
    """画像から人物・顔・頭部などを検出し、ピクセル座標の矩形を返す。

    画像の EXIF 向き補正は行わない。呼び出し側が表示向きに補正した (EXIF 適用済みの)
    PIL 画像を渡し、返る座標もその画像の座標系になる。推論は CPU で行う。
    初回呼び出し時に Hugging Face Hub からモデルを取得し、セッションをキャッシュする。

    Args:
        images: 対象の PIL 画像。1 枚以上。
        detector: 検出器名 (`list_detectors()` の `name`)。
        classes: 返すクラス名。None なら検出器の全クラス。
        min_score: 残す最小 score (0.0 から 1.0)。None なら検出器の既定値。
        nms_iou: NMS の IoU 閾値 (0.0 より大きく 1.0 以下)。

    Returns:
        入力と同じ順序の ImageDetections のリスト。検出 0 件の画像は detections が空。

    Raises:
        ModelNotFoundError: 未知の検出器名。
        InvalidInputError: 画像が 0 枚・PIL 画像でない・未知のクラス名・閾値が範囲外。
        ModelLoadError: モデルの取得またはセッション生成に失敗した場合。
        OutOfMemoryError: 推論中にメモリ確保に失敗した場合。
        ModelExecutionError: 推論に失敗した場合。
        InvalidOutputError: 検出器出力の形状が不正な場合。
    """
    info = _resolve_detector(detector)
    image_list = _validate_images(images)
    allowed_classes = _validate_classes(info, classes)
    score_threshold = info.default_min_score if min_score is None else min_score
    _validate_thresholds(score_threshold, nms_iou)

    session = _get_session(info)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    started = time.perf_counter()
    results: list[ImageDetections] = []
    for image in image_list:
        tensor, transform = letterbox(image, info.input_size)
        raw_output = _run_session(session, info, output_name, {input_name: tensor})
        detections = decode_detections(
            raw_output,
            info.classes,
            transform,
            min_score=score_threshold,
            nms_iou=nms_iou,
            allowed_classes=allowed_classes,
        )
        logger.debug(f"検出: detector={info.name}, size={image.size}, 件数={len(detections)}")
        results.append(
            ImageDetections(detector=info.name, image_size=image.size, detections=tuple(detections))
        )

    total = sum(len(result.detections) for result in results)
    elapsed = time.perf_counter() - started
    logger.info(
        f"物体検出完了: detector={info.name}, 画像={len(results)}枚, 検出={total}件, {elapsed:.2f}秒"
    )
    return results


def release_detectors(names: Iterable[str] | None = None) -> int:
    """キャッシュ済みの検出器セッションを破棄する。

    Args:
        names: 破棄する検出器名。None なら全て。キャッシュに無い名前は無視する。

    Returns:
        破棄したセッション数。
    """
    with _SESSION_LOCK:
        targets = list(_SESSION_CACHE) if names is None else [n for n in names if n in _SESSION_CACHE]
        for name in targets:
            del _SESSION_CACHE[name]
    if targets:
        logger.debug(f"検出器セッションを破棄: {targets}")
    return len(targets)


def _resolve_detector(detector: str) -> DetectorInfo:
    info = BUILTIN_DETECTORS.get(detector)
    if info is None:
        raise ModelNotFoundError(detector)
    return info


def _validate_images(images: Sequence[Image.Image]) -> list[Image.Image]:
    image_list = list(images)
    if not image_list:
        raise InvalidInputError("検出対象の画像が 0 枚です")
    for index, image in enumerate(image_list):
        if not isinstance(image, Image.Image):
            raise InvalidInputError(
                f"images[{index}] が PIL.Image.Image ではありません ({type(image).__name__})"
            )
        if image.width <= 0 or image.height <= 0:
            raise InvalidInputError(f"images[{index}] のサイズが不正です ({image.size})")
    return image_list


def _validate_classes(info: DetectorInfo, classes: Iterable[str] | None) -> frozenset[str]:
    if classes is None:
        return frozenset(info.classes)
    if isinstance(classes, str):
        # 文字列を渡すと 1 文字ずつに分解されるため、誤用として明示的に弾く
        raise InvalidInputError("classes には文字列ではなくクラス名の列を渡してください")
    requested = frozenset(classes)
    if not requested:
        raise InvalidInputError("classes が空です")
    unknown = sorted(requested - set(info.classes))
    if unknown:
        raise InvalidInputError(
            f"検出器 '{info.name}' に存在しないクラス: {unknown} (利用可能: {list(info.classes)})"
        )
    return requested


def _validate_thresholds(min_score: float, nms_iou: float) -> None:
    if not 0.0 <= min_score <= 1.0:
        raise InvalidInputError(f"min_score は 0.0 から 1.0 の範囲で指定してください ({min_score})")
    if not 0.0 < nms_iou <= 1.0:
        raise InvalidInputError(f"nms_iou は 0.0 より大きく 1.0 以下で指定してください ({nms_iou})")


def _get_session(info: DetectorInfo) -> OnnxSession:
    """キャッシュ済みセッションを返す。無ければ生成してキャッシュする。"""
    with _SESSION_LOCK:
        session = _SESSION_CACHE.get(info.name)
        if session is None:
            session = _create_session(info)
            _SESSION_CACHE[info.name] = session
        return session


def _create_session(info: DetectorInfo) -> OnnxSession:
    """モデルを取得し CPU 実行の ONNX セッションを生成する。"""
    model_path = _download_model(info)
    import onnxruntime as ort

    try:
        session: OnnxSession = ort.InferenceSession(str(model_path), providers=_CPU_PROVIDERS)
    except _ort_error_types() as error:
        raise ModelLoadError(
            f"検出器 '{info.name}' の ONNX セッション生成に失敗しました: {error}"
        ) from error
    logger.info(f"検出器 '{info.name}' をロードしました (CPUExecutionProvider)")
    return session


def _download_model(info: DetectorInfo) -> Path:
    """Hugging Face Hub から ONNX ファイルを取得する (HF キャッシュを再利用)。"""
    import httpx
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return Path(hf_hub_download(info.repo_id, info.filename))
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError) as error:
        raise ModelLoadError(f"検出器 '{info.name}' のモデル取得に失敗しました: {error}") from error


def _run_session(
    session: OnnxSession,
    info: DetectorInfo,
    output_name: str,
    input_feed: dict[str, npt.NDArray[np.float32]],
) -> npt.NDArray[np.generic]:
    try:
        outputs = session.run([output_name], input_feed)
    except _ort_error_types() as error:
        message = str(error).lower()
        if "failed to allocate memory" in message or "out of memory" in message:
            raise OutOfMemoryError(f"検出器 '{info.name}' の推論でメモリ不足") from error
        raise ModelExecutionError(f"検出器 '{info.name}' の推論に失敗しました: {error}") from error
    return np.asarray(outputs[0])


def _ort_error_types() -> tuple[type[BaseException], ...]:
    """onnxruntime が送出する例外型 (RuntimeError を含む) を返す。"""
    from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

    found: list[type[BaseException]] = [RuntimeError]
    for name in _ORT_ERROR_NAMES:
        candidate = getattr(ort_state, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    return tuple(found)
