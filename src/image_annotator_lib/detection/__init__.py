"""物体検出 API (`annotate()` とは別の型付き API)。

人物・顔・頭部の検出器で矩形を返す。onnxruntime / huggingface_hub は推論時まで import しない。
設計判断は ADR 0014 を参照。
"""

from .catalog import BUILTIN_DETECTORS
from .runtime import DEFAULT_NMS_IOU, detect_objects, list_detectors, release_detectors
from .types import BoundingBox, Detection, DetectorInfo, ImageDetections

__all__ = [
    "BUILTIN_DETECTORS",
    "DEFAULT_NMS_IOU",
    "BoundingBox",
    "Detection",
    "DetectorInfo",
    "ImageDetections",
    "detect_objects",
    "list_detectors",
    "release_detectors",
]
