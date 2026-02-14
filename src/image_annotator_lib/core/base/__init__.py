"""基底クラスモジュール - 各フレームワーク用の基底クラスをエクスポート"""

from .annotator import BaseAnnotator
from .clip import ClipBaseAnnotator
from .onnx import ONNXBaseAnnotator
from .pipeline import PipelineBaseAnnotator
from .tensorflow import TensorflowBaseAnnotator
from .transformers import TransformersBaseAnnotator
from .webapi import WebApiBaseAnnotator

__all__ = [
    "BaseAnnotator",
    "ClipBaseAnnotator",
    "ONNXBaseAnnotator",
    "PipelineBaseAnnotator",
    "TensorflowBaseAnnotator",
    "TransformersBaseAnnotator",
    "WebApiBaseAnnotator",
]
