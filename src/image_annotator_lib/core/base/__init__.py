"""基底クラスモジュール - 各フレームワーク用の基底クラスをエクスポート

Note:
    ADR 0023 Phase 1 (Issue #35) で WebAPI 系は `WebApiAnnotator`
    (`webapi/annotator.py`) に統合された。旧 `WebApiBaseAnnotator` は廃止。
"""

from .annotator import BaseAnnotator
from .clip import ClipBaseAnnotator
from .onnx import ONNXBaseAnnotator
from .pipeline import PipelineBaseAnnotator
from .tensorflow import TensorflowBaseAnnotator
from .transformers import TransformersBaseAnnotator

__all__ = [
    "BaseAnnotator",
    "ClipBaseAnnotator",
    "ONNXBaseAnnotator",
    "PipelineBaseAnnotator",
    "TensorflowBaseAnnotator",
    "TransformersBaseAnnotator",
]
