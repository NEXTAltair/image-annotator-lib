"""フレームワーク固有モデルローダーパッケージ。

LoaderBase を基底クラスとして、各フレームワーク (Transformers, ONNX,
TensorFlow, CLIP) の具象ローダーを提供する。
"""

from .clip_loader import CLIPLoader
from .loader_base import LoaderBase
from .onnx_loader import ONNXLoader
from .tensorflow_loader import TensorFlowLoader
from .transformers_loader import TransformersLoader, TransformersPipelineLoader

__all__ = [
    "CLIPLoader",
    "LoaderBase",
    "ONNXLoader",
    "TensorFlowLoader",
    "TransformersLoader",
    "TransformersPipelineLoader",
]
