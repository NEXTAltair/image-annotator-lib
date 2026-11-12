import os
from collections.abc import Iterable, Sequence

from PIL import Image

from .config_policy import CONFIG_READ_ONLY_ENV, ReadOnlyConfigError, config_read_only_enabled

# NOTE: TensorFlowの表示抑制用､見ててうざいだろ
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from .api import PHashAnnotationResults as PHashAnnotationResults
from .api import list_annotator_info
from .core.config import config_registry
from .core.constants import (
    DEFAULT_PATHS,
    MODEL_RUNTIME_CACHE_PATH,
    SYSTEM_CONFIG_PATH,
    USER_CONFIG_PATH,
)
from .core.inference_runtime import InferenceRuntimeInfo, get_inference_runtime_info
from .core.model_factory import ModelLoad
from .core.registry import (
    initialize_registry,
    list_available_annotators,
)
from .core.tensorflow_runtime import set_tensorflow_cpu_threads
from .core.types import AnnotationErrorCode, AnnotationResult, AnnotatorInfo, ModelType
from .core.utils import init_logger
from .detection.types import BoundingBox, Detection, DetectorInfo, ImageDetections
from .exceptions.errors import (
    AnnotatorError,
    ModelInstallCancelledError,
    ModelLoadError,
    ModelNotFoundError,
    OutOfMemoryError,
)
from .model_installer import ModelInstallProgress, install_model, is_model_installed
from .webapi.api_model_discovery import (
    discover_available_vision_models,
    get_available_models,
    is_model_deprecated,
    list_all_models,
)
from .webapi.batch import (
    BatchErrorPhase,
    BatchFetchResult,
    BatchItemError,
    BatchItemStatus,
    BatchJobError,
    BatchJobHandle,
    BatchModelInfo,
    BatchProviderItemStatus,
    BatchResultItem,
    BatchStatus,
    BatchStatusResult,
    BatchSubmitItem,
    BatchSubmitRequest,
    BatchSubmitResult,
    cancel_batch,
    fetch_batch_results,
    list_batch_capable_models,
    retrieve_batch,
    submit_batch,
)

# --- Public API ---
# ADR 0023 Phase 1: create_agent は廃止 (SimplifiedAgentFactory 全廃)。
# get_available_models / list_all_models / is_model_deprecated は LiteLLM runtime call に切替。
__all__ = [
    "CONFIG_READ_ONLY_ENV",
    "DEFAULT_PATHS",
    "MODEL_RUNTIME_CACHE_PATH",
    "SYSTEM_CONFIG_PATH",
    "USER_CONFIG_PATH",
    "AnnotationErrorCode",
    "AnnotationResult",
    "AnnotatorError",
    "AnnotatorInfo",
    "BatchErrorPhase",
    "BatchFetchResult",
    "BatchItemError",
    "BatchItemStatus",
    "BatchJobError",
    "BatchJobHandle",
    "BatchModelInfo",
    "BatchProviderItemStatus",
    "BatchResultItem",
    "BatchStatus",
    "BatchStatusResult",
    "BatchSubmitItem",
    "BatchSubmitRequest",
    "BatchSubmitResult",
    "BoundingBox",
    "Detection",
    "DetectorInfo",
    "ImageDetections",
    "InferenceRuntimeInfo",
    "ModelInstallCancelledError",
    "ModelInstallProgress",
    "ModelLoad",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelType",
    "OutOfMemoryError",
    "PHashAnnotationResults",
    "ReadOnlyConfigError",
    "annotate",
    "cancel_batch",
    "config_registry",
    "detect_objects",
    "discover_available_vision_models",
    "fetch_batch_results",
    "get_available_models",
    "get_inference_runtime_info",
    "install_model",
    "is_model_deprecated",
    "is_model_installed",
    "list_all_models",
    "list_annotator_info",
    "list_available_annotators",
    "list_batch_capable_models",
    "list_detectors",
    "release_detectors",
    "retrieve_batch",
    "set_tensorflow_cpu_threads",
    "submit_batch",
]

# モジュールレベルのキャッシュ
# NOTE: 遅延インポートにして必要なときだけannotatorをインポートしないと他のテストが激遅になる
_cached_annotate = None

init_logger()
if not config_read_only_enabled():
    initialize_registry()


def annotate(
    images_list: list[Image.Image],
    model_name_list: list[str],
    phash_list: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
    additional_prompt: str | None = None,
) -> PHashAnnotationResults:
    """
    指定されたモデルを使用して画像のリストにアノテーションを付けます。

    Args:
        images_list: アノテーションを付けるPIL Imageオブジェクトのリスト。
        model_name_list: 使用するアノテーターモデル名のリスト。
        phash_list: 画像のpHash値のリスト。
        api_keys: WebAPIモデル用のAPIキー辞書（オプション）。
                 例: {"openai": "sk-...", "anthropic": "sk-ant-..."}
                 指定された場合、環境変数より優先されます。
        additional_prompt: WebAPI モデルの BASE_PROMPT 末尾に追記するプロンプト。
                 None または空文字列の場合は追記しない。ローカル ML モデルには無視される。

    Returns:
        pHash をキーとし、その値がモデル名をキーとする ModelResultDict の辞書。
        例: {'pHash1': {'model_a': {'tags': ['tag1'], ...}, 'model_b': {'score': 0.8, ...}}, ...}
        # NOTE: 実際の戻り値の型は PHashAnnotationResults です。
    """
    global _cached_annotate
    if _cached_annotate is None:
        from .api import annotate as _annotate_impl

        _cached_annotate = _annotate_impl

    return _cached_annotate(images_list, model_name_list, phash_list, api_keys, additional_prompt)


def detect_objects(
    images: Sequence[Image.Image],
    detector: str,
    classes: Iterable[str] | None = None,
    min_score: float | None = None,
    nms_iou: float = 0.5,
) -> list[ImageDetections]:
    """画像から人物・顔・頭部などを検出し、ピクセル座標の矩形を返す。

    `image_annotator_lib.detection.detect_objects` への遅延 import ラッパー。
    EXIF の向き補正は行わないため、呼び出し側が EXIF 適用済みの画像を渡す。

    Args:
        images: 対象の PIL 画像。1 枚以上。
        detector: 検出器名 (`list_detectors()` の `name`)。
        classes: 返すクラス名。None なら検出器の全クラス。
        min_score: 残す最小 score。None なら検出器の既定値。
        nms_iou: NMS の IoU 閾値。

    Returns:
        入力と同じ順序の ImageDetections のリスト。
    """
    from .detection.runtime import detect_objects as _detect_objects_impl

    return _detect_objects_impl(images, detector, classes=classes, min_score=min_score, nms_iou=nms_iou)


def list_detectors() -> list[DetectorInfo]:
    """利用可能な検出器のメタデータを名前順で返す (遅延 import ラッパー)。

    Returns:
        DetectorInfo のリスト。
    """
    from .detection.runtime import list_detectors as _list_detectors_impl

    return _list_detectors_impl()


def release_detectors(names: Iterable[str] | None = None) -> int:
    """キャッシュ済みの検出器セッションを破棄する (遅延 import ラッパー)。

    Args:
        names: 破棄する検出器名。None なら全て。

    Returns:
        破棄したセッション数。
    """
    from .detection.runtime import release_detectors as _release_detectors_impl

    return _release_detectors_impl(names)


# You might want to add version information here later
# __version__ = "0.1.0"
