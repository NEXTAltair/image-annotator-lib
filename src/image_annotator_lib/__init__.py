import os

from PIL import Image

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
from .core.model_factory import ModelLoad
from .core.registry import (
    initialize_registry,
    list_available_annotators,
)
from .core.types import AnnotationResult, AnnotatorInfo, ModelType
from .core.utils import init_logger
from .exceptions.errors import AnnotatorError, ModelLoadError, ModelNotFoundError, OutOfMemoryError
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
    "DEFAULT_PATHS",
    "MODEL_RUNTIME_CACHE_PATH",
    "SYSTEM_CONFIG_PATH",
    "USER_CONFIG_PATH",
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
    "ModelLoad",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelType",
    "OutOfMemoryError",
    "PHashAnnotationResults",
    "annotate",
    "cancel_batch",
    "config_registry",
    "discover_available_vision_models",
    "fetch_batch_results",
    "get_available_models",
    "is_model_deprecated",
    "list_all_models",
    "list_annotator_info",
    "list_available_annotators",
    "list_batch_capable_models",
    "retrieve_batch",
    "submit_batch",
]

# モジュールレベルのキャッシュ
# NOTE: 遅延インポートにして必要なときだけannotatorをインポートしないと他のテストが激遅になる
_cached_annotate = None

init_logger()
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

    return _cached_annotate(
        images_list, model_name_list, phash_list, api_keys, additional_prompt
    )


# You might want to add version information here later
# __version__ = "0.1.0"
