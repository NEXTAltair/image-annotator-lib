import os

from PIL import Image

# NOTE: TensorFlowの表示抑制用､見ててうざいだろ
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from .api import PHashAnnotationResults as PHashAnnotationResults
from .core.api_model_discovery import discover_available_vision_models
from .core.config import config_registry
from .core.constants import (
    AVAILABLE_API_MODELS_CONFIG_PATH,
    DEFAULT_PATHS,
    SYSTEM_CONFIG_PATH,
    USER_CONFIG_PATH,
)
from .core.model_factory import ModelLoad
from .core.registry import (
    initialize_registry,
    list_available_annotators,
    list_available_annotators_with_metadata,
)
from .core.simplified_agent_factory import create_agent, get_available_models, is_model_deprecated, list_all_models
from .core.types import AnnotationResult
from .core.utils import init_logger
from .exceptions.errors import AnnotatorError, ModelLoadError, ModelNotFoundError, OutOfMemoryError

# --- Public API ---
__all__ = [
    "AVAILABLE_API_MODELS_CONFIG_PATH",
    "DEFAULT_PATHS",
    "SYSTEM_CONFIG_PATH",
    "USER_CONFIG_PATH",
    "AnnotationResult",
    "AnnotatorError",
    "ModelLoad",
    "ModelLoadError",
    "ModelNotFoundError",
    "OutOfMemoryError",
    "PHashAnnotationResults",
    "annotate",
    "config_registry",
    "create_agent",
    "discover_available_vision_models",
    "get_available_models",
    "is_model_deprecated",
    "list_all_models",
    "list_available_annotators",
    "list_available_annotators_with_metadata",
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

    Returns:
        pHash をキーとし、その値がモデル名をキーとする ModelResultDict の辞書。
        例: {'pHash1': {'model_a': {'tags': ['tag1'], ...}, 'model_b': {'score': 0.8, ...}}, ...}
        # NOTE: 実際の戻り値の型は PHashAnnotationResults です。
    """
    global _cached_annotate
    if _cached_annotate is None:
        from .api import annotate as _annotate_impl

        _cached_annotate = _annotate_impl

    return _cached_annotate(images_list, model_name_list, phash_list, api_keys)


# You might want to add version information here later
# __version__ = "0.1.0"
