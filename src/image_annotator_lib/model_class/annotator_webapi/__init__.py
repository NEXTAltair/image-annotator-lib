# モジュール: image_annotator_lib.model_class.annotator_webapi
# 目的: 各WebAPIアノテータークラス・共通型を再エクスポートし、外部からのimport互換性を維持する
# 外部依存: 各APIごとの具象クラス、共通型・定数

from .anthropic_api import AnthropicApiAnnotator
from .google_api import GoogleApiAnnotator
from .openai_api_chat import OpenRouterApiAnnotator
from .openai_api_response import OpenAIApiAnnotator
from .webapi_shared import (
    BASE_PROMPT,
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    AnnotationSchema,
    FormattedOutput,
    Responsedict,
)

# __all__で明示的にエクスポート対象を指定(IDE補完・型解決のため推奨)
__all__ = [
    "BASE_PROMPT",
    "JSON_SCHEMA",
    "SYSTEM_PROMPT",
    "AnnotationSchema",
    "AnthropicApiAnnotator",
    "FormattedOutput",
    "GoogleApiAnnotator",
    "OpenAIApiAnnotator",
    "OpenRouterApiAnnotator",
    "Responsedict",
]