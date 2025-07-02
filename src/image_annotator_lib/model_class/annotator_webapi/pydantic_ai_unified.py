"""PydanticAI統一WebAPIアノテーター実装

このモジュールは、PydanticAIの最新機能を活用してすべてのWebAPIプロバイダーを
統一的に扱うPydanticAIWebAPIAnnotatorクラスを提供します。
"""

from ...core.base.pydantic_ai_annotator import PydanticAIWebAPIAnnotator

# 公開API
__all__ = ["PydanticAIWebAPIAnnotator"]
