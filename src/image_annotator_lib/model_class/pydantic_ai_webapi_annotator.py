"""PydanticAI統一WebAPIアノテーター

このモジュールは、PydanticAIの最新機能を活用してすべてのWebAPIプロバイダーを
統一的に扱うPydanticAIWebAPIAnnotatorクラスを提供します。

このクラスはGoogle、OpenAI、Anthropic、OpenRouter等すべてのWebAPIプロバイダーを
単一の実装で処理し、従来のプロバイダー固有クラスを置き換えます。
"""

# PydanticAI統一実装をインポート
from ..core.base.pydantic_ai_annotator import PydanticAIWebAPIAnnotator

# 公開API - 旧クラス名との互換性のためエイリアスも提供
__all__ = [
    "PydanticAIWebAPIAnnotator",
]

# 旧クラスとの互換性のためのエイリアス（必要に応じて）
# GoogleApiAnnotator = PydanticAIWebAPIAnnotator
# OpenAIApiAnnotator = PydanticAIWebAPIAnnotator
# AnthropicApiAnnotator = PydanticAIWebAPIAnnotator
# OpenRouterApiAnnotator = PydanticAIWebAPIAnnotator
