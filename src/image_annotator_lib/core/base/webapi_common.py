"""WebAPI Annotator共通処理モジュール。

4プロバイダ（OpenRouter, Anthropic, Google, OpenAI）で完全に
重複していたraw_output処理とエラーハンドリングを共通化する。
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from ..types import TaskCapability, UnifiedAnnotationResult


def convert_to_raw_output(response_content: Any) -> dict[str, Any] | None:
    """APIレスポンスを安全にraw_output辞書に変換する。

    4段階のフォールバック:
    1. MagicMock検出（テスト環境対策）
    2. Pydantic model_dump()
    3. __dict__ 変換
    4. 文字列フォールバック

    Args:
        response_content: APIからのレスポンスオブジェクト。

    Returns:
        raw_output辞書。変換失敗時もエラー情報を含む辞書を返す。
    """
    try:
        # MagicMock検出（unittest.mockのMagicMockオブジェクト）
        if str(type(response_content)).find("MagicMock") != -1:
            return {"mock_type": "MagicMock", "mock_str": str(response_content)}
        if hasattr(response_content, "model_dump") and callable(response_content.model_dump):
            return response_content.model_dump()
        if hasattr(response_content, "__dict__"):
            try:
                return dict(response_content.__dict__)
            except Exception:
                return {
                    "object_type": str(type(response_content)),
                    "content": str(response_content),
                }
        return {"fallback_content": str(response_content)}
    except Exception:
        return {
            "error": "Failed to serialize response",
            "type": str(type(response_content)),
        }


def build_success_result(
    model_name: str,
    capabilities: set[TaskCapability],
    response_content: Any,
    provider_name: str,
    raw_output: dict[str, Any] | None,
) -> UnifiedAnnotationResult:
    """成功レスポンスからUnifiedAnnotationResultを構築する。

    Args:
        model_name: モデル名。
        capabilities: タスク能力セット。
        response_content: APIレスポンスオブジェクト（tags/captions/score属性を持つ）。
        provider_name: プロバイダ識別名。
        raw_output: convert_to_raw_outputの結果。

    Returns:
        構築されたUnifiedAnnotationResult。
    """
    return UnifiedAnnotationResult(
        model_name=model_name,
        capabilities=capabilities,
        tags=response_content.tags if TaskCapability.TAGS in capabilities else None,
        captions=response_content.captions if TaskCapability.CAPTIONS in capabilities else None,
        scores={"score": response_content.score}
        if TaskCapability.SCORES in capabilities and response_content.score
        else None,
        provider_name=provider_name,
        framework="api",
        raw_output=raw_output,
    )


def build_error_result(
    model_name: str,
    capabilities: set[TaskCapability],
    error: str,
    provider_name: str,
) -> UnifiedAnnotationResult:
    """エラー情報からUnifiedAnnotationResultを構築する。

    Args:
        model_name: モデル名。
        capabilities: タスク能力セット。
        error: エラーメッセージ。
        provider_name: プロバイダ識別名。

    Returns:
        エラーを含むUnifiedAnnotationResult。
    """
    return UnifiedAnnotationResult(
        model_name=model_name,
        capabilities=capabilities,
        error=error,
        provider_name=provider_name,
        framework="api",
    )


def handle_inference_error(
    exc: Exception,
    model_name: str,
    capabilities: set[TaskCapability],
    provider_name: str,
    provider_display: str,
) -> UnifiedAnnotationResult:
    """推論中の例外を3段階で処理してエラー結果を返す。

    Args:
        exc: 発生した例外。
        model_name: モデル名。
        capabilities: タスク能力セット。
        provider_name: プロバイダ識別名（結果に格納される小文字名）。
        provider_display: ログ表示用のプロバイダ名。

    Returns:
        エラーを含むUnifiedAnnotationResult。
    """
    if isinstance(exc, ModelHTTPError):
        error_message = f"{provider_display} HTTP {exc.status_code}: {exc.body or str(exc)}"
    elif isinstance(exc, UnexpectedModelBehavior):
        error_message = f"{provider_display} API Error: Unexpected model behavior: {exc!s}"
    else:
        error_message = f"{provider_display} API Error: {exc!s}"

    logger.error(f"{provider_display} API 推論エラー: {error_message}")
    return build_error_result(model_name, capabilities, error_message, provider_name)
