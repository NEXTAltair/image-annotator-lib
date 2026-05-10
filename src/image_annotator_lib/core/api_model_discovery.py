"""利用可能な Vision API モデルの runtime 取得 (ADR 0023 Phase 1)。

LiteLLM 同梱 DB を runtime SSoT として直接参照する。旧 `available_api_models.toml`
キャッシュ、OpenRouter fallback fetch、TTL refresh、background refresh thread は
すべて廃止された。`SUPPORTED_PROVIDERS` (`core/model_id.py`) に含まれる provider のみ
を扱う。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

import os
from typing import Any

# LiteLLM が import 時に remote cost map を取りに行かないよう local backup を強制する。
# image-annotator-lib は同梱 DB のみを使う。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

from .model_id import SUPPORTED_PROVIDERS
from .utils import logger

_SUPPORTED_LITELLM_MODES: frozenset[str] = frozenset({"chat", "responses"})

# ADR 0023 Phase 1.10 (Issue #52): LiteLLM 同梱 DB は Anthropic 直接モデルを `claude-*` の
# bare 名で格納するため、`anthropic/` プレフィックスを補完して `_BUILDER_DISPATCH` 対象に乗せる。
# スコープ拡張時はこの tuple に prefix を追加すれば `_canonicalize_litellm_id()` 経路で対応可能。
_ANTHROPIC_BARE_PREFIXES: tuple[str, ...] = ("claude-",)


def _canonicalize_litellm_id(model_id: str) -> str | None:
    """LiteLLM 同梱 DB のキーを `provider/model` 形式に正規化する (ADR 0023 Phase 1.10)。

    LiteLLM 同梱 DB は Anthropic 直接モデルを `claude-opus-4-6` のような bare 名で
    格納している。`is_allowed_provider()` の `/` チェックで全除外されると Anthropic
    直接プロバイダー経路が registry に登録されないため、`anthropic/<bare>` 形式に補完する。

    Args:
        model_id: LiteLLM 同梱 DB のキー (slash 入り or bare)。

    Returns:
        正規化後の `provider/model` 形式 ID。スコープ外の bare 名 (`claude-` 以外で
        始まる、または slash を含まないが `_ANTHROPIC_BARE_PREFIXES` に該当しない形式)
        は None。
    """
    if "/" in model_id:
        return model_id
    if model_id.startswith(_ANTHROPIC_BARE_PREFIXES):
        return f"anthropic/{model_id}"
    return None


def is_allowed_provider(model_id: str) -> bool:
    """model_id の provider prefix が `SUPPORTED_PROVIDERS` に含まれるか判定する。

    `core/model_id.py` の dispatch table を SSoT とし、本ファイル独自の allowlist は持たない。

    ADR 0023 Phase 1.10 (Issue #52): bare `claude-*` (LiteLLM JSON で `/` 無しで格納される
    Anthropic 直接モデル) は `_canonicalize_litellm_id()` で `anthropic/<bare>` に補完して
    判定する。
    """
    canonical = _canonicalize_litellm_id(model_id)
    if canonical is None:
        return False
    provider = canonical.split("/", 1)[0]
    return provider in SUPPORTED_PROVIDERS


def _is_litellm_model_annotation_compatible(info: dict[str, Any]) -> bool:
    """画像アノテーションに適したモデルか判定する (Vision + Tool/Function calling)。

    ADR 0023 Phase 1 (Issue #45): structured output は PydanticAI default Tool Output
    で得るため、`supports_response_schema` ではなく `supports_function_calling` を
    主条件にする。`supports_response_schema` は NativeOutput 最適化の参考用 metadata
    として LiteLLM 側にあるが、本判定では使わない。
    """
    mode = info.get("mode", "chat")
    return (
        info.get("supports_vision") is True
        and info.get("supports_function_calling") is True
        and mode in _SUPPORTED_LITELLM_MODES
    )


def _format_litellm_metadata(model_id: str, info: dict[str, Any]) -> dict[str, Any] | None:
    """LiteLLM の `get_model_info()` 結果を共通 metadata 形式に整形する。

    Issue #51 (ADR 0023 Phase 1.9): `model_name_short` は LiteLLM 同梱 DB のオリジナルキー
    と同一の完全 ID を保持する。旧実装は `split("/", 1)[1]` で provider prefix を剥がしていた
    ため、`openrouter/<inner>/<model>` (例: `openrouter/z-ai/glm-4.7`) で prefix 欠落形が
    registry / CLI / DB に伝播し、推論時に `_BUILDER_DISPATCH` 未知 prefix で
    `UnknownProviderError` が発生していた。

    Issue #52 (ADR 0023 Phase 1.10): bare `claude-*` 形式 (LiteLLM JSON で Anthropic 直接
    モデルが格納される形式) は `_canonicalize_litellm_id()` で `anthropic/<bare>` に正規化
    した値を `model_name_short` / `display_name` に設定する。

    Returns:
        フォーマット済 dict。`_canonicalize_litellm_id()` がスコープ外と判定した model_id
        (`/` 無しかつ `claude-` で始まらない形式) は None。
    """
    canonical = _canonicalize_litellm_id(model_id)
    if canonical is None:
        return None

    provider_raw, _ = canonical.split("/", 1)
    provider = "OpenAI" if provider_raw == "openai" else provider_raw.capitalize()

    return {
        "provider": provider,
        "model_name_short": canonical,
        "display_name": canonical,
        "mode": info.get("mode", "chat"),
        "max_tokens": info.get("max_tokens"),
        "max_input_tokens": info.get("max_input_tokens"),
        "max_output_tokens": info.get("max_output_tokens"),
        "supports_vision": info.get("supports_vision"),
        "supports_function_calling": info.get("supports_function_calling"),
        "supports_tool_choice": info.get("supports_tool_choice"),
        "supports_parallel_function_calling": info.get("supports_parallel_function_calling"),
        "input_cost_per_token": info.get("input_cost_per_token"),
        "output_cost_per_token": info.get("output_cost_per_token"),
        "deprecation_date": info.get("deprecation_date"),
    }


def _collect_models(
    *,
    require_compatible: bool = True,
    exclude_deprecated: bool = True,
) -> dict[str, dict[str, Any]]:
    """LiteLLM `model_cost` を allowlist / capability / deprecation で filter する。

    Args:
        require_compatible: True なら `supports_vision` + `supports_function_calling` 必須
            (ADR 0023 Phase 1 / Issue #45)。
        exclude_deprecated: True なら `deprecation_date` が設定されている entry を除外。

    Returns:
        `<正規化後 litellm_model_id> -> metadata` のマッピング。Issue #52 (ADR 0023
        Phase 1.10) 以降、bare `claude-*` は `anthropic/claude-*` に正規化された
        キーで格納される。slash 入り ID はそのまま LiteLLM オリジナルキーがキーとなる
        (Issue #51 / Phase 1.9)。
    """
    metadata: dict[str, dict[str, Any]] = {}
    for model_id in litellm.model_cost.keys():
        if not is_allowed_provider(model_id):
            continue
        try:
            info = litellm.get_model_info(model_id)
        except Exception:
            # LiteLLM 側で部分的に metadata が欠ける entry はスキップする。
            continue
        if info is None:
            continue
        if require_compatible and not _is_litellm_model_annotation_compatible(info):
            continue
        if exclude_deprecated and info.get("deprecation_date"):
            continue
        formatted = _format_litellm_metadata(model_id, info)
        if formatted:
            # ADR 0023 Phase 1.10 (Issue #52): dict キーは正規化後 ID で統一。
            # registry / CLI / LoRAIro DB 列に伝播する `model_name_short` と一致させる。
            metadata[formatted["model_name_short"]] = formatted
    return metadata


def discover_available_vision_models() -> dict[str, Any]:
    """利用可能な Vision 対応モデルの一覧を runtime LiteLLM call で取得する。

    Returns:
        ``{"models": list[str], "metadata": dict[str, dict[str, Any]]}``
        - "models": deprecated 除外済みの利用可能モデル ID リスト
        - "metadata": 各モデル ID -> metadata の dict (空の場合あり)
    """
    metadata = _collect_models(require_compatible=True, exclude_deprecated=True)
    logger.info(f"利用可能 Vision モデル: {len(metadata)} 件 (LiteLLM runtime)")
    return {"models": list(metadata.keys()), "metadata": metadata}


def get_available_models() -> list[str]:
    """deprecated 除外済みの利用可能 Vision モデル ID リストを返す。

    旧 `SimplifiedAgentFactory.get_available_models()` の互換 helper。
    """
    metadata = _collect_models(require_compatible=True, exclude_deprecated=True)
    return list(metadata.keys())


def list_all_models() -> list[str]:
    """deprecated 含む全 Vision 対応モデル ID リストを返す。

    旧 `SimplifiedAgentFactory.list_all_models()` の互換 helper。
    """
    metadata = _collect_models(require_compatible=True, exclude_deprecated=False)
    return list(metadata.keys())


def is_model_deprecated(model_id: str) -> bool:
    """指定モデルが LiteLLM の `deprecation_date` に基づいて deprecated か判定する。

    旧 `SimplifiedAgentFactory.is_model_deprecated()` の互換 helper。
    """
    if not is_allowed_provider(model_id):
        return False
    try:
        info = litellm.get_model_info(model_id)
    except Exception:
        return False
    if info is None:
        return False
    return bool(info.get("deprecation_date"))


__all__ = [
    "discover_available_vision_models",
    "get_available_models",
    "is_allowed_provider",
    "is_model_deprecated",
    "list_all_models",
]
