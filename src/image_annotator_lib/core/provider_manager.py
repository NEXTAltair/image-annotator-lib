"""WebAPI 推論経路の中核 (ADR 0023 Phase 1)。

LiteLLM ID から PydanticAI native provider/model を構築し、Agent を毎回新規作成して
推論を実行する。Agent / Provider / Model はキャッシュしない。`os.environ` は一切
mutate しない (`api_keys` 経由の明示注入のみ)。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from PIL import Image
from pydantic_ai import Agent

from ..exceptions.errors import (
    ContentPolicyRefusalError,
    InferenceError,
    MissingApiKeyError,
    SafetyRefusalError,
)
from ..model_class.annotator_webapi.webapi_shared import BASE_PROMPT
from .http_retry import build_retry_http_client
from .image_preprocess import preprocess_images_to_binary
from .model_id import build_pydantic_model, resolve_model_ref
from .output_normalization import normalize_annotation_output
from .result_adapter import to_annotation_result
from .types import AnnotationResult
from .utils import calculate_phash, logger

# ADR 0023 retry policy:
# - output normalization / schema validation failure: PydanticAI `output_retries=1` で 1 回再生成
#   (`ModelRetry` 経由)。
# - HTTP/API transient failure: Phase 1.8 (Issue #46) で `core/http_retry.py` の
#   `AsyncTenacityTransport` が provider HTTP client 層で initial+2 retries (max 3 attempts)。
# 両 retry は独立した経路で機能し、互いに干渉しない。
_OUTPUT_RETRIES = 1

# Agent.run の user message として渡す短い指示。BASE_PROMPT は system_prompt 側で詳細を伝える。
_USER_PROMPT_TEXT = "Analyze this image and provide annotations as specified."

# LoRAIro #274: cause/context chain を辿る最大段数 (循環・過剰深さの安全弁)。
_EXCEPTION_CHAIN_MAX_DEPTH = 5


def _format_exception_chain(exc: BaseException, *, max_depth: int = _EXCEPTION_CHAIN_MAX_DEPTH) -> str:
    """例外の cause/context chain を 1 行の診断文字列に整形する (LoRAIro #274)。

    ``openai.APIConnectionError`` 等の SDK ラッパー例外は ``str(exc)`` が
    ``"Connection error."`` のような短い定型文に潰れ、根本原因 (httpx の
    ``ConnectTimeout`` / ``ConnectError`` 等) が ``AnnotationResult.error`` から
    失われる。``__cause__`` / ``__context__`` を辿って各層の型名とメッセージを
    連結し、原因特定に足る error 文字列を生成する。

    Args:
        exc: 整形対象の例外。
        max_depth: 辿る chain の最大段数。

    Returns:
        ``"openai.APIConnectionError: Connection error. <- caused by httpx.ConnectError: ..."``
        形式の 1 行文字列。chain が無ければ単層の ``"型名: message"``。
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and id(current) not in seen and depth < max_depth:
        seen.add(id(current))
        cls = type(current)
        qualified = (
            cls.__name__ if cls.__module__ in ("builtins", "") else f"{cls.__module__}.{cls.__name__}"
        )
        message = str(current).strip() or "(no message)"
        parts.append(f"{qualified}: {message}")
        # __cause__ (明示的 raise from) を優先。無く __context__ が suppress されて
        # いなければ暗黙の context を辿る。
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
        depth += 1
    return " <- caused by ".join(parts)


class ProviderManager:
    """LiteLLM ID ベースで PydanticAI 推論を実行するクラスメソッド集。

    Agent / Provider / Model object はキャッシュせず、推論呼び出しごとに新規作成する。
    `_resolve_api_key` は `api_keys` dict と `config_registry` のみを参照し、
    環境変数 fallback / `os.environ` mutate は行わない。
    """

    @classmethod
    async def run_inference_with_model_async(
        cls,
        *,
        model_name: str,
        images_list: list[Image.Image],
        litellm_model_id: str,
        api_keys: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        _test_agent: Agent | None = None,
    ) -> dict[str, AnnotationResult]:
        """非同期推論の中核実装。

        Args:
            model_name: registry 経由のモデル名 (registry 登録モデル / 直接 LiteLLM ID)。
            images_list: 推論対象の PIL Image リスト。
            litellm_model_id: LiteLLM 形式 ID (`openai/gpt-4o` 等)。
            api_keys: provider 名 (`openai` / `anthropic` / `google` / `openrouter`) 単位の API key dict。
            config: provider 固有の追加設定 (OpenRouter `referer` / `app_name` 等)。
            _test_agent: pytest fixture からの Agent 注入専用 (本番では None)。

        Returns:
            `dict[phash, AnnotationResult]`。画像ごとに 1 entry。
        """
        # ADR 0023 Phase 1.8 (Issue #46): transport retry 付き AsyncClient を推論呼び出しごとに
        # 新規生成する。Agent / Provider / Model キャッシュなし方針 (ADR 0023 Agent ライフサイクル)
        # に揃え、httpx.AsyncClient も同じ寿命で aclose() する。test 経路 (`_test_agent` 注入) では
        # provider 構築自体を skip するため http_client は不要 (None のまま)。
        http_client = None
        if _test_agent is not None:
            agent = _test_agent
        else:
            ref = resolve_model_ref(litellm_model_id, config)
            # ADR 0023 Phase 1 (Issue #45): capability check は discovery 段階で完結。
            # registry 経由の通常パスでは登録時に supports_vision / supports_function_calling
            # が確認済みのため、推論直前 fail-fast は冗長として削除した。
            api_key = cls._resolve_api_key(model_name, ref.provider, api_keys)
            http_client = build_retry_http_client()
            try:
                model = build_pydantic_model(ref, api_key, config, http_client=http_client)
                agent = Agent(
                    model=model,
                    output_type=normalize_annotation_output,
                    system_prompt=BASE_PROMPT,
                    output_retries=_OUTPUT_RETRIES,
                )
            except BaseException:
                # provider/model/Agent 構築途中で失敗した場合は、開いた AsyncClient を確実に閉じる。
                await http_client.aclose()
                raise

        try:
            binary_contents = preprocess_images_to_binary(images_list)
            results: dict[str, AnnotationResult] = {}
            for index, (image, binary) in enumerate(zip(images_list, binary_contents, strict=True)):
                phash = calculate_phash(image) or f"unknown_image_{index}"
                try:
                    run_result = await agent.run([_USER_PROMPT_TEXT, binary])
                    results[phash] = to_annotation_result(run_result.output, phash)
                except Exception as exc:
                    refusal_exc = _classify_refusal(exc, litellm_model_id, phash)
                    if refusal_exc is not None:
                        # ADR 0023 Phase 1.5 (Issue #42): UnifiedAnnotationResult.error に
                        # exception type 名 prefix で構造的伝搬。LoRAIro 側で error_records に
                        # decode する (annotation_save_service)。
                        logger.warning(
                            f"WebAPI safety refusal: model={model_name}, "
                            f"litellm_model_id={litellm_model_id}, phash={phash}, "
                            f"reason={refusal_exc.provider_refusal_reason or 'no reason'}"
                        )
                        results[phash] = to_annotation_result(
                            None, phash, error=f"{type(refusal_exc).__name__}: {refusal_exc}"
                        )
                        continue
                    # ADR 0023 Phase 1.8 (Issue #46): transport retry 枯渇時は
                    # tenacity が httpx 例外を reraise する (RetryConfig(reraise=True))。
                    # 既存の str(exc) 文字列伝搬経路でそのまま LoRAIro 側に流れる。
                    #
                    # LoRAIro #274: openai.APIConnectionError 等は str(exc) が
                    # "Connection error." に潰れ、根本原因 (httpx の ConnectTimeout 等)
                    # が AnnotationResult.error から失われる。cause/context chain を
                    # 辿った診断文字列を log・result.error の双方に伝搬する。
                    error_detail = _format_exception_chain(exc)
                    # Issue #69 / LoRAIro #275: `logger.error(message, exc_info=True)` は
                    # loguru 内部で `message.format(*args, **kwargs)` を呼ぶため、f-string
                    # 結果に str(exc) 由来の `{'type': ...}` dict 表現が含まれると
                    # placeholder と誤認識され `KeyError("'type'")` が leak する。
                    # `.opt(exception=exc).error(message)` パターンに変更して loguru の
                    # message 再フォーマットを回避する。
                    logger.opt(exception=exc).error(
                        f"WebAPI 推論失敗: model={model_name}, "
                        f"litellm_model_id={litellm_model_id}, error={error_detail}"
                    )
                    results[phash] = to_annotation_result(None, phash, error=error_detail)
            return results
        finally:
            if http_client is not None:
                await http_client.aclose()

    @classmethod
    def run_inference_with_model(
        cls,
        *,
        model_name: str,
        images_list: list[Image.Image],
        litellm_model_id: str,
        api_keys: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        _test_agent: Agent | None = None,
    ) -> dict[str, AnnotationResult]:
        """`run_inference_with_model_async()` の sync wrapper。

        Running asyncio loop の中で呼ばれた場合は `InferenceError` を raise する
        (thread fallback はしない)。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                cls.run_inference_with_model_async(
                    model_name=model_name,
                    images_list=images_list,
                    litellm_model_id=litellm_model_id,
                    api_keys=api_keys,
                    config=config,
                    _test_agent=_test_agent,
                )
            )
        raise InferenceError(
            "ProviderManager.run_inference_with_model() called from a running event loop. "
            "Use run_inference_with_model_async() in async context.",
            litellm_model_id=litellm_model_id,
        )

    @classmethod
    def _resolve_api_key(
        cls,
        model_name: str,
        provider: str,
        api_keys: dict[str, str] | None,
    ) -> str:
        """API key を `api_keys` dict から解決する (ADR 0023 Phase 1)。

        ADR 0023: API key は provider object への明示注入のみを許容する。env mutate /
        env fallback / `config_registry.get(model_name, "api_key")` legacy fallback は
        いずれも行わない。`api_keys` dict に該当 provider のキーが無ければ即
        `MissingApiKeyError` を raise する。

        Args:
            model_name: 呼び出し元モデル名 (logging 用、解決には使わない)。
            provider: `core/model_id.SUPPORTED_PROVIDERS` の provider 名。
            api_keys: provider -> API key の dict。
        """
        if api_keys and provider in api_keys and api_keys[provider]:
            return api_keys[provider]

        raise MissingApiKeyError(provider=provider)


# --- Refusal 分類のための内部 helper ---
#
# ADR 0023 Phase 1.5 (Codex P1 r3212094633): PydanticAI 統合の利点を活かし、
# exception の string を grep する旧方式から、`exc.body` の構造化ペイロード
# (JSON / dict) を再帰探索して `finish_reason` / `stop_reason` フィールドを
# 直接取得する設計に変更。文字列 grep は legacy fallback としてのみ残す。
#
# 構造化ファースト方式の利点:
# - JSON quote 形式 (`"finishReason": "SAFETY"`) も provider 横断で一様に処理
# - `finish_reason` フィールド限定検査で偽陽性 (e.g. "safety filter applied" 等の
#   誤マッチ) を構造的に排除
# - Provider 別 response 構造 (`candidates[].finishReason` / `choices[].finish_reason`
#   / 直下 `stop_reason`) を再帰 walker で吸収

_REASON_KEY_NORMALIZED = frozenset({"finishreason", "stopreason"})

# Legacy fallback: body が無い / JSON でない exception 経路用 (PydanticAI 以外の
# 例外型、または旧 SDK 直叩き経路)。`.lower()` 適用後の文字列で snake_case / camelCase
# / `:` / `=` を吸収する。引用符を許容するため `["']*` を含む。
_FINISH_REASON_SAFETY_RE = re.compile(r"""finish_?reason["']?\s*[:=]\s*["']?safety""")
_STOP_REASON_REFUSAL_RE = re.compile(r"""stop_?reason["']?\s*[:=]\s*["']?refusal""")
_FINISH_REASON_CONTENT_FILTER_RE = re.compile(r"""finish_?reason["']?\s*[:=]\s*["']?content_filter""")


def _walk_for_reasons(obj: Any, reasons: list[str] | None = None) -> list[str]:
    """dict / list を再帰的に walk して `finishReason`/`stopReason` の値を **全て** 収集する。

    Codex P2 (PR #44 r3212139750): 単一値しか返さないと dict 走査順依存で
    precedence が崩れる (e.g. body に `stop_reason="refusal"` と
    `finish_reason="content_filter"` が両立する場合、本来 ContentPolicy 優先
    すべきところ traversal order によって SafetyRefusal を返す不具合)。
    全件収集して caller 側 (`_classify_reasons`) で precedence 判定する設計に分離。

    Args:
        obj: 探索対象 (dict / list / その他 — その他は素通り)。
        reasons: 累積バッファ (再帰用、外部呼び出しでは None 推奨)。

    Returns:
        traversal 順での全 reason 値 (lowercase) のリスト。見つからなければ空。
    """
    if reasons is None:
        reasons = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                normalized_key = key.lower().replace("_", "")
                if normalized_key in _REASON_KEY_NORMALIZED and isinstance(value, str):
                    reasons.append(value.lower())
            _walk_for_reasons(value, reasons)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_reasons(item, reasons)
    return reasons


def _extract_reasons_from_body(body: Any) -> list[str]:
    """PydanticAI exception の `body` 属性から refusal reason を全件抽出する。

    `UnexpectedModelBehavior(message, body)` / `ModelHTTPError(..., body)` は
    body 属性に response payload を持つ。JSON 文字列 / dict / list / object を
    安全に parse して `finishReason` / `stop_reason` 等を取り出す。

    Args:
        body: PydanticAI exception の body 属性 (str / dict / list / None)。

    Returns:
        traversal 順での全 reason 値リスト (lowercase)。
        body が None / 非 JSON 文字列 / 非構造化 object なら空リスト。
    """
    if body is None:
        return []

    parsed: Any = body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            # JSON でない body (plain text 等) は legacy fallback に任せる
            return []

    return _walk_for_reasons(parsed)


# Refusal classification の precedence (Codex P2 r3212139750):
# legacy regex 経路と整合させる: ContentPolicy (`content_filter`) > Safety
# (`refusal` / `safety` / `blocked`) の順で判定する。両者が body に共存
# する場合 (e.g. `stop_reason="refusal"` + `finish_reason="content_filter"`)
# でも ContentPolicy が優先される。
_CONTENT_FILTER_REASONS = frozenset({"content_filter"})
_SAFETY_REASONS = frozenset({"safety", "refusal", "blocked"})


def _classify_reasons(
    reasons: list[str],
) -> type[SafetyRefusalError] | type[ContentPolicyRefusalError] | None:
    """抽出された reason 集合を refusal exception type に precedence 付きで分類する。

    Codex P2 (PR #44 r3212139750): `_walk_for_reasons` が全件返すため、ここで
    legacy regex 経路と同じ precedence (ContentPolicy > Safety) を enforce する。

    Args:
        reasons: 抽出された全 reason 値リスト (lowercase)。

    Returns:
        対応する exception type、または分類対象外なら None。
    """
    if not reasons:
        return None
    reasons_set = set(reasons)
    if reasons_set & _CONTENT_FILTER_REASONS:
        return ContentPolicyRefusalError
    if reasons_set & _SAFETY_REASONS:
        return SafetyRefusalError
    return None


def _classify_refusal(
    exc: BaseException,
    litellm_model_id: str,
    image_phash: str,
) -> SafetyRefusalError | ContentPolicyRefusalError | None:
    """PydanticAI / provider SDK exception から safety/content refusal を検出する。

    判定は 3 段階の優先度で行う (Codex P1 r3212094633 / Issue #42):

    1. **PydanticAI `exc.body` 構造化ペイロード**: `UnexpectedModelBehavior`/
       `ModelHTTPError` が body 属性で provider response (JSON / dict) を保持
       するため、これを再帰 walk して `finishReason`/`stop_reason` を直接抽出。
       - `finish_reason=content_filter` → ContentPolicyRefusalError
       - `finish_reason=safety` / `stop_reason=refusal` / blocked → SafetyRefusalError
       Provider 横断で一様に動作し、引用符 / 区切りの違いに影響されない。

    2. **Exception type 名 fallback**: PydanticAI 以外の SDK で provider 専用例外
       型を持つ場合 (e.g. OpenAI `ContentFilterFinishReasonError`)。

    3. **String regex fallback (legacy)**: body 属性が無い経路 / 非 JSON body
       のため、message 文字列を case-insensitive 部分マッチ。snake_case/camelCase
       および `:`/`=` 区切り、引用符の有無を正規表現で吸収。

    Args:
        exc: catch された例外
        litellm_model_id: 推論対象モデル ID
        image_phash: 対象画像の pHash

    Returns:
        分類された refusal exception、または該当しなければ None
        (caller が generic error として扱う)。
    """
    # Priority 1: PydanticAI exc.body 構造化ペイロード
    body = getattr(exc, "body", None)
    body_reasons = _extract_reasons_from_body(body)
    refusal_cls = _classify_reasons(body_reasons)
    if refusal_cls is not None:
        # provider_refusal_reason には全検出 reason を含める (debug / log 用途)
        reason_summary = ",".join(body_reasons)
        return refusal_cls(
            litellm_model_id=litellm_model_id,
            image_phash=image_phash,
            provider_refusal_reason=f"finish_reason={reason_summary}"[:200],
        )

    # Priority 2: Exception type 名 (OpenAI ContentFilterFinishReasonError 等)
    exc_type_name = type(exc).__name__.lower()
    if "contentfilter" in exc_type_name:
        return ContentPolicyRefusalError(
            litellm_model_id=litellm_model_id,
            image_phash=image_phash,
            provider_refusal_reason=str(exc)[:200],
        )
    if "refusal" in exc_type_name:
        return SafetyRefusalError(
            litellm_model_id=litellm_model_id,
            image_phash=image_phash,
            provider_refusal_reason=str(exc)[:200],
        )

    # Priority 3: String regex fallback (legacy / 非 PydanticAI 経路)
    # message 本体に加え、body が非 JSON で priority 1 が failed した場合の
    # body 文字列も搜索対象に含める (PydanticAI 以外の plain-text body 救済)。
    exc_message = str(exc).lower()
    if body is not None and not body_reasons:
        body_str = body if isinstance(body, str) else str(body)
        exc_message = f"{exc_message} {body_str.lower()}"

    is_content_filter = (
        "content_filter" in exc_message or _FINISH_REASON_CONTENT_FILTER_RE.search(exc_message) is not None
    )
    is_safety_refusal = (
        _STOP_REASON_REFUSAL_RE.search(exc_message) is not None
        or _FINISH_REASON_SAFETY_RE.search(exc_message) is not None
        or "blocked due to safety" in exc_message
        or "blockedreason" in exc_message
    )

    if is_content_filter:
        return ContentPolicyRefusalError(
            litellm_model_id=litellm_model_id,
            image_phash=image_phash,
            provider_refusal_reason=str(exc)[:200],
        )
    if is_safety_refusal:
        return SafetyRefusalError(
            litellm_model_id=litellm_model_id,
            image_phash=image_phash,
            provider_refusal_reason=str(exc)[:200],
        )
    return None


__all__ = ["ProviderManager"]
