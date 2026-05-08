"""WebAPI 推論経路の中核 (ADR 0023 Phase 1)。

LiteLLM ID から PydanticAI native provider/model を構築し、Agent を毎回新規作成して
推論を実行する。Agent / Provider / Model はキャッシュしない。`os.environ` は一切
mutate しない (`api_keys` 経由の明示注入のみ)。

参考: docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md
"""

from __future__ import annotations

import asyncio
from typing import Any

import litellm
from PIL import Image
from pydantic_ai import Agent

from ..exceptions.errors import (
    ContentPolicyRefusalError,
    InferenceError,
    MissingApiKeyError,
    SafetyRefusalError,
    VisionUnsupportedError,
)
from ..model_class.annotator_webapi.webapi_shared import BASE_PROMPT
from .image_preprocess import preprocess_images_to_binary
from .model_id import build_pydantic_model, resolve_model_ref
from .result_adapter import to_annotation_result
from .types import AnnotationResult, AnnotationSchema
from .utils import calculate_phash, logger

# ADR 0023 Phase 1 retry policy: structured output validation failure を 1 回再生成
_OUTPUT_RETRIES = 1

# Agent.run の user message として渡す短い指示。BASE_PROMPT は system_prompt 側で詳細を伝える。
_USER_PROMPT_TEXT = "Analyze this image and provide annotations as specified."


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
        if _test_agent is not None:
            agent = _test_agent
        else:
            ref = resolve_model_ref(litellm_model_id, config)
            # ADR 0023: 推論実行直前に LiteLLM の capability で fail-fast
            if not litellm.supports_vision(ref.litellm_model_id):
                raise VisionUnsupportedError(litellm_model_id=ref.litellm_model_id)
            api_key = cls._resolve_api_key(model_name, ref.provider, api_keys)
            model = build_pydantic_model(ref, api_key, config)
            agent = Agent(
                model=model,
                output_type=AnnotationSchema,
                system_prompt=BASE_PROMPT,
                output_retries=_OUTPUT_RETRIES,
            )

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
                logger.error(
                    f"WebAPI 推論失敗: model={model_name}, "
                    f"litellm_model_id={litellm_model_id}, error={exc}",
                    exc_info=True,
                )
                results[phash] = to_annotation_result(None, phash, error=str(exc))
        return results

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


def _classify_refusal(
    exc: BaseException,
    litellm_model_id: str,
    image_phash: str,
) -> SafetyRefusalError | ContentPolicyRefusalError | None:
    """PydanticAI / provider SDK exception から safety/content refusal を検出する。

    検出パターン (ADR 0023 Phase 1.5):
    - exception type 名に "ContentFilter" 含む (OpenAI) → ContentPolicyRefusalError
    - exception message に "content_filter" 含む → ContentPolicyRefusalError
    - exception type 名に "Refusal" 含む (Anthropic) → SafetyRefusalError
    - exception message の signature 検査:
      - "stop_reason=refusal" / "stop_reason: refusal" → SafetyRefusalError
      - "finishreason: safety" / "finish_reason=safety" → SafetyRefusalError
      - "blocked due to safety" / "blockedreason" → SafetyRefusalError
    - 該当しなければ None (caller が generic error として扱う)

    PydanticAI / provider SDK の実 surface に追従するため、case-insensitive な
    部分マッチで検出する。実例外の signature が判明次第 follow-up で精度向上可能。

    Args:
        exc: catch された例外
        litellm_model_id: 推論対象モデル ID
        image_phash: 対象画像の pHash

    Returns:
        分類された refusal exception、または None
    """
    exc_type_name = type(exc).__name__.lower()
    exc_message = str(exc).lower()

    # ContentPolicy (OpenAI 系の content_filter) を優先判定
    is_content_filter = "contentfilter" in exc_type_name or "content_filter" in exc_message

    is_safety_refusal = (
        "refusal" in exc_type_name
        or "stop_reason=refusal" in exc_message
        or "stop_reason: refusal" in exc_message
        or "finishreason: safety" in exc_message
        or "finish_reason=safety" in exc_message
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
