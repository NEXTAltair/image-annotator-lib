"""実 provider API key を使った WebAPI on-demand validation (ADR 0001 amended)。

`_BUILDER_DISPATCH` (`core/model_id.py`) で対応する 4 provider (OpenAI / Anthropic /
Google / OpenRouter) を各 1 model でカバーし、`image_annotator_lib.annotate()` public
API 経由で実 WebAPI request を送る。PydanticAI / LiteLLM / provider SDK の breaking
change を開発者ローカルで早期検知することが目的。

Runtime cost:
    1 回の test run で 4 provider × 1 推論 ≒ ~$0.001 (合計)。CI 不経由なので積算は
    開発者の手元、月数 USD 以下を想定 (Tier 2 / LoRAIro #278 参照)。

Skip 仕様 (ADR 0001 amended):
    API key 環境変数 (`GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
    `OPENROUTER_API_KEY`) が未設定の provider は明示的に `pytest.skip` する。
    product failure ではないことを reason に明記。

Marker:
    `@pytest.mark.calls_real_webapi` (CI 不経由、ローカル only)。

Related:
    LoRAIro #274 (OpenAI Connection error) / LoRAIro #275 (Anthropic KeyError 'type') の
    regression 検出 path。既知バグは FAIL 出力を Issue にコメントして追跡する。

    LoRAIro umbrella #276 / Tier 2 #278 / iam-lib #71 / ADR 0001 amended 2026-05-18
"""

import os
from pathlib import Path

import pytest
from PIL import Image

from image_annotator_lib import annotate

_RESOURCE_IMG = Path(__file__).parent.parent / "resources" / "img" / "1_img" / "file07.webp"

# (model_name, provider key in api_keys dict, env var name).
# provider key は `core/model_id.py:_BUILDER_DISPATCH` のキー (`gemini` は `google` に
# 正規化されるため `api_keys` には `"google"` を渡す)。
_PROVIDER_KEY_MAP: dict[str, tuple[str, str]] = {
    "gemini/gemini-flash-lite-latest": ("google", "GEMINI_API_KEY"),
    "openai/gpt-4o-mini": ("openai", "OPENAI_API_KEY"),
    "anthropic/claude-haiku-4-5": ("anthropic", "ANTHROPIC_API_KEY"),
    "openrouter/anthropic/claude-haiku-4.5": ("openrouter", "OPENROUTER_API_KEY"),
}


@pytest.mark.calls_real_webapi
@pytest.mark.parametrize("model_name", list(_PROVIDER_KEY_MAP.keys()))
def test_real_webapi_runtime(model_name: str) -> None:
    """4 provider の各 1 model で実 WebAPI request を送り、output が non-empty かを確認する。

    API key が未設定の provider は skip。`error is None` かつ tags / captions / scores
    のいずれかが non-empty であることを smoke 条件とする。
    """
    provider, key_env = _PROVIDER_KEY_MAP[model_name]
    api_key = os.environ.get(key_env)
    if not api_key:
        pytest.skip(f"{key_env} not set (provider={provider})")
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    img = Image.open(_RESOURCE_IMG).convert("RGB")
    result = annotate(
        images_list=[img],
        model_name_list=[model_name],
        api_keys={provider: api_key},
    )

    assert len(result) == 1, f"expected 1 phash entry, got {len(result)}"
    for _phash, models in result.items():
        assert model_name in models, (
            f"{model_name} missing in result keys: {list(models.keys())}"
        )
        ann = models[model_name]
        assert ann.error is None, f"{model_name} returned error: {ann.error}"
        output_present = bool(ann.tags) or bool(ann.captions) or bool(ann.scores) or bool(
            ann.score_labels
        )
        assert output_present, (
            f"{model_name}: all output fields empty "
            f"(tags={ann.tags!r}, captions={ann.captions!r}, "
            f"scores={ann.scores!r}, score_labels={ann.score_labels!r})"
        )
