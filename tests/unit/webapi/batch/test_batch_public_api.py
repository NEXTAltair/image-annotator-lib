from __future__ import annotations

import pytest

import image_annotator_lib
from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.webapi.batch import (
    BatchErrorPhase,
    BatchJobError,
    BatchSubmitRequest,
    list_batch_capable_models,
    service,
    submit_batch,
)


def test_public_batch_api_exports() -> None:
    assert image_annotator_lib.BatchSubmitRequest is BatchSubmitRequest
    assert callable(image_annotator_lib.submit_batch)
    assert callable(image_annotator_lib.retrieve_batch)
    assert callable(image_annotator_lib.cancel_batch)
    assert callable(image_annotator_lib.fetch_batch_results)
    assert callable(image_annotator_lib.list_batch_capable_models)


def test_submit_rejects_unsupported_provider() -> None:
    request = BatchSubmitRequest(
        provider="openrouter",
        endpoint="messages",
        litellm_model_id="openrouter/anthropic/claude",
        prompt_profile="default",
        description=None,
        api_keys={},
        items=[],
    )

    try:
        submit_batch(request)
    except BatchJobError as exc:
        assert exc.phase is BatchErrorPhase.PREPARE
        assert exc.code == "unsupported_provider"
    else:
        raise AssertionError("Expected BatchJobError")


def test_submit_rejects_denylisted_model() -> None:
    """#152: cost-safety denylist は submit 経路でも強制する (Codex P2)。

    list_batch_capable_models() から隠すだけでは、BatchSubmitRequest を直接組み立てる
    呼び出しや一覧更新前のキャッシュ選択が denylisted model を素通りで dispatch できる。
    `gpt-5.5-pro` 系は submit_batch() でも PREPARE phase で拒否する。
    """
    request = BatchSubmitRequest(
        provider="openai",
        endpoint="/v1/chat/completions",
        litellm_model_id="openai/gpt-5.5-pro",
        prompt_profile="default",
        description=None,
        api_keys={"openai": "sk-test"},
        items=[],
    )

    try:
        submit_batch(request)
    except BatchJobError as exc:
        assert exc.phase is BatchErrorPhase.PREPARE
        assert exc.code == "denylisted_model"
    else:
        raise AssertionError("Expected BatchJobError")


def test_list_batch_capable_models_returns_anthropic_metadata(monkeypatch) -> None:
    monkeypatch.setattr(service, "list_available_annotators", lambda: ["claude"])
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "claude": {
                "provider": "anthropic",
                "litellm_model_id": "anthropic/claude-3-5-haiku-latest",
                "capabilities": ["tags", "captions", "scores", "ratings"],
            },
        }.get(name),
    )

    models = list_batch_capable_models()

    assert len(models) == 1
    assert models[0].provider == "anthropic"
    assert models[0].litellm_model_id == "anthropic/claude-3-5-haiku-latest"
    assert TaskCapability.RATINGS in models[0].capabilities
    assert models[0].metadata["result_retention_days"] == 29
    assert models[0].metadata["zero_data_retention_eligible"] is False


def test_list_batch_capable_models_includes_openai_non_ratings_model(monkeypatch) -> None:
    """#152: RATINGS 必須フィルタ撤去後、chat-completions 系 OpenAI モデルも一覧に含む。

    旧実装は OpenAI に RATINGS capability を要求し、tags/captions/scores しか持たない
    chat-completions モデルを誤って除外していた (moderation endpoint 専用 adapter として
    追加された経緯の残存)。adapter-availability gate では provider に adapter があれば
    capability に依らず batch-capable として返す。
    """
    monkeypatch.setattr(service, "list_available_annotators", lambda: ["gpt"])
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "gpt": {
                "provider": "openai",
                "litellm_model_id": "openai/gpt-4o",
                "capabilities": ["tags", "captions", "scores"],
            },
        }.get(name),
    )

    models = list_batch_capable_models()

    assert len(models) == 1
    assert models[0].provider == "openai"
    assert models[0].litellm_model_id == "openai/gpt-4o"
    assert TaskCapability.RATINGS not in models[0].capabilities
    assert models[0].metadata["provider_batch_api"] == "openai_batch"


def test_list_batch_capable_models_excludes_gpt_5_5_pro_denylist(monkeypatch) -> None:
    """#152: `gpt-5.5-pro` family は cost-safety denylist で除外する (ADR 0005)。

    非 pro の `gpt-5.5` は対象外で一覧に残る。
    """
    monkeypatch.setattr(
        service, "list_available_annotators", lambda: ["gpt-5.5-pro", "gpt-5.5-pro-dated", "gpt-5.5"]
    )
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "gpt-5.5-pro": {
                "provider": "openai",
                "litellm_model_id": "openai/gpt-5.5-pro",
                "capabilities": ["tags", "captions", "scores"],
            },
            "gpt-5.5-pro-dated": {
                "provider": "openai",
                "litellm_model_id": "openai/gpt-5.5-pro-2026-04-23",
                "capabilities": ["tags", "captions", "scores"],
            },
            "gpt-5.5": {
                "provider": "openai",
                "litellm_model_id": "openai/gpt-5.5",
                "capabilities": ["tags", "captions", "scores"],
            },
        }.get(name),
    )

    models = list_batch_capable_models()

    assert {model.litellm_model_id for model in models} == {"openai/gpt-5.5"}


def test_list_batch_capable_models_skips_models_without_webapi_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ローカル ML モデルは get_webapi_metadata() が None を返す (iam-lib #128)。

    list_available_annotators() はローカル ML モデル (WDTagger 等) も列挙するが、
    それらは _WEBAPI_MODEL_METADATA 未登録のため get_webapi_metadata() が None を返す。
    None を skip せず .get() を呼ぶと AttributeError でループ全体が落ち、batch-capable
    一覧が常に空になる。None メタデータのモデルは skip し WebAPI モデルだけ返すこと。
    """
    monkeypatch.setattr(service, "list_available_annotators", lambda: ["wd-tagger", "claude"])
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "claude": {
                "provider": "anthropic",
                "litellm_model_id": "anthropic/claude-3-5-haiku-latest",
                "capabilities": ["tags", "captions", "scores", "ratings"],
            }
        }.get(name),  # "wd-tagger" は None (ローカル ML モデルを再現)
    )

    models = list_batch_capable_models()

    assert {model.litellm_model_id for model in models} == {"anthropic/claude-3-5-haiku-latest"}


def test_list_batch_capable_models_includes_openai_ratings_model(monkeypatch) -> None:
    monkeypatch.setattr(service, "list_available_annotators", lambda: ["claude", "gpt"])
    monkeypatch.setattr(
        service,
        "get_webapi_metadata",
        lambda name: {
            "claude": {
                "provider": "anthropic",
                "litellm_model_id": "anthropic/claude-3-5-haiku-latest",
                "capabilities": ["tags", "captions", "scores", "ratings"],
            },
            "gpt": {
                "provider": "openai",
                "litellm_model_id": "openai/omni-moderation-latest",
                "capabilities": ["ratings"],
            },
        }.get(name),
    )

    models = list_batch_capable_models()
    providers = {model.provider for model in models}

    assert providers == {"anthropic", "openai"}
    openai_model = next(model for model in models if model.provider == "openai")
    assert openai_model.litellm_model_id == "openai/omni-moderation-latest"
    assert openai_model.metadata["provider_batch_api"] == "openai_batch"
    assert openai_model.metadata["zero_data_retention_eligible"] is True
