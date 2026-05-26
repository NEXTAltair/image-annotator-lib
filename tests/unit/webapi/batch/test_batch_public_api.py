from __future__ import annotations

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


def test_list_batch_capable_models_returns_anthropic_metadata(monkeypatch) -> None:
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
                "litellm_model_id": "openai/gpt-4o",
                "capabilities": ["tags"],
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
