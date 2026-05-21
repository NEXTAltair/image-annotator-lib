"""Issue #47: ProviderManager uses PydanticAI output normalization."""

from __future__ import annotations

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from image_annotator_lib.core.output_normalization import normalize_annotation_output
from image_annotator_lib.core.provider_manager import ProviderManager


def test_provider_manager_returns_normalized_annotation_result() -> None:
    agent = Agent(
        model=TestModel(
            custom_output_args={
                "tags": "cat, dog",
                "captions": "a cat outside",
                "score": "0.82",
            }
        ),
        output_type=normalize_annotation_output,
        output_retries=1,
    )
    image = Image.new("RGB", (8, 8), color="white")

    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=[image],
        litellm_model_id="openai/gpt-4o",
        _test_agent=agent,
    )

    result = next(iter(results.values()))
    assert result["error"] is None
    assert result["tags"] == ["cat", "dog"]
    assert result["formatted_output"] == {
        "tags": ["cat", "dog"],
        "captions": ["a cat outside"],
        "score": 0.82,
        "ratings": [],
    }
