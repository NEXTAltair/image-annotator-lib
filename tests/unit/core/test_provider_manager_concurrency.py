from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from image_annotator_lib.core.types import AnnotationSchema, TaskCapability
from image_annotator_lib.webapi import provider_manager as provider_manager_module
from image_annotator_lib.webapi.provider_manager import ProviderManager


def _images(count: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for index in range(count):
        image = Image.new("RGB", (8, 8), color=(index, index, index))
        image.info["phash"] = f"phash-{index}"
        images.append(image)
    return images


def _patch_phash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_manager_module,
        "calculate_phash",
        lambda image: image.info["phash"],
    )


class _DelayedAgent:
    def __init__(self, delays: list[float]) -> None:
        self._delays = delays
        self._next_index = 0
        self.active = 0
        self.peak_active = 0
        self.started: list[int] = []

    async def run(self, *_args: object, **_kwargs: object) -> object:
        index = self._next_index
        self._next_index += 1
        self.started.append(index)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(self._delays[index])
            return SimpleNamespace(
                output=AnnotationSchema(tags=[f"tag-{index}"], captions=[], score=None, ratings=[])
            )
        finally:
            self.active -= 1


class _ScriptedAgent:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self._next_index = 0

    async def run(self, *_args: object, **_kwargs: object) -> object:
        index = self._next_index
        self._next_index += 1
        outcome = self._outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(output=outcome)


def test_webapi_requests_are_bounded_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_phash(monkeypatch)
    agent = _DelayedAgent([0.03, 0.03, 0.03, 0.03, 0.03])

    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=_images(5),
        litellm_model_id="openai/gpt-4o-mini",
        capabilities={TaskCapability.TAGS},
        max_concurrency=2,
        _test_agent=agent,
    )

    assert agent.peak_active == 2
    assert list(results) == [f"phash-{index}" for index in range(5)]
    assert [results[f"phash-{index}"]["tags"][0] for index in range(5)] == [
        f"tag-{index}" for index in range(5)
    ]


def test_webapi_result_order_is_stable_when_completion_order_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_phash(monkeypatch)
    agent = _DelayedAgent([0.04, 0.01, 0.02])

    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=_images(3),
        litellm_model_id="openai/gpt-4o-mini",
        capabilities={TaskCapability.TAGS},
        max_concurrency=3,
        _test_agent=agent,
    )

    assert list(results) == ["phash-0", "phash-1", "phash-2"]
    assert results["phash-0"]["tags"] == ["tag-0"]
    assert results["phash-1"]["tags"] == ["tag-1"]
    assert results["phash-2"]["tags"] == ["tag-2"]


def test_webapi_concurrent_path_preserves_per_image_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_phash(monkeypatch)
    agent = _ScriptedAgent(
        [
            AnnotationSchema(tags=["ok"], captions=[], score=None, ratings=[]),
            AnnotationSchema(tags=[], captions=[], score=None, ratings=[]),
            RuntimeError("stop_reason=refusal"),
            RuntimeError("network timeout"),
        ]
    )

    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=_images(4),
        litellm_model_id="openai/gpt-4o-mini",
        capabilities={TaskCapability.TAGS},
        max_concurrency=4,
        _test_agent=agent,
    )

    assert results["phash-0"]["error"] is None
    assert results["phash-0"]["tags"] == ["ok"]
    assert results["phash-1"]["error_code"] == "EMPTY_ANNOTATION"
    assert results["phash-1"]["retryable"] is False
    assert results["phash-2"]["error_code"] == "SAFETY_REFUSAL"
    assert results["phash-2"]["retryable"] is False
    assert "RuntimeError: network timeout" in results["phash-3"]["error"]


@pytest.mark.parametrize("max_concurrency", [0, -1, True, 1.5])
def test_webapi_rejects_invalid_max_concurrency(max_concurrency: Any) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        ProviderManager.run_inference_with_model(
            model_name="test-webapi",
            images_list=_images(1),
            litellm_model_id="openai/gpt-4o-mini",
            max_concurrency=max_concurrency,
            _test_agent=_DelayedAgent([0]),
        )


def test_webapi_max_concurrency_one_allows_serial_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_phash(monkeypatch)
    agent = _DelayedAgent([0, 0, 0])

    results = ProviderManager.run_inference_with_model(
        model_name="test-webapi",
        images_list=_images(3),
        litellm_model_id="openai/gpt-4o-mini",
        capabilities={TaskCapability.TAGS},
        max_concurrency=1,
        _test_agent=agent,
    )

    assert agent.peak_active == 1
    assert list(results) == ["phash-0", "phash-1", "phash-2"]
