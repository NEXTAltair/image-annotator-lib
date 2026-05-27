from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
from PIL import Image

from image_annotator_lib.core.types import RatingPrediction
from image_annotator_lib.exceptions.errors import MissingApiKeyError
from image_annotator_lib.webapi.provider_manager import ProviderManager


class _FakeModerationResponse:
    def model_dump(self, *, mode: str = "python") -> dict:
        assert mode == "json"
        return {
            "results": [
                {
                    "category_scores": {
                        "sexual": 0.91,
                        "violence": 0.05,
                    }
                }
            ]
        }


class _FakeModerations:
    def __init__(self, calls: list[dict], exc: Exception | None = None) -> None:
        self._calls = calls
        self._exc = exc

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _FakeModerationResponse()


class _FakeAsyncOpenAI:
    calls: ClassVar[list[dict]] = []
    exc: ClassVar[Exception | None] = None

    def __init__(self, *, api_key: str, http_client) -> None:
        assert api_key == "test-key"
        assert http_client is not None
        self.moderations = _FakeModerations(self.calls, self.exc)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, exc: Exception | None = None) -> list[dict]:
    calls: list[dict] = []
    fake_cls = type(
        "FakeAsyncOpenAI",
        (_FakeAsyncOpenAI,),
        {"calls": calls, "exc": exc},
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=fake_cls),
    )
    return calls


def test_openai_moderation_dispatch_returns_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_openai(monkeypatch)
    image = Image.new("RGB", (8, 8), color="white")

    results = ProviderManager.run_inference_with_model(
        model_name="openai/omni-moderation-latest",
        images_list=[image],
        litellm_model_id="openai/omni-moderation-latest",
        api_keys={"openai": "test-key"},
    )

    result = next(iter(results.values()))
    assert result["error"] is None
    rating = result["formatted_output"]["ratings"][0]
    assert isinstance(rating, RatingPrediction)
    assert rating.raw_label == "x"
    assert rating.source_scheme == "openai_moderation_v1"
    assert calls[0]["model"] == "omni-moderation-latest"
    assert calls[0]["input"][0]["image_url"]["url"].startswith("data:image/webp;base64,")


def test_openai_moderation_requires_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)

    with pytest.raises(MissingApiKeyError):
        ProviderManager.run_inference_with_model(
            model_name="openai/omni-moderation-latest",
            images_list=[Image.new("RGB", (8, 8), color="white")],
            litellm_model_id="openai/omni-moderation-latest",
            api_keys={},
        )


def test_openai_moderation_api_error_is_wrapped_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ValueError("socket closed")
    exc = RuntimeError("moderation failed")
    exc.__cause__ = cause
    _install_fake_openai(monkeypatch, exc=exc)

    results = ProviderManager.run_inference_with_model(
        model_name="openai/omni-moderation-latest",
        images_list=[Image.new("RGB", (8, 8), color="white")],
        litellm_model_id="openai/omni-moderation-latest",
        api_keys={"openai": "test-key"},
    )

    result = next(iter(results.values()))
    assert result["tags"] == []
    assert "RuntimeError: moderation failed" in result["error"]
    assert "ValueError: socket closed" in result["error"]
