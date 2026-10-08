"""Offline tests for the public decision boundary and local llama.cpp wire contract."""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from image_annotator_lib.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionErrorCode,
    DecisionRequest,
    LocalDecisionClient,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    local,
)

pytestmark = [pytest.mark.unit, pytest.mark.standard]


@pytest.fixture(autouse=True)
def mock_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def session(settings: Any, timeout: float) -> Iterator[str]:
        yield "http://127.0.0.1:11437"

    monkeypatch.setattr(local, "runtime_session", session)


def _local_client(**changes: Any) -> LocalDecisionClient:
    options = {"server_path": __file__, "model_path": __file__, "mmproj_path": __file__, **changes}
    return LocalDecisionClient(**options)


def _request(**changes: Any) -> DecisionRequest:
    return replace(
        DecisionRequest(
            "image-42-review", {"tag": "dog"}, {"tag_000": NoulQuestion("Does the image support this tag?")}
        ),
        **changes,
    )


def _response(answers: Any = None, **changes: Any) -> dict[str, Any]:
    result = {
        "model": "clef-flash",
        "answers": {"tag_000": {"type": "noul", "noul": 0.03}} if answers is None else answers,
    }
    result.update(changes)
    return result


def _client(response: Any = None, *, status: int = 200, **changes: Any) -> LocalDecisionClient:
    return _local_client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                status, content=json.dumps(_response() if response is None else response).encode("utf-8")
            )
        ),
        **changes,
    )


def test_all_answer_types_keep_reference_ids_probabilities_and_raw_score() -> None:
    request = _request(
        questions={
            "tag_000": NoulQuestion("Supported?"),
            "crop": ChoiceQuestion("Best crop?", {"whole": "Full image", "face": "Face"}),
            "severity": ScoreQuestion("How severe?", ["None", "Minor", "Major"]),
        }
    )
    answers = {
        "tag_000": {"type": "noul", "noul": 0.03},
        "crop": {
            "type": "choice",
            "choice": "face",
            "confidence": 0.9,
            "probabilities": {"whole": 0.1, "face": 0.9},
        },
        "severity": {
            "type": "score",
            "score": 1.4,
            "confidence": 0.5,
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
    }
    result = _client(_response(answers)).evaluate(request)
    assert result.error is None
    assert result.request_id == "image-42-review"
    assert result.provider == "llamacpp"
    assert result.model_name == "clef-flash"
    assert result.answers == {
        "tag_000": NoulAnswer(0.03),
        "crop": ChoiceAnswer("face", {"whole": 0.1, "face": 0.9}, 0.9),
        "severity": ScoreAnswer(1.4, {"0": 0.1, "1": 0.4, "2": 0.5}, 0.5, (0.0, 2.0)),
    }


def test_injected_stateful_transport_remains_owned_by_the_caller() -> None:
    class StatefulTransport(httpx.BaseTransport):
        def __init__(self) -> None:
            self.closed = False
            self.requests = 0
            self.close_calls = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if self.closed:
                raise httpx.ConnectError("Transport is closed", request=request)
            self.requests += 1
            return httpx.Response(200, json=_response())

        def close(self) -> None:
            self.closed = True
            self.close_calls += 1

    transport = StatefulTransport()
    client = _local_client(transport=transport)
    for request_id in ("first", "second"):
        result = client.evaluate(_request(request_id=request_id))
        assert result.error is None
        assert result.request_id == request_id
        assert result.answers == {"tag_000": NoulAnswer(0.03)}
    assert transport.requests == 2
    assert not transport.closed
    assert transport.close_calls == 0

    transport.close()
    result = client.evaluate(_request())
    assert result.error is not None and result.error.code == DecisionErrorCode.TRANSPORT
    assert transport.requests == 2
    assert transport.closed
    assert transport.close_calls == 1


@pytest.mark.parametrize(
    "state", ["plain state", ["record", {"ready": True}], {"nested": {"value": None}, "finite": 1.5}]
)
def test_wire_body_stays_on_loopback_without_credentials(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    before = dict(os.environ)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_response())

    client = _local_client(transport=httpx.MockTransport(handler))
    assert client.evaluate(_request(state=state)).error is None
    assert len(captured) == 1
    wire = captured[0]
    assert str(wire.url) == "http://127.0.0.1:11437/v1/systemone"
    assert "Authorization" not in wire.headers
    assert json.loads(wire.content) == {
        "model": "clef-flash",
        "state": state,
        "questions": {"tag_000": {"type": "noul", "instructions": "Does the image support this tag?"}},
    }
    assert "private-api-token" not in repr(client)
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "image_format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_local_image_mime_and_bytes_are_preserved(tmp_path: Path, image_format: str, mime: str) -> None:
    path = tmp_path / "image.data"
    Image.new("RGB", (10, 20), "blue").save(path, format=image_format)
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    client = _local_client(transport=httpx.MockTransport(handler))
    assert client.evaluate(_request(images=[path])).error is None
    prefix, data = captured[0]["images"][0].split(",", 1)
    assert prefix == f"data:{mime};base64"
    assert base64.b64decode(data) == path.read_bytes()


@pytest.mark.parametrize(
    "mode,color,encoded_mode",
    [
        ("RGB", (12, 34, 56), "RGB"),
        ("RGBA", (12, 34, 56, 78), "RGBA"),
        ("LA", (31, 128), "LA"),
        ("CMYK", (0, 255, 255, 0), "RGB"),
        ("RGBa", (12, 34, 56, 78), "RGBA"),
    ],
)
def test_pil_image_encoded_as_png_without_mutating_input(
    mode: str, color: tuple[int, ...], encoded_mode: str
) -> None:
    image = Image.new(mode, (10, 20), color)
    pixels = image.tobytes()
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    result = _local_client(transport=httpx.MockTransport(handler)).evaluate(_request(images=[image]))
    assert result.error is None
    prefix, data = captured[0]["images"][0].split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(data))) as encoded:
        assert encoded.format == "PNG"
        assert encoded.mode == encoded_mode
        assert encoded.size == image.size
        assert encoded.tobytes() == image.convert(encoded_mode).tobytes()
    assert image.mode == mode
    assert image.size == (10, 20)
    assert image.tobytes() == pixels


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": ""},
        {"questions": {}},
        {"questions": {str(i): NoulQuestion("Valid?") for i in range(65)}},
        {"questions": {"bad id": NoulQuestion("Valid?")}},
        {"questions": {"x" * 101: NoulQuestion("Valid?")}},
        {"questions": {"id": NoulQuestion(" ")}},
        {"questions": {"id": {"type": "noul"}}},
        {"questions": {"id": ChoiceQuestion("Pick?", {"only": "One"})}},
        {"questions": {"id": ChoiceQuestion("Pick?", {str(i): "Option" for i in range(256)})}},
        {"questions": {"id": ChoiceQuestion("Pick?", {"": "Empty", "b": "Other"})}},
        {"questions": {"id": ChoiceQuestion("Pick?", {"a": "", "b": "Other"})}},
        {"questions": {"id": ScoreQuestion("Rate?", ["Only"])}},
        {"questions": {"id": ScoreQuestion("Rate?", ["Level"] * 11)}},
        {"questions": {"id": ScoreQuestion("Rate?", ["", "Good"])}},
        {"state": True},
        {"state": {"nan": float("nan")}},
        {"state": {"bad": object()}},
        {"state": {1: "Non-string key"}},
        {"state": {"tuple": (1, 2)}},
    ],
)
def test_invalid_input_never_reaches_network(changes: dict[str, Any]) -> None:
    def no_network(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid input sent to provider")

    result = _local_client(transport=httpx.MockTransport(no_network)).evaluate(_request(**changes))
    assert result.error is not None
    assert result.error.code == DecisionErrorCode.INVALID_REQUEST
    assert result.answers == {}
    assert result.error.retryable is False


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_maximum_question_and_option_limits_are_accepted(kind: str) -> None:
    question = {
        "choice": ChoiceQuestion("Pick?", {str(i): "Option" for i in range(255)}),
        "score": ScoreQuestion("Rate?", ["Level"] * 10),
        "noul": NoulQuestion("Valid?"),
    }[kind]
    body, _ = local._body(_request(questions={str(i): question for i in range(64)}), "clef-flash")
    assert len(json.loads(body)["questions"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"server_path": ""},
        {"model_path": "missing.gguf"},
        {"mmproj_path": "missing-projector.gguf"},
        {"server_path": Path(__file__).parent},
        {"model_name": "remote-clef"},
        {"n_gpu_layers": True},
        {"n_gpu_layers": -1},
        {"n_gpu_layers": 1000},
        {"context_size": 511},
        {"context_size": 131073},
        {"context_size": 4096.0},
        {"timeout": float("inf")},
        {"timeout": 0},
        {"timeout": True},
    ],
)
def test_configuration_errors_are_sanitized(changes: dict[str, Any]) -> None:
    result = _local_client(**changes).evaluate(_request())
    assert result.error is not None and result.error.code == DecisionErrorCode.CONFIGURATION
    assert "private-api-token" not in result.error.message
    assert result.answers == {}


@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (401, DecisionErrorCode.PROVIDER, False),
        (403, DecisionErrorCode.PROVIDER, False),
        (429, DecisionErrorCode.PROVIDER, True),
        (500, DecisionErrorCode.PROVIDER, True),
        (400, DecisionErrorCode.PROVIDER, False),
        (413, DecisionErrorCode.PROVIDER, False),
        (404, DecisionErrorCode.CONFIGURATION, False),
        (302, DecisionErrorCode.PROVIDER, False),
    ],
)
def test_http_failures_have_typed_errors_without_provider_body(
    status: int, code: DecisionErrorCode, retryable: bool
) -> None:
    result = _client({"error": "private-api-token confidential provider body"}, status=status).evaluate(
        _request()
    )
    assert result.error is not None
    assert result.error.code == code and result.error.retryable == retryable
    assert "private-api-token" not in repr(result)
    assert "confidential" not in repr(result)
    assert result.request_id == "image-42-review" and result.answers == {}


def test_redirects_never_send_images_off_loopback_and_requests_are_not_retried() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example/steal"})

    result = _local_client(transport=httpx.MockTransport(handler)).evaluate(_request())
    assert result.error is not None
    assert len(requests) == 1
    assert requests[0].url.host == "127.0.0.1"


@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
def test_transport_failures_sanitize_exception_messages(error: type[httpx.RequestError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("private-api-token confidential detail", request=request)

    result = _local_client(transport=httpx.MockTransport(handler)).evaluate(_request())
    assert result.error is not None
    assert result.error.code == DecisionErrorCode.TRANSPORT and result.error.retryable
    assert "private-api-token" not in repr(result)


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"error": {"message": "invalid local request"}},
        _response(answers={}),
        _response(answers={"other": {"type": "noul", "noul": 0.5}}),
        _response(answers={"tag_000": {"type": "choice", "noul": 0.5}}),
        _response(model="unrecognized"),
        _response(model="clef"),
        _response(model=None),
        _response(answers=[]),
        {"success": True, "result": []},
    ],
)
def test_invalid_or_failed_provider_envelopes_are_explicit(response: Any) -> None:
    result = _client(response).evaluate(_request())
    assert result.error is not None
    assert result.error.code in {DecisionErrorCode.INVALID_RESPONSE, DecisionErrorCode.PROVIDER}
    assert result.answers == {}
    assert "private-api-token" not in repr(result)


@pytest.mark.parametrize("value", [True, "0.3", None, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_noul_probabilities_are_rejected(value: Any) -> None:
    result = _client(_response(answers={"tag_000": {"type": "noul", "noul": value}})).evaluate(_request())
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert result.answers == {}


@pytest.mark.parametrize(
    "kind,answer",
    [
        (
            "choice",
            {"type": "choice", "choice": "other", "confidence": 0.6, "probabilities": {"a": 0.4, "b": 0.6}},
        ),
        (
            "choice",
            {"type": "choice", "choice": "a", "confidence": 1.2, "probabilities": {"a": 0.4, "b": 0.6}},
        ),
        ("choice", {"type": "choice", "choice": "a", "confidence": 0.8, "probabilities": {"a": 1.0}}),
        (
            "choice",
            {"type": "choice", "choice": "a", "confidence": 0.8, "probabilities": {"a": 0.2, "b": 0.2}},
        ),
        (
            "choice",
            {"type": "choice", "choice": "a", "confidence": 0.8, "probabilities": {"a": True, "b": 0}},
        ),
        (
            "score",
            {"type": "score", "score": 1.2, "confidence": 0.6, "probabilities": {"0": 0.4, "1": 0.6}},
        ),
        (
            "score",
            {"type": "score", "score": 0.6, "confidence": 0.8, "probabilities": {"low": 0.4, "high": 0.6}},
        ),
        (
            "score",
            {"type": "score", "score": "0.6", "confidence": 0.6, "probabilities": {"0": 0.4, "1": 0.6}},
        ),
    ],
)
def test_invalid_choice_and_score_answers_are_rejected(kind: str, answer: Any) -> None:
    question = (
        ChoiceQuestion("Pick?", {"a": "First", "b": "Second"})
        if kind == "choice"
        else ScoreQuestion("Rate?", ["Low", "High"])
    )
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": question})
    )
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert result.answers == {}


@pytest.mark.parametrize(
    "kind,answer",
    [
        (
            "choice",
            {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.4, "b": 0.6}},
        ),
        (
            "choice",
            {"type": "choice", "choice": "b", "confidence": 0.7, "probabilities": {"a": 0.4, "b": 0.6}},
        ),
        (
            "score",
            {"type": "score", "score": 0.3, "confidence": 0.6, "probabilities": {"0": 0.4, "1": 0.6}},
        ),
        (
            "score",
            {"type": "score", "score": 0.6, "confidence": 0.7, "probabilities": {"0": 0.4, "1": 0.6}},
        ),
    ],
)
def test_contradictory_choice_and_score_fields_are_rejected(kind: str, answer: dict[str, Any]) -> None:
    question = (
        ChoiceQuestion("Pick?", {"a": "First", "b": "Second"})
        if kind == "choice"
        else ScoreQuestion("Rate?", ["Low", "High"])
    )
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": question})
    )
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert result.answers == {}


@pytest.mark.parametrize("selected", ["a", "b"])
def test_choice_accepts_any_provider_rounded_maximum_tie(selected: str) -> None:
    answer = {
        "type": "choice",
        "choice": selected,
        "confidence": 0.5,
        "probabilities": {"a": 0.5, "b": 0.5},
    }
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ChoiceQuestion("Pick?", {"a": "First", "b": "Second"})})
    )
    assert result.error is None
    assert result.answers["tag_000"] == ChoiceAnswer(selected, {"a": 0.5, "b": 0.5}, 0.5)


def test_choice_accepts_provider_rounding_at_maximum_option_count() -> None:
    # round(1 / 255, 4) is 0.0039: all displayed probabilities are tied.
    probabilities = {str(i): 0.0039 for i in range(255)}
    answer = {"type": "choice", "choice": "254", "confidence": 0.0039, "probabilities": probabilities}
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ChoiceQuestion("Pick?", {str(i): "Option" for i in range(255)})})
    )
    assert result.error is None
    assert result.answers["tag_000"] == ChoiceAnswer("254", probabilities, 0.0039)


def test_choice_accepts_four_decimal_rounding_above_legacy_sum_tolerance() -> None:
    original = [0.0048499] * 205 + [0.0000499] * 49
    original.append(1.0 - math.fsum(original))
    probabilities = {str(i): round(value, 4) for i, value in enumerate(original)}
    assert math.fsum(probabilities.values()) == pytest.approx(0.9873)
    answer = {"type": "choice", "choice": "0", "confidence": 0.0048, "probabilities": probabilities}
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ChoiceQuestion("Pick?", {str(i): "Option" for i in range(255)})})
    )
    assert result.error is None
    assert result.answers["tag_000"] == ChoiceAnswer("0", probabilities, 0.0048)


@pytest.mark.parametrize("common,last", [(0.0039, 0.0005), (0.004, 0.0008)])
def test_choice_rejects_sum_error_beyond_maximum_option_rounding_bound(common: float, last: float) -> None:
    probabilities = {str(i): common for i in range(253)} | {"253": last, "254": 0.0}
    answer = {"type": "choice", "choice": "0", "confidence": common, "probabilities": probabilities}
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ChoiceQuestion("Pick?", {str(i): "Option" for i in range(255)})})
    )
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert result.answers == {}


@pytest.mark.parametrize(
    "probabilities,score",
    [
        # Original probabilities 0.333349, 0.666651 yield score 0.6667.
        ({"0": 0.3333, "1": 0.6667}, 0.6667),
        # First five probabilities are 0.100049, last five 0.099951.
        # Their weighted score is 4.498775, although rounded probabilities are 0.1.
        ({str(i): 0.1 for i in range(10)}, 4.4988),
    ],
)
def test_score_preserves_provider_rounded_weighted_levels(
    probabilities: dict[str, float], score: float
) -> None:
    confidence = max(probabilities.values())
    answer = {"type": "score", "score": score, "confidence": confidence, "probabilities": probabilities}
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ScoreQuestion("Rate?", ["Level"] * len(probabilities))})
    )
    assert result.error is None
    assert result.answers["tag_000"] == ScoreAnswer(
        score, probabilities, confidence, (0.0, float(len(probabilities) - 1))
    )


@pytest.mark.parametrize("levels,expected,tolerance", [(2, 0.5, 0.0001), (10, 4.5, 0.0023)])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("extra,accepted", [(0.0, True), (1e-8, False)])
def test_score_rounding_error_bound_scales_with_levels_and_rejects_beyond_it(
    levels: int, expected: float, tolerance: float, direction: int, extra: float, accepted: bool
) -> None:
    probabilities = {str(i): 1.0 / levels for i in range(levels)}
    answer = {
        "type": "score",
        "score": expected + direction * (tolerance + extra),
        "confidence": 1.0 / levels,
        "probabilities": probabilities,
    }
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": ScoreQuestion("Rate?", ["Level"] * levels)})
    )
    if accepted:
        assert result.error is None
    else:
        assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
        assert result.answers == {}


@pytest.mark.parametrize("kind", ["choice", "score"])
@pytest.mark.parametrize(
    "second,accepted",
    [
        (0.49, True),
        (0.51, True),
        (0.489999999999, False),
        (0.510000000001, False),
    ],
)
def test_probability_sum_tolerance_includes_only_the_rounding_boundaries(
    kind: str, second: float, accepted: bool
) -> None:
    if kind == "choice":
        question: ChoiceQuestion | ScoreQuestion = ChoiceQuestion("Pick?", {"a": "First", "b": "Second"})
        probabilities = {"a": 0.5, "b": second}
        answer: dict[str, Any] = {"type": "choice", "choice": "b" if second > 0.5 else "a"}
    else:
        question = ScoreQuestion("Rate?", ["Low", "High"])
        probabilities = {"0": 0.5, "1": second}
        answer = {"type": "score", "score": second}
    answer.update(confidence=max(probabilities.values()), probabilities=probabilities)
    result = _client(_response(answers={"tag_000": answer})).evaluate(
        _request(questions={"tag_000": question})
    )
    if accepted:
        assert result.error is None
        assert isinstance(result.answers["tag_000"], (ChoiceAnswer, ScoreAnswer))
        assert result.answers["tag_000"].probabilities == probabilities
    else:
        assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
        assert result.answers == {}


def test_json_parse_error_is_sanitized() -> None:
    client = _local_client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"private-api-token invalid JSON")
        ),
    )
    result = client.evaluate(_request())
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert "private-api-token" not in repr(result)


def test_actual_deeply_nested_json_returns_sanitized_invalid_response() -> None:
    # Python 3.13's JSON C-stack guard is independent of the Python recursion limit.
    depth = max(20_000, sys.getrecursionlimit() + 100)
    content = b'{"private-api-token":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    with pytest.raises(RecursionError):
        json.loads(content)
    client = _local_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )
    result = client.evaluate(_request())
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_RESPONSE
    assert result.answers == {}
    assert "private-api-token" not in repr(result)


def test_recursion_error_outside_json_parsing_is_not_suppressed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RecursionError("Programming error in caller transport")

    client = _local_client(transport=httpx.MockTransport(handler))
    with pytest.raises(RecursionError, match="Programming error"):
        client.evaluate(_request())


@pytest.mark.parametrize(
    "image_kind", ["missing", "garbage", "gif", "too_large", "too_many", "too_many_pixels", "remote_url"]
)
def test_bad_images_are_rejected_before_network(tmp_path: Path, image_kind: str) -> None:
    path = tmp_path / "image.data"
    image = Image.new("RGB", (4, 4))
    if image_kind == "garbage":
        path.write_bytes(b"confidential garbage")
    elif image_kind == "gif":
        image.save(path, format="GIF")
    elif image_kind == "too_large":
        path.write_bytes(b"x" * (local.MAX_IMAGE_BYTES + 1))
    images: Any = {
        "too_many": [image] * 5,
        "too_many_pixels": [Image.new("1", (4001, 4000))],
        "remote_url": ["https://example.com/image.png"],
    }.get(image_kind, [path])

    def no_network(request: httpx.Request) -> httpx.Response:
        pytest.fail("Bad image sent to provider")

    result = _local_client(transport=httpx.MockTransport(no_network)).evaluate(_request(images=images))
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_IMAGE
    assert "confidential" not in result.error.message


def test_exact_image_limit_total_limit_and_body_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    original = path.read_bytes()
    path.write_bytes(original + b"\0" * (local.MAX_IMAGE_BYTES - len(original)))
    assert _client().evaluate(_request(images=[path, path])).error is None
    result = _client().evaluate(_request(images=[path, path, path]))
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_IMAGE
    result = _client().evaluate(_request(state="x" * local.MAX_REQUEST_BYTES))
    assert result.error is not None and result.error.code == DecisionErrorCode.INVALID_REQUEST
    monkeypatch.setattr(local, "MAX_IMAGE_PIXELS", 16)
    assert _client().evaluate(_request(images=[Image.new("RGB", (4, 4))])).error is None
    assert _client().evaluate(_request(images=[Image.new("RGB", (5, 4))])).error is not None


def test_public_import_and_mock_transport_in_fresh_process() -> None:
    source = Path(__file__).resolve().parents[3] / "src"
    script = """
import sys, httpx
from contextlib import contextmanager
from image_annotator_lib.decisions import LocalDecisionClient, DecisionRequest, NoulQuestion, local
@contextmanager
def mock_session(settings, timeout):
    yield "http://127.0.0.1:11437"
local.runtime_session = mock_session
assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'onnxruntime', 'transformers'))
client = LocalDecisionClient(sys.executable, sys.executable, sys.executable, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'model': 'clef-flash', 'answers': {'caption_000': {'type': 'noul', 'noul': 0.91}}})))
result = client.evaluate(DecisionRequest('image-1', {'caption': 'a dog'}, {'caption_000': NoulQuestion('Supported?')}))
assert result.error is None
assert result.answers['caption_000'].probability == 0.91
print('public decision boundary passed')
"""
    env = {
        **os.environ,
        "IMAGE_ANNOTATOR_CONFIG_READ_ONLY": "1",
        "PYTHONPATH": os.pathsep.join([str(source), os.environ.get("PYTHONPATH", "")]),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=True
    )
    assert "public decision boundary passed" in completed.stdout
