"""Cloudflare Workers AI Clef transport and strict decision normalization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from image_annotator_lib.webapi.image_payload import build_base64_data_url

from .types import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionError,
    DecisionErrorCode,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 13 * 1024 * 1024
_MODELS = {"@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"}
_QUESTION_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_IMAGE_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class _InvalidDecision(ValueError):
    def __init__(self, message: str, code: DecisionErrorCode = DecisionErrorCode.INVALID_REQUEST) -> None:
        super().__init__(message)
        self.code = code


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _questions(questions: dict[str, DecisionQuestion]) -> dict[str, Any]:
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise _InvalidDecision("A request must contain 1 to 64 questions.")
    prepared: dict[str, Any] = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not _QUESTION_ID.fullmatch(question_id):
            raise _InvalidDecision(
                "Question IDs must use letters, digits, underscores, dots or hyphens (1-100 characters)."
            )
        prepared[question_id] = _question(question)
    return prepared


def _question(question: DecisionQuestion) -> dict[str, Any]:
    if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        raise _InvalidDecision("A question must be a typed noul, choice or score question.")
    if not _text(question.instructions):
        raise _InvalidDecision("Question instructions must be non-empty text.")
    item: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
    if isinstance(question, ChoiceQuestion):
        if not isinstance(question.criteria, dict) or not 2 <= len(question.criteria) <= 255:
            raise _InvalidDecision("Choice questions require 2 to 255 options.")
        if any(not _text(key) or not _text(value) for key, value in question.criteria.items()):
            raise _InvalidDecision("Choice option IDs and descriptions must be non-empty text.")
        item["criteria"] = dict(question.criteria)
    elif isinstance(question, ScoreQuestion):
        if not isinstance(question.criteria, list) or not 2 <= len(question.criteria) <= 10:
            raise _InvalidDecision("Score questions require 2 to 10 ordered levels.")
        if any(not _text(value) for value in question.criteria):
            raise _InvalidDecision("Score level descriptions must be non-empty text.")
        item["criteria"] = list(question.criteria)
    return item


def _image_payload(image: Path | Image.Image) -> tuple[bytes, str]:
    try:
        if isinstance(image, Path):
            # Read one byte past the bound, including when a file changes after stat().
            with image.open("rb") as source:
                payload = source.read(MAX_IMAGE_BYTES + 1)
            if len(payload) > MAX_IMAGE_BYTES:
                raise _InvalidDecision("Each image must be at most 4 MiB.", DecisionErrorCode.INVALID_IMAGE)
            with Image.open(BytesIO(payload)) as decoded:
                mime = _IMAGE_MIME.get(decoded.format or "")
                if mime is None:
                    raise _InvalidDecision(
                        "Images must be PNG, JPEG or WebP.", DecisionErrorCode.INVALID_IMAGE
                    )
                _image_dimensions(decoded)
                decoded.verify()
            with Image.open(BytesIO(payload)) as decoded:
                decoded.load()
            return payload, mime
        if not isinstance(image, Image.Image):
            raise _InvalidDecision(
                "Images must be local paths or PIL images.", DecisionErrorCode.INVALID_IMAGE
            )
        _image_dimensions(image)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        if len(payload) > MAX_IMAGE_BYTES:
            raise _InvalidDecision("Each image must be at most 4 MiB.", DecisionErrorCode.INVALID_IMAGE)
        return payload, "image/png"
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        if isinstance(exc, _InvalidDecision):
            raise
        raise _InvalidDecision(
            "Image could not be read or encoded.", DecisionErrorCode.INVALID_IMAGE
        ) from None


def _image_dimensions(image: Image.Image) -> None:
    if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
        raise _InvalidDecision(
            "Each image must contain 1 to 16 million pixels.", DecisionErrorCode.INVALID_IMAGE
        )


def _images(images: Sequence[Path | Image.Image]) -> list[str]:
    if not isinstance(images, Sequence) or len(images) > 4:
        raise _InvalidDecision(
            "A request may contain at most four images.", DecisionErrorCode.INVALID_IMAGE
        )
    result: list[str] = []
    total = 0
    for image in images:
        payload, mime = _image_payload(image)
        total += len(payload)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise _InvalidDecision(
                "Total image bytes must be at most 8 MiB.", DecisionErrorCode.INVALID_IMAGE
            )
        result.append(build_base64_data_url(payload, mime))
    return result


def _body(request: DecisionRequest, model_name: str) -> tuple[bytes, dict[str, Any]]:
    if not _text(request.request_id):
        raise _InvalidDecision("Request ID must be non-empty text.")
    if not isinstance(request.state, (str, dict, list)):
        raise _InvalidDecision("State must be text, a JSON object or a JSON array.")
    questions = _questions(request.questions)
    body: dict[str, Any] = {
        "model": model_name.rsplit("/", 1)[-1],
        "state": request.state,
        "questions": questions,
    }
    images = _images(request.images)
    if images:
        body["images"] = images
    try:
        _json_state(request.state)
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise _InvalidDecision("State and questions must contain valid finite JSON data.") from None
    if len(payload) > MAX_REQUEST_BYTES:
        raise _InvalidDecision("Request body must be at most 13 MiB.")
    return payload, questions


def _json_state(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _json_state(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_state(item)
        return
    raise _InvalidDecision("State must contain only JSON values with string object keys.")


def _number(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidDecision(
            "Decision response contains a non-numeric value.", DecisionErrorCode.INVALID_RESPONSE
        )
    try:
        number = float(value)
    except OverflowError:
        raise _InvalidDecision(
            "Decision response contains an invalid numeric value.", DecisionErrorCode.INVALID_RESPONSE
        ) from None
    if not math.isfinite(number) or not low <= number <= high:
        raise _InvalidDecision(
            "Decision response contains an invalid numeric value.", DecisionErrorCode.INVALID_RESPONSE
        )
    return number


def _answer(raw: Any, question: dict[str, Any]) -> DecisionAnswer:
    if not isinstance(raw, dict) or raw.get("type") != question["type"]:
        raise _InvalidDecision(
            "Answer type does not match its question.", DecisionErrorCode.INVALID_RESPONSE
        )
    if question["type"] == "noul":
        return NoulAnswer(probability=_number(raw.get("noul")))
    keys = (
        set(question["criteria"])
        if question["type"] == "choice"
        else {str(i) for i in range(len(question["criteria"]))}
    )
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != keys:
        raise _InvalidDecision(
            "Answer probability IDs do not match its options.", DecisionErrorCode.INVALID_RESPONSE
        )
    values = {key: _number(value) for key, value in probabilities.items()}
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=0.01):
        raise _InvalidDecision(
            "Answer probabilities do not sum to one.", DecisionErrorCode.INVALID_RESPONSE
        )
    confidence = _number(raw.get("confidence"))
    if question["type"] == "choice":
        choice = raw.get("choice")
        if not isinstance(choice, str) or choice not in keys:
            raise _InvalidDecision(
                "Answer choice is not an allowed option.", DecisionErrorCode.INVALID_RESPONSE
            )
        return ChoiceAnswer(choice=choice, probabilities=values, confidence=confidence)
    high = float(len(keys) - 1)
    return ScoreAnswer(
        score=_number(raw.get("score"), high=high),
        probabilities=values,
        confidence=confidence,
        value_range=(0.0, high),
    )


def _result(response: Any, questions: dict[str, Any], request_id: str) -> DecisionResult:
    if not isinstance(response, dict):
        raise _InvalidDecision("Decision response must be an object.", DecisionErrorCode.INVALID_RESPONSE)
    if response.get("success") is False:
        raise _InvalidDecision("Cloudflare reported a failed decision request.", DecisionErrorCode.PROVIDER)
    if "success" in response and response["success"] is not True:
        raise _InvalidDecision(
            "Decision response has an invalid success flag.", DecisionErrorCode.INVALID_RESPONSE
        )
    raw = response.get("result", response)
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise _InvalidDecision(
            "Decision response has no answers object.", DecisionErrorCode.INVALID_RESPONSE
        )
    answers = raw["answers"]
    if set(answers) != set(questions):
        raise _InvalidDecision(
            "Answer question IDs do not match the request.", DecisionErrorCode.INVALID_RESPONSE
        )
    model = raw.get("model")
    if model in ("clef", "clef-flash"):
        model = f"@cf/cloudflare/{model}"
    if not isinstance(model, str) or model not in _MODELS:
        raise _InvalidDecision(
            "Decision response has no recognized model identifier.", DecisionErrorCode.INVALID_RESPONSE
        )
    return DecisionResult(
        request_id=request_id,
        model_name=model,
        answers={key: _answer(answers[key], question) for key, question in questions.items()},
    )


class CloudflareDecisionClient:
    """Evaluate typed questions through Workers AI, without automatic retries.

    Credentials are explicit and are never read from, or written to, the
    environment. Redirects are disabled so authorization stays at Cloudflare.
    ``transport`` allows deterministic offline tests with ``httpx.MockTransport``.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model_name: str = "@cf/cloudflare/clef-flash",
        timeout: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self.model_name = model_name
        self._timeout = timeout
        self._transport = transport

    def _configuration(self) -> None:
        if not isinstance(self._account_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self._account_id):
            raise _InvalidDecision(
                "Cloudflare account ID is missing or invalid.", DecisionErrorCode.CONFIGURATION
            )
        if not isinstance(self._api_token, str) or not re.fullmatch(r"[\x21-\x7e]+", self._api_token):
            raise _InvalidDecision(
                "Cloudflare API token is missing or invalid.", DecisionErrorCode.CONFIGURATION
            )
        if not isinstance(self.model_name, str) or self.model_name not in _MODELS:
            raise _InvalidDecision("Select Cloudflare Clef or Clef-flash.", DecisionErrorCode.CONFIGURATION)
        if (
            isinstance(self._timeout, bool)
            or not isinstance(self._timeout, (int, float))
            or not math.isfinite(self._timeout)
            or self._timeout <= 0
        ):
            raise _InvalidDecision("Timeout must be finite and positive.", DecisionErrorCode.CONFIGURATION)

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        """Return validated answers, or an explicit sanitized typed failure."""
        try:
            self._configuration()
            payload, questions = _body(request, self.model_name)
            url = (
                f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/run/{self.model_name}"
            )
            with httpx.Client(
                timeout=self._timeout, transport=self._transport, follow_redirects=False
            ) as client:
                response = client.post(
                    url,
                    content=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_token}",
                        "Content-Type": "application/json",
                    },
                )
            if not 200 <= response.status_code < 300:
                return self._failure(request, _http_error(response.status_code))
            try:
                raw = response.json()
            except (ValueError, UnicodeError):
                raise _InvalidDecision(
                    "Decision response is not valid JSON.", DecisionErrorCode.INVALID_RESPONSE
                ) from None
            return _result(raw, questions, request.request_id)
        except _InvalidDecision as exc:
            return self._failure(request, DecisionError(code=exc.code, message=str(exc)))
        except httpx.RequestError:
            return self._failure(
                request,
                DecisionError(
                    code=DecisionErrorCode.TRANSPORT,
                    message="Decision request failed or timed out.",
                    retryable=True,
                ),
            )

    def _failure(self, request: DecisionRequest, error: DecisionError) -> DecisionResult:
        return DecisionResult(
            request_id=request.request_id, model_name=self.model_name, answers={}, error=error
        )


def _http_error(status: int) -> DecisionError:
    if status in (401, 403):
        return DecisionError(
            code=DecisionErrorCode.AUTHENTICATION,
            message="Cloudflare rejected the credentials or account permissions.",
        )
    if status == 429:
        return DecisionError(
            code=DecisionErrorCode.RATE_LIMIT, message="Cloudflare rate limit was reached.", retryable=True
        )
    return DecisionError(
        code=DecisionErrorCode.PROVIDER,
        message=f"Cloudflare decision request failed (HTTP {status}).",
        retryable=status >= 500 or status == 408,
    )
