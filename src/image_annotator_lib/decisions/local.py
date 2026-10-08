"""Local llama.cpp Clef transport and strict decision normalization."""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from image_annotator_lib.webapi.image_payload import build_base64_data_url

from .runtime import LocalRuntimeError, RuntimeSettings, runtime_session
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
_MODELS = {"clef", "clef-flash"}
_QUESTION_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_IMAGE_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_PNG_MODES = {"1", "L", "LA", "I", "I;16", "I;16B", "P", "RGB", "RGBA"}
_FOUR_DECIMAL_ROUNDING_ERROR = 0.00005


class _InvalidDecision(ValueError):
    def __init__(self, message: str, code: DecisionErrorCode = DecisionErrorCode.INVALID_REQUEST) -> None:
        super().__init__(message)
        self.code = code


class _BorrowedTransport(httpx.BaseTransport):
    """Forward requests without entering or closing the caller's transport."""

    def __init__(self, transport: httpx.BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._transport.handle_request(request)

    def close(self) -> None:
        """The caller owns the wrapped transport's lifetime."""


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
        encoded_image = image
        if image.mode not in _PNG_MODES:
            alpha = "A" in image.getbands() or "a" in image.getbands()
            encoded_image = image.convert("RGBA" if alpha else "RGB")
        encoded_image.save(buffer, format="PNG")
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
        "model": model_name,
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
    # Each provider probability is rounded to four decimals. Preserve the legacy
    # tolerance, allowing the larger rounding bound for choice's many options.
    sum_tolerance = max(0.01, len(values) * _FOUR_DECIMAL_ROUNDING_ERROR)
    if not math.isclose(
        math.fsum(values.values()), 1.0, rel_tol=0.0, abs_tol=sum_tolerance + math.ulp(1.0)
    ):
        raise _InvalidDecision(
            "Answer probabilities do not sum to one.", DecisionErrorCode.INVALID_RESPONSE
        )
    confidence = _number(raw.get("confidence"))
    maximum = max(values.values())
    if question["type"] == "choice":
        choice = raw.get("choice")
        if not isinstance(choice, str) or choice not in keys:
            raise _InvalidDecision(
                "Answer choice is not an allowed option.", DecisionErrorCode.INVALID_RESPONSE
            )
        if values[choice] != maximum:
            raise _InvalidDecision(
                "Answer choice is not a maximum-probability option.", DecisionErrorCode.INVALID_RESPONSE
            )
        return ChoiceAnswer(choice=choice, probabilities=values, confidence=confidence)
    high = float(len(keys) - 1)
    score = _number(raw.get("score"), high=high)
    weighted = math.fsum(int(level) * value for level, value in values.items())
    # The returned score and each probability have at most 0.00005 rounding
    # error. Probability errors are weighted by their level (0 through N-1).
    score_tolerance = _FOUR_DECIMAL_ROUNDING_ERROR * (1 + len(values) * (len(values) - 1) // 2)
    if not math.isclose(score, weighted, rel_tol=0.0, abs_tol=score_tolerance + math.ulp(high)):
        raise _InvalidDecision(
            "Answer score does not match its weighted probabilities.", DecisionErrorCode.INVALID_RESPONSE
        )
    return ScoreAnswer(
        score=score,
        probabilities=values,
        confidence=confidence,
        value_range=(0.0, high),
    )


def _result(response: Any, questions: dict[str, Any], request_id: str) -> DecisionResult:
    if not isinstance(response, dict):
        raise _InvalidDecision("Decision response must be an object.", DecisionErrorCode.INVALID_RESPONSE)
    raw = response
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
    if not isinstance(model, str) or model not in _MODELS:
        raise _InvalidDecision(
            "Decision response has no recognized model identifier.", DecisionErrorCode.INVALID_RESPONSE
        )
    return DecisionResult(
        request_id=request_id,
        model_name=model,
        answers={key: _answer(answers[key], question) for key, question in questions.items()},
    )


class LocalDecisionClient:
    """Evaluate typed questions using an automatically managed local llama.cpp server.

    Model loading happens on the first valid evaluation. Clients share one
    runtime per Python process; inference is serialized, and changed settings
    replace the previous model. Only loopback traffic is sent; environment
    proxies and redirects are disabled. No remote service or user credentials exist.
    An injected ``transport`` is borrowed, and does not bypass model startup.
    """

    def __init__(
        self,
        server_path: str | Path,
        model_path: str | Path,
        mmproj_path: str | Path,
        *,
        model_name: str = "clef-flash",
        n_gpu_layers: int = 10,
        context_size: int = 4096,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._paths = (server_path, model_path, mmproj_path)
        self.model_name = model_name
        self._n_gpu_layers = n_gpu_layers
        self._context_size = context_size
        self._timeout = timeout
        self._transport = transport

    def _configuration(self) -> RuntimeSettings:
        if not isinstance(self.model_name, str) or self.model_name not in _MODELS:
            raise _InvalidDecision("Select local Clef or Clef-flash.", DecisionErrorCode.CONFIGURATION)
        if (
            isinstance(self._timeout, bool)
            or not isinstance(self._timeout, (int, float))
            or not math.isfinite(self._timeout)
            or self._timeout <= 0
            or self._timeout > threading.TIMEOUT_MAX
        ):
            raise _InvalidDecision(
                "Timeout must be positive and within the platform's supported wait range.",
                DecisionErrorCode.CONFIGURATION,
            )
        for value, low, high, label in (
            (self._n_gpu_layers, 0, 999, "GPU layers"),
            (self._context_size, 512, 131072, "Context size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise _InvalidDecision(
                    f"{label} must be an integer between {low} and {high}.",
                    DecisionErrorCode.CONFIGURATION,
                )
        paths: list[Path] = []
        versions: list[tuple[int, int, int]] = []
        for path_value, label in zip(
            self._paths, ("llama-server executable", "Clef GGUF model", "vision projector"), strict=True
        ):
            try:
                if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
                    raise ValueError
                path = Path(path_value).resolve(strict=True)
                if not path.is_file():
                    raise ValueError
                stat = path.stat()
            except (OSError, ValueError, RuntimeError):
                raise _InvalidDecision(
                    f"Select an existing local {label} file.", DecisionErrorCode.CONFIGURATION
                ) from None
            paths.append(path)
            versions.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return RuntimeSettings(
            server_path=paths[0],
            model_path=paths[1],
            mmproj_path=paths[2],
            model_name=self.model_name,
            n_gpu_layers=self._n_gpu_layers,
            context_size=self._context_size,
            file_versions=tuple(versions),
        )

    def evaluate(self, request: DecisionRequest) -> DecisionResult:
        """Return validated answers, or an explicit sanitized typed failure."""
        try:
            settings = self._configuration()
            payload, questions = _body(request, self.model_name)
            transport = _BorrowedTransport(self._transport) if self._transport is not None else None
            with (
                runtime_session(settings, self._timeout) as endpoint,
                httpx.Client(
                    timeout=self._timeout, transport=transport, follow_redirects=False, trust_env=False
                ) as client,
            ):
                response = client.post(
                    f"{endpoint.base_url}/v1/systemone",
                    content=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {endpoint.api_key}",
                    },
                )
            if not 200 <= response.status_code < 300:
                return self._failure(request, _http_error(response.status_code))
            try:
                raw = response.json()
            except (ValueError, UnicodeError, RecursionError):
                raise _InvalidDecision(
                    "Decision response is not valid JSON.", DecisionErrorCode.INVALID_RESPONSE
                ) from None
            result = _result(raw, questions, request.request_id)
            if result.model_name != self.model_name:
                raise _InvalidDecision(
                    "Local Clef response model does not match the requested model.",
                    DecisionErrorCode.INVALID_RESPONSE,
                )
            return result
        except (_InvalidDecision, LocalRuntimeError) as exc:
            return self._failure(
                request,
                DecisionError(
                    code=exc.code,
                    message=str(exc),
                    retryable=exc.code == DecisionErrorCode.TRANSPORT,
                ),
            )
        except httpx.RequestError:
            return self._failure(
                request,
                DecisionError(
                    code=DecisionErrorCode.TRANSPORT,
                    message="Local Clef evaluation failed or timed out.",
                    retryable=True,
                ),
            )

    def _failure(self, request: DecisionRequest, error: DecisionError) -> DecisionResult:
        return DecisionResult(
            request_id=request.request_id, model_name=self.model_name, answers={}, error=error
        )


def _http_error(status: int) -> DecisionError:
    if status == 501:
        return DecisionError(
            code=DecisionErrorCode.CONFIGURATION,
            message="The selected model or projector does not support Clef decisions. "
            "Select a Clef GGUF and its matching vision projector.",
        )
    if status == 404:
        return DecisionError(
            code=DecisionErrorCode.CONFIGURATION,
            message="This llama-server does not support Clef /v1/systemone. Select a compatible build.",
        )
    if status in (400, 413):
        return DecisionError(
            code=DecisionErrorCode.PROVIDER,
            message=f"Local Clef rejected the input (HTTP {status}). "
            "Reduce the questions/image size or increase the context size.",
        )
    return DecisionError(
        code=DecisionErrorCode.PROVIDER,
        message=f"Local Clef evaluation failed (HTTP {status}).",
        retryable=status >= 500 or status in (408, 429),
    )
