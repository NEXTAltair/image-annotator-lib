"""Public decision contracts, independent of annotation capabilities and storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from PIL import Image


@dataclass(frozen=True)
class NoulQuestion:
    """Ask a yes/no question; its answer is the probability of yes."""

    instructions: str
    type: Literal["noul"] = field(default="noul", init=False)


@dataclass(frozen=True)
class ChoiceQuestion:
    """Select an option while retaining all caller-provided option IDs."""

    instructions: str
    criteria: dict[str, str]
    type: Literal["choice"] = field(default="choice", init=False)


@dataclass(frozen=True)
class ScoreQuestion:
    """Evaluate ordered levels, indexed from zero, without score normalization."""

    instructions: str
    criteria: list[str]
    type: Literal["score"] = field(default="score", init=False)


type DecisionQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class DecisionRequest:
    """One decision evaluation; reference IDs belong to the caller.

    Images are local PNG/JPEG/WebP paths or PIL images (encoded as PNG).
    State and question text are data supplied by the application. The library
    does not define warning thresholds or attach database-specific meanings.
    """

    request_id: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, DecisionQuestion]
    images: Sequence[Path | Image.Image] = ()


@dataclass(frozen=True)
class NoulAnswer:
    """Probability that the corresponding question is true, not tag confidence."""

    probability: float
    type: Literal["noul"] = field(default="noul", init=False)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: Literal["choice"] = field(default="choice", init=False)


@dataclass(frozen=True)
class ScoreAnswer:
    """Provider's weighted level and range; never an annotation/aesthetic score."""

    score: float
    probabilities: dict[str, float]
    confidence: float
    value_range: tuple[float, float]
    type: Literal["score"] = field(default="score", init=False)


type DecisionAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class DecisionErrorCode(StrEnum):
    CONFIGURATION = "configuration"
    INVALID_REQUEST = "invalid_request"
    INVALID_IMAGE = "invalid_image"
    TRANSPORT = "transport"
    PROVIDER = "provider"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class DecisionError:
    """Sanitized failure information; retryable never implies an automatic retry."""

    code: DecisionErrorCode
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class DecisionResult:
    """Complete answers on success, or an explicit error with no answers.

    Absence of a result means not evaluated. A failed evaluation must never be
    interpreted as a negative answer or as the absence of warnings.
    """

    request_id: str
    model_name: str
    answers: dict[str, DecisionAnswer]
    error: DecisionError | None = None
    provider: str = "llamacpp"
