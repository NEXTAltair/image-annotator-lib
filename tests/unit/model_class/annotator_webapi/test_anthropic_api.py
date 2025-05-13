from typing import cast

import anthropic
import httpx
import pytest

from image_annotator_lib.core.types import AnnotationSchema, RawOutput, WebApiFormattedOutput
from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator

DUMMY_BASE64 = "dGVzdGltYWdlYmFzZTY0Ig=="

class ToolUseBlock:
    def __init__(self, input_data):
        self.input = input_data

class DummyMessage:
    def __init__(self, content, stop_reason=None):
        self.content = content
        self.stop_reason = stop_reason

@pytest.fixture
def annotator():
    ann = AnthropicApiAnnotator(model_name="test-model")
    ann.client = anthropic.Anthropic(api_key="test")
    ann.api_model_id = "claude-3-test"
    ann.config = {"max_output_tokens": 100, "temperature": 0.1}
    return ann

def test_run_inference_success(annotator, monkeypatch) -> None:
    input_dict = {"tags": ["cat"], "captions": ["A cat"], "score": 0.9}
    dummy_block = ToolUseBlock(input_dict)
    dummy_message = DummyMessage([dummy_block])
    monkeypatch.setattr(annotator.client.messages, "create", lambda **kwargs: dummy_message)
    results = annotator._run_inference([DUMMY_BASE64])
    assert isinstance(results, list)
    assert results[0]["error"] is None
    assert isinstance(results[0]["response"], AnnotationSchema)
    assert results[0]["response"].tags == ["cat"]

def test_run_inference_error(annotator) -> None:
    def raise_exc(*args, **kwargs):
        dummy_request = httpx.Request("GET", "http://dummy")
        raise anthropic.APIConnectionError(message="API error", request=dummy_request)
    annotator.client.messages.create = raise_exc
    results = annotator._run_inference([DUMMY_BASE64])
    assert isinstance(results, list)
    assert results[0]["response"] is None
    assert results[0]["error"] is not None
    assert "API error" in str(results[0]["error"])

def test_format_predictions_success() -> None:
    ann = AnthropicApiAnnotator(model_name="test-model")
    input_dict = {"tags": ["dog"], "captions": ["A dog"], "score": 1.0}
    schema = AnnotationSchema(
        tags=cast(list[str], input_dict["tags"]),
        captions=cast(list[str], input_dict["captions"]),
        score=cast(float, input_dict["score"])
    )
    rawoutput = RawOutput(response=schema, error=None)
    formatted: list[WebApiFormattedOutput] = ann._format_predictions([rawoutput])
    assert isinstance(formatted, list)
    assert formatted[0]['error'] is None
    annotation = formatted[0]['annotation']
    assert annotation is not None
    assert annotation['tags'] == ["dog"]
    assert annotation['captions'] == ["A dog"]
    assert annotation['score'] == 1.0

def test_format_predictions_invalid_type() -> None:
    ann = AnthropicApiAnnotator(model_name="test-model")
    rawoutput = RawOutput(response=None, error="Invalid response type: str")
    formatted = ann._format_predictions([rawoutput])
    assert formatted[0]['annotation'] is None
    assert formatted[0]['error'] is not None
    assert "Invalid response type" in str(formatted[0]['error'])

def test_format_predictions_content_error() -> None:
    ann = AnthropicApiAnnotator(model_name="test-model")
    rawoutput = RawOutput(response=None, error="応答コンテンツが無効")
    formatted = ann._format_predictions([rawoutput])
    assert formatted[0]['annotation'] is None
    assert formatted[0]['error'] is not None
    assert "応答コンテンツが無効" in str(formatted[0]['error'])
