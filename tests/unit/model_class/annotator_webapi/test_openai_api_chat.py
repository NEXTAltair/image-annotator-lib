from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import AnnotationSchema, FormattedOutput

DUMMY_BASE64 = "dGVzdGltYWdlYmFzZTY0"
DUMMY_MODEL = "test-model"

DUMMY_JSON = {
    "tags": ["cat", "animal"],
    "captions": ["A cat sitting on a mat."],
    "score": 8.75,
}
DUMMY_JSON_STR = '{"tags":["cat","animal"],"captions":["A cat sitting on a mat."],"score":8.75}'

class DummyChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)

class DummyResponse:
    def __init__(self, content):
        self.choices = [DummyChoice(content)]

@pytest.fixture
def annotator():
    with patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.config_registry") as mock_cfg:
        mock_cfg.get.side_effect = lambda model, key, default=None: {
            "temperature": 0.7,
            "max_output_tokens": 1800,
            "timeout": 60.0,
            "json_schema_supported": True,
            "referer": None,
            "app_name": None,
        }.get(key, default)
        annotator = OpenRouterApiAnnotator(DUMMY_MODEL)
        yield annotator

def test_run_inference_success(annotator):
    annotator.client = OpenAI(api_key="sk-test")
    annotator.api_model_id = DUMMY_MODEL
    annotator._wait_for_rate_limit = MagicMock()
    with patch.object(annotator, "_call_openrouter_with_json_schema", return_value=DummyResponse(DUMMY_JSON_STR)):
        results = annotator._run_inference([DUMMY_BASE64])
    assert isinstance(results, list)
    assert results[0]["error"] is None
    assert isinstance(results[0]["response"], AnnotationSchema)
    assert results[0]["response"].tags == ["cat", "animal"]

def test_run_inference_content_empty(annotator):
    annotator.client = OpenAI(api_key="sk-test")
    annotator.api_model_id = DUMMY_MODEL
    annotator._wait_for_rate_limit = MagicMock()
    annotator._call_openrouter_with_json_schema = MagicMock(
        return_value=DummyResponse(None)
    )
    results = annotator._run_inference([DUMMY_BASE64])
    assert results[0]["response"] is None
    assert "メッセージコンテンツが空" in results[0]["error"]

def test_run_inference_choices_empty(annotator):
    class DummyResponseNoChoices:
        choices = []
    annotator.client = OpenAI(api_key="sk-test")
    annotator.api_model_id = DUMMY_MODEL
    annotator._wait_for_rate_limit = MagicMock()
    annotator._call_openrouter_with_json_schema = MagicMock(
        return_value=DummyResponseNoChoices()
    )
    results = annotator._run_inference([DUMMY_BASE64])
    assert results[0]["response"] is None
    assert "choicesが空" in results[0]["error"]

def test_run_inference_invalid_json(annotator):
    annotator.client = OpenAI(api_key="sk-test")
    annotator.api_model_id = DUMMY_MODEL
    annotator._wait_for_rate_limit = MagicMock()
    annotator._call_openrouter_with_json_schema = MagicMock(
        return_value=DummyResponse("{invalid json}")
    )
    results = annotator._run_inference([DUMMY_BASE64])
    assert results[0]["response"] is None
    assert "AnnotationSchema変換失敗" in results[0]["error"]

def test_run_inference_api_error(annotator):
    annotator.client = OpenAI(api_key="sk-test")
    annotator.api_model_id = DUMMY_MODEL
    annotator._wait_for_rate_limit = MagicMock()
    annotator._call_openrouter_with_json_schema = MagicMock(
        side_effect=Exception("API error!")
    )
    results = annotator._run_inference([DUMMY_BASE64])
    assert results[0]["response"] is None
    assert "Unexpected error" in results[0]["error"]

def test_format_predictions_success():
    output = [{"response": AnnotationSchema(tags=["cat"], captions=["A cat"], score=1.0), "error": None}]
    annotator = OpenRouterApiAnnotator(DUMMY_MODEL)
    formatted = annotator._format_predictions(output)  # type: ignore[arg-type]
    assert isinstance(formatted[0], FormattedOutput)
    assert formatted[0].annotation is not None
    assert formatted[0].annotation.tags == ["cat"]
    assert formatted[0].error is None

def test_format_predictions_type_error():
    output = [{"response": "not_schema", "error": None}]
    annotator = OpenRouterApiAnnotator(DUMMY_MODEL)
    formatted = annotator._format_predictions(output)  # type: ignore[arg-type]
    assert formatted[0].annotation is None
    assert formatted[0].error is not None
    assert "Invalid response type" in formatted[0].error

def test_format_predictions_error():
    output = [{"response": None, "error": "some error"}]
    annotator = OpenRouterApiAnnotator(DUMMY_MODEL)
    formatted = annotator._format_predictions(output)  # type: ignore[arg-type]
    assert formatted[0].annotation is None
    assert formatted[0].error == "some error"
