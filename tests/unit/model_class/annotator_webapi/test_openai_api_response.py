from unittest.mock import MagicMock

import pytest

from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import (
    AnnotationSchema,
    Responsedict,
)

# ダミーのbase64画像文字列
DUMMY_BASE64 = "dGVzdGltYWdlYmFzZTY0Ig=="

class DummyResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed

@pytest.fixture
def annotator():
    ann = OpenAIApiAnnotator(model_name="test-model")
    # clientとapi_model_idをモック
    ann.client = MagicMock()
    ann.api_model_id = "gpt-4.1"
    ann.config = {"max_output_tokens": 100, "temperature": 0.5}
    return ann

def test_run_inference_success(annotator):
    # AnnotationSchemaのダミー
    parsed = AnnotationSchema(tags=["cat", "animal"], captions=["A cat on a mat"], score=8.5)
    annotator.client.responses.create.return_value = DummyResponse(parsed)
    result = annotator._run_inference([DUMMY_BASE64])
    assert isinstance(result, list)
    assert result[0]["error"] is None
    assert isinstance(result[0]["response"], AnnotationSchema)
    assert result[0]["response"].tags == ["cat", "animal"]

def test_run_inference_api_error(annotator):
    annotator.client.responses.create.side_effect = Exception("API error!")
    result = annotator._run_inference([DUMMY_BASE64])
    assert result[0]["response"] is None
    assert "API error" in result[0]["error"]

def test_format_predictions_success():
    parsed = AnnotationSchema(tags=["dog"], captions=["A dog"], score=7.0)
    responsedict = Responsedict(response=parsed, error=None)
    ann = OpenAIApiAnnotator(model_name="test-model")
    formatted = ann._format_predictions([responsedict])
    assert isinstance(formatted, list)
    assert formatted[0]["error"] is None
    annotation = formatted[0]["annotation"]
    assert annotation is not None
    assert annotation["tags"] == ["dog"]
    assert annotation["caption"] == "A dog"
    assert annotation["score"] == 7.0

def test_format_predictions_error():
    responsedict = Responsedict(response=None, error="some error")
    ann = OpenAIApiAnnotator(model_name="test-model")
    formatted = ann._format_predictions([responsedict])
    assert formatted[0]["annotation"] is None
    assert formatted[0]["error"] == "some error"
