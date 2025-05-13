from unittest.mock import MagicMock

import openai  # openai をインポート
import pytest

from image_annotator_lib.core.types import AnnotationSchema, RawOutput
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

# ダミーのbase64画像文字列
DUMMY_BASE64 = "dGVzdGltYWdlYmFzZTY0Ig=="

class DummyResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed

@pytest.fixture
def annotator():
    ann = OpenAIApiAnnotator(model_name="test-model")
    # clientとapi_model_idをモック
    ann.client = MagicMock(spec=openai.OpenAI)
    ann.client.__class__ = openai.OpenAI
    ann.client.responses = MagicMock() # responses 属性も MagicMock にする
    ann.client.responses.parse = MagicMock() # parse メソッドをモック
    ann.api_model_id = "gpt-4.1"
    ann.config = {"max_output_tokens": 100, "temperature": 0.5}
    return ann

def test_run_inference_success(annotator):
    # AnnotationSchemaのダミー
    parsed = AnnotationSchema(tags=["cat", "animal"], captions=["A cat on a mat"], score=8.5)
    # client.responses.create ではなく client.responses.parse をモックする
    annotator.client.responses.parse.return_value = DummyResponse(parsed)
    result = annotator._run_inference([DUMMY_BASE64])
    assert isinstance(result, list)
    assert result[0]["error"] is None
    assert isinstance(result[0]["response"], AnnotationSchema)
    assert result[0]["response"].tags == ["cat", "animal"]

def test_run_inference_api_error(annotator):
    # client.responses.create ではなく client.responses.parse をモックする
    annotator.client.responses.parse.side_effect = Exception("API error!")
    result = annotator._run_inference([DUMMY_BASE64])
    assert result[0]["response"] is None
    assert "API error" in result[0]["error"]

def test_format_predictions_error(annotator):
    responsedict = RawOutput(response=None, error="some error")
    formatted = annotator._format_predictions([responsedict])
    assert formatted[0]["annotation"] is None
    assert formatted[0]["error"] == "some error"
