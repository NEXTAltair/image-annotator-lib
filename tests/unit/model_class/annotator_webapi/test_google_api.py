from unittest.mock import MagicMock

import pytest

from image_annotator_lib.core.base import WebApiAnnotationOutput
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import AnnotationSchema, FormattedOutput


class DummyClient:
    class Models:
        def generate_content(self, model, contents, config):
            # 正常系: AnnotationSchemaに適合するJSON文字列を返す
            return '{"tags": ["cat"], "captions": ["A cat"], "score": 8.5}'
    models = Models()

def test_run_inference_success(monkeypatch):
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    annotator.client = DummyClient()
    annotator.api_model_id = "dummy-model"
    # ダミー画像バイト列
    images = [b"dummybytes"]
    results = annotator._run_inference(images)
    assert isinstance(results, list)
    assert results[0]["error"] is None
    assert isinstance(results[0]["annotation"], dict)
    assert results[0]["annotation"]["tags"] == ["cat"]
    assert results[0]["annotation"]["captions"] == ["A cat"]
    assert results[0]["annotation"]["score"] == 8.5

def test_run_inference_schema_error(monkeypatch):
    class BadClient:
        class Models:
            def generate_content(self, model, contents, config):
                # スキーマ不一致: 必須フィールド欠落
                return '{"tags": ["cat"]}'
        models = Models()
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    annotator.client = BadClient()
    annotator.api_model_id = "dummy-model"
    images = [b"dummybytes"]
    results = annotator._run_inference(images)
    assert results[0]["annotation"] is None
    assert "スキーマ不一致" in results[0]["error"]

def test_format_predictions_success():
    # dictからAnnotationSchemaへ変換されるか
    raw_outputs = [
        {"annotation": {"tags": ["cat"], "captions": ["A cat"], "score": 8.5}, "error": None}
    ]
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    formatted = annotator._format_predictions(raw_outputs)
    assert isinstance(formatted, list)
    assert isinstance(formatted[0], FormattedOutput)
    assert formatted[0].annotation.tags == ["cat"]
    assert formatted[0].annotation.captions == ["A cat"]
    assert formatted[0].annotation.score == 8.5

def test_format_predictions_error():
    raw_outputs = [
        {"annotation": None, "error": "some error"}
    ]
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    formatted = annotator._format_predictions(raw_outputs)
    assert formatted[0].annotation is None
    assert formatted[0].error == "some error"
