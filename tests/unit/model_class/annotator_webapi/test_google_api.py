from typing import cast

from image_annotator_lib.core.types import AnnotationSchema, RawOutput, WebApiFormattedOutput
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator


class DummyClient:
    class Models:
        def generate_content(self, model, contents, config):
            # AnnotationSchema(Pydanticモデル)を返す
            return AnnotationSchema(tags=["cat"], captions=["A cat"], score=8.5)
    models = Models()

def test_run_inference_success(monkeypatch):
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    annotator.client = DummyClient()
    annotator.api_model_id = "dummy-model"
    images = [b"dummybytes"]
    results = annotator._run_inference(images)
    assert isinstance(results, list)
    assert results[0].get("error") is None
    response = results[0].get("response")
    assert isinstance(response, AnnotationSchema)
    assert response.tags == ["cat"]
    assert response.captions == ["A cat"]
    assert response.score == 8.5

def test_run_inference_schema_error(monkeypatch):
    class BadClient:
        class Models:
            def generate_content(self, model, contents, config):
                # スキーマ不一致: 必須フィールド欠落
                return '{"tags": ["cat"]}'  # str型で返すことでバリデーションエラーを誘発
        models = Models()
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    annotator.client = BadClient()
    annotator.api_model_id = "dummy-model"
    images = [b"dummybytes"]
    try:
        annotator._run_inference(images)
    except Exception as e:
        assert "未対応のレスポンス型" in str(e) or "スキーマ不一致" in str(e)

def test_format_predictions_success() -> None:
    # RawOutput型(response: AnnotationSchema, error: None)からWebApiFormattedOutput(annotation: dict)へ変換されるか
    raw_outputs: list[RawOutput] = [
        {"response": AnnotationSchema(tags=["cat"], captions=["A cat"], score=8.5), "error": None}
    ]
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    formatted = annotator._format_predictions(raw_outputs)
    assert isinstance(formatted, list)
    assert isinstance(formatted[0]['annotation'], dict)
    assert formatted[0]['annotation']["tags"] == ["cat"]
    assert formatted[0]['annotation']["captions"] == ["A cat"]
    assert formatted[0]['annotation']["score"] == 8.5

def test_format_predictions_error() -> None:
    raw_outputs: list[RawOutput] = [
        {"response": None, "error": "some error"}
    ]
    annotator = GoogleApiAnnotator(model_name="dummy-model")
    formatted = annotator._format_predictions(raw_outputs)
    assert formatted[0]['annotation'] is None
    assert formatted[0]['error'] == "some error"
