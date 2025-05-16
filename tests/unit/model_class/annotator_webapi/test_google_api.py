from unittest.mock import MagicMock

import pytest

from image_annotator_lib.core.types import AnnotationSchema, WebApiInput
from image_annotator_lib.exceptions.errors import WebApiError
from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT

MODEL_NAME = "test_google_model"
DUMMY_IMAGE_BYTES = b"dummy_image_bytes_for_google_test"

@pytest.fixture
def mock_google_adapter():
    adapter = MagicMock()
    adapter.call_api = MagicMock()
    return adapter

@pytest.fixture
def annotator(mock_google_adapter):
    ann = GoogleApiAnnotator(model_name=MODEL_NAME)
    ann.client = mock_google_adapter
    ann.api_model_id = "gemini-pro-vision-test"
    return ann

def test_run_inference_success(annotator, mock_google_adapter):
    expected_annotation = AnnotationSchema(tags=["test_tag"], captions=["test_caption"], score=0.99)
    mock_google_adapter.call_api.return_value = expected_annotation

    images = [DUMMY_IMAGE_BYTES]
    results = annotator._run_inference(images)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].get("error") is None
    response = results[0].get("response")
    assert isinstance(response, AnnotationSchema)
    assert response == expected_annotation

    expected_model_id = "gemini-pro-vision-test"
    expected_web_api_input = WebApiInput(image_bytes=DUMMY_IMAGE_BYTES)
    expected_params = {
        "prompt": BASE_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 1800
    }
    expected_output_schema = AnnotationSchema

    print("DEBUG: Expected model_id:", expected_model_id)
    print("DEBUG: Expected web_api_input:", expected_web_api_input)
    print("DEBUG: Expected params:", expected_params)
    print("DEBUG: Expected output_schema:", expected_output_schema)

    actual_call_args = mock_google_adapter.call_api.call_args
    if actual_call_args:
        actual_args, actual_kwargs = actual_call_args
        print("DEBUG: Actual args:", actual_args)
        print("DEBUG: Actual kwargs model_id:", actual_kwargs.get("model_id"))
        print("DEBUG: Actual kwargs web_api_input:", actual_kwargs.get("web_api_input"))
        print("DEBUG: Actual kwargs params:", actual_kwargs.get("params"))
        print("DEBUG: Actual kwargs output_schema:", actual_kwargs.get("output_schema"))
    else:
        print("DEBUG: Actual call_args is None (API was not called as expected)")

    mock_google_adapter.call_api.assert_called_once_with(
        model_id=expected_model_id,
        web_api_input=expected_web_api_input,
        params=expected_params,
        output_schema=expected_output_schema
    )

def test_run_inference_api_error(annotator, mock_google_adapter):
    error_msg = "Google Unit Test API Error!"
    mock_google_adapter.call_api.side_effect = WebApiError(error_msg, provider_name="Google")

    images = [DUMMY_IMAGE_BYTES]
    results = annotator._run_inference(images)

    assert len(results) == 1
    assert results[0].get("response") is None
    assert f"Google API Adapter Error: Google API エラー: {error_msg}" in results[0].get("error")
    mock_google_adapter.call_api.assert_called_once()

# _format_predictions 関連のテストは一旦コメントアウト
# def test_format_predictions_success() -> None:
#     # RawOutput型(response: AnnotationSchema, error: None)からWebApiFormattedOutput(annotation: dict)へ変換されるか
#     raw_outputs: list[RawOutput] = [
#         {"response": AnnotationSchema(tags=["cat"], captions=["A cat"], score=8.5), "error": None}
#     ]
#     annotator = GoogleApiAnnotator(model_name="dummy-model")
#     formatted = annotator._format_predictions(raw_outputs)
#     assert isinstance(formatted, list)
#     assert isinstance(formatted[0]['annotation'], dict)
#     assert formatted[0]['annotation']["tags"] == ["cat"]
#     assert formatted[0]['annotation']["captions"] == ["A cat"]
#     assert formatted[0]['annotation']["score"] == 8.5
#
# def test_format_predictions_error() -> None:
#     raw_outputs: list[RawOutput] = [
#         {"response": None, "error": "some error"}
#     ]
#     annotator = GoogleApiAnnotator(model_name="dummy-model")
#     formatted = annotator._format_predictions(raw_outputs)
#     assert formatted[0]['annotation'] is None
#     assert formatted[0]['error'] == "some error"
