from unittest.mock import MagicMock

import openai
import pytest

from image_annotator_lib.core.types import AnnotationSchema, WebApiInput
from image_annotator_lib.exceptions.errors import WebApiError
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT

MODEL_NAME = "test_openai_model"

@pytest.fixture
def mock_openai_adapter():
    """OpenAIAdapter のモックを返すフィクスチャ"""
    adapter = MagicMock()
    adapter.call_api = MagicMock()
    return adapter

@pytest.fixture
def annotator(mock_openai_adapter):
    """テスト用 OpenAIApiAnnotator インスタンスを返すフィクスチャ"""
    annotator_instance = OpenAIApiAnnotator(model_name=MODEL_NAME)
    annotator_instance.client = mock_openai_adapter
    annotator_instance.api_model_id = "gpt-4o-test"
    return annotator_instance

def test_init(annotator):
    """初期化が正しく行われるかテスト"""
    assert annotator.model_name == MODEL_NAME
    assert annotator.api_model_id == "gpt-4o-test"
    assert annotator.client is not None


def test_run_inference_success(annotator, mock_openai_adapter):
    """_run_inference が成功するケースをテスト"""
    image_data = ["base64_image_data_1"]
    expected_annotation = AnnotationSchema(tags=["tag1"], captions=["caption1"], score=0.9)
    mock_openai_adapter.call_api.return_value = expected_annotation

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["error"] is None
    assert result[0]["response"] == expected_annotation
    mock_openai_adapter.call_api.assert_called_once_with(
        model_id="gpt-4o-test",
        web_api_input=WebApiInput(image_b64="base64_image_data_1"),
        params=pytest.approx({
            "prompt": BASE_PROMPT,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0.7,
            "max_output_tokens": 2000,
            "use_responses_parse": True
        }),
        output_schema=AnnotationSchema
    )


def test_run_inference_api_error(annotator, mock_openai_adapter):
    """API エラーが発生するケースをテスト"""
    image_data = ["base64_image_data_1"]
    mock_openai_adapter.call_api.side_effect = WebApiError("Adapter API Error Message")

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAI Adapter Error: API エラー: Adapter API Error Message" in result[0]["error"]
    mock_openai_adapter.call_api.assert_called_once()

def test_run_inference_refusal(annotator, mock_openai_adapter):
    """API が拒否 (refusal) するケースをテスト"""
    image_data = ["base64_image_data_1"]
    mock_openai_adapter.call_api.side_effect = WebApiError("Request refused by API")

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAI Adapter Error: API エラー: Request refused by API" in result[0]["error"]
    mock_openai_adapter.call_api.assert_called_once()


def test_run_inference_unexpected_response_type(annotator, mock_openai_adapter):
    """Adapter が予期せぬ型を返した場合"""
    image_data = ["base64_image_data_1"]
    mock_openai_adapter.call_api.side_effect = WebApiError("Adapter returned unexpected type")

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAI Adapter Error: API エラー: Adapter returned unexpected type" in result[0]["error"]
    mock_openai_adapter.call_api.assert_called_once()

def test_run_inference_empty_or_no_output(annotator, mock_openai_adapter):
    """Adapter が有効なレスポンスを返さなかった場合"""
    image_data = ["base64_image_data_1"]
    mock_openai_adapter.call_api.side_effect = WebApiError("Adapter returned no valid output")

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAI Adapter Error: API エラー: Adapter returned no valid output" in result[0]["error"]
    mock_openai_adapter.call_api.assert_called_once()


def test_run_inference_invalid_input_type(annotator):
    """不正な入力型 (bytes) が渡された場合のテスト"""
    invalid_data = [b"bytes_data"]
    with pytest.raises(WebApiError) as excinfo:
        annotator._run_inference(invalid_data)
    assert "expects a list of base64 encoded image strings" in str(excinfo.value)