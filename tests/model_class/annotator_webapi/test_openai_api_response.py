from unittest.mock import MagicMock

import openai
import pytest

from image_annotator_lib.core.types import AnnotationSchema
from image_annotator_lib.exceptions.errors import WebApiError
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

MODEL_NAME = "test_openai_model"

@pytest.fixture
def mock_openai_client():
    """openai.OpenAI クライアントのモックを返すフィクスチャ"""
    client = MagicMock(spec=openai.OpenAI)
    # client.responses を MagicMock インスタンスとして作成
    client.responses = MagicMock()
    # その上で client.responses.parse をモック
    client.responses.parse = MagicMock()
    return client

@pytest.fixture
def annotator(mock_openai_client): # mock_config を削除 (config は直接アクセスされるため)
    """テスト用 OpenAIApiAnnotator インスタンスを返すフィクスチャ"""
    annotator_instance = OpenAIApiAnnotator(model_name=MODEL_NAME)
    annotator_instance.client = mock_openai_client
    annotator_instance.api_model_id = "gpt-4o-test"
    # config_registry を直接モックする必要はない (通常は pytest.ini などで設定される)
    return annotator_instance

def test_init(annotator):
    """初期化が正しく行われるかテスト"""
    assert annotator.model_name == MODEL_NAME
    assert annotator.api_model_id == "gpt-4o-test"
    assert annotator.client is not None


def test_run_inference_success(annotator, mock_openai_client):
    """_run_inference が成功するケースをテスト"""
    image_data = ["base64_image_data_1"]
    # モックの戻り値を設定 (AnnotationSchema インスタンスを直接返す)
    mock_response = MagicMock()
    expected_annotation = AnnotationSchema(tags=["tag1"], captions=["caption1"], score=0.9)
    mock_response.output_parsed = expected_annotation
    # refusal や error 属性がないことを確認
    del mock_response.refusal
    del mock_response.error
    mock_openai_client.responses.parse.return_value = mock_response

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["error"] is None
    assert result[0]["response"] == expected_annotation
    mock_openai_client.responses.parse.assert_called_once()


def test_run_inference_api_error(annotator, mock_openai_client):
    """API エラーが発生するケースをテスト"""
    image_data = ["base64_image_data_1"]
    # parse が APIError を送出するように設定
    mock_openai_client.responses.parse.side_effect = openai.APIError("API Error Message", request=MagicMock(), body=None)

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAI API Error: API Error Message" in result[0]["error"]
    mock_openai_client.responses.parse.assert_called_once()

def test_run_inference_refusal(annotator, mock_openai_client):
    """API が拒否 (refusal) するケースをテスト"""
    image_data = ["base64_image_data_1"]
    # refusal を含むレスポンスを返すように設定
    mock_response = MagicMock()
    mock_response.refusal = "Request refused due to safety reasons."
    del mock_response.output_parsed # output_parsed はない想定
    del mock_response.error
    mock_openai_client.responses.parse.return_value = mock_response

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAIがリクエストを拒否しました: Request refused due to safety reasons." in result[0]["error"]
    mock_openai_client.responses.parse.assert_called_once()


def test_run_inference_unexpected_response_type(annotator, mock_openai_client):
    """output_parsed が予期せぬ型で返るケースをテスト"""
    image_data = ["base64_image_data_1"]
    # output_parsed に AnnotationSchema ではないオブジェクトを設定
    mock_response = MagicMock()
    mock_response.output_parsed = {"invalid": "data"} # 辞書型など
    del mock_response.refusal
    del mock_response.error
    mock_openai_client.responses.parse.return_value = mock_response

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAIレスポンスのoutput_parsedが予期せぬ型です: <class 'dict'>" in result[0]["error"]
    mock_openai_client.responses.parse.assert_called_once()

def test_run_inference_empty_or_no_output(annotator, mock_openai_client):
    """レスポンスに output_parsed も refusal も error もないエッジケース"""
    image_data = ["base64_image_data_1"]
    mock_response = MagicMock()
    # すべての関連属性を削除
    del mock_response.output_parsed
    del mock_response.refusal
    del mock_response.error
    # 必要であれば to_dict() もモック
    mock_response.to_dict.return_value = {"some": "other_data"}
    mock_openai_client.responses.parse.return_value = mock_response

    result = annotator._run_inference(image_data)

    assert len(result) == 1
    assert result[0]["response"] is None
    assert "OpenAIから予期せぬレスポンス形式" in result[0]["error"]
    mock_openai_client.responses.parse.assert_called_once()


def test_run_inference_invalid_input_type(annotator):
    """不正な入力型 (bytes) が渡された場合のテスト"""
    invalid_data = [b"bytes_data"]
    with pytest.raises(WebApiError) as excinfo:
        annotator._run_inference(invalid_data)
    assert "expects a list of base64 encoded image strings" in str(excinfo.value)