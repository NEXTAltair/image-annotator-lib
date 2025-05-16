from unittest.mock import MagicMock

import pytest

from image_annotator_lib.core.types import AnnotationSchema, RawOutput, WebApiInput # WebApiInput をインポート
from image_annotator_lib.exceptions.errors import WebApiError # WebApiError をインポート
from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
# webapi_shared から BASE_PROMPT と SYSTEM_PROMPT をインポート
from image_annotator_lib.model_class.annotator_webapi.webapi_shared import BASE_PROMPT, SYSTEM_PROMPT

# ダミーのbase64画像文字列
DUMMY_BASE64 = "dGVzdGltYWdlYmFzZTY0Ig=="

# DummyResponse クラスは不要なので削除

MODEL_NAME = "test_openai_model" # MODEL_NAME を定義 (test_openai_api_response.py から流用)

@pytest.fixture
def mock_openai_adapter(): # OpenAIAdapter のモックを返すフィクスチャ
    adapter = MagicMock()
    adapter.call_api = MagicMock()
    return adapter

@pytest.fixture
def annotator(mock_openai_adapter): # mock_openai_adapter を使用
    ann = OpenAIApiAnnotator(model_name=MODEL_NAME)
    ann.client = mock_openai_adapter # client に Adapter のモックを設定
    ann.api_model_id = "gpt-4o-unit-test" # APIモデルIDを設定
    # ann.config は _run_inference 内で config_registry から取得するため、ここでは設定不要
    return ann

def test_run_inference_success(annotator, mock_openai_adapter): # mock_openai_adapter を引数に追加
    # AnnotationSchemaのダミー
    expected_annotation = AnnotationSchema(tags=["cat", "animal"], captions=["A cat on a mat"], score=0.85) # scoreをfloatに変更
    mock_openai_adapter.call_api.return_value = expected_annotation # call_api の戻り値を設定

    result = annotator._run_inference([DUMMY_BASE64])

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["error"] is None
    assert result[0]["response"] == expected_annotation
    mock_openai_adapter.call_api.assert_called_once_with(
        model_id="gpt-4o-unit-test",
        web_api_input=WebApiInput(image_b64=DUMMY_BASE64),
        params=pytest.approx({
            "prompt": BASE_PROMPT,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0.7, # _run_inference内のデフォルト値に合わせる
            "max_output_tokens": 2000, # _run_inference内のデフォルト値に合わせる
            "use_responses_parse": True
        }),
        output_schema=AnnotationSchema
    )

def test_run_inference_api_error(annotator, mock_openai_adapter): # mock_openai_adapter を引数に追加
    error_msg = "Unit Test API error!"
    mock_openai_adapter.call_api.side_effect = WebApiError(error_msg) # WebApiError を送出

    result = annotator._run_inference([DUMMY_BASE64])

    assert len(result) == 1
    assert result[0]["response"] is None
    assert f"OpenAI Adapter Error: API エラー: {error_msg}" in result[0]["error"]
    mock_openai_adapter.call_api.assert_called_once()