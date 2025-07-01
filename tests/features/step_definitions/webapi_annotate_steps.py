import logging
import os

import pytest
from pytest_bdd import given, parsers, then, when
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from image_annotator_lib.api import annotate

logger = logging.getLogger(__name__)


# --- Given ---
@given("PydanticAI テスト環境が設定されている")
def pydantic_ai_test_environment():
    """PydanticAI テスト環境を設定: 実APIコールを有効化"""
    models.ALLOW_MODEL_REQUESTS = True
    logger.info("PydanticAI E2Eテスト環境設定完了: 実APIリクエストを有効化")


@given(parsers.parse("{provider} プロバイダーが利用可能になっている"))
def provider_available(provider):
    """指定されたプロバイダーのAPIキーが.envまたは環境変数に設定されていることを確認"""
    # PydanticAIが.envファイルから自動的にAPIキーを読み込む
    key_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Google": "GOOGLE_API_KEY"
    }
    
    required_key = key_map.get(provider)
    if not required_key:
        raise ValueError(f"サポートされていないプロバイダー: {provider}")
    
    # .envファイルまたは環境変数からAPIキーをチェック
    api_key = os.getenv(required_key)
    if not api_key:
        pytest.skip(f"{provider} APIキー ({required_key}) が設定されていないためスキップします")
    
    logger.info(f"{provider} プロバイダーのAPIキーが確認されました")
    return True


@given("PydanticAI認証が失敗するよう設定されている")
def pydantic_ai_auth_failure(monkeypatch):
    """PydanticAI認証エラーをシミュレート"""
    # 一時的にAPIキーを削除して認証失敗を引き起こす
    keys_to_remove = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    
    for key in keys_to_remove:
        if os.environ.get(key):
            monkeypatch.delenv(key, raising=False)
    
    logger.info("PydanticAI認証失敗をシミュレート: APIキーを一時的に削除")


@given("PydanticAIモデルが利用不可になっている")
def pydantic_ai_model_unavailable(monkeypatch):
    """PydanticAIモデル利用不可をシミュレート"""
    # 無効なAPIキーを設定して認証エラーを引き起こす
    monkeypatch.setenv("OPENAI_API_KEY", "invalid_key_for_testing")
    logger.info("PydanticAIモデル利用不可をシミュレート: 無効なAPIキーを設定")


@given("PydanticAI TestModelが設定されている")
def pydantic_ai_test_model():
    """PydanticAI TestModelを設定してモック実行を有効化"""
    models.ALLOW_MODEL_REQUESTS = False
    logger.info("PydanticAI TestModelが設定されました: モック実行モード")


@given("APIがタイムアウトするよう設定されている")
def api_timeout(monkeypatch):
    """PydanticAI APIタイムアウトをシミュレート"""
    # E2Eテスト環境では実際のAPIリクエストを有効化する
    models.ALLOW_MODEL_REQUESTS = True
    
    # テスト用の非常に短いタイムアウトを設定
    monkeypatch.setenv("ANTHROPIC_API_TIMEOUT", "0.001")  # 1msの極短タイムアウト
    monkeypatch.setenv("OPENAI_API_TIMEOUT", "0.001")
    monkeypatch.setenv("GOOGLE_API_TIMEOUT", "0.001")
    
    logger.info("E2Eテスト用の極短タイムアウト(1ms)を設定してタイムアウトエラーをシミュレート")


@given("APIからエラーレスポンスが返されるよう設定されている")
def api_error_response(monkeypatch):
    """PydanticAI APIエラーレスポンスをシミュレート"""
    # E2Eテスト環境では実際のAPIリクエストを有効化
    models.ALLOW_MODEL_REQUESTS = True
    
    # 無効な形式のAPIキーを設定してAPIエラーを発生させる
    monkeypatch.setenv("OPENAI_API_KEY", "invalid_test_key_format")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid_test_key_format")
    monkeypatch.setenv("GOOGLE_API_KEY", "invalid_test_key_format")
    
    logger.info("E2Eテスト用の無効なAPIキーを設定してAPIエラーをシミュレート")


# --- When ---
@when(
    parsers.parse("画像を指定して{model_name}でアノテーションを実行する"),
    target_fixture="annotation_result",
)
def run_annotation_with_model(model_name, single_image):
    """指定されたmodel_nameで実際のアノテーションを実行"""
    logger.info(f"E2Eアノテーション実行: モデル={model_name}")
    result = annotate(images_list=[single_image], model_name_list=[model_name])
    logger.info(f"アノテーション結果: {result}")
    return result


@when("画像を指定してアノテーションを実行する", target_fixture="annotation_result")
def run_annotation_expect_error(single_image):
    """エラーシナリオ用のアノテーション実行"""
    model_to_use = "gpt-4o-mini"  # PydanticAIが自動でOpenAIプロバイダーを推測
    logger.info(f"エラーシナリオ用のアノテーション実行: モデル={model_to_use}")
    result = annotate(images_list=[single_image], model_name_list=[model_to_use])
    return result


# --- Then ---
@then("アノテーション結果に期待通りの内容が含まれる")
def annotation_result_is_expected(annotation_result):
    assert annotation_result is not None
    for model_results in annotation_result.values():
        for model_name, res in model_results.items():
            err = res.get("error")
            if err:
                # API側の問題や予期せぬ応答で、コードの問題ではない場合にスキップする条件
                skip_conditions = [
                    "Google API: レスポンスが空です",  # Google API が空応答
                    "Google API: GenerateContentResponse内のtextコンテンツが空です",  # Google API が空コンテンツ
                    "Google API: GenerateContentResponseの構造が予期されるものではありませんでした",  # Google API 予期せぬ構造
                    "FinishReason.OTHER",  # Google API 予期せぬ終了理由
                    # 必要に応じて他のAPIの一時的な問題を示すエラーメッセージを追加
                ]
                # OpenRouterのJSONエラーなどはコード側の問題の可能性があるため、スキップしない
                if any(cond in err for cond in skip_conditions):
                    pytest.skip(
                        f"モデル '{model_name}' でAPIから予期せぬ応答/空応答があったためスキップ: {err[:100]}..."
                    )
                # スキップ条件に合致しないエラーはアサーションエラーとして失敗させる
                raise AssertionError(f"モデル '{model_name}' で予期せぬAPIエラーが発生しました: {err}")
            # エラーがない場合は、主要なキーが存在することを確認
            assert (
                res.get("tags") is not None
                or res.get("captions") is not None
                or res.get("score") is not None
            ), f"モデル '{model_name}' の結果に tags, captions, score のいずれも含まれていません: {res}"


@then("エラーは発生していない")
def no_error_occurred(annotation_result):
    for model_results in (
        annotation_result.values() if isinstance(annotation_result, dict) else annotation_result
    ):
        for res in model_results.values() if isinstance(model_results, dict) else [model_results]:
            assert res["error"] is None


@then("認証エラーメッセージが返される")
def authentication_error_returned(annotation_result):
    """PydanticAI認証エラーが返されることを確認"""
    assert annotation_result is not None
    found_auth_error = False
    
    for image_hash, model_results_dict in annotation_result.items():
        for model_name, model_result in model_results_dict.items():
            error_message = model_result.get("error")
            if error_message and ("api key" in error_message.lower() or "authentication" in error_message.lower()):
                found_auth_error = True
                logger.info(f"認証エラーを確認: {error_message}")
    
    assert found_auth_error, "認証エラーメッセージが見つかりませんでした"


@then("モデル利用不可エラーメッセージが返される")
def model_unavailable_error_returned(annotation_result):
    """PydanticAIモデル利用不可エラーが返されることを確認"""
    assert annotation_result is not None
    found_model_error = False
    
    for image_hash, model_results_dict in annotation_result.items():
        for model_name, model_result in model_results_dict.items():
            error_message = model_result.get("error")
            if error_message and ("invalid" in error_message.lower() or "unauthorized" in error_message.lower()):
                found_model_error = True
                logger.info(f"モデル利用不可エラーを確認: {error_message}")
    
    assert found_model_error, "モデル利用不可エラーメッセージが見つかりませんでした"


@then("TestModelによるモック結果が返される")
def test_model_result_returned(annotation_result):
    """PydanticAI TestModelによるモック結果が返されることを確認"""
    assert annotation_result is not None
    
    for image_hash, model_results_dict in annotation_result.items():
        for model_name, model_result in model_results_dict.items():
            assert model_result.get("error") is None, f"TestModelでエラーが発生: {model_result.get('error')}"
            tags = model_result.get("tags", [])
            assert len(tags) > 0, "TestModelからタグが返されませんでした"
            logger.info(f"TestModelの結果を確認: {tags}")


@then("結果のタグリストは空である")
def tag_list_is_empty(annotation_result):
    assert annotation_result is not None
    for image_key_results in annotation_result.values():
        for model_result in image_key_results.values():
            tags = model_result.get("tags", [])
            assert isinstance(tags, list)
            assert tags == []


@then("タイムアウトエラーメッセージが返される")
def timeout_error_message(annotation_result):
    assert annotation_result is not None
    for image_key_results in annotation_result.values():
        for model_result in image_key_results.values():
            assert model_result["error"]
            assert "タイムアウト" in model_result["error"] or "timeout" in model_result["error"]


@then("APIエラーメッセージが結果に含まれる")
def api_error_message_in_result(annotation_result):
    assert annotation_result is not None
    for image_key_results in annotation_result.values():
        for model_result in image_key_results.values():
            assert model_result["error"]
            # 大文字･小文字を区別せずに "error" が含まれているか確認
            assert "error" in model_result["error"].lower() or "エラー" in model_result["error"]
