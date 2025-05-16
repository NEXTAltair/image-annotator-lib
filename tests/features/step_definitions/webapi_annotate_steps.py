import logging
import os

import pytest
from pytest_bdd import given, parsers, then, when

from image_annotator_lib.api import annotate
from image_annotator_lib.core import model_factory
from image_annotator_lib.core.model_factory import _get_api_key
from image_annotator_lib.exceptions.errors import ApiAuthenticationError
from image_annotator_lib.model_class.annotator_webapi import google_api

logger = logging.getLogger(__name__)

# --- Given ---
@given("APIキーが環境変数に設定されている", target_fixture="api_key")
def api_key_is_set():
    # 主要APIキーのいずれかがセットされていることを検証し、値を返す
    key_candidates = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "DUMMY_API_KEY",
    ]
    for key in key_candidates:
        value = os.environ.get(key)
        if value:
            return value
    raise AssertionError("いずれのAPIキーも環境変数に設定されていません")

@given(parsers.parse("{provider} APIが利用可能な状態になっている"), target_fixture="api_key")
def provider_api_available(provider):
    # プロジェクト実装の_get_api_key関数を使ってAPIキーを取得
    api_key = _get_api_key(provider, "")
    assert api_key is not None
    return api_key

@given("APIキーが未設定の状態になっている")
def api_key_is_unset(monkeypatch):
    # target_provider = "Google" # 不要なため削除
    # target_env_var = "GOOGLE_API_KEY" # 不要なため削除

    logger.info("APIキー未設定をシミュレート: model_factory._get_api_key が ApiAuthenticationError を送出するようにモックします。")

    # _get_api_key が呼び出されたら、指定されたエラーを送出する関数を定義
    def mock_get_api_key_error(provider_name: str, api_model_id: str):
        # 実際の provider_name に基づいてエラーメッセージを作成
        env_var_map = {
            "Google": "GOOGLE_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "OpenRouter": "OPENROUTER_API_KEY",
        }
        assumed_env_var = env_var_map.get(provider_name, "UNKNOWN_ENV_VAR")
        error_message = f"環境変数 '{assumed_env_var}' が設定されていないか、空です。 (プロバイダー: {provider_name}) - Mocked Error"
        raise ApiAuthenticationError(provider_name=provider_name, message=error_message)

    # monkeypatch.setattr を使って model_factory._get_api_key を上記の関数で置き換え
    monkeypatch.setattr(model_factory, "_get_api_key", mock_get_api_key_error)

    # (オプション) 環境変数も削除しておくことで、より確実に未設定状態を模倣
    keys_to_unset = [
        "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY", "DUMMY_API_KEY",
    ]
    logger.debug("念のため、関連する環境変数も削除します (monkeypatch使用)。")
    for key in keys_to_unset:
        if os.environ.get(key) is not None:
            monkeypatch.delenv(key, raising=False)

    logger.info("monkeypatch適用後の環境変数チェック:")
    for key in keys_to_unset:
        logger.info(f"  '{key}': {os.environ.get(key)}")

@given("APIがタイムアウトするよう設定されている")
def api_timeout(monkeypatch):
    # このシナリオでは Gemini (Google API) が使われることを前提として、
    # GoogleApiAnnotator の _run_inference をモックしてタイムアウトをシミュレートする。

    # original_run_inference = google_api.GoogleApiAnnotator._run_inference # コメントアウトされたまま

    def mock_run_inference_timeout(self, processed_images: list[str] | list[bytes]):
        logger.info("タイムアウトをシミュレート: google_api.GoogleApiAnnotator._run_inference で RuntimeError を送出します。")
        # 実際の _run_inference が返す型に合わせて、エラー発生時は適切な値を返すか、
        # 例外を送出する。ここでは RuntimeError を使用。
        # (メッセージはテストで検証される内容に合わせて調整が必要な場合がある)
        raise RuntimeError("Simulated API timeout for Google API") # google_genai_errors.APIError から変更

    monkeypatch.setattr(
        google_api.GoogleApiAnnotator,
        "_run_inference",
        mock_run_inference_timeout
    )
    logger.info("google_api.GoogleApiAnnotator._run_inference がタイムアウトするようにモックしました。")

@given("APIからエラーレスポンスが返されるよう設定されている")
def api_error_response(monkeypatch):
    # このシナリオでは Gemini (Google API) が主にテストされると仮定し、
    # GoogleApiAnnotator の _run_inference をモックして汎用的なAPIエラーをシミュレートする。
    def mock_run_inference_api_error(self, processed_images: list[str] | list[bytes]):
        error_message = "Simulated API Processing Error"
        logger.info(f"汎用APIエラーをシミュレート: google_api.GoogleApiAnnotator._run_inference で RuntimeError('{error_message}') を送出します。")
        raise RuntimeError(error_message)

    monkeypatch.setattr(
        google_api.GoogleApiAnnotator,
        "_run_inference",
        mock_run_inference_api_error
    )
    logger.info("google_api.GoogleApiAnnotator._run_inference が汎用APIエラーメッセージを含むエラーを送出するようにモックしました。")

# --- When ---
@when(parsers.parse("画像を指定して{model_alias}モデルでアノテーションを実行する"), target_fixture="annotation_result")
def run_annotation_with_model(model_alias, single_image):
    model_name = model_alias if model_alias else "Gemini 2.5 Pro Preview"
    result = annotate([single_image], [model_name])
    return result

@when("画像を指定してアノテーションを実行する", target_fixture="annotation_result")
def run_annotation_expect_error(single_image):
    # annotate を呼び出し、結果を返す
    # エラーシナリオでは特定のモデルに固定するか、設定ファイル等から取得するか検討。
    # ここではフィーチャーファイルでモデル指定がないため、代表的なモデルを使用。
    # 以前の pytest.raises で指定していたモデルに合わせる。
    model_to_use = "Gemini 2.5 Pro Preview"
    logger.info(f"エラーシナリオ用のアノテーション実行（モデル: {model_to_use}）")
    result = annotate([single_image], [model_to_use])
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
                    "Google API: レスポンスが空です", # Google API が空応答
                    "Google API: GenerateContentResponse内のtextコンテンツが空です", # Google API が空コンテンツ
                    "Google API: GenerateContentResponseの構造が予期されるものではありませんでした", # Google API 予期せぬ構造
                    "FinishReason.OTHER", # Google API 予期せぬ終了理由
                    # 必要に応じて他のAPIの一時的な問題を示すエラーメッセージを追加
                ]
                # OpenRouterのJSONエラーなどはコード側の問題の可能性があるため、スキップしない
                if any(cond in err for cond in skip_conditions):
                    pytest.skip(f"モデル '{model_name}' でAPIから予期せぬ応答/空応答があったためスキップ: {err[:100]}...")
                # スキップ条件に合致しないエラーはアサーションエラーとして失敗させる
                raise AssertionError(f"モデル '{model_name}' で予期せぬAPIエラーが発生しました: {err}")
            # エラーがない場合は、主要なキーが存在することを確認
            assert res.get("tags") is not None or res.get("captions") is not None or res.get("score") is not None, \
                   f"モデル '{model_name}' の結果に tags, captions, score のいずれも含まれていません: {res}"

@then("エラーは発生していない")
def no_error_occurred(annotation_result):
    for model_results in annotation_result.values() if isinstance(annotation_result, dict) else annotation_result:
        for res in model_results.values() if isinstance(model_results, dict) else [model_results]:
            assert res["error"] is None

@then(parsers.parse('"{expected_error_type}" のエラーメッセージが返される'))
def api_specific_error_message_is_returned(annotation_result, expected_error_type):
    assert annotation_result is not None, "アノテーション結果がNoneであってはなりません。"

    found_at_least_one_model_result = False
    if not annotation_result: # 結果自体が空の辞書の場合
        # found_at_least_one_model_result が False のままなので、最終的なアサーションで失敗する
        logger.warning("アノテーション結果が空の辞書です。")

    for image_hash, model_results_dict in annotation_result.items():
        assert isinstance(model_results_dict, dict), \
            f"画像ハッシュ '{image_hash}' の結果 (model_results_dict) が辞書ではありません。"

        if not model_results_dict:
            logger.warning(f"画像ハッシュ '{image_hash}' にはモデルごとの結果が含まれていません。")
            continue

        for model_name, model_result in model_results_dict.items():
            found_at_least_one_model_result = True
            error_message = model_result.get("error")

            assert error_message is not None, \
                (f"モデル '{model_name}' (画像ハッシュ: {image_hash}) の結果にエラーメッセージが含まれていません。"
                 f"取得した結果: {model_result}")

            # エラーメッセージに期待されるエラータイプ名が含まれているかを確認
            assert expected_error_type in error_message, \
                (f"モデル '{model_name}' (画像ハッシュ: {image_hash}) のエラーメッセージ '{error_message}' に "
                 f"期待されるエラータイプ '{expected_error_type}' が含まれていません。")

            # ApiAuthenticationError の場合は、APIキー関連の文言もチェック
            if expected_error_type == "ApiAuthenticationError":
                assert "APIキー" in error_message or "API key" in error_message or "環境変数" in error_message, \
                    (f"モデル '{model_name}' (画像ハッシュ: {image_hash}) の ApiAuthenticationError メッセージ '{error_message}' に "
                     f"APIキー関連のキーワード（APIキー, API key, 環境変数）が含まれていません。")

    assert found_at_least_one_model_result, \
        f"アノテーション結果内に、'{expected_error_type}' のエラーメッセージを検証できるモデル結果が1つも見つかりませんでした。"

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
