import logging
import os

import pytest
from pydantic_ai import models
from pytest_bdd import given, parsers, then, when

from image_annotator_lib.api import annotate

logger = logging.getLogger(__name__)


# --- Given ---
@given("アノテーション環境が設定されている")
def annotation_environment_setup():
    """アノテーション環境を設定: 実APIコールを有効化"""
    from image_annotator_lib.core.registry import initialize_registry
    
    models.ALLOW_MODEL_REQUESTS = True
    
    # レジストリを初期化してモデルが利用可能になるようにする
    initialize_registry()
    
    logger.info("アノテーション環境設定完了: 実APIリクエストを有効化、レジストリ初期化完了")


@given(parsers.parse("{model_name}が利用可能になっている"))
def model_available(model_name):
    """指定されたモデルが利用可能であることを確認"""
    # モデル名からプロバイダーを推測
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        required_key = "OPENAI_API_KEY"
    elif model_name.startswith("claude-"):
        required_key = "ANTHROPIC_API_KEY"
    elif model_name.startswith("gemini-"):
        required_key = "GOOGLE_API_KEY"
    else:
        # その他のモデルの場合はOpenRouterをチェック
        required_key = "OPENROUTER_API_KEY"

    # APIキーをチェック
    api_key = os.getenv(required_key)
    if not api_key:
        pytest.skip(f"{model_name} 用のAPIキー ({required_key}) が設定されていないためスキップします")

    logger.info(f"{model_name} が利用可能であることを確認しました")
    return True


@given("全てのAPIキーが未設定になっている")
def all_api_keys_unset(monkeypatch):
    """全てのAPIキーを未設定にする"""
    from image_annotator_lib.core.registry import initialize_registry
    
    all_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]

    for key in all_keys:
        if os.environ.get(key):
            monkeypatch.delenv(key, raising=False)

    # レジストリを初期化（APIキー未設定でもモデルクラスは利用可能にする）
    initialize_registry()
    
    logger.info("全てのAPIキーを未設定にしました（レジストリは初期化済み）")


@given("モデルが利用不可になっている")
def model_unavailable(monkeypatch):
    """モデルが利用不可になるよう設定"""
    from image_annotator_lib.core.registry import initialize_registry
    
    # 無効なAPIキーを設定してモデル利用不可を引き起こす
    monkeypatch.setenv("OPENAI_API_KEY", "invalid_key_for_testing")
    
    # レジストリを初期化（無効なAPIキーでもモデルクラスは利用可能にする）
    initialize_registry()
    
    logger.info("モデル利用不可をシミュレート: 無効なAPIキーを設定（レジストリは初期化済み）")


@given("OpenAI、Anthropic、GoogleのAPIキーが未設定になっている")
def main_providers_api_keys_unset(monkeypatch):
    """主要プロバイダーのAPIキーを未設定にする"""
    from image_annotator_lib.core.registry import initialize_registry
    
    main_provider_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]

    for key in main_provider_keys:
        if os.environ.get(key):
            monkeypatch.delenv(key, raising=False)

    # レジストリを初期化（主要プロバイダーAPIキー未設定でもモデルクラスは利用可能にする）
    initialize_registry()
    
    logger.info("主要プロバイダー(OpenAI, Anthropic, Google)のAPIキーを未設定にしました（レジストリは初期化済み）")


@given("OpenRouterのAPIキーが設定されている")
def openrouter_api_key_available():
    """OpenRouterのAPIキーが利用可能であることを確認"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter APIキー (OPENROUTER_API_KEY) が設定されていないためスキップします")

    logger.info("OpenRouterのAPIキーが確認されました")
    return True


@given("複数のモデルが利用可能になっている")
def multiple_models_available():
    """複数のモデルが利用可能であることを確認"""
    # 少なくとも1つのAPIキーがあることを確認
    available_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]

    has_key = any(os.getenv(key) for key in available_keys)
    if not has_key:
        pytest.skip("利用可能なAPIキーが1つもないためスキップします")

    logger.info("複数モデルテスト用の環境が確認されました")
    return True


@given("APIがタイムアウトするよう設定されている")
def api_timeout(monkeypatch):
    """PydanticAI APIタイムアウトをシミュレート"""
    from image_annotator_lib.core.registry import initialize_registry
    
    # E2Eテスト環境では実際のAPIリクエストを有効化する
    models.ALLOW_MODEL_REQUESTS = True

    # テスト用の非常に短いタイムアウトを設定
    monkeypatch.setenv("ANTHROPIC_API_TIMEOUT", "0.001")  # 1msの極短タイムアウト
    monkeypatch.setenv("OPENAI_API_TIMEOUT", "0.001")
    monkeypatch.setenv("GOOGLE_API_TIMEOUT", "0.001")

    # レジストリを初期化（タイムアウト設定でもモデルクラスは利用可能にする）
    initialize_registry()

    logger.info("E2Eテスト用の極短タイムアウト(1ms)を設定してタイムアウトエラーをシミュレート（レジストリは初期化済み）")


@given("APIからエラーレスポンスが返されるよう設定されている")
def api_error_response(monkeypatch):
    """PydanticAI APIエラーレスポンスをシミュレート"""
    from image_annotator_lib.core.registry import initialize_registry
    
    # E2Eテスト環境では実際のAPIリクエストを有効化
    models.ALLOW_MODEL_REQUESTS = True

    # 無効な形式のAPIキーを設定してAPIエラーを発生させる
    monkeypatch.setenv("OPENAI_API_KEY", "invalid_test_key_format")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid_test_key_format")
    monkeypatch.setenv("GOOGLE_API_KEY", "invalid_test_key_format")

    # レジストリを初期化（無効なAPIキーでもモデルクラスは利用可能にする）
    initialize_registry()

    logger.info("E2Eテスト用の無効なAPIキーを設定してAPIエラーをシミュレート（レジストリは初期化済み）")


# --- When ---
@when(
    parsers.parse("画像を指定して{model_name}でアノテーションを実行する"),
    target_fixture="annotation_result",
)
def run_annotation_with_model(model_name, single_image):
    """指定されたmodel_nameで実際のアノテーションを実行"""
    from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY, initialize_registry
    
    logger.info(f"E2Eアノテーション実行: モデル={model_name}")
    
    # デバッグ: レジストリ状態を確認
    logger.info(f"レジストリの現在の状態: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    
    # レジストリが空の場合は再初期化
    if not _MODEL_CLASS_OBJ_REGISTRY:
        logger.warning("レジストリが空です。再初期化を実行します。")
        initialize_registry()
        logger.info(f"再初期化後のレジストリ: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    
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


@when("画像を指定してgpt-4o-miniでアノテーションを実行する", target_fixture="annotation_result")
def run_annotation_with_gpt_4o_mini(single_image):
    """gpt-4o-miniで実際のアノテーションを実行"""
    from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY, initialize_registry
    
    model_to_use = "gpt-4o-mini"
    logger.info(f"gpt-4o-mini アノテーション実行")
    
    # デバッグ: レジストリ状態を確認
    logger.info(f"レジストリの現在の状態: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    
    # レジストリが空の場合は再初期化
    if not _MODEL_CLASS_OBJ_REGISTRY:
        logger.warning("レジストリが空です。再初期化を実行します。")
        initialize_registry()
        logger.info(f"再初期化後のレジストリ: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    
    result = annotate(images_list=[single_image], model_name_list=[model_to_use])
    logger.info(f"アノテーション結果: {result}")
    return result


@when("画像を指定してinvalid-modelでアノテーションを実行する", target_fixture="annotation_result")
def run_annotation_with_invalid_model(single_image):
    """invalid-modelで実際のアノテーションを実行"""
    model_to_use = "invalid-model"
    logger.info(f"invalid-model アノテーション実行")
    result = annotate(images_list=[single_image], model_name_list=[model_to_use])
    logger.info(f"アノテーション結果: {result}")
    return result


@when("画像を指定して複数モデルでアノテーションを実行する", target_fixture="annotation_result")
def run_annotation_with_multiple_models(single_image):
    """複数モデルでのアノテーション実行"""
    models_to_test = ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash"]
    logger.info(f"複数モデルアノテーション実行: モデル={models_to_test}")
    result = annotate(images_list=[single_image], model_name_list=models_to_test)
    logger.info(f"複数モデルアノテーション結果: {result}")
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
            if error_message and (
                "api key" in error_message.lower() or "authentication" in error_message.lower()
            ):
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
            if error_message and (
                "invalid" in error_message.lower() or "unauthorized" in error_message.lower()
            ):
                found_model_error = True
                logger.info(f"モデル利用不可エラーを確認: {error_message}")

    assert found_model_error, "モデル利用不可エラーメッセージが見つかりませんでした"


@then("タイムアウトエラーメッセージが返される")
def timeout_error_returned(annotation_result):
    """タイムアウトエラーが返されることを確認"""
    assert annotation_result is not None
    found_timeout_error = False

    for image_hash, model_results_dict in annotation_result.items():
        for model_name, model_result in model_results_dict.items():
            error_message = model_result.get("error")
            if error_message and ("timeout" in error_message.lower() or "タイムアウト" in error_message):
                found_timeout_error = True
                logger.info(f"タイムアウトエラーを確認: {error_message}")

    assert found_timeout_error, "タイムアウトエラーメッセージが見つかりませんでした"


@then("全てのモデルからアノテーション結果が返される")
def all_models_return_results(annotation_result):
    """全てのモデルから結果が返されることを確認"""
    assert annotation_result is not None
    expected_models = ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash"]

    for image_hash, model_results_dict in annotation_result.items():
        for model_name in expected_models:
            assert model_name in model_results_dict, f"モデル '{model_name}' の結果が見つかりません"
            model_result = model_results_dict[model_name]

            # エラーがある場合はスキップ条件をチェック
            error = model_result.get("error")
            if error:
                skip_conditions = ["APIキー", "api key", "authentication", "利用不可", "unavailable"]
                if any(cond in error.lower() for cond in skip_conditions):
                    logger.info(f"モデル '{model_name}' はAPIキー未設定によりスキップ: {error}")
                    continue
                else:
                    pytest.fail(f"モデル '{model_name}' で予期しないエラー: {error}")

            # 成功した場合は内容をチェック
            assert (
                model_result.get("tags") is not None
                or model_result.get("captions") is not None
                or model_result.get("score") is not None
            ), f"モデル '{model_name}' の結果に有効な内容が含まれていません"

            logger.info(f"モデル '{model_name}' から正常な結果を確認")


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
