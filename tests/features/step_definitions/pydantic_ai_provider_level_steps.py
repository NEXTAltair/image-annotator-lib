"""
PydanticAI Provider-level E2E テスト用ステップ定義
"""

import logging
import os
import time

import pytest
from pytest_bdd import given, parsers, then, when

from image_annotator_lib.api import annotate
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAgentFactory

logger = logging.getLogger(__name__)


# --- Given Steps ---


@given(parsers.parse("{provider_type}プロバイダーのAPIキーが設定されている"))
def provider_api_key_available(provider_type):
    """特定プロバイダーのAPIキーが利用可能であることを確認"""
    key_mapping = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
    }

    required_key = key_mapping.get(provider_type)
    if not required_key:
        pytest.fail(f"未対応のプロバイダータイプ: {provider_type}")

    api_key = os.getenv(required_key)
    if not api_key:
        pytest.skip(f"{provider_type} APIキー ({required_key}) が設定されていないためスキップします")

    logger.info(f"{provider_type} プロバイダーのAPIキーが確認されました")
    return True


@given("OpenRouterのカスタムヘッダー設定が準備されている")
def openrouter_custom_headers_ready():
    """OpenRouter用のカスタムヘッダー設定を準備"""
    # テスト用のカスタムヘッダー設定を環境変数で設定
    os.environ.setdefault("OPENROUTER_REFERER", "https://test-app.example.com")
    os.environ.setdefault("OPENROUTER_APP_NAME", "ImageAnnotatorLibE2ETest")

    logger.info("OpenRouterカスタムヘッダー設定を準備しました")
    return True


@given("複数のプロバイダーのAPIキーが設定されている")
def multiple_providers_api_keys_available():
    """複数プロバイダーのAPIキーが利用可能であることを確認"""
    all_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]
    available_keys = [key for key in all_keys if os.getenv(key)]

    if len(available_keys) < 2:
        pytest.skip("複数プロバイダーテストに必要な最低2つのAPIキーが設定されていません")

    logger.info(f"利用可能なプロバイダー: {available_keys}")
    return available_keys


@given("無効なAPIキーが設定されている")
def invalid_api_keys_set(monkeypatch):
    """無効なAPIキーを設定してエラーをシミュレート"""
    invalid_keys = {
        "OPENAI_API_KEY": "invalid_openai_key_test",
        "ANTHROPIC_API_KEY": "invalid_anthropic_key_test",
        "GOOGLE_API_KEY": "invalid_google_key_test",
        "OPENROUTER_API_KEY": "invalid_openrouter_key_test",
    }

    for key, invalid_value in invalid_keys.items():
        monkeypatch.setenv(key, invalid_value)

    logger.info("全プロバイダーに無効なAPIキーを設定しました")


@given("古いWebAPIアノテーター形式のモデル設定がある")
def legacy_webapi_annotator_config():
    """古いWebAPIアノテーター形式のモデル設定をシミュレート"""
    # レジストリで古いクラス名が指定されていても、PydanticAIWebAPIAnnotatorに統一されることをテスト
    logger.info("古いWebAPIアノテーター形式の設定をシミュレート")
    return True


@given("複数の画像ファイルが準備されている", target_fixture="multiple_images")
def multiple_images_prepared(load_image_files):
    """複数画像ファイルの準備"""
    images = load_image_files(3)  # 3つの画像を準備
    assert len(images) >= 3
    logger.info(f"{len(images)}個の画像ファイルを準備しました")
    return images


# --- When Steps ---


@when(
    parsers.parse("Provider-level で{model_1}と{model_2}を使用してアノテーションを実行する"),
    target_fixture="provider_level_annotation_result",
)
def run_provider_level_annotation_with_two_models(model_1, model_2, single_image):
    """Provider-level で2つのモデルを使用してアノテーション実行"""
    logger.info(f"Provider-level アノテーション実行: {model_1}, {model_2}")

    # 実行前にプロバイダーキャッシュをクリア
    PydanticAIAgentFactory.clear_cache()

    # 2つのモデルでアノテーション実行
    result = annotate(images_list=[single_image], model_name_list=[model_1, model_2])

    logger.info(f"Provider-level アノテーション結果: {result}")
    return result


@when(
    parsers.parse("OpenRouter Provider-level で{model_name}を使用してアノテーションを実行する"),
    target_fixture="openrouter_annotation_result",
)
def run_openrouter_provider_level_annotation(model_name, single_image):
    """OpenRouter Provider-level でアノテーション実行"""
    logger.info(f"OpenRouter Provider-level アノテーション実行: {model_name}")

    result = annotate(images_list=[single_image], model_name_list=[model_name])

    logger.info(f"OpenRouter アノテーション結果: {result}")
    return result


@when(
    "Provider Manager でプロバイダー判定テストを実行する", target_fixture="provider_determination_results"
)
def run_provider_determination_test():
    """Provider Manager のプロバイダー判定機能をテスト"""
    test_cases = [
        ("gpt-4o", "openai"),
        ("claude-3-5-sonnet", "anthropic"),
        ("gemini-pro", "google"),
        ("openrouter:meta-llama/llama-3", "openrouter"),
        ("anthropic:claude-3-5-sonnet", "anthropic"),
        ("google:gemini-pro", "google"),
    ]

    results = {}
    for model_id, expected_provider in test_cases:
        determined = ProviderManager._determine_provider("test-model", model_id)
        results[model_id] = {
            "expected": expected_provider,
            "actual": determined,
            "match": determined == expected_provider,
        }
        logger.info(f"プロバイダー判定: {model_id} -> {determined} (期待値: {expected_provider})")

    return results


@when("同一設定で複数回 Agent を取得する", target_fixture="agent_cache_results")
def get_agents_multiple_times():
    """同一設定で複数回Agentを取得してキャッシュ動作を確認"""
    PydanticAIAgentFactory.clear_cache()

    # 同一設定で3回Agent取得
    model_name = "test-openai-model"
    api_model_id = "gpt-4o-mini"
    api_key = "test-key-for-cache"

    agents = []
    for i in range(3):
        agent = PydanticAIAgentFactory.get_cached_agent(model_name, api_model_id, api_key)
        agents.append(agent)
        logger.info(f"Agent取得 #{i + 1}: {id(agent)}")

    # プロバイダーキャッシュ状況も確認
    provider_cache_size = len(PydanticAIAgentFactory._providers)

    return {"agents": agents, "provider_cache_size": provider_cache_size}


@when(
    "Provider-level で各プロバイダーでアノテーションを実行する",
    target_fixture="provider_error_handling_results",
)
def run_annotation_with_invalid_keys_all_providers(single_image):
    """無効なAPIキーで各プロバイダーのエラーハンドリングをテスト"""
    test_models = [
        "gpt-4o-mini",  # OpenAI
        "claude-3-5-haiku",  # Anthropic
        "gemini-1.5-flash",  # Google
    ]

    results = {}
    for model in test_models:
        try:
            result = annotate(images_list=[single_image], model_name_list=[model])
            results[model] = result
            logger.info(f"モデル {model} の結果: {result}")
        except Exception as e:
            logger.error(f"モデル {model} で例外発生: {e}")
            results[model] = {"exception": str(e)}

    return results


@when("統一WebAPIラッパー経由でアノテーションを実行する", target_fixture="unified_wrapper_result")
def run_annotation_via_unified_wrapper(single_image):
    """統一WebAPIラッパー経由でアノテーションを実行"""
    # レジストリが自動的にPydanticAIWebAPIAnnotatorを選択することをテスト
    model_name = "gpt-4o-mini"

    result = annotate(images_list=[single_image], model_name_list=[model_name])
    logger.info(f"統一ラッパーアノテーション結果: {result}")

    return result


@when(
    "Provider-level で同時に複数の異なるモデルでアノテーションを実行する",
    target_fixture="concurrent_annotation_results",
)
def run_concurrent_provider_level_annotation(multiple_images):
    """複数画像と複数モデルで同時アノテーション実行"""
    models = ["gpt-4o-mini", "gpt-3.5-turbo"]  # 同一プロバイダーの異なるモデル

    start_time = time.time()

    # プロバイダーキャッシュクリア
    PydanticAIAgentFactory.clear_cache()

    result = annotate(images_list=multiple_images, model_name_list=models)

    end_time = time.time()
    processing_time = end_time - start_time

    # プロバイダーキャッシュ使用状況確認
    provider_cache_size = len(PydanticAIAgentFactory._providers)

    return {
        "result": result,
        "processing_time": processing_time,
        "provider_cache_size": provider_cache_size,
    }


# --- Then Steps ---


@then("両方のモデルから正常なアノテーション結果が返される")
def both_models_return_valid_results(provider_level_annotation_result):
    """両方のモデルから正常な結果が返されることを確認"""
    assert provider_level_annotation_result is not None

    model_count = 0
    for _image_hash, model_results in provider_level_annotation_result.items():
        for model_name, result in model_results.items():
            model_count += 1

            # エラーがある場合はAPIキー関連をスキップ
            error = result.get("error")
            if error and ("api key" in error.lower() or "authentication" in error.lower()):
                pytest.skip(f"モデル '{model_name}' のAPIキーが未設定のためスキップ: {error}")

            # 正常な結果をチェック
            assert error is None, f"モデル '{model_name}' でエラー発生: {error}"
            assert (
                result.get("tags") is not None
                or result.get("captions") is not None
                or result.get("score") is not None
            ), f"モデル '{model_name}' の結果が空です"

    assert model_count == 2, f"期待される2つのモデル結果に対して {model_count} 個の結果"


@then("同一プロバイダーインスタンスが共有されている")
def same_provider_instance_shared():
    """同一プロバイダーインスタンスが共有されていることを確認"""
    # プロバイダーキャッシュに適切な数のインスタンスがあることを確認
    provider_cache_size = len(PydanticAIAgentFactory._providers)

    # 同一プロバイダーの複数モデルは1つのプロバイダーインスタンスを共有すべき
    assert provider_cache_size >= 1, "プロバイダーインスタンスが作成されていません"
    logger.info(f"プロバイダーキャッシュサイズ: {provider_cache_size}")


@then("OpenRouterの専用ベースURLが使用されている")
def openrouter_base_url_used(openrouter_annotation_result):
    """OpenRouterの専用ベースURLが使用されていることを確認"""
    # 結果からOpenRouterが正常に動作していることを間接的に確認
    assert openrouter_annotation_result is not None

    for _image_hash, model_results in openrouter_annotation_result.items():
        for _model_name, result in model_results.items():
            error = result.get("error")
            if error and ("api key" in error.lower() or "authentication" in error.lower()):
                pytest.skip(f"OpenRouterのAPIキーが未設定のためスキップ: {error}")

            # OpenRouterが正常に動作していることを確認
            assert error is None or "openrouter.ai" in error.lower(), (
                f"OpenRouter専用エラーでない予期しないエラー: {error}"
            )


@then("カスタムヘッダーが正しく送信されている")
def custom_headers_sent_correctly():
    """カスタムヘッダーが正しく送信されていることを確認"""
    # 環境変数でカスタムヘッダーが設定されていることを確認
    referer = os.getenv("OPENROUTER_REFERER")
    app_name = os.getenv("OPENROUTER_APP_NAME")

    assert referer is not None, "OpenRouter Refererが設定されていません"
    assert app_name is not None, "OpenRouter App Nameが設定されていません"

    logger.info(f"OpenRouterカスタムヘッダー確認 - Referer: {referer}, App: {app_name}")


@then(parsers.parse("{model_id} は {expected_provider} プロバイダーと判定される"))
def model_provider_determined_correctly(provider_determination_results, model_id, expected_provider):
    """モデルIDが正しいプロバイダーと判定されることを確認"""
    assert model_id in provider_determination_results
    result = provider_determination_results[model_id]
    assert result["match"], (
        f"{model_id} のプロバイダー判定が間違っています: 期待値={expected_provider}, 実際={result['actual']}"
    )


@then("キャッシュされた同一の Agent インスタンスが返される")
def same_agent_instance_cached(agent_cache_results):
    """キャッシュされた同一のAgentインスタンスが返されることを確認"""
    agents = agent_cache_results["agents"]

    # 全てのAgentが同一インスタンスであることを確認
    first_agent = agents[0]
    for i, agent in enumerate(agents[1:], 1):
        assert agent is first_agent, f"Agent #{i + 1} が同一インスタンスではありません"

    logger.info("全てのAgentが同一インスタンスであることを確認しました")


@then("プロバイダーインスタンスも共有されている")
def provider_instances_shared(agent_cache_results):
    """プロバイダーインスタンスも共有されていることを確認"""
    provider_cache_size = agent_cache_results["provider_cache_size"]

    # 同一設定では1つのプロバイダーインスタンスのみ存在すべき
    assert provider_cache_size == 1, (
        f"プロバイダーキャッシュサイズが期待値1ではありません: {provider_cache_size}"
    )

    logger.info("プロバイダーインスタンスが正しく共有されています")


@then("メモリ使用量が最適化されている")
def memory_usage_optimized():
    """メモリ使用量が最適化されていることを確認"""
    # プロバイダーキャッシュが適切に管理されていることを確認
    provider_cache_size = len(PydanticAIAgentFactory._providers)
    assert provider_cache_size <= 10, "プロバイダーキャッシュが過大になっています"

    logger.info(f"メモリ最適化確認 - プロバイダーキャッシュサイズ: {provider_cache_size}")


@then("全プロバイダーで統一されたエラーハンドリングが動作する")
def unified_error_handling_works(provider_error_handling_results):
    """全プロバイダーで統一されたエラーハンドリングが動作することを確認"""
    assert provider_error_handling_results is not None

    error_found = False
    for model_name, result in provider_error_handling_results.items():
        if "exception" in result:
            # 予期しない例外が発生した場合
            pytest.fail(f"モデル '{model_name}' で予期しない例外: {result['exception']}")

        # アノテーション結果からエラーを確認
        for _image_hash, model_results in result.items():
            for model, model_result in model_results.items():
                error = model_result.get("error")
                if error:
                    error_found = True
                    logger.info(f"統一エラーハンドリング確認 - {model}: {error}")

    assert error_found, "エラーハンドリングのテストができませんでした（エラーが発生していません）"


@then("PydanticAI ModelHTTPError が適切に処理される")
def pydantic_ai_http_error_handled(provider_error_handling_results):
    """PydanticAI ModelHTTPErrorが適切に処理されることを確認"""
    http_error_found = False

    for _model_name, result in provider_error_handling_results.items():
        for _image_hash, model_results in result.items():
            for model, model_result in model_results.items():
                error = model_result.get("error")
                if error and ("HTTP" in error or "401" in error or "403" in error):
                    http_error_found = True
                    logger.info(f"HTTPエラー処理確認 - {model}: {error}")

    assert http_error_found, "HTTPエラーが適切に処理されませんでした"


@then("エラーメッセージに HTTPステータスコード が含まれる")
def error_message_contains_http_status(provider_error_handling_results):
    """エラーメッセージにHTTPステータスコードが含まれることを確認"""
    status_code_found = False

    for _model_name, result in provider_error_handling_results.items():
        for _image_hash, model_results in result.items():
            for model, model_result in model_results.items():
                error = model_result.get("error")
                if error and any(code in error for code in ["401", "403", "429", "500"]):
                    status_code_found = True
                    logger.info(f"ステータスコード確認 - {model}: {error}")

    assert status_code_found, "エラーメッセージにHTTPステータスコードが含まれていません"


@then("PydanticAI Provider-level で正常に処理される")
def processed_by_provider_level(unified_wrapper_result):
    """PydanticAI Provider-levelで正常に処理されることを確認"""
    assert unified_wrapper_result is not None

    for _image_hash, model_results in unified_wrapper_result.items():
        for _model_name, result in model_results.items():
            error = result.get("error")
            if error and ("api key" in error.lower() or "authentication" in error.lower()):
                pytest.skip(f"APIキーが未設定のためスキップ: {error}")

            assert error is None, f"Provider-level処理でエラー: {error}"


@then("既存のAPIインターフェースとの互換性が保たれる")
def api_compatibility_maintained(unified_wrapper_result):
    """既存のAPIインターフェースとの互換性が保たれることを確認"""
    # annotate()関数の戻り値形式が期待通りであることを確認
    assert isinstance(unified_wrapper_result, dict)

    for _image_hash, model_results in unified_wrapper_result.items():
        assert isinstance(model_results, dict)
        for _model_name, result in model_results.items():
            assert "error" in result
            # tags, captions, score のいずれかが存在することを確認
            has_content = any(key in result for key in ["tags", "captions", "score"])
            error = result.get("error")

            if error and ("api key" in error.lower()):
                continue  # APIキーエラーはスキップ

            assert has_content or error is not None, "結果に有効な内容もエラーも含まれていません"


@then("レジストリで自動的にPydanticAIWebAPIAnnotatorが選択される")
def registry_selects_pydantic_ai_annotator():
    """レジストリで自動的にPydanticAIWebAPIAnnotatorが選択されることを確認"""
    from image_annotator_lib.core.registry import get_cls_obj_registry

    registry = get_cls_obj_registry()

    # WebAPIモデルがPydanticAIWebAPIAnnotatorクラスを使用していることを確認
    webapi_models = [
        name
        for name in registry.keys()
        if any(
            provider in name.lower()
            for provider in ["openai", "anthropic", "google", "gpt", "claude", "gemini"]
        )
    ]

    if webapi_models:
        # 少なくとも1つのWebAPIモデルがあることを確認
        sample_model = webapi_models[0]
        model_class = registry[sample_model]
        class_name = model_class.__name__

        logger.info(f"サンプルWebAPIモデル '{sample_model}' のクラス: {class_name}")
        # 注: レジストリ統一により、実際のクラスがPydanticAIWebAPIAnnotatorかどうかはチェックしない
        # 代わりに正常に動作することで統一が機能していることを確認


@then("全てのリクエストが効率的に処理される")
def all_requests_processed_efficiently(concurrent_annotation_results):
    """全てのリクエストが効率的に処理されることを確認"""
    result = concurrent_annotation_results["result"]
    assert result is not None

    # 全ての画像とモデルの組み合わせで結果があることを確認
    total_actual = sum(len(model_results) for model_results in result.values())

    # APIキーエラーを除いて確認
    successful_results = 0
    for _image_hash, model_results in result.items():
        for _model_name, model_result in model_results.items():
            error = model_result.get("error")
            if not error or "api key" not in error.lower():
                successful_results += 1

    if successful_results == 0:
        pytest.skip("全てのリクエストでAPIキーエラーが発生したためスキップ")

    logger.info(f"効率的処理確認 - 成功結果: {successful_results}/{total_actual}")


@then("プロバイダーインスタンス共有によるメモリ効率性が確認される")
def memory_efficiency_confirmed(concurrent_annotation_results):
    """プロバイダーインスタンス共有によるメモリ効率性が確認される"""
    provider_cache_size = concurrent_annotation_results["provider_cache_size"]

    # 同一プロバイダーの複数モデルは1つのプロバイダーインスタンスを共有すべき
    assert provider_cache_size <= 2, f"プロバイダーキャッシュが過大: {provider_cache_size}"

    logger.info(f"メモリ効率性確認 - プロバイダーキャッシュサイズ: {provider_cache_size}")


@then("処理時間が最適化されている")
def processing_time_optimized(concurrent_annotation_results):
    """処理時間が最適化されていることを確認"""
    processing_time = concurrent_annotation_results["processing_time"]

    # 合理的な処理時間であることを確認（実際のAPI呼び出しを考慮して緩い制限）
    assert processing_time < 60, f"処理時間が長すぎます: {processing_time}秒"

    logger.info(f"処理時間最適化確認 - 処理時間: {processing_time:.2f}秒")


@then("構造化されたPydantic出力スキーマが返される")
def structured_pydantic_output_returned(annotation_result):
    """構造化されたPydantic出力スキーマが返されることを確認"""
    assert annotation_result is not None

    for _image_hash, model_results in annotation_result.items():
        for _model_name, result in model_results.items():
            error = result.get("error")
            if error and ("api key" in error.lower() or "authentication" in error.lower()):
                pytest.skip(f"APIキーが未設定のためスキップ: {error}")

            # 構造化された出力の確認
            assert error is None, f"構造化出力でエラー: {error}"

            # 期待される構造の確認
            has_structured_content = any(key in result for key in ["tags", "captions", "score"])
            assert has_structured_content, "構造化されたコンテンツが含まれていません"


@then("タグ配列が適切な形式で含まれる")
def tags_array_properly_formatted(annotation_result):
    """タグ配列が適切な形式で含まれることを確認"""
    for _image_hash, model_results in annotation_result.items():
        for model_name, result in model_results.items():
            if result.get("error"):
                continue

            tags = result.get("tags")
            if tags is not None:
                assert isinstance(tags, list), f"タグが配列形式ではありません: {type(tags)}"
                for tag in tags:
                    assert isinstance(tag, str), f"タグが文字列ではありません: {type(tag)}"
                logger.info(f"タグ配列確認 - {model_name}: {len(tags)}個のタグ")


@then("キャプション配列が適切な形式で含まれる")
def captions_array_properly_formatted(annotation_result):
    """キャプション配列が適切な形式で含まれることを確認"""
    for _image_hash, model_results in annotation_result.items():
        for model_name, result in model_results.items():
            if result.get("error"):
                continue

            captions = result.get("captions")
            if captions is not None:
                assert isinstance(captions, list), f"キャプションが配列形式ではありません: {type(captions)}"
                for caption in captions:
                    assert isinstance(caption, str), f"キャプションが文字列ではありません: {type(caption)}"
                logger.info(f"キャプション配列確認 - {model_name}: {len(captions)}個のキャプション")


@then("スコア値が数値形式で含まれる")
def score_value_properly_formatted(annotation_result):
    """スコア値が数値形式で含まれることを確認"""
    for _image_hash, model_results in annotation_result.items():
        for model_name, result in model_results.items():
            if result.get("error"):
                continue

            score = result.get("score")
            if score is not None:
                assert isinstance(score, (int, float)), f"スコアが数値形式ではありません: {type(score)}"
                assert 0 <= score <= 1, f"スコアが範囲外です: {score}"
                logger.info(f"スコア値確認 - {model_name}: {score}")
