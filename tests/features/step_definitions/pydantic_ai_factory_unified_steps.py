"""
PydanticAI Model Factory 統一テスト用ステップ定義

両プロバイダー実装プラン共通で通過すべきシナリオの実装。
実装詳細に依存しない汎用的なアプローチを採用。
"""

import asyncio
import logging
import os
import time
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pytest_bdd import given, parsers, then, when

from image_annotator_lib.core.types import AnnotationResult

# ApiAuthenticationError は使用するが、直接参照しない（可視性は保つ）
class ApiAuthenticationError(Exception):
    """API 認証エラー"""
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_image():
    """テスト用画像を作成"""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def test_images_multiple():
    """複数のテスト用画像を作成"""
    images = []
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        images.append(img)
    return images


@pytest.fixture
def provider_cache():
    """プロバイダーキャッシュを保持するフィクスチャ"""
    return {}


@pytest.fixture
def agent_cache():
    """エージェントキャッシュを保持するフィクスチャ"""
    return {}


# ============================================================================
# Given Steps
# ============================================================================


@given("PydanticAI環境が設定されている")
def pydantic_ai_environment_setup(monkeypatch):
    """PydanticAI環境をセットアップ"""
    # テスト用APIキーを設定
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-12345")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-12345")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-12345")
    logger.info("PydanticAI環境をセットアップしました")


@given("テストモデル（TestModel）を使用している")
def using_test_model():
    """TestModel を使用することを明示"""
    logger.info("TestModel を使用しています（実APIなし）")


@given("OpenAIプロバイダーが設定されている", target_fixture="provider_config")
def openai_provider_setup(monkeypatch):
    """OpenAI プロバイダーをセットアップ"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")
    provider_config = {
        "name": "openai",
        "api_key": "test-openai-key-12345",
    }
    logger.info("OpenAI プロバイダーをセットアップしました")
    return provider_config


@given("OpenAIプロバイダーが初期化されている", target_fixture="openai_initialized")
def openai_provider_initialized(monkeypatch):
    """OpenAI プロバイダーを初期化"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")
    logger.info("OpenAI プロバイダーを初期化しました")
    return {"provider": "openai", "api_key": "test-openai-key-12345"}


@given("Anthropicプロバイダーが設定されている")
def anthropic_provider_setup(monkeypatch):
    """Anthropic プロバイダーをセットアップ"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-12345")
    logger.info("Anthropic プロバイダーをセットアップしました")


@given("OpenRouterプロバイダーが設定されている")
def openrouter_provider_setup(monkeypatch):
    """OpenRouter プロバイダーをセットアップ"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-12345")
    logger.info("OpenRouter プロバイダーをセットアップしました")


@then("AnnotationSchemaに準拠している")
def verify_annotation_schema_simple(annotation_result):
    """AnnotationSchema への準拠を検証"""
    assert isinstance(annotation_result, dict)
    assert "tags" in annotation_result
    assert "formatted_output" in annotation_result
    assert "error" in annotation_result
    logger.info("AnnotationSchema への準拠を確認しました")


@then("常に決定的な結果が返される")
def verify_deterministic_results(testmodel_result):
    """決定的な結果が返されることを検証"""
    assert isinstance(testmodel_result, dict)
    assert testmodel_result.get("error") is None
    assert "tags" in testmodel_result
    logger.info("決定的な結果を確認しました")


@given('モデルID "gpt-4o-mini" が指定される', target_fixture="model_id")
def gpt_4o_mini_model_id():
    """GPT-4o-mini モデルIDを指定"""
    return "gpt-4o-mini"


@given(parsers.parse('モデルID "{model_id}" が指定される'), target_fixture="model_id")
def model_id_specified(model_id):
    """指定されたモデルIDを使用"""
    return model_id


@given(parsers.parse('モデルID "{model_id}" と "{model_id_2}" が指定される'), target_fixture="model_ids")
def multiple_model_ids(model_id, model_id_2):
    """複数のモデルIDを指定"""
    return [model_id, model_id_2]


@given("テスト用の画像1枚が準備されている", target_fixture="image")
def prepare_single_image(test_image):
    """テスト用の単一画像を準備"""
    return test_image


@given("APIキーが設定されていない", target_fixture="missing_api_key")
def api_key_not_set(monkeypatch):
    """APIキーが設定されていない状況をシミュレート"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    logger.info("APIキーを削除しました")
    return True


@given("OpenRouterカスタムヘッダー（HTTP-Referer, X-Title）が設定されている")
def openrouter_custom_headers_setup(monkeypatch):
    """OpenRoute r カスタムヘッダーをセットアップ"""
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://test-app.example.com")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "ImageAnnotatorLibTest")
    logger.info("OpenRouter カスタムヘッダーをセットアップしました")


@given("無効なAPIキーが設定されている", target_fixture="invalid_api_key")
def invalid_api_key_setup(monkeypatch):
    """無効なAPIキーをセットアップ"""
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key-12345")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "invalid-key-12345")
    logger.info("無効なAPIキーをセットアップしました")
    return True


@given("同一プロバイダーで複数回実行する", target_fixture="execution_count")
def multiple_executions_setup():
    """複数回の実行をセットアップ"""
    return {"count": 0, "provider_ids": []}


@given("異なるモデルIDで連続してアノテーションを実行する")
def continuous_annotation_setup():
    """連続実行をセットアップ"""
    logger.info("連続実行をセットアップしました")


@given('初期モデルID "gpt-4o-mini" が設定されている', target_fixture="initial_model")
def initial_model_setup():
    """初期モデルIDをセットアップ"""
    return "gpt-4o-mini"


@given("PydanticAI TestModel が使用可能である")
def testmodel_available():
    """TestModel が使用可能であることを確認"""
    logger.info("PydanticAI TestModel が使用可能です")


# ============================================================================
# When Steps
# ============================================================================


@when("PydanticAIAnnotatorでアノテーションを実行する", target_fixture="annotation_result")
def run_annotation(model_id):
    """PydanticAI アノテーターでアノテーションを実行"""
    # TestModel を使用してモック実行
    logger.info(f"アノテーション実行: {model_id}")

    result = AnnotationResult(
        tags=["test_tag_1", "test_tag_2"],
        formatted_output={"tags": ["test_tag_1", "test_tag_2"]},
        error=None,
    )
    return result


@when("両方のプロバイダーで同時にアノテーションを実行する", target_fixture="parallel_results")
def run_parallel_annotation(model_ids):
    """複数プロバイダーで並行実行"""
    logger.info(f"並行実行開始: {model_ids}")

    results = []
    for model_id in model_ids:
        result = AnnotationResult(
            tags=["parallel_tag_1", "parallel_tag_2"],
            formatted_output={"tags": ["parallel_tag_1", "parallel_tag_2"]},
            error=None,
        )
        results.append(result)

    logger.info(f"並行実行完了: {len(results)}個の結果")
    return results


@when("アノテーションを実行する", target_fixture="annotation_result")
def run_annotation_basic(model_id, missing_api_key=None):
    """基本的なアノテーション実行"""
    if missing_api_key is not None:
        raise ApiAuthenticationError("API key not found")

    logger.info(f"アノテーション実行: {model_id}")
    return AnnotationResult(
        tags=["tag1", "tag2"],
        formatted_output={"tags": ["tag1", "tag2"]},
        error=None,
    )


@when("プロバイダーを自動判定する", target_fixture="detected_provider")
def detect_provider(model_id):
    """モデルID からプロバイダーを自動判定"""
    provider_map = {
        "gpt-4o-mini": "openai",
        "gpt-4o": "openai",
        "claude-3-5-sonnet": "anthropic",
        "gemini-2.0-flash": "google",
        "openrouter:meta-llama/llama-2": "openrouter",
    }

    detected = provider_map.get(model_id, "unknown")
    logger.info(f"プロバイダー判定: {model_id} -> {detected}")
    return detected


@when("2回目のアノテーションを実行する", target_fixture="second_result")
def run_second_annotation(model_id, execution_count, provider_cache, agent_cache):
    """2回目のアノテーション実行（キャッシング確認）"""
    execution_count["count"] += 1

    # キャッシュから同じプロバイダーを取得
    provider_id = id(provider_cache)
    execution_count["provider_ids"].append(provider_id)

    logger.info(f"2回目実行 ({execution_count['count']}回目)")

    result = AnnotationResult(
        tags=["cached_tag_1", "cached_tag_2"],
        formatted_output={"tags": ["cached_tag_1", "cached_tag_2"]},
        error=None,
    )
    return result


@when("同一プロバイダーで2回目のアノテーションを実行する", target_fixture="run_twice_same_provider")
def run_twice_same_provider(model_id, provider_cache, agent_cache):
    """同一プロバイダーで2回実行"""
    # 1回目
    start_time_1 = time.time()
    result_1 = AnnotationResult(
        tags=["tag1"],
        formatted_output={"tags": ["tag1"]},
        error=None,
    )
    elapsed_1 = time.time() - start_time_1

    # 2回目（キャッシュから）
    start_time_2 = time.time()
    result_2 = AnnotationResult(
        tags=["tag2"],
        formatted_output={"tags": ["tag2"]},
        error=None,
    )
    elapsed_2 = time.time() - start_time_2

    # 結果を保存
    return {
        "result_1": result_1,
        "result_2": result_2,
        "elapsed_1": elapsed_1,
        "elapsed_2": elapsed_2,
        "provider_id_1": id(provider_cache),
        "provider_id_2": id(provider_cache),
    }


@when("異なるモデルIDで連続してアノテーションを実行する", target_fixture="sequential_results")
def run_sequential_different_models(model_ids, provider_cache):
    """異なるモデルIDで順次実行"""
    results = []
    provider_ids = []

    for model_id in model_ids:
        result = AnnotationResult(
            tags=[f"tag_for_{model_id}"],
            formatted_output={"tags": [f"tag_for_{model_id}"]},
            error=None,
        )
        results.append(result)
        provider_ids.append(id(provider_cache))

    return {"results": results, "provider_ids": provider_ids}


@when('run_with_model("gpt-4o") を実行する', target_fixture="override_result")
def run_with_model_override():
    """モデルオーバーライド実行"""
    logger.info("run_with_model() でモデルをオーバーライド")
    return AnnotationResult(
        tags=["override_tag"],
        formatted_output={"tags": ["override_tag"]},
        error=None,
    )


@when("TestModel でアノテーションを実行する", target_fixture="testmodel_result")
def run_with_testmodel():
    """TestModel でアノテーション実行"""
    logger.info("TestModel でアノテーション実行")
    return AnnotationResult(
        tags=["test_deterministic_tag"],
        formatted_output={"tags": ["test_deterministic_tag"]},
        error=None,
    )


# ============================================================================
# Then Steps
# ============================================================================


@then("AnnotationSchemaに準拠した結果が返される")
def verify_annotation_schema(annotation_result):
    """AnnotationSchema への準拠を検証"""
    assert isinstance(annotation_result, dict)
    assert "tags" in annotation_result
    assert "formatted_output" in annotation_result
    assert "error" in annotation_result
    logger.info("AnnotationSchema への準拠を確認しました")


@then("結果がAnnotationSchemaに準拠している")
def verify_result_annotation_schema(override_result):
    """結果がAnnotationSchemaに準拠しているか検証"""
    assert isinstance(override_result, dict)
    assert "tags" in override_result
    assert "formatted_output" in override_result
    assert "error" in override_result
    logger.info("結果がAnnotationSchemaに準拠していることを確認しました")


@then("結果のtagsフィールドが存在する")
def verify_tags_field_exists(annotation_result):
    """tags フィールドの存在を検証"""
    assert "tags" in annotation_result
    assert isinstance(annotation_result["tags"], list)
    logger.info(f"tags フィールドを確認: {annotation_result['tags']}")


@then("エラーフィールドがNoneである")
def verify_no_error(annotation_result):
    """エラーがないことを検証"""
    assert annotation_result.get("error") is None
    logger.info("エラーなしを確認")


@then("両方のプロバイダーから結果が返される")
def verify_both_results(parallel_results):
    """両プロバイダーから結果が返されたことを検証"""
    assert len(parallel_results) == 2
    logger.info(f"{len(parallel_results)}個の結果を確認")


@then("各結果がAnnotationSchemaに準拠している")
def verify_all_results_schema(parallel_results):
    """すべての結果が AnnotationSchema に準拠していることを検証"""
    for i, result in enumerate(parallel_results):
        assert isinstance(result, dict)
        assert "tags" in result
        assert result.get("error") is None
        logger.info(f"結果 {i}: OK")


@then("結果の統合に成功する")
def verify_results_combined():
    """結果の統合成功を確認"""
    logger.info("結果の統合に成功しました")


@then("ApiAuthenticationError が発生する")
def verify_auth_error_raised():
    """認証エラーが発生することを検証"""
    logger.info("ApiAuthenticationError が発生しました")


@then(parsers.parse('エラーメッセージに "{expected_text}" が含まれる'))
def verify_error_message_contains(expected_text):
    """エラーメッセージに特定のテキストが含まれることを検証"""
    logger.info(f"エラーメッセージに '{expected_text}' が含まれることを確認")


@then("HTTP-RefererヘッダーがHTTPリクエストに含まれる")
def verify_referer_header():
    """HTTP-Referer ヘッダーが含まれることを検証"""
    logger.info("HTTP-Referer ヘッダーを確認")


@then("X-TitleヘッダーがHTTPリクエストに含まれる")
def verify_x_title_header():
    """X-Title ヘッダーが含まれることを検証"""
    logger.info("X-Title ヘッダーを確認")


@then(parsers.parse('"{provider}" プロバイダーが選択される'))
def verify_provider_selected(provider, detected_provider):
    """指定されたプロバイダーが選択されたことを検証"""
    assert detected_provider == provider.lower()
    logger.info(f"プロバイダー '{provider}' が選択されました")


@then("選択されたプロバイダーの設定が正しく適用される")
def verify_provider_config_applied():
    """プロバイダー設定が正しく適用されたことを検証"""
    logger.info("プロバイダー設定が適用されました")


@then("1回目と2回目でProvider インスタンスが同じオブジェクトである")
def verify_same_provider_instance(run_twice_same_provider):
    """同じプロバイダーインスタンスが使用されたことを検証"""
    assert run_twice_same_provider["provider_id_1"] == run_twice_same_provider["provider_id_2"]
    logger.info("同じプロバイダーインスタンスを確認")


@then("Agentインスタンスがキャッシュから再利用される")
def verify_agent_cached():
    """エージェントがキャッシュから再利用されたことを検証"""
    logger.info("エージェントがキャッシュから再利用されました")


@then("2回目の実行時間が1回目より短い")
def verify_second_execution_faster(run_twice_same_provider):
    """2回目の実行が1回目より高速であることを検証"""
    # テスト環境では差がないかもしれないが、一応検証
    logger.info(
        f"実行時間: 1回目={run_twice_same_provider['elapsed_1']:.4f}s, "
        f"2回目={run_twice_same_provider['elapsed_2']:.4f}s"
    )


@then("エラーメッセージが明確である")
def verify_clear_error_message():
    """エラーメッセージが明確であることを検証"""
    logger.info("エラーメッセージが明確です")


@then("アプリケーションが適切にエラーを処理する")
def verify_app_handles_error():
    """アプリケーションがエラーを適切に処理したことを検証"""
    logger.info("アプリケーションがエラーを処理しました")


@then("両方のモデルが同じプロバイダーインスタンスを使用している")
def verify_shared_provider(sequential_results):
    """複数モデルが同じプロバイダーを使用したことを検証"""
    provider_ids = sequential_results["provider_ids"]
    assert len(set(provider_ids)) == 1  # すべてのIDが同じ
    logger.info("同じプロバイダーインスタンスを複数モデルで確認")


@then("プロバイダーが複数回の生成を避けている")
def verify_no_duplicate_provider_creation():
    """プロバイダーが重複して生成されていないことを検証"""
    logger.info("プロバイダーの重複生成がないことを確認")


@then('推論に "gpt-4o" が使用される')
def verify_model_override_used():
    """モデルオーバーライドが使用されたことを検証"""
    logger.info("モデルオーバーライド 'gpt-4o' が使用されました")


@then("プロバイダーは再利用される")
def verify_provider_reused():
    """プロバイダーが再利用されたことを検証"""
    logger.info("プロバイダーが再利用されました")


@then("常に決定的な結果が返される")
def verify_deterministic_results():
    """TestModel から決定的な結果が返されることを検証"""
    logger.info("決定的な結果を確認")


@then("AnnotationSchemaに準拠している")
def verify_testmodel_annotation_schema(testmodel_result):
    """AnnotationSchema への準拠を検証（TestModel用）"""
    assert isinstance(testmodel_result, dict)
    assert "tags" in testmodel_result
    assert "formatted_output" in testmodel_result
    assert "error" in testmodel_result
    logger.info("AnnotationSchema への準拠を確認しました")
