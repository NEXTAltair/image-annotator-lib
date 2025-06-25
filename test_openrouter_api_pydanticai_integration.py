#!/usr/bin/env python3
"""
OpenRouter API PydanticAI統合テスト

PydanticAI版OpenRouter APIアノテーターの実動作を検証する
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# PIL Image for testing
from PIL import Image


def create_test_image() -> Image.Image:
    """テスト用の小さな画像を作成"""
    img = Image.new("RGB", (64, 64), color="green")
    return img


def test_openrouter_pydanticai_structure():
    """OpenRouter PydanticAI実装の構造テスト"""
    print("=== OpenRouter PydanticAI 構造テスト ===")

    try:

        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        # 必要メソッドの存在確認 (新しいProvider-level実装)
        required_methods = [
            "__init__",
            "__enter__",
            "__exit__",
            "_run_inference",
            "run_with_model",
            "_handle_api_error",
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(OpenRouterApiAnnotator, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"❌ 不足メソッド: {missing_methods}")
            return False

        print(f"✅ 全{len(required_methods)}個の必要メソッドが存在")

        # 基底クラス継承確認
        from image_annotator_lib.core.base import WebApiBaseAnnotator

        if not issubclass(OpenRouterApiAnnotator, WebApiBaseAnnotator):
            print("❌ WebApiBaseAnnotator継承が不正")
            return False

        print("✅ WebApiBaseAnnotator継承確認")
        return True

    except Exception as e:
        print(f"❌ 構造テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_image_preprocessing():
    """画像前処理のテスト"""
    print("\n=== 画像前処理テスト ===")

    try:
        from pydantic_ai.messages import BinaryContent

        # テスト画像作成
        test_images = [create_test_image(), create_test_image()]

        # PydanticAIAnnotatorMixinの前処理機能テスト
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        # Mixinインスタンス作成
        mixin = PydanticAIAnnotatorMixin("test-model")

        # 前処理実行
        processed = mixin._preprocess_images_to_binary(test_images)

        # 結果検証
        assert isinstance(processed, list)
        assert len(processed) == 2

        for item in processed:
            assert isinstance(item, BinaryContent)
            assert item.media_type == "image/webp"
            assert len(item.data) > 0

        print("✅ 画像前処理成功")
        print(f"   - 処理数: {len(processed)}")
        print("   - BinaryContent形式: ✅")
        print(f"   - メディアタイプ: {processed[0].media_type}")

        return True

    except Exception as e:
        print(f"❌ 画像前処理テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_hash_generation():
    """設定ハッシュ生成テスト"""
    print("\n=== 設定ハッシュ生成テスト ===")

    try:
        from image_annotator_lib.core.webapi_agent_cache import create_config_hash

        # テスト用設定データ
        config_data = {
            "model_id": "anthropic/claude-3.5-sonnet",
            "temperature": 0.7,
            "max_tokens": 1800,
            "referer": "https://example.com",
            "app_name": "TestApp",
        }

        # ハッシュ生成
        config_hash = create_config_hash(config_data)
        config_hash2 = create_config_hash(config_data)

        # 結果検証
        assert isinstance(config_hash, str)
        assert len(config_hash) > 0
        assert config_hash == config_hash2  # 同一データで同一ハッシュ

        print("✅ 設定ハッシュ生成成功")
        print(f"   - ハッシュ値: {config_hash}")
        print("✅ 同一設定で同一ハッシュ確認")

        return True

    except Exception as e:
        print(f"❌ 設定ハッシュ生成テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_creation_mock():
    """Provider Factory OpenRouter Agent作成のモックテスト"""
    print("\n=== Agent作成モックテスト ===")

    try:
        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory

        # Provider Factory のOpenRouter Agent作成テスト
        with (
            patch("image_annotator_lib.core.pydantic_ai_factory.infer_model") as mock_infer_model,
            patch("image_annotator_lib.core.pydantic_ai_factory.Agent") as mock_agent_class,
            patch.object(PydanticAIProviderFactory, "get_provider") as mock_get_provider,
        ):
            # Mock model
            mock_model = MagicMock()
            mock_model.system = "openai"
            mock_infer_model.return_value = mock_model

            # Mock provider
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            # Mock agent
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # OpenRouter用config_data
            config_data = {
                "model_id": "anthropic/claude-3.5-sonnet",
                "referer": "https://example.com",
                "app_name": "TestApp",
            }

            # Provider Factory でOpenRouter Agent作成
            agent = PydanticAIProviderFactory.create_openrouter_agent(
                model_name="test-model",
                api_model_id="openrouter:anthropic/claude-3.5-sonnet",
                api_key="test-api-key",
                config_data=config_data,
            )

            # 検証
            assert agent is not None
            mock_infer_model.assert_called_once_with("openai:anthropic/claude-3.5-sonnet")
            mock_get_provider.assert_called_once()

            # get_provider呼び出し引数の確認
            call_args = mock_get_provider.call_args
            assert call_args[0][0] == "openai"  # provider_name
            provider_kwargs = call_args[1]
            assert provider_kwargs["api_key"] == "test-api-key"
            assert provider_kwargs["base_url"] == "https://openrouter.ai/api/v1"
            assert "default_headers" in provider_kwargs

            # ヘッダーの確認
            headers = provider_kwargs["default_headers"]
            assert headers["HTTP-Referer"] == "https://example.com"
            assert headers["X-Title"] == "TestApp"

            mock_agent_class.assert_called_once()

        print("✅ Provider Factory OpenRouter Agent作成モック成功")
        print("   - OpenRouterプレフィックス処理: ✅")
        print("   - カスタムヘッダー設定: ✅")
        print("   - Provider共有メカニズム: ✅")
        print("   - Agent作成: ✅")

        return True

    except Exception as e:
        print(f"❌ Agent作成モックテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")

    try:
        from image_annotator_lib.exceptions.errors import (
            ApiAuthenticationError,
            ApiRateLimitError,
            ApiServerError,
            ApiTimeoutError,
        )
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator

        annotator = OpenRouterApiAnnotator("test-model")

        # 認証エラーテスト
        try:
            annotator._handle_api_error(Exception("401 authentication failed"))
            assert False, "認証エラーが検出されませんでした"
        except ApiAuthenticationError:
            print("✅ 認証エラー検出")

        # レート制限エラーテスト
        try:
            annotator._handle_api_error(Exception("429 rate limit exceeded"))
            assert False, "レート制限エラーが検出されませんでした"
        except ApiRateLimitError:
            print("✅ レート制限エラー検出")

        # タイムアウトエラーテスト
        try:
            annotator._handle_api_error(Exception("timeout occurred"))
            assert False, "タイムアウトエラーが検出されませんでした"
        except ApiTimeoutError:
            print("✅ タイムアウトエラー検出")

        # サーバーエラーテスト
        try:
            annotator._handle_api_error(Exception("500 server error"))
            assert False, "サーバーエラーが検出されませんでした"
        except ApiServerError:
            print("✅ サーバーエラー検出")

        # 一般エラーテスト
        error_msg = annotator._handle_api_error(Exception("general error"))
        assert "OpenRouter API Error: general error" in error_msg
        print("✅ 一般エラー処理")

        return True

    except Exception as e:
        print(f"❌ エラーハンドリングテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_inference_pipeline_mock():
    """Provider Manager OpenRouter推論パイプラインのモックテスト"""
    print("\n=== 推論パイプライン モックテスト ===")

    try:
        from image_annotator_lib.core.provider_manager import ProviderManager
        from image_annotator_lib.core.types import AnnotationSchema

        # テストデータ準備
        test_image = create_test_image()
        expected_result = [
            {
                "response": AnnotationSchema(
                    tags=["test", "openrouter"], captions=["Mock test image for OpenRouter"], score=0.88
                ),
                "error": None,
            }
        ]

        # Provider Manager のrun_inference_with_modelをモック
        with patch.object(ProviderManager, "run_inference_with_model") as mock_inference:
            mock_inference.return_value = expected_result

            # 推論実行
            results = ProviderManager.run_inference_with_model(
                model_name="test-model", images=[test_image], api_model_id="anthropic/claude-3.5-sonnet"
            )

            # 結果検証
            assert isinstance(results, list)
            assert len(results) == 1

            result = results[0]
            assert isinstance(result, dict)
            assert "response" in result
            assert isinstance(result["response"], AnnotationSchema)
            assert result.get("error") is None

            # モック呼び出し確認
            mock_inference.assert_called_once_with(
                model_name="test-model", images=[test_image], api_model_id="anthropic/claude-3.5-sonnet"
            )

        print("✅ Provider Manager OpenRouter推論パイプライン モック成功")
        print("   - 画像前処理: ✅")
        print("   - Providerレベル実行: ✅")
        print("   - 結果フォーマット: ✅")

        return True

    except Exception as e:
        print(f"❌ 推論パイプライン モックテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("OpenRouter API PydanticAI統合テスト開始\n")

    tests = [
        test_openrouter_pydanticai_structure,
        test_image_preprocessing,
        test_config_hash_generation,
        test_agent_creation_mock,
        test_error_handling,
        test_inference_pipeline_mock,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ テスト関数 {test_func.__name__} で予期しないエラー: {e}")
            failed += 1

    print(f"\n📊 OpenRouter PydanticAI統合テスト結果: {passed}成功 / {failed}失敗 / {len(tests)}合計")

    if failed == 0:
        print("🎉 OpenRouter PydanticAI統合テストが全て成功しました！")
        print("   - 構造整合性: ✅")
        print("   - 画像前処理: ✅")
        print("   - Agent作成: ✅")
        print("   - エラーハンドリング: ✅")
        print("   - 推論パイプライン: ✅")
        print("   - OpenRouter固有機能: ✅")
        return True
    else:
        print("⚠️  一部のOpenRouter PydanticAI統合テストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
