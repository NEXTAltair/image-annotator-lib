#!/usr/bin/env python3
"""
統一Provider-level PydanticAI統合テスト

全WebAPIプロバイダー（OpenAI, Anthropic, Google, OpenRouter）の
Provider-level実装の統一性とデザインパターンを検証する
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# PIL Image for testing
from PIL import Image


def create_test_image() -> Image.Image:
    """テスト用の小さな画像を作成"""
    img = Image.new("RGB", (64, 64), color="red")
    return img


def test_all_providers_structure():
    """全プロバイダーのProvider-level構造統一性テスト"""
    print("=== 全プロバイダー構造統一性テスト ===")

    providers = [
        (
            "OpenAI",
            "image_annotator_lib.model_class.annotator_webapi.openai_api_response",
            "OpenAIApiAnnotator",
        ),
        (
            "Anthropic",
            "image_annotator_lib.model_class.annotator_webapi.anthropic_api",
            "AnthropicApiAnnotator",
        ),
        ("Google", "image_annotator_lib.model_class.annotator_webapi.google_api", "GoogleApiAnnotator"),
        (
            "OpenRouter",
            "image_annotator_lib.model_class.annotator_webapi.openai_api_chat",
            "OpenRouterApiAnnotator",
        ),
    ]

    # 共通の必要メソッド
    required_methods = ["__init__", "__enter__", "__exit__", "run_with_model", "_handle_api_error"]

    success_count = 0

    for provider_name, module_path, class_name in providers:
        try:
            # モジュールとクラスをインポート
            module = __import__(module_path, fromlist=[class_name])
            annotator_class = getattr(module, class_name)

            # 必要メソッドの存在確認
            missing_methods = []
            for method in required_methods:
                if not hasattr(annotator_class, method):
                    missing_methods.append(method)

            if missing_methods:
                print(f"❌ {provider_name}: 不足メソッド {missing_methods}")
                continue

            # PydanticAIAnnotatorMixin継承確認
            from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

            if not issubclass(annotator_class, PydanticAIAnnotatorMixin):
                print(f"❌ {provider_name}: PydanticAIAnnotatorMixin継承不正")
                continue

            print(f"✅ {provider_name}: 構造統一性確認")
            success_count += 1

        except Exception as e:
            print(f"❌ {provider_name}: 構造テスト失敗 - {e}")

    print(f"構造統一性: {success_count}/{len(providers)} プロバイダー成功")
    return success_count == len(providers)


def test_provider_manager_integration():
    """ProviderManager統合テスト"""
    print("\n=== ProviderManager統合テスト ===")

    try:
        from image_annotator_lib.core.provider_manager import ProviderManager

        # プロバイダー判定テストケース
        test_cases = [
            ("gpt-4o", "openai"),
            ("claude-3-5-sonnet", "anthropic"),
            ("gemini-pro", "google"),
            ("anthropic:claude-3-5-sonnet", "anthropic"),
            ("openai:gpt-4o", "openai"),
            ("google:gemini-pro", "google"),
            ("openrouter:meta-llama/llama-3", "openrouter"),
        ]

        for api_model_id, expected_provider in test_cases:
            determined_provider = ProviderManager._determine_provider("test-model", api_model_id)
            if determined_provider != expected_provider:
                print(
                    f"❌ プロバイダー判定失敗: {api_model_id} -> 期待:{expected_provider}, 実際:{determined_provider}"
                )
                return False

        # ProviderManager必要メソッド確認
        required_manager_methods = [
            "get_provider_instance",
            "run_inference_with_model",
            "_determine_provider",
        ]

        missing_manager_methods = []
        for method in required_manager_methods:
            if not hasattr(ProviderManager, method):
                missing_manager_methods.append(method)

        if missing_manager_methods:
            print(f"❌ ProviderManager不足メソッド: {missing_manager_methods}")
            return False

        print("✅ ProviderManager統合確認")
        print(f"   - プロバイダー判定テスト: {len(test_cases)}個成功")
        return True

    except Exception as e:
        print(f"❌ ProviderManager統合テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_provider_factory_integration():
    """PydanticAIProviderFactory統合テスト"""
    print("\n=== PydanticAIProviderFactory統合テスト ===")

    try:
        from image_annotator_lib.core.pydantic_ai_factory import (
            PydanticAIAnnotatorMixin,
            PydanticAIProviderFactory,
        )

        # Factory必要メソッド確認
        required_factory_methods = [
            "get_provider",
            "create_agent",
            "get_cached_agent",
            "create_openrouter_agent",
            "_extract_provider_name",
        ]

        missing_factory_methods = []
        for method in required_factory_methods:
            if not hasattr(PydanticAIProviderFactory, method):
                missing_factory_methods.append(method)

        if missing_factory_methods:
            print(f"❌ Factory不足メソッド: {missing_factory_methods}")
            return False

        # Mixin必要メソッド確認
        required_mixin_methods = [
            "__init__",
            "_preprocess_images_to_binary",
            "_run_inference_with_model",
            "_setup_agent",
            "_get_provider_name",
        ]

        missing_mixin_methods = []
        for method in required_mixin_methods:
            if not hasattr(PydanticAIAnnotatorMixin, method):
                missing_mixin_methods.append(method)

        if missing_mixin_methods:
            print(f"❌ Mixin不足メソッド: {missing_mixin_methods}")
            return False

        print("✅ PydanticAIProviderFactory統合確認")
        return True

    except Exception as e:
        print(f"❌ PydanticAIProviderFactory統合テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_unified_image_preprocessing():
    """統一画像前処理テスト"""
    print("\n=== 統一画像前処理テスト ===")

    try:
        from pydantic_ai.messages import BinaryContent

        from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAnnotatorMixin

        # テスト画像作成
        test_images = [create_test_image(), create_test_image()]

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

        print("✅ 統一画像前処理成功")
        print(f"   - 処理数: {len(processed)}")
        print("   - BinaryContent形式: ✅")
        print(f"   - メディアタイプ: {processed[0].media_type}")

        return True

    except Exception as e:
        print(f"❌ 統一画像前処理テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api_wrapper_backward_compatibility():
    """API Wrapper後方互換性テスト"""
    print("\n=== API Wrapper後方互換性テスト ===")

    try:
        from image_annotator_lib.api import PydanticAIWebAPIWrapper, _is_pydantic_ai_webapi_annotator
        from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

        # 全プロバイダーのPydanticAI判定テスト
        pydantic_annotators = [
            ("Anthropic", AnthropicApiAnnotator),
            ("Google", GoogleApiAnnotator),
            ("OpenAI", OpenAIApiAnnotator),
        ]

        for provider_name, annotator_class in pydantic_annotators:
            if not _is_pydantic_ai_webapi_annotator(annotator_class):
                print(f"❌ {provider_name}: PydanticAI WebAPIアノテーター判定失敗")
                return False

        # ラッパーインスタンス作成テスト
        wrapper = PydanticAIWebAPIWrapper("test-model", AnthropicApiAnnotator)

        # 必要メソッドの確認
        required_wrapper_methods = ["__init__", "__enter__", "__exit__", "predict"]

        missing_wrapper_methods = []
        for method in required_wrapper_methods:
            if not hasattr(wrapper, method):
                missing_wrapper_methods.append(method)

        if missing_wrapper_methods:
            print(f"❌ Wrapper不足メソッド: {missing_wrapper_methods}")
            return False

        print("✅ API Wrapper後方互換性確認")
        print(f"   - PydanticAI判定: {len(pydantic_annotators)}プロバイダー成功")
        return True

    except Exception as e:
        print(f"❌ API Wrapper後方互換性テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_unified_error_handling():
    """統一エラーハンドリングテスト"""
    print("\n=== 統一エラーハンドリングテスト ===")

    try:
        from image_annotator_lib.exceptions.errors import (
            ApiAuthenticationError,
            ApiRateLimitError,
            ApiServerError,
            ApiTimeoutError,
        )

        # 全プロバイダーのエラーハンドリングテスト
        providers_with_modules = [
            (
                "OpenAI",
                "image_annotator_lib.model_class.annotator_webapi.openai_api_response",
                "OpenAIApiAnnotator",
            ),
            (
                "Anthropic",
                "image_annotator_lib.model_class.annotator_webapi.anthropic_api",
                "AnthropicApiAnnotator",
            ),
            ("Google", "image_annotator_lib.model_class.annotator_webapi.google_api", "GoogleApiAnnotator"),
        ]

        error_test_cases = [
            ("authentication failed", ApiAuthenticationError),
            ("rate limit exceeded", ApiRateLimitError),
            ("timeout occurred", ApiTimeoutError),
            ("server error", ApiServerError),
        ]

        success_count = 0

        for provider_name, module_path, class_name in providers_with_modules:
            try:
                # モジュールとクラスをインポート
                module = __import__(module_path, fromlist=[class_name])
                annotator_class = getattr(module, class_name)

                annotator = annotator_class("test-model")

                # 各エラータイプのテスト
                for error_message, expected_exception in error_test_cases:
                    try:
                        annotator._handle_api_error(Exception(error_message))
                        print(f"❌ {provider_name}: {expected_exception.__name__}が検出されませんでした")
                        break
                    except expected_exception:
                        # 期待されたエラーが発生 = 正常
                        pass
                    except Exception as e:
                        print(f"❌ {provider_name}: 予期しないエラー {type(e).__name__}: {e}")
                        break
                else:
                    # 全エラーテストが成功
                    print(f"✅ {provider_name}: エラーハンドリング統一性確認")
                    success_count += 1

            except Exception as e:
                print(f"❌ {provider_name}: エラーハンドリングテスト失敗 - {e}")

        print(f"エラーハンドリング統一性: {success_count}/{len(providers_with_modules)} プロバイダー成功")
        return success_count == len(providers_with_modules)

    except Exception as e:
        print(f"❌ 統一エラーハンドリングテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("統一Provider-level PydanticAI統合テスト開始\n")

    tests = [
        ("全プロバイダー構造統一性", test_all_providers_structure),
        ("ProviderManager統合", test_provider_manager_integration),
        ("PydanticAIProviderFactory統合", test_provider_factory_integration),
        ("統一画像前処理", test_unified_image_preprocessing),
        ("API Wrapper後方互換性", test_api_wrapper_backward_compatibility),
        ("統一エラーハンドリング", test_unified_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️  {test_name}テストが失敗しました。")
        except Exception as e:
            print(f"❌ {test_name}テスト実行エラー: {e}")

    print(
        f"\n📊 統一Provider-level PydanticAI統合テスト結果: {passed}成功 / {total - passed}失敗 / {total}合計"
    )

    if passed == total:
        print("🎉 全ての統一Provider-level PydanticAI統合テストが成功しました！")
        print("   - 構造統一性: ✅")
        print("   - Provider Manager: ✅")
        print("   - Provider Factory: ✅")
        print("   - 画像前処理統一: ✅")
        print("   - 後方互換性: ✅")
        print("   - エラーハンドリング統一: ✅")
        return True
    else:
        print("⚠️  一部の統一Provider-level PydanticAI統合テストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
