#!/usr/bin/env python3
"""
OpenAI PydanticAI統合の簡単なテストスクリプト

既存のWebApiBaseAnnotatorインターフェースとの互換性を検証する
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_openai_annotator_structure():
    """OpenAIApiAnnotatorの基本構造をテストする"""
    print("=== OpenAIApiAnnotator構造テスト ===")

    try:
        # 最小限のインポートテスト
        from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator

        print("✅ OpenAIApiAnnotator インポート成功")

        # インスタンス作成テスト
        annotator = OpenAIApiAnnotator("test-model")
        print("✅ OpenAIApiAnnotator インスタンス作成成功")

        # 必要な属性をチェック
        assert hasattr(annotator, "agent"), "agent属性が存在しません"
        assert annotator.agent is None, "初期状態でagentはNoneであるべきです"
        print("✅ agent属性チェック成功")

        # 必要なメソッドをチェック
        required_methods = [
            "_load_configuration",
            "_create_agent",
            "_preprocess_images",
            "_run_inference",
            "_format_predictions",
            "_generate_tags",
            "_wait_for_rate_limit",
            "_handle_api_error",
        ]

        for method_name in required_methods:
            assert hasattr(annotator, method_name), f"{method_name}メソッドが存在しません"
        print(f"✅ 必要メソッド {len(required_methods)}個 チェック成功")

        # WebApiBaseAnnotatorからの継承チェック
        from image_annotator_lib.core.base import WebApiBaseAnnotator

        assert isinstance(annotator, WebApiBaseAnnotator), "WebApiBaseAnnotatorから継承していません"
        print("✅ WebApiBaseAnnotator継承チェック成功")

        print("\n🎉 OpenAIApiAnnotator構造テスト完了!")
        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pydantic_ai_imports():
    """PydanticAI関連のインポートをテストする"""
    print("\n=== PydanticAI インポートテスト ===")

    try:
        from pydantic_ai import Agent

        print("✅ pydantic_ai.Agent インポート成功")

        from pydantic_ai.models.openai import OpenAIModel

        print("✅ pydantic_ai.models.OpenAIModel インポート成功")

        from pydantic_ai.messages import BinaryContent

        print("✅ pydantic_ai.messages.BinaryContent インポート成功")

        print("\n🎉 PydanticAI インポートテスト完了!")
        return True

    except Exception as e:
        print(f"❌ PydanticAI インポート失敗: {e}")
        return False


def test_annotation_schema_compatibility():
    """AnnotationSchemaとの互換性をテストする"""
    print("\n=== AnnotationSchema互換性テスト ===")

    try:
        from image_annotator_lib.core.types import AnnotationSchema

        print("✅ AnnotationSchema インポート成功")

        # サンプルデータでAnnotationSchemaを作成
        sample_data = {"tags": ["test", "sample", "image"], "captions": ["テスト画像です"], "score": 0.85}

        annotation = AnnotationSchema(**sample_data)
        print("✅ AnnotationSchema インスタンス作成成功")

        # model_dump()メソッドの確認
        dumped = annotation.model_dump()
        assert isinstance(dumped, dict), "model_dump()はdictを返すべきです"
        assert "tags" in dumped, "tagsフィールドが存在しません"
        assert "captions" in dumped, "captionsフィールドが存在しません"
        assert "score" in dumped, "scoreフィールドが存在しません"
        print("✅ AnnotationSchema.model_dump()チェック成功")

        print("\n🎉 AnnotationSchema互換性テスト完了!")
        return True

    except Exception as e:
        print(f"❌ AnnotationSchema互換性テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("OpenAI PydanticAI統合テスト開始\n")

    tests = [
        test_pydantic_ai_imports,
        test_annotation_schema_compatibility,
        test_openai_annotator_structure,
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

    print(f"\n📊 テスト結果: {passed}成功 / {failed}失敗 / {len(tests)}合計")

    if failed == 0:
        print("🎉 すべてのテストが成功しました！OpenAI PydanticAI統合は正常に動作しています。")
        return True
    else:
        print("⚠️  一部のテストが失敗しました。問題を修正してください。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
