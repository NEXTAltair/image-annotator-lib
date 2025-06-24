#!/usr/bin/env python3
"""
Google API PydanticAI統合テスト

PydanticAI版Google APIアノテーターの実動作を検証する
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# PIL Image for testing
from PIL import Image


def create_test_image() -> Image.Image:
    """テスト用の小さな画像を作成"""
    img = Image.new('RGB', (64, 64), color='red')
    return img


def test_google_pydanticai_structure():
    """Google PydanticAI実装の構造テスト"""
    print("=== Google PydanticAI 構造テスト ===")

    try:
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.core.types import AnnotationSchema
        from pydantic_ai.messages import BinaryContent
        
        # 必要メソッドの存在確認
        required_methods = [
            "__init__", "__enter__", "__exit__",
            "_load_configuration", "_create_agent", "_get_config_hash",
            "_preprocess_images", "_run_inference", 
            "_run_inference_sync", "_run_inference_async",
            "_handle_api_error"
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(GoogleApiAnnotator, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"❌ 不足メソッド: {missing_methods}")
            return False

        print(f"✅ 全{len(required_methods)}個の必要メソッドが存在")
        
        # 基底クラス継承確認
        from image_annotator_lib.core.base import WebApiBaseAnnotator
        if not issubclass(GoogleApiAnnotator, WebApiBaseAnnotator):
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
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from pydantic_ai.messages import BinaryContent
        
        # テスト画像作成
        test_images = [create_test_image(), create_test_image()]
        
        # Annotatorインスタンス作成（設定なし）
        annotator = GoogleApiAnnotator("test-model")
        
        # 前処理実行
        processed = annotator._preprocess_images(test_images)
        
        # 結果検証
        assert isinstance(processed, list)
        assert len(processed) == 2
        
        for item in processed:
            assert isinstance(item, BinaryContent)
            assert item.media_type == "image/webp"
            assert len(item.data) > 0
            
        print("✅ 画像前処理成功")
        print(f"   - 処理数: {len(processed)}")
        print(f"   - BinaryContent形式: ✅")
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
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.core.webapi_agent_cache import create_config_hash
        
        # Annotatorインスタンス作成
        annotator = GoogleApiAnnotator("test-model")
        annotator.api_model_id = "gemini-2.0-flash"
        
        # 設定ハッシュ生成（モックconfig_registry使用）
        with patch("image_annotator_lib.core.config.config_registry") as mock_registry:
            mock_registry.get.side_effect = lambda name, key, default=None: {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 32,
                "max_output_tokens": 1800,
            }.get(key, default)
            
            config_hash = annotator._get_config_hash()
            
        # 結果検証
        assert isinstance(config_hash, str)
        assert len(config_hash) > 0
        
        print("✅ 設定ハッシュ生成成功")
        print(f"   - ハッシュ値: {config_hash}")
        
        # 同じ設定で同じハッシュが生成されることを確認
        with patch("image_annotator_lib.core.config.config_registry") as mock_registry:
            mock_registry.get.side_effect = lambda name, key, default=None: {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 32,
                "max_output_tokens": 1800,
            }.get(key, default)
            
            config_hash2 = annotator._get_config_hash()
        
        assert config_hash == config_hash2
        print("✅ 同一設定で同一ハッシュ確認")
        
        return True

    except Exception as e:
        print(f"❌ 設定ハッシュ生成テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation_mock():
    """Agent作成のモックテスト"""
    print("\n=== Agent作成モックテスト ===")

    try:
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from pydantic_ai import Agent
        from pydantic import SecretStr
        
        # Annotatorインスタンス作成
        annotator = GoogleApiAnnotator("test-model")
        annotator.api_model_id = "gemini-2.0-flash"
        annotator.api_key = SecretStr("test-api-key")
        
        # Agent作成をモック（正しいモジュールパスで）
        with patch("image_annotator_lib.model_class.annotator_webapi.google_api.GoogleProvider") as mock_provider_class, \
             patch("image_annotator_lib.model_class.annotator_webapi.google_api.GoogleModel") as mock_model_class, \
             patch("image_annotator_lib.model_class.annotator_webapi.google_api.Agent") as mock_agent_class:
            
            # モック設定
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Agent作成実行
            agent = annotator._create_agent()
            
            # 呼び出し確認
            mock_provider_class.assert_called_once_with(api_key="test-api-key")
            mock_model_class.assert_called_once_with(model_name="gemini-2.0-flash", provider=mock_provider)
            mock_agent_class.assert_called_once()
            
            assert agent == mock_agent
            
        print("✅ Agent作成モックテスト成功")
        print("   - GoogleProvider作成: ✅")
        print("   - GoogleModel作成: ✅") 
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
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.exceptions.errors import (
            ApiAuthenticationError, ApiRateLimitError, 
            ApiTimeoutError, ApiServerError
        )
        
        annotator = GoogleApiAnnotator("test-model")
        
        # 認証エラーテスト
        try:
            annotator._handle_api_error(Exception("authentication failed"))
            assert False, "認証エラーが検出されませんでした"
        except ApiAuthenticationError:
            print("✅ 認証エラー検出")
        
        # レート制限エラーテスト
        try:
            annotator._handle_api_error(Exception("rate limit exceeded"))
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
        assert "Google API Error: general error" in error_msg
        print("✅ 一般エラー処理")
        
        return True

    except Exception as e:
        print(f"❌ エラーハンドリングテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_pipeline_mock():
    """推論パイプライン全体のモックテスト"""
    print("\n=== 推論パイプライン モックテスト ===")

    try:
        from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
        from image_annotator_lib.core.types import AnnotationSchema, RawOutput
        from pydantic_ai.messages import BinaryContent
        
        # テストデータ準備
        test_image = create_test_image()
        expected_result = AnnotationSchema(
            tags=["test", "mock"],
            captions=["Mock test image"],
            score=0.95
        )
        
        # Annotator作成
        annotator = GoogleApiAnnotator("test-model")
        
        # Agentを設定（推論実行に必要）
        mock_agent = MagicMock()
        annotator.agent = mock_agent
        
        # 前処理実行（実際の処理）
        binary_contents = annotator._preprocess_images([test_image])
        
        # 推論実行をモック
        with patch.object(annotator, '_run_inference_sync') as mock_inference:
            mock_inference.return_value = expected_result
            
            # 推論実行
            results = annotator._run_inference(binary_contents)
            
            # 結果検証
            assert len(results) == 1
            result = results[0]
            
            # RawOutputは TypedDict (辞書形式) なので、キーでアクセス
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'error' in result
            assert result['response'] == expected_result
            assert result['error'] is None
            
            # モック呼び出し確認
            mock_inference.assert_called_once()
            call_args = mock_inference.call_args[0]
            assert isinstance(call_args[0], BinaryContent)
        
        print("✅ 推論パイプライン モックテスト成功")
        print("   - 画像前処理: ✅")
        print("   - Agent初期化: ✅")
        print("   - 推論実行: ✅")
        print("   - 結果フォーマット: ✅")
        
        return True

    except Exception as e:
        print(f"❌ 推論パイプライン モックテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Google API PydanticAI統合テスト開始\n")

    tests = [
        test_google_pydanticai_structure,
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

    print(f"\n📊 Google PydanticAI統合テスト結果: {passed}成功 / {failed}失敗 / {len(tests)}合計")

    if failed == 0:
        print("🎉 Google PydanticAI統合テストが全て成功しました！")
        print("   - 構造整合性: ✅")
        print("   - 画像前処理: ✅")
        print("   - Agent作成: ✅")
        print("   - エラーハンドリング: ✅")
        print("   - 推論パイプライン: ✅")
        return True
    else:
        print("⚠️  一部のGoogle PydanticAI統合テストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)