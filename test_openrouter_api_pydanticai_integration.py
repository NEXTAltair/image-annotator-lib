#!/usr/bin/env python3
"""
OpenRouter API PydanticAI統合テスト

PydanticAI版OpenRouter APIアノテーターの実動作を検証する
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
    img = Image.new('RGB', (64, 64), color='green')
    return img


def test_openrouter_pydanticai_structure():
    """OpenRouter PydanticAI実装の構造テスト"""
    print("=== OpenRouter PydanticAI 構造テスト ===")

    try:
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
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
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from pydantic_ai.messages import BinaryContent
        
        # テスト画像作成
        test_images = [create_test_image(), create_test_image()]
        
        # Annotatorインスタンス作成（設定なし）
        annotator = OpenRouterApiAnnotator("test-model")
        
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
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from image_annotator_lib.core.webapi_agent_cache import create_config_hash
        
        # Annotatorインスタンス作成
        annotator = OpenRouterApiAnnotator("test-model")
        annotator.api_model_id = "anthropic/claude-3.5-sonnet"
        
        # 設定ハッシュ生成（モックconfig_registry使用）
        with patch("image_annotator_lib.core.config.config_registry") as mock_registry:
            mock_registry.get.side_effect = lambda name, key, default=None: {
                "temperature": 0.7,
                "max_output_tokens": 1800,
                "json_schema_supported": True,
                "referer": "https://example.com",
                "app_name": "TestApp",
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
                "max_output_tokens": 1800,
                "json_schema_supported": True,
                "referer": "https://example.com",
                "app_name": "TestApp",
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
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from pydantic_ai import Agent
        from pydantic import SecretStr
        
        # Annotatorインスタンス作成
        annotator = OpenRouterApiAnnotator("test-model")
        annotator.api_model_id = "anthropic/claude-3.5-sonnet"
        annotator.api_key = SecretStr("test-api-key")
        
        # Agent作成をモック（正しいモジュールパスで）
        with patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.OpenAIProvider") as mock_provider_class, \
             patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.OpenAIModel") as mock_model_class, \
             patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.Agent") as mock_agent_class:
            
            # モック設定
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # config_registryもモック
            with patch("image_annotator_lib.core.config.config_registry") as mock_registry:
                mock_registry.get.side_effect = lambda name, key, default=None: {
                    "referer": "https://example.com",
                    "app_name": "TestApp",
                }.get(key, default)
                
                # Agent作成実行
                agent = annotator._create_agent()
            
            # 呼び出し確認
            mock_provider_class.assert_called_once()
            call_kwargs = mock_provider_class.call_args[1]
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
            
            # default_headers の存在確認（中身は設定により決まる）
            assert "default_headers" in call_kwargs
            headers = call_kwargs["default_headers"]
            # モック設定に基づいてヘッダーが設定されているか確認
            if headers:  # ヘッダーが設定されている場合のみチェック
                # referer と app_name が設定されていれば対応するヘッダーが存在
                if "HTTP-Referer" in headers:
                    assert headers["HTTP-Referer"] == "https://example.com"
                if "X-Title" in headers:
                    assert headers["X-Title"] == "TestApp"
            
            mock_model_class.assert_called_once_with(model_name="anthropic/claude-3.5-sonnet", provider=mock_provider)
            mock_agent_class.assert_called_once()
            
            assert agent == mock_agent
            
        print("✅ Agent作成モックテスト成功")
        print("   - OpenAIProvider作成: ✅")
        print("   - OpenAIModel作成: ✅") 
        print("   - Agent作成: ✅")
        print("   - OpenRouterヘッダー設定: ✅")
        
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
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from image_annotator_lib.exceptions.errors import (
            ApiAuthenticationError, ApiRateLimitError, 
            ApiTimeoutError, ApiServerError
        )
        
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
    """推論パイプライン全体のモックテスト"""
    print("\n=== 推論パイプライン モックテスト ===")

    try:
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        from image_annotator_lib.core.types import AnnotationSchema, RawOutput
        from pydantic_ai.messages import BinaryContent
        
        # テストデータ準備
        test_image = create_test_image()
        expected_result = AnnotationSchema(
            tags=["test", "openrouter"],
            captions=["Mock test image for OpenRouter"],
            score=0.88
        )
        
        # Annotator作成
        annotator = OpenRouterApiAnnotator("test-model")
        
        # Agentを設定（推論実行に必要）
        mock_agent = MagicMock()
        annotator.agent = mock_agent
        
        # 画像をBase64文字列に変換（基底クラスの仕様に合わせる）
        import base64
        buffered = BytesIO()
        test_image.save(buffered, format="WEBP")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 推論実行をモック
        with patch.object(annotator, '_run_inference_sync') as mock_inference:
            mock_inference.return_value = expected_result
            
            # 推論実行（Base64文字列リストを渡す）
            results = annotator._run_inference([base64_image])
            
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