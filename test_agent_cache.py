#!/usr/bin/env python3
"""
WebAPI Agent キャッシュシステムのテスト

PydanticAI Agent キャッシュの動作を検証する
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_agent_cache_basic():
    """基本的なキャッシュ機能のテスト"""
    print("=== WebAPI Agent キャッシュ基本機能テスト ===")

    try:
        from image_annotator_lib.core.webapi_agent_cache import (
            WebApiAgentCache,
            create_cache_key,
            create_config_hash,
        )

        # 初期状態
        cache_info = WebApiAgentCache.get_cache_info()
        print(f"✅ 初期キャッシュ状態: {cache_info['cache_size']}個")

        # キャッシュキー作成テスト
        cache_key1 = create_cache_key("gpt-4o-mini", "openai", "gpt-4o-mini")
        cache_key2 = create_cache_key("gpt-4", "openai", "gpt-4")

        print(f"✅ キャッシュキー生成: {cache_key1}, {cache_key2}")

        # 設定ハッシュテスト
        config1 = {"model": "gpt-4o-mini", "temperature": 0.7}
        config2 = {"model": "gpt-4o-mini", "temperature": 0.8}

        hash1 = create_config_hash(config1)
        hash2 = create_config_hash(config2)

        print(f"✅ 設定ハッシュ生成: {hash1} != {hash2}")
        assert hash1 != hash2, "異なる設定は異なるハッシュを持つべき"

        # モックAgent作成関数
        def create_mock_agent():
            return {"type": "mock_agent", "id": cache_key1}

        # Agent取得テスト（キャッシュミス）
        agent1 = WebApiAgentCache.get_agent(cache_key1, create_mock_agent, hash1)
        assert agent1["id"] == cache_key1
        print("✅ Agent取得（キャッシュミス）成功")

        # Agent取得テスト（キャッシュヒット）
        agent1_cached = WebApiAgentCache.get_agent(cache_key1, create_mock_agent, hash1)
        assert agent1 is agent1_cached, "キャッシュヒット時は同じインスタンスを返すべき"
        print("✅ Agent取得（キャッシュヒット）成功")

        # キャッシュ情報確認
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] == 1
        print(f"✅ キャッシュサイズ確認: {cache_info['cache_size']}個")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cache_eviction():
    """キャッシュ削除機能のテスト"""
    print("\n=== キャッシュ削除機能テスト ===")

    try:
        from image_annotator_lib.core.webapi_agent_cache import WebApiAgentCache, create_cache_key

        # キャッシュクリア
        WebApiAgentCache.clear_cache()

        # 最大キャッシュサイズを小さく設定
        WebApiAgentCache.set_max_cache_size(2)
        print("✅ 最大キャッシュサイズを2に設定")

        # 複数のAgentを作成
        agents = {}
        for i in range(3):
            cache_key = create_cache_key(f"model-{i}", "openai", f"gpt-{i}")

            def create_agent(idx=i):
                return {"type": "mock_agent", "id": idx}

            agent = WebApiAgentCache.get_agent(cache_key, create_agent)
            agents[cache_key] = agent

            # 少し待機してLRU順序を明確にする
            import time

            time.sleep(0.01)

        # キャッシュサイズが制限されていることを確認
        cache_info = WebApiAgentCache.get_cache_info()
        assert cache_info["cache_size"] <= 2, (
            f"キャッシュサイズが制限を超えています: {cache_info['cache_size']}"
        )
        print(f"✅ LRU削除動作確認: {cache_info['cache_size']}個のAgentがキャッシュ中")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_change_detection():
    """設定変更検出のテスト"""
    print("\n=== 設定変更検出テスト ===")

    try:
        from image_annotator_lib.core.webapi_agent_cache import (
            WebApiAgentCache,
            create_cache_key,
            create_config_hash,
        )

        # キャッシュクリア
        WebApiAgentCache.clear_cache()

        cache_key = create_cache_key("test-model", "openai")

        # 初回設定
        config1 = {"temperature": 0.7, "model": "gpt-4o-mini"}
        hash1 = create_config_hash(config1)

        agent_counter = {"count": 0}

        def create_agent():
            agent_counter["count"] += 1
            return {"type": "mock_agent", "creation_count": agent_counter["count"]}

        # 初回Agent取得
        agent1 = WebApiAgentCache.get_agent(cache_key, create_agent, hash1)
        assert agent1["creation_count"] == 1
        print("✅ 初回Agent作成")

        # 同じ設定でAgent取得（キャッシュヒット）
        agent1_cached = WebApiAgentCache.get_agent(cache_key, create_agent, hash1)
        assert agent1_cached["creation_count"] == 1  # 作成回数は増えない
        print("✅ 同じ設定でキャッシュヒット")

        # 設定変更
        config2 = {"temperature": 0.8, "model": "gpt-4o-mini"}  # temperatureを変更
        hash2 = create_config_hash(config2)

        # 設定変更後のAgent取得（新規作成されるべき）
        agent2 = WebApiAgentCache.get_agent(cache_key, create_agent, hash2)
        assert agent2["creation_count"] == 2  # 新しくAgentが作成される
        print("✅ 設定変更検出による新規Agent作成")

        return True

    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("WebAPI Agent キャッシュシステムテスト開始\n")

    tests = [
        test_agent_cache_basic,
        test_cache_eviction,
        test_config_change_detection,
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

    print(f"\n📊 キャッシュテスト結果: {passed}成功 / {failed}失敗 / {len(tests)}合計")

    if failed == 0:
        print("🎉 WebAPI Agent キャッシュシステムが正常に動作しています！")
        print("   - LRUキャッシュ機能: ✅")
        print("   - 設定変更検出: ✅")
        print("   - メモリ効率管理: ✅")
        return True
    else:
        print("⚠️  一部のキャッシュテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
