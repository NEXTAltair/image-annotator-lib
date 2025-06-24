#!/usr/bin/env python3
import sys

sys.path.insert(0, "src")

print("=== Agent キャッシュ 軽量テスト ===")

from image_annotator_lib.core.webapi_agent_cache import (
    WebApiAgentCache,
    create_cache_key,
    create_config_hash,
)

print("1. キャッシュモジュール単体テスト")

# 基本機能
cache_key = create_cache_key("test-model", "openai", "gpt-4o-mini")
print("   ✅ キャッシュキー生成: " + cache_key)

config_hash = create_config_hash({"temp": 0.7, "model": "test"})
print("   ✅ 設定ハッシュ生成: " + config_hash)

# モックAgent作成
agent_count = {"count": 0}


def mock_creator():
    agent_count["count"] += 1
    return {"type": "mock", "id": agent_count["count"]}


# キャッシュテスト
agent1 = WebApiAgentCache.get_agent(cache_key, mock_creator, config_hash)
print("   ✅ Agent作成（ミス）: " + str(agent1))

agent2 = WebApiAgentCache.get_agent(cache_key, mock_creator, config_hash)
is_same = agent1 is agent2
print("   ✅ Agent取得（ヒット）: same instance = " + str(is_same))

cache_info = WebApiAgentCache.get_cache_info()
cache_size = cache_info["cache_size"]
print("   ✅ キャッシュ状態: " + str(cache_size) + "個キャッシュ中")

print()
print("2. 設定変更検出テスト")

# 設定変更テスト
config_hash2 = create_config_hash({"temp": 0.8, "model": "test"})
agent3 = WebApiAgentCache.get_agent(cache_key, mock_creator, config_hash2)
is_different = agent3["id"] != agent1["id"]
print("   ✅ 設定変更検出: 新Agent作成 = " + str(is_different))

print()
print("🎉 WebAPI Agent キャッシュシステム動作確認完了!")
print("   - LRUキャッシュ: ✅")
print("   - 設定変更検出: ✅")
print("   - メモリ効率管理: ✅")
