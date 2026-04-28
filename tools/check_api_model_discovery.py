"""LiteLLM 駆動のモデル発見機能の動作確認スクリプト。

LiteLLM のローカル DB から Vision 対応モデルを取得し、
available_api_models.toml を更新して結果を検証する。
"""

import pprint
import sys
import time

import litellm

from image_annotator_lib import discover_available_vision_models
from image_annotator_lib.core.constants import AVAILABLE_API_MODELS_CONFIG_PATH


def show_litellm_stats() -> None:
    """LiteLLM ローカル DB の Vision モデル統計を表示する。"""
    all_ids = [k for k in litellm.model_cost.keys() if "/" in k]
    vision_ids = [k for k in all_ids if litellm.supports_vision(k)]
    print(f"LiteLLM 総モデル数 (provider-prefixed): {len(all_ids)}")
    print(f"LiteLLM Vision 対応モデル数: {len(vision_ids)}")
    print("Vision モデル (先頭10件):")
    pprint.pprint(vision_ids[:10])
    print(f"  ... (全 {len(vision_ids)} 件)")


def print_result(title: str, result: dict) -> None:
    """結果を整形して表示するヘルパー関数。"""
    print(f"--- {title} ---")
    if "error" in result:
        print(f"エラー: {result['error']}")
    elif "models" in result:
        print(f"成功: {len(result['models'])} 件の Vision モデルが見つかりました。")
        print("モデルリスト (先頭10件):")
        pprint.pprint(result["models"][:10])
    else:
        print("不明な結果形式です。")
        pprint.pprint(result)
    print("-" * (len(title) + 6))


if __name__ == "__main__":
    print(f"モデル情報 TOML パス: {AVAILABLE_API_MODELS_CONFIG_PATH}")

    print("\n=== LiteLLM ローカル DB 統計 ===")
    show_litellm_stats()

    print("\n=== 初回実行 (強制リフレッシュ) ===")
    start = time.monotonic()
    result1 = discover_available_vision_models(force_refresh=True)
    elapsed = time.monotonic() - start
    print_result(f"強制リフレッシュ ({elapsed:.2f}秒)", result1)

    if "error" in result1:
        print(f"\nエラーが発生したため終了します: {result1['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n=== 2回目実行 (キャッシュ利用) ===")
    start = time.monotonic()
    result2 = discover_available_vision_models(force_refresh=False)
    elapsed = time.monotonic() - start
    print_result(f"キャッシュ利用 ({elapsed:.2f}秒)", result2)

    print(f"\n完了。結果は {AVAILABLE_API_MODELS_CONFIG_PATH} に保存されています。")
