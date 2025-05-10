"""API モデル発見機能の動作確認用サンプルスクリプト。

実行した結果 OpenRouter API が Vision タスクに対応したモデルの一覧を取得できるかを確認するためのスクリプト。
結果は PROJECTROOT/config/available_api_models.toml に保存される。
"""

import pprint
import time

from image_annotator_lib import discover_available_vision_models
from image_annotator_lib.core.constants import AVAILABLE_API_MODELS_CONFIG_PATH


def print_result(title: str, result: dict):
    """結果を整形して表示するヘルパー関数。"""
    print(f"--- {title} ---")
    if "error" in result:
        print(f"エラー: {result['error']}")
    elif "models" in result:
        print(f"成功: {len(result['models'])} 件の Vision モデルが見つかりました。")
        # 最初のいくつかを表示 (オプション)
        # print("モデルリスト (一部):")
        # pprint.pprint(result['models'][:10])
        # print("...")
    else:
        print("不明な結果形式です。")
        pprint.pprint(result)
    print("-" * (len(title) + 6))


if __name__ == "__main__":
    print(f"モデル情報 TOML パス: {AVAILABLE_API_MODELS_CONFIG_PATH}")
    # 必要なら設定ファイルパスを指定して config_registry をロードし直す
    # config_registry.load(config_path="path/to/your/config/annotator_config.toml")

    # 1. 初回実行 (キャッシュなし)
    start_time = time.monotonic()
    result1 = discover_available_vision_models()
    end_time = time.monotonic()
    print_result(f"初回実行 ({end_time - start_time:.2f}秒)", result1)

    # 2. 2回目の実行 (キャッシュ利用を期待)
    time.sleep(1)  # 念のため少し待つ
    start_time = time.monotonic()
    result2 = discover_available_vision_models()
    end_time = time.monotonic()
    print_result(f"2回目実行 (キャッシュ利用, {end_time - start_time:.2f}秒)", result2)

    # 3. 強制リフレッシュ実行
    start_time = time.monotonic()
    result3 = discover_available_vision_models(force_refresh=True)
    end_time = time.monotonic()
    print_result(f"強制リフレッシュ実行 ({end_time - start_time:.2f}秒)", result3)

    print("\n完了。")
    print(f"結果は {AVAILABLE_API_MODELS_CONFIG_PATH} に保存されているはずです。")
