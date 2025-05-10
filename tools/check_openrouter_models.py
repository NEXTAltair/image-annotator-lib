import os
import traceback

import requests
import json
from dotenv import load_dotenv

# .env ファイルから API キーをロード
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("エラー: 環境変数 OPENROUTER_API_KEY が設定されていません。")
else:
    try:
        print("利用可能な OpenRouter Vision 対応モデル情報:")

        # OpenRouter APIのエンドポイント
        url = "https://openrouter.ai/api/v1/models"

        # API リクエストを実行
        response = requests.get(url)

        response.raise_for_status()  # ステータスコードが 200 OK でない場合に例外を発生

        models_data = response.json()
        with open("models_data.json", "w") as f:
            json.dump(models_data, f, indent=4)
        vision_models_found = False

        # 応答からモデルデータのリストを取得
        if "data" in models_data and isinstance(models_data["data"], list):
            model_list = models_data["data"]

            for model in model_list:
                # --- Vision 対応かチェック ---
                architecture = model.get("architecture")
                is_vision_model = False
                if architecture and isinstance(architecture, dict):
                    input_modalities = architecture.get("input_modalities")
                    if (
                        input_modalities
                        and isinstance(input_modalities, list)
                        and "image" in input_modalities
                    ):
                        is_vision_model = True

                # Vision 対応モデルでなければスキップ
                if not is_vision_model:
                    continue
                # --- チェックここまで ---

                vision_models_found = True
                # Vision 対応モデルの情報のみ表示
                print("-" * 20)
                print(f"モデル ID (id): {model.get('id', 'N/A')}")
                print(f"  名前 (name): {model.get('name', 'N/A')}")
                # print(f"  説明 (description): {model.get('description', 'N/A')}") # 長いのでコメントアウト
                print(f"  コンテキスト長 (context_length): {model.get('context_length', 'N/A')}")

                # 特に重要な architecture フィールド
                if architecture and isinstance(architecture, dict):
                    print(f"  アーキテクチャ:")
                    print(f"    入力形式 (input_modalities): {architecture.get('input_modalities', 'N/A')}")
                    print(
                        f"    出力形式 (output_modalities): {architecture.get('output_modalities', 'N/A')}"
                    )
                #     print(
                #         f"    トークナイザ (tokenizer): {architecture.get('tokenizer', 'N/A')}"
                #     )  # 不要な情報のためコメントアウト
                # else:
                #     print("  アーキテクチャ情報: なし")

                # 全データ表示
                print(f"  全データ (raw): {model}")

            if not vision_models_found:
                print("Vision 対応モデルが見つかりませんでした。")

        else:
            print("応答形式が予期したものと異なります。'data' キーが見つからないか、リストではありません。")
            print(f"受信した応答: {models_data}")

    except requests.exceptions.RequestException as e:
        # 接続エラー、タイムアウトなど
        print(f"APIへの接続中にエラーが発生しました: {e}")
        # print(traceback.format_exc()) # デバッグ時以外は不要かも
    except requests.exceptions.HTTPError as e:
        # APIからのエラー応答 (4xx, 5xx)
        print(f"OpenRouter APIエラー (ステータスコード: {e.response.status_code}): {e.response.text}")
        # print(traceback.format_exc()) # デバッグ時以外は不要かも
    except Exception as e:
        print(f"モデル一覧の取得中に予期せぬエラーが発生しました: {e}")
        print(traceback.format_exc())
