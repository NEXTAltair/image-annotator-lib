# Image Annotator Lib

`image-annotator-lib` は、様々な画像アノテーションモデル(タガー、スコアラーなど)を統一されたインターフェースで利用するための Python ライブラリです。`scorer_wrapper_lib` と `tagger_wrapper_lib` を統合し、コードの重複削減、API の統一、メンテナンス性の向上、機能拡張の容易化を目指しています。

## 主な機能

- 複数の画像タギングモデルと画像スコアリングモデルをサポート
- 統一された API (`annotate`) による複数モデル･複数画像の一括処理
- pHash に基づく画像と結果の紐付け
- 設定ファイル (`annotator_config.toml`) によるモデル選択と設定
- メモリ使用量に基づく効率的なモデルキャッシュ管理 (LRU, CPU 退避/CUDA 復元)

## インストール

本ライブラリは [uv](https://github.com/astral-sh/uv) を使用したパッケージ管理を推奨しています。

1.  **仮想環境の作成 (推奨):**
    プロジェクトルートで以下を実行します。
    ```bash
    uv venv
    ```
    作成された仮想環境を有効化します。
    (Windows: `.venv\Scripts\activate`, Linux/macOS: `source .venv/bin/activate`)

2.  **依存関係のインストール:**
    ```bash
    # 通常の利用
    uv pip sync

    # 開発用にソースからインストール
    uv pip sync --dev
    ```

## Getting Started / 基本的な使い方

### 1. ライブラリのインポート

```python
import logging
from pathlib import Path
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators

# ロギング設定 (任意、詳細表示用)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_annotator_lib")
logger.setLevel(logging.DEBUG) # DEBUGレベルで詳細ログを出力
```

### 2. 利用可能なモデルの確認

設定ファイル (`annotator_config.toml`) で利用可能になっているモデルを確認します。

```python
available_models = list_available_annotators()
print("Available models:", available_models)
# 例: ['wd-v1-4-vit-tagger-v2', 'aesthetic-shadow-v2', 'blip-large-captioning', ...]
```

### 3. 画像の準備

アノテーションしたい画像を PIL (Pillow) を使って読み込みます。`annotate` 関数は PIL Image オブジェクトのリストを受け取ります。

```python
try:
    # 処理したい画像のパスを指定
    image_path = Path("path/to/your/image.jpg")
    img = Image.open(image_path)
    images_to_process = [img] # 単一画像をリストに入れる
except FileNotFoundError:
    print(f"Error: 画像ファイルが見つかりません {image_path}")
    exit()
except Exception as e:
    print(f"Error opening image {image_path}: {e}")
    exit()
```

### 4. 単一モデルでのアノテーション

利用可能なモデルリストからモデル名を一つ選び、`annotate` 関数に渡します。

```python
# モデルを選択 (例: WD Tagger)
model_name = "wd-v1-4-vit-tagger-v2"
models_to_use = [model_name]

# 選択したモデルが利用可能か確認 (任意だが推奨)
if model_name not in available_models:
    print(f"Warning: 選択されたモデル '{model_name}' は利用できません。")
    exit()

# アノテーション実行
print(f"Annotating image with model: {model_name}...")
results = annotate(images_to_process, models_to_use)
print("Annotation complete.")

# 結果の処理
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")
    if not model_results:
        print("  この画像の処理結果はありません。")
        continue

    # 特定モデルの結果を取得
    result_for_model = model_results.get(model_name)

    if result_for_model:
        if result_for_model.get("error"):
            print(f"  Error: {result_for_model['error']}")
        else:
            tags = result_for_model.get('tags', []) # タグのリストを取得
            formatted_output = result_for_model.get('formatted_output') # モデル固有の整形済み出力
            print(f"  Tags: {tags}")
            # print(f"  Formatted Output: {formatted_output}") # 詳細出力が必要な場合
    else:
        print(f"  モデル '{model_name}' の結果が見つかりません。")
```

### 5. 複数モデルでのアノテーション

複数の画像と複数のモデルを同時に処理できます。

```python
# 複数画像の準備
try:
    image_path1 = Path("path/to/your/image1.jpg")
    image_path2 = Path("path/to/your/image2.png")
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    multiple_images_to_process = [img1, img2]
except FileNotFoundError as e:
    print(f"Error: 画像ファイルが見つかりません: {e}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# 複数モデルを選択 (例: WD Tagger と Aesthetic Scorer)
models_to_use_multi = ["wd-v1-4-vit-tagger-v2", "aesthetic-shadow-v2"]

# 利用可能か確認 (任意)
unavailable = [m for m in models_to_use_multi if m not in available_models]
if unavailable:
    print(f"Warning: 次のモデルは利用できません: {', '.join(unavailable)}")
    models_to_use_multi = [m for m in models_to_use_multi if m in available_models]
    if not models_to_use_multi:
        print("Error: 利用可能なモデルが選択されていません。")
        exit()

# アノテーション実行
print(f"Annotating {len(multiple_images_to_process)} images with models: {', '.join(models_to_use_multi)}...")
multi_results = annotate(multiple_images_to_process, models_to_use_multi)
print("Annotation complete.")

# 複数モデルの結果処理
for phash, model_results in multi_results.items():
    print(f"--- Image (pHash: {phash}) ---")

    # 各モデルの結果にアクセス
    for model_name, result in model_results.items():
        print(f"  Model: {model_name}")
        if result.get("error"):
            print(f"    Error: {result['error']}")
        else:
            tags = result.get('tags', [])
            formatted_output = result.get('formatted_output')
            if tags:
                print(f"    Tags (Top 5): {tags[:5]}") # 例: 上位5件表示
            if formatted_output:
                # スコアラーの例 (formatted_output または tags に含まれる可能性あり)
                if isinstance(formatted_output, (float, int)): # 単純なスコア
                     print(f"    Score: {formatted_output:.4f}")
                elif isinstance(formatted_output, dict): # より複雑な出力
                     print(f"    Formatted Output: {formatted_output}")
                 # 他にスコアが見つからない場合、タグのパターンを確認
                elif any(t.startswith("[SCORE]") for t in tags):
                    score_tag = next((t for t in tags if t.startswith("[SCORE]")), None)
                    if score_tag:
                        try:
                            score_value = float(score_tag.split("[SCORE]")[1])
                            print(f"    Score: {score_value:.4f}")
                        except (IndexError, ValueError):
                            print(f"    タグからスコアをパースできませんでした: {score_tag}")

    print() # 画像ごとに改行
```

### 6. 結果の構造

`annotate` 関数は、画像の **知覚ハッシュ (pHash)** をキーとする辞書を返します。これにより、処理順序が変わったりエラーが発生したりしても、結果を正しい画像に関連付けることができます。

各 pHash キーは、アノテーションに使用された **モデル名** をキーとする別の辞書にマッピングされます。その値には、特定の画像とモデルに対するアノテーション結果が含まれます。

```python
{
    "image1_phash": { # 最初の画像の知覚ハッシュ
        "model1_name": { # 最初のモデルの結果
            "tags": ["tagA", "tagB", ...], # 生成されたタグ/スコアのリスト
            "formatted_output": {...}, # モデル固有の詳細な整形済み出力
            "error": None # 成功時は None、エラー時はエラーメッセージ文字列
        },
        "model2_name": { ... } # 最初の画像に対する2番目のモデルの結果
    },
    "image2_phash": { ... }, # 2番目の画像の結果
    # pHash 計算に失敗した場合の特殊キー
    "unknown_image_0": { ... }
}
```

### 7. エラーハンドリング

モデル実行中のエラー(モデルロード失敗、推論エラーなど)は、特定モデル･画像の `error` キーの下にある結果辞書内に捕捉されます。`annotate` 関数自体は、個々のモデルの失敗に対して通常は例外を発生させず、部分的な結果を返すことができます。

```python
# エラーチェックの例
results = annotate([img], ["non-existent-model"]) # 失敗するモデルを指定
for phash, model_results in results.items():
    for model_name, result in model_results.items():
        if result.get("error"):
            print(f"Error processing image {phash} with model {model_name}: {result['error']}")
```

## ドキュメント

より詳細な情報については、`docs/` ディレクトリ内の以下のドキュメントを参照してください。

-   [**製品要求仕様書 (Product Requirement Document)**](./docs/product_requirement_docs.md): プロジェクトの目標、対象ユーザー、主要機能など。
-   [**システムアーキテクチャ (System Architecture)**](./docs/architecture.md): ライブラリの構造、主要コンポーネント、ワークフロー、設計決定など。
-   [**技術仕様書 (Technical Specifications)**](./docs/technical.md): 開発環境、技術スタック、依存関係、コーディング規約、モデル追加･テスト･ロギング手順など。

(古いドキュメントへのリンクは削除されました。)
テスト完了
