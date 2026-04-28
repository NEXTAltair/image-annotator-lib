# Image Annotator Lib

`image-annotator-lib` は、様々な画像アノテーションモデル(タガー、スコアラーなど)を統一されたインターフェースで利用するための Python ライブラリです。`scorer_wrapper_lib` と `tagger_wrapper_lib` を統合し、コードの重複削減、API の統一、メンテナンス性の向上、機能拡張の容易化を目指しています。

## 主な機能

- 複数の画像タギングモデルと画像スコアリングモデルをサポート
- 統一された API (`annotate`) による複数モデル･複数画像の一括処理
- pHash に基づく画像と結果の紐付け
- 設定ファイル (`annotator_config.toml`) によるモデル選択と設定（ローカル ML モデル専用）
- メモリ使用量に基づく効率的なモデルキャッシュ管理 (LRU, CPU 退避/CUDA 復元)
- **LiteLLM による WebAPI モデル自動検出**（追加設定不要）

## WebAPI モデル自動検出

本ライブラリは [LiteLLM](https://github.com/BerriAI/litellm) のローカル DB を使用して、OpenAI / Anthropic / Google などの WebAPI モデルを自動的に検出します。

### 動作フロー

1. **初回起動時**: LiteLLM のローカル DB から Vision 対応モデルを検出し、`config/available_api_models.toml` に保存
2. **以降の起動時**: `[meta] last_refresh` の TTL（デフォルト 7 日）を確認し、超過時はバックグラウンドで再取得
3. **週次 CI**: `refresh-models.yml` が最新 LiteLLM で `available_api_models.toml` を更新し PR を自動作成

### 環境変数

| 変数名 | 既定値 | 説明 |
|---|---|---|
| `IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS` | `7` | モデルリストの有効期間（日数） |
| `IMAGE_ANNOTATOR_SKIP_API_DISCOVERY` | `false` | `true` に設定すると起動時の API discovery を完全スキップ |

### プログラムからの利用

```python
from image_annotator_lib import discover_available_vision_models
from image_annotator_lib.core.api_model_discovery import should_refresh, trigger_background_refresh

# キャッシュから読み込み（TTL 内なら再取得しない）
result = discover_available_vision_models()
if "models" in result:
    print(f"{len(result['models'])} 件の Vision モデルが利用可能")

# TTL 超過確認と手動 refresh
if should_refresh():
    trigger_background_refresh()  # 非同期・起動をブロックしない

# 強制再取得
result = discover_available_vision_models(force_refresh=True)
```

詳細は [`docs/integrations.md`](docs/integrations.md) を参照してください。

## インストール

本ライブラリは [uv](https://github.com/astral-sh/uv) を使用したパッケージ管理を推奨しています。

1. **仮想環境の作成 (推奨):**
   プロジェクトルートで以下を実行します。

   ```bash
   uv venv
   ```

   作成された仮想環境を有効化します。
   (Windows: `.venv\Scripts\activate`, Linux/macOS: `source .venv/bin/activate`)
2. **依存関係のインストール:**

   ```bash
   # 通常の利用
   uv sync

   # 開発用にソースからインストール
   uv sync --dev
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

## アノテーターの追加方法

新しい画像アノテーションモデルを追加する手順：

### 1. 適切なベースクラスの選択

モデルの種類に応じて、適切なベースクラスを継承してください：

- **Web API モデル** (PydanticAI): `WebApiBaseAnnotator` + `PydanticAIAnnotatorMixin`
- **ONNX モデル**: `ONNXBaseAnnotator`
- **Transformers モデル**: `TransformersBaseAnnotator`
- **TensorFlow モデル**: `TensorflowBaseAnnotator`
- **CLIP ベースモデル**: `ClipBaseAnnotator`

### 2. 必要なメソッドの実装

実装が必要な抽象メソッド：

- `_generate_tags()`: アノテーション結果からタグを生成
- `_run_inference()`: モデル推論の実行
- PydanticAI モデルの場合: `run_with_model()` (プロバイダーレベル実行)

### 3. 設定ファイルへの登録

`config/annotator_config.toml` に新しいモデルの設定を追加：

```toml
[your-model-name]
model_path = "huggingface/repo-name"  # または URL、ローカルパス
class = "YourModelClassName"
device = "cuda"  # または "cpu"
estimated_size_gb = 1.5

# Web API モデルの場合
api_model_id = "provider-model-id"
model_name_on_provider = "provider-model-name"
```

### 4. テストの追加

適切なテストカテゴリにテストを追加：

- `tests/unit/` - ユニットテスト
- `tests/integration/` - 統合テスト（実際のモデル使用）
- `tests/model_class/` - モデル固有のテスト

詳細な開発ガイドラインは `CLAUDE.md` を参照してください。

## 開発者向け情報

### テスト実行

```bash
# 全テスト実行
pytest

# 特定カテゴリのテスト実行
pytest -m unit        # ユニットテストのみ
pytest -m integration # 統合テストのみ
pytest -m webapi      # Web APIテストのみ
pytest -m scorer      # スコアラーモデルテストのみ
pytest -m tagger      # タガーモデルテストのみ
```

### コード品質チェック

```bash
# リンティングとフォーマット
ruff check
ruff format

# 型チェック
mypy src/
```
