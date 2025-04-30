# 技術仕様書

## 1. 開発環境

### 1.1 パス情報注意点

ファイルパス指定ミス防止のため、主要パス情報を以下記載。

- **プロジェクトルート:** `image-annotator-lib` (リポジトリルート)
    - 全ファイル操作(read_file, write_to_file等)は原則本ディレクトリからの相対パス指定。
- **`image-annotator-lib` パッケージソースルート:** `src/image_annotator_lib`
    - ライブラリ内部モジュール参照・編集時の基点。
    - 例: `src/image_annotator_lib/core/base.py`

### 1.2 技術スタック

- Python >= 3.12
- PyTorch (Transformers, CLIP)
- ONNX Runtime
- TensorFlow (DeepDanbooru)
- TOML (設定ファイル)
- Ruff (フォーマッター、リンター)
- Mypy (型チェック)
- uv (パッケージ管理)

### 1.3 主要依存関係

依存関係は `pyproject.toml` に定義、`uv` で管理。主要ライブラリは以下:

- `toml`
- `requests`
- `huggingface_hub`
- `transformers`
- `onnxruntime` / `onnxruntime-gpu`
- `tensorflow`
- `Pillow`
- `numpy`
- `tqdm`
- `pytest` (テスト用)
- `psutil` (メモリ管理用)
- `python-dotenv` (APIキー管理用)
- `anthropic` (API用)
- `google-genai` (API用)
- `openai` (API用)

### 1.4 ディレクトリ構造 (ソースコード)

ライブラリ主要ソースコードは `src/image_annotator_lib/` 以下配置。

```
src/
└── image_annotator_lib/   # ライブラリパッケージ
    ├── __init__.py        # パッケージ初期化、主要APIエクスポート
    ├── api.py             # ユーザー向けAPI関数 (annotate)
    ├── py.typed           # PEP 561 準拠マーカーファイル
    ├── config/            # (空の可能性、設定ファイルは resources/ に移動?)
    ├── core/              # Tagger/Scorer共通基盤モジュール
    │   ├── api_model_discovery.py # Web API モデル情報取得・管理
    │   ├── base.py          # BaseAnnotator, フレームワーク別基底クラス, 型定義
    │   ├── config.py        # ModelConfigRegistry (設定管理)
    │   ├── constants.py     # 定数定義
    │   ├── factory.pyi      # (型スタブファイル)
    │   ├── model_factory.py # ModelLoad (モデルロード/キャッシュ管理)
    │   ├── registry.py      # モデル登録/取得関連
    │   └── utils.py         # 共通ユーティリティ (ロギング、ファイルI/O等)
    ├── exceptions/        # カスタム例外クラス
    │   ├── __init__.py
    │   ├── errors.py
    │   └── errors.pyi     # (型スタブファイル)
    ├── model_class/       # 各モデル実装 (具象クラス)
    │   ├── annotator_webapi.py # Web API ベースアノテーター基底・実装
    │   ├── pipeline_scorers.py # Pipeline ベーススコアラー
    │   ├── scorer_clip.py      # CLIP ベーススコアラー
    │   ├── tagger_onnx.py      # ONNX ベースタガー
    │   ├── tagger_tensorflow.py # TensorFlow ベースタガー
    │   └── tagger_transformers.py # Transformers ベースタガー
    └── resources/         # 設定ファイル等リソース
        └── system/
            ├── annotator_config.toml # システムデフォルト設定
            └── openrouter_json_compatible_models.py # OpenRouterモデル情報
```
*(注意: 以前ドキュメントの `resources/user/` ディレクトリは現リストに含まず)*

## 2. コーディング規約

### 2.1 基本規約

- **言語:** 日本語 (エラーメッセージ、ログ、コメント、Docstring)
- **フォーマッター:** Ruff format (`ruff format .`)
- **リンター:** Ruff (`ruff check . --fix`)
- **型チェック:** Mypy (`mypy src/`)
- **Docstring:** Google スタイル (日本語)
- **ルール参照:** `.cursor/rules` ディレクトリ内ルールファイル参照しコーディング。変更必要時はユーザー判断仰ぐ。

### 2.2 特に重要なルール (AI 向け)

- **型ヒント:**
    - **モダンな型:** `typing.List` や `typing.Dict`, `Optional` でなく、Python 3.9 以降組込型 (`list`, `dict`) や `collections.abc` 型使用 (`Union` は `|` 使用)。
    - **`Any` 回避:** `Any` 型使用は最小限、可能限り具体的型指定。
    - **エラー抑制禁止:** Mypy/Ruff エラー/警告は `# type: ignore` や `# noqa` で抑制せず、根本解決。
- **半角文字:** コード、コメント、ドキュメント内では、**絶対全角英数字・全角記号不使用**。
- **原則違反通知:**
    - AIが定義原則違反コード生成・編集せざるを得ない場合、**一度作業停止、必ずユーザーに旨・理由説明、指示仰ぐ。**
- **問題解決プロセス:**
    - エラー/警告発生時は以下手順対応。
        1.  **解決策検討:** 問題原因分析、最低3つ以上異解決策検討。
        2.  **最適解決策選択:** 検討解決策中、最適と考えられるもの選択、理由記録。
        3.  **試行・反復:** 選択解決策適用、問題解決確認。解決しない場合、別解決策試行、更なる別解決策検討。
        4.  **エスカレーション:** 上記試行3回以上繰返しても問題未解決の場合、作業中断、ユーザーに状況、試行解決策、考原因説明、判断要求。

## 3. 主要技術決定 (履歴)

以下はプロジェクト開発中主要技術決定記録。詳細背景・理由は `.cursor/rules/lessons-learned.mdc` や関連コミットログ参照。

- **ログ出力多重化問題修正 (2025-04-19):**
    - `core/utils.py` logger 初期化処理複数回実行問題修正。`init_logger` 関数分離、`__init__.py` で一度のみ呼出変更。
- **レジストリ・logger初期化明示化 (2025-04-19):**
    - `core/registry.py` import 時自動初期化(ログ出力、`register_annotators()` 呼出)廃止。初期化はエントリーポイント等で明示実行設計変更、import 時副作用排除。
- **CUDA非対応環境CPUフォールバック実装 (2025-04-19):**
    - `core/utils.py` に `determine_effective_device` 関数追加、環境応じ利用可能デバイス(`cuda` or `cpu`)判定。CUDA 利用不可でも CPU 動作可能モデルは自動フォールバック修正。
- **`annotator_config.toml` キー設計維持決定 (2024-07-28):**
    - Web API モデル設定セクションキーとして、可読性・実装シンプルさから `model_name_short` (例: `"Gemini 1.5 Pro"`) 維持決定。プレフィックス除去済 ID (例: `"gemini-pro-1.5"`) 不採用。
- **Web API アノテーター初期化フロー変更 (2024-07-28):**
    - Web API アノテーター `__init__` は `model_name` のみ受取統一。API コール使用最終モデル ID (`api_model_id`) 解決・加工は `__enter__` メソッド内実行変更。

## 4. 新モデル追加方法

本セクションでは、新画像アノテーションモデル(従来MLモデル・Web APIベースモデル含)を `image-annotator-lib` に追加手順説明。

### 4.1 アーキテクチャ・クラス階層理解

ライブラリはコード重複最小限抑えつつ多様アノテーター管理のため、3層クラス階層採用。詳細は `docs/architecture.md` 参照。

- **`BaseAnnotator` (`core/base.py`):** 全アノテーター抽象基底クラス。共通機能(ロギング、設定読込、コンテキスト管理、**バッチ処理**、pHash計算、エラーハンドリング、**標準化結果生成 (`_generate_result`)**)提供。
- **フレームワーク/タイプ別基底クラス (`core/base.py`, `model_class/annotator_webapi.py`):** `BaseAnnotator` 継承。特定フレームワーク(ONNX, Transformers, TensorFlow, CLIP等)またはタイプ(Web API等)共通ロジック実装。例: `ONNXBaseAnnotator`, `BaseWebApiAnnotator`。
- **具象モデルクラス (`model_class/`):** フレームワーク/タイプ別基底クラス継承。個別モデル固有ロジックのみ実装(主に `_generate_tags`)。例: `WDTagger`, `GoogleApiAnnotator`。

### 4.2 具象モデルクラス実装

適切ディレクトリ(`src/image_annotator_lib/model_class/`)に新モデル用Pythonクラス作成。

1.  **適切基底クラス選択:** モデルフレームワーク/タイプ適合基底クラス選択。
2.  **クラス定義:**
    *   選択基底クラス継承。
    *   `__init__(self, model_name)` 実装:
        *   `super().__init__(model_name)` 呼出。
        *   モデル固有設定は `config_registry.get(self.model_name, "your_key", default_value)` 使用取得(`src/image_annotator_lib/core/config.py` 共有インスタンス `config_registry` 利用)。**`self.config` 直接アクセス不可。**
        *   モデル固有初期化(タグリスト読込、閾値設定等)実行。Web APIモデルの場合、APIキーは基底クラス `_load_api_key` で処理。
    *   **必要抽象メソッドオーバーライド:** 通常、モデル固有データ処理・タグ/スコア生成ロジック実装。
        *   `_preprocess_images(self, images: list[Image.Image]) -> Any`: PIL画像リストをモデル期待形式へ前処理。
        *   `_run_inference(self, processed: Any) -> Any`: 前処理済データで推論実行、生モデル出力返却。Web APIモデルはAPIコール実行。
        *   `_format_predictions(self, raw_outputs: Any) -> list[Any]`: (任意) 生モデル出力を `_generate_tags` 消費容易形式へ整形。
        *   `_generate_tags(self, formatted_output: Any) -> list[str]`: 整形済出力から最終タグリスト(または `["[SCORE]0.95"]` 等スコア文字列)生成。**多場合、実装必要主要メソッド。**
    *   **`_generate_result` 通常オーバーライド不可:** `BaseAnnotator` が標準化 `AnnotationResult` 生成のため、本メソッドオーバーライド非推奨。
    *   **バッチ処理:** `BaseAnnotator.predict` が入力画像チャンク(バッチ)分割処理。`_preprocess_images` と `_run_inference` はバッチデータ受入設計。
    *   **結果・エラー:** `BaseAnnotator.predict` が結果・エラー集約。サブクラスメソッド内発生エラーは、可能なら結果構造一部として返却、または例外伝播(基底クラス捕捉し `AnnotationResult` `error` フィールド記録)。

### 4.3 設定ファイルエントリ追加

新モデル利用可能化のため、設定ファイル(`src/image_annotator_lib/resources/system/annotator_config.toml` またはユーザー設定ファイル)にエントリ追加。ユーザー設定ファイルはプロジェクトルート `config/annotator_config.toml` 配置推奨。

- **セクション名 (`[model_unique_name]`):** ライブラリ内モデル識別用一意名。
- `class` (必須): 実装具象モデルクラス名(文字列)。
- `model_path` (ローカル/DLモデル必須): モデルファイル/リポジトリパス/URL。APIモデルはプロバイダー上モデルID等。
- `estimated_size_gb` (ローカル/DLモデル推奨): メモリ管理用モデルサイズ概算値(GB)。APIモデル通常0。
- `device` (任意): モデル使用デバイス(`"cuda"`, `"cpu"` 等)。
- 他モデル固有設定キー。

**設定例:**

```toml
[my-new-onnx-tagger-v1]
class = "MyNewONNXTagger"
model_path = "path/to/your/model.onnx"
estimated_size_gb = 0.5
tags_file_path = "path/to/tags.txt"
threshold = 0.45
```

### 4.4 機能検証

ライブラリ使用し新モデル正動作テスト。

```python
from image_annotator_lib import annotate, list_available_annotators
from PIL import Image

available = list_available_annotators()
print(available)
assert "my-new-onnx-tagger-v1" in available

img = Image.open("path/to/test/image.jpg")
results = annotate([img], ["my-new-onnx-tagger-v1"])
print(results)
```

新モデルクラス対単体テスト追加も検討。

## 5. テスト実行方法

`pytest` と `pytest-bdd` 使用テスト実行手順。

### 5.1 準備

開発依存関係インストール(プロジェクトルート実行)。

```bash
# 仮想環境有効化
# uv pip install -e .[dev] または pip install -e .[dev]
uv pip install -e .[dev]
```

### 5.2 テスト実行コマンド

プロジェクトルートから `pytest` コマンド実行。

- **全テスト実行:** `pytest`
- **詳細出力:** `pytest -v`
- **BDD形式出力:** `pytest --gherkin-terminal-reporter`
- **特定ファイル/ディレクトリ:** `pytest tests/unit/test_api.py` または `pytest tests/integration/`
- **名前指定:** `pytest -k "annotate"` または `pytest -k "model loading"`
- **カバレッジ測定:** `pytest --cov=src/image_annotator_lib tests/`

### 5.3 VSCode実行

"BDD - Cucumber/Gherkin Full Support" 拡張機能使用で、`.feature` ファイルから直接シナリオ実行/デバッグ可。

## 6. ロギング設定方法

ライブラリは標準 `logging` モジュール使用。

### 6.1 基本使い方

モジュールレベルでロガー取得。

```python
import logging
logger = logging.getLogger(__name__)
logger.info("情報メッセージ")
```

### 6.2 ロガー設定 (`setup_logger`)

基本設定(レベル、フォーマット、ハンドラ)は `core/utils.py` `setup_logger` 関数実行(通常内部呼出)。

- **ログレベル:** デフォルト `logging.INFO`。
- **フォーマット:** `%(asctime)s - %(name)s - %(levelname)s - %(message)s`。
- **ハンドラ:** デフォルトコンソール(`StreamHandler`)出力。ファイル出力も設定可。

### 6.3 ログレベル変更

環境変数 `LOG_LEVEL` 設定でライブラリ全体ログレベル変更可(例: `export LOG_LEVEL=DEBUG` または `.env` ファイルに `LOG_LEVEL=DEBUG` 記述)。