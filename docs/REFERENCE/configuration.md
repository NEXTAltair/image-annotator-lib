# リファレンス: 設定ファイル仕様

このドキュメントでは、`image-annotator-lib` が使用する設定ファイルのフォーマットと設定可能な項目について説明します。

## 概要

`image-annotator-lib` は、利用可能なアノテーションモデルとその設定を TOML ファイルから読み込みます。このファイルによって、どのモデルをライブラリに認識させ、どのように動作させるかを定義します。
ライブラリは、`ModelConfigRegistry` クラス (インスタンス名: `config_registry`) を通じて設定を管理します。

## ファイルの場所

設定ファイルは、以下のデフォルトパスから読み込まれます。`ModelConfigRegistry` の `load` メソッドでパスを明示的に指定することも可能です。

*   **システム設定 (デフォルト):** `<プロジェクトルート>/config/annotator_config.toml`
    *   ライブラリがデフォルトで読み込む基本設定ファイルです。
    *   **注意:** このファイルが存在しない場合、ライブラリは初回ロード時にパッケージ内部のテンプレート (`src/image_annotator_lib/resources/system/template_config.toml`) からこのパスに設定ファイルを自動的にコピーします。
*   **ユーザー設定 (デフォルト, 任意):** `<プロジェクトルート>/config/user_config.toml`
    *   このファイルが存在する場合、システム設定の内容を上書き・追加できます。特定のモデルの設定を変更したり、独自モデルを追加したりする場合に使用します。

ライブラリは起動時にこれらのパスから設定を読み込みます (`config_registry.load()` 経由)。

## ファイルフォーマット

設定ファイルは TOML (Tom's Obvious, Minimal Language) 形式で記述されます。

*   ファイル全体が UTF-8 でエンコードされている必要があります。
*   各アノテーションモデルの設定は、トップレベルの **テーブル (セクション)** として定義されます。
*   テーブル名は、ライブラリ内でモデルを一意に識別するための **モデル名** となります。この名前は `list_available_annotators()` で返され、`annotate` 関数の `model_names` 引数で指定するものです。
    *   **重要:** モデル名にドット (`.`) が含まれる場合は、TOML の仕様に従い、モデル名をダブルクォーテーション (`"`) で囲む必要があります (例: `["my.model-v1.0"]`)。
*   各テーブル内には、そのモデルの設定をキーと値のペアで記述します。

## 設定項目

各モデルのテーブル (セクション) 内で以下のキーを設定できます。

*   **`class` (文字列, 必須)**
    *   このモデルに対応する Python クラスの名前。`src/image_annotator_lib/models/` または `src/image_annotator_lib/model_class/` 内に実装されている具象クラス (例: `WDTagger`, `AestheticShadowV2`, `GoogleApiAnnotator`) を指定します。
    *   ライブラリ内部のファクトリがこの名前を使って対応するクラスをインスタンス化します。

*   **`model_path` (文字列, 必須)**
    *   モデルファイルまたはモデルリポジトリの場所を指定します。以下の形式がサポートされています。
        *   **Hugging Face Hub リポジトリ ID:** (例: `"Salesforce/blip-image-captioning-large"`) ライブラリが自動的に必要なファイルをダウンロードします。
        *   **直接ダウンロード URL:** (例: `"https://github.com/path/to/model.zip"`) 指定された URL からファイルをダウンロードします。ZIP ファイルの場合は自動的に解凍されます。
        *   **ローカルファイルパス:** (例: `"C:/path/to/my_model.onnx"` または `"./local_models/model.safetensors"`) ローカルにあるモデルファイルまたはディレクトリへのパス。絶対パスまたは相対パスで指定できます。

*   **`estimated_size_gb` (数値, 推奨)**
    *   モデルの推定メモリ使用量 (GB単位)。ライブラリのメモリ管理機能 (`ModelLoad` クラス) が、モデルのロード/アンロードを判断する際に使用します。
    *   正確な値でなくても構いませんが、指定しておくとメモリ不足エラーの防止に役立ちます。指定しない場合や0以下の場合は、メモリチェックの一部がスキップされる可能性があります。

*   **`device` (文字列, 任意)**
    *   モデルの推論に使用するデバイスを指定します (例: `"cuda"`, `"cpu"`, `"cuda:0"`)。
    *   省略した場合、または `"cuda"` が指定されたが CUDA が利用できない場合、ライブラリは `core/utils.py` の `determine_effective_device` 関数を通じて、利用可能な最適なデバイス (CUDA GPU があればデフォルトで `"cuda"`、なければ `"cpu"`) を自動的に選択しようとします。
    *   特定のモデルを CPU で実行したい場合などに `"cpu"` を明示的に指定します。

*   **その他のカスタム設定 (任意)**
    *   モデルクラスの実装によっては、上記以外のカスタム設定キーを受け付ける場合があります (例: `threshold`, `tag_file_path`, `api_endpoint`, `retry_count`)。
    *   これらのキーは、モデルクラスの `__init__` メソッド内で `config_registry.get(self.model_name, "your_key", default_value)` のようにして読み込む必要があります。
    *   カスタム設定を追加する場合は、対応するモデルクラスのドキュメントやコードを確認してください。
    *   **推奨:** モデル固有のパラメータ（閾値など）は、可能な限りモデルクラスのインスタンス変数としてハードコーディングするか、デフォルト値を設定し、設定ファイルでの指定は任意とすることが望ましいです。

## 設定例

```toml
# src/image_annotator_lib/resources/system/annotator_config.toml

# --- Tagger Models ---

[wd-v1-4-vit-tagger-v2]
class = "WDTagger"
model_path = "SmilingWolf/wd-v1-4-vit-tagger-v2" # Hugging Face Repo ID
estimated_size_gb = 1.695
# device = "cuda" # 省略可能

["z3d-e621-tagger-v1"] # モデル名にハイフンがあるので引用符は不要
class = "Z3D_E621Tagger"
model_path = "zhonger/z3d-e621-tagger-v1"
estimated_size_gb = 1.695

[blip-large-captioning]
class = "BLIPTagger"
model_path = "Salesforce/blip-image-captioning-large"
estimated_size_gb = 2.1

["deepdanbooru-v3-20211112-sgd-e28"]
class = "DeepDanbooruTagger"
model_path = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip" # URL
estimated_size_gb = 0.723
device = "cpu" # このモデルは CPU で実行する指定

# --- Scorer Models ---

[aesthetic-shadow-v1]
class = "AestheticShadowV1"
model_path = "shadowlilac/aesthetic-shadow"
estimated_size_gb = 0.6 # 値を更新

[aesthetic-shadow-v2]
class = "AestheticShadowV2"
model_path = "NEXTAltair/cache_aestheic-shadow-v2" # ミラーリポジトリ
estimated_size_gb = 1.1 # 仮の値

[cafe-aesthetic]
class = "CafePredictor"
model_path = "cafeai/cafe_aesthetic"
estimated_size_gb = 0.086 # 仮の値

# --- Web API Models ---

# [google-gemini-pro-vision]
# class = "GoogleApiAnnotator"
# model_path = "gemini-pro-vision" # Provider上のモデル名
# estimated_size_gb = 0 # APIモデルは通常メモリ使用量を考慮しない
# prompt_template = "Please describe the image and suggest tags for web image search. Format the output as JSON: {'Annotation': {'caption': '...', 'tags': ['tag1', 'tag2', ...]}}."
# model_name_on_provider = "gemini-pro-vision"

# --- Custom Model Example ---
# # ユーザー設定ファイル (user/annotator_config.toml) に記述する例
# ["my.custom-tagger.v1.0"] # モデル名にドットが含まれる場合は引用符で囲む
# class = "MyCustomTagger"
# model_path = "./models/custom_tagger_v1" # ローカルパス
# estimated_size_gb = 0.2
# custom_threshold = 0.4 # モデル固有のカスタム設定
# device = "cpu" # ユーザー設定でデバイスを上書き
```