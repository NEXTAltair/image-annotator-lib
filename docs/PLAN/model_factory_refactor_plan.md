# model_factory.py リファクタリング計画 (インターフェース維持版)

## 1. 目的

`src/image_annotator_lib/core/model_factory.py` の内部構造を改善し、保守性、可読性、堅牢性を向上させる。
外部モジュール (例: `base.py`) から利用されている `ModelLoad` クラスの静的メソッドインターフェースは変更せず、後方互換性を維持する。

## 2. 背景 / 現状の課題

-   各ローダークラス (`TransformersLoader`, `ONNXLoader` など) 内で、モデルサイズ計算、キャッシュチェック、メモリチェック、状態更新などの共通ロジックが重複しており、コードの冗長性が高い。
-   `ModelLoad` クラス、各ローダークラス、`BaseModelLoader` 間での責務が分散・重複しており、コードの見通しが悪く、変更時の影響範囲が読みにくい。
-   全体としてコードの分量が多く、可読性・保守性が低い状態にある。

## 3. 方針

1.  **`ModelLoad` 内部の責務分離と共通化:** `ModelLoad` クラス内に、サイズ管理、キャッシュ管理、状態管理、ロード処理実行のための内部ヘルパーメソッドまたは内部クラスを導入し、ロジックを集約する。
2.  **ローダーの内部化:** 各フレームワーク固有のローダークラス (`TransformersLoader` など) を `ModelLoad` の内部実装詳細 (`_TransformersLoader` など) とし、外部から直接参照されないようにする。
3.  **インターフェース維持:** `ModelLoad` の既存の静的メソッド (`load_..._components`, `cache_...`, `restore_...` など) はそのまま維持し、内部で新しい構造を呼び出すように修正する。
4.  **依存関係の整理:** `utils` モジュールへの依存を見直し、可能であれば `huggingface_hub` などの標準ライブラリに置き換える。
5.  **コード改善:** CLIPモデルの構造推測ロジックの改善や、`model_factory.py` 内のLinterエラーの修正も行う。
6.  **シンプルさの維持:** ライブラリへの過度な依存や、既存処理への不必要な処理の追加を避け、シンプルで理解しやすいコードを目指す (YAGNI原則)。

## 4. 非目標 (スコープ外)

今回のリファクタリングでは以下の項目は **行わない**:

-   `ModelLoad` クラスの外部インターフェース（静的メソッド）の変更。(`base.py` など呼び出し元への影響を避けるため)
-   新しいモデルタイプ（フレームワーク）のサポート追加。
-   `src/image_annotator_lib/core/base.py` ファイルの変更。
-   推論パフォーマンス自体の最適化。

## 5. テスト方針

リファクタリングによるデグレード（意図しない挙動の変化）を防ぐため、以下のテストを行う:

-   **動作確認:** リファクタリングの各段階および完了後に `example/example_lib.py` を実行し、主要なモデル（Scorer, Tagger, Captioner など、異なるタイプを含む）について、エラーなくアノテーションが実行され、リファクタリング前と同様の結果（タグ、スコア、キャプション等）が得られることを **手動で確認** する。
-   **既存ユニットテスト:** (もしあれば) 既存の関連するユニットテストを実行し、すべてパスすることを確認する。

## 6. リスクと対策

-   **リスク1:** 内部ロジック (特にキャッシュ制御、エラーハンドリング) の変更によるデグレード。
    -   **対策:**
        -   計画に基づき段階的にリファクタリングを実施する。
        -   各段階で `example_lib.py` による動作確認を徹底する。
        -   コードレビューを実施し、ロジックの妥当性を検証する。
-   **リスク2:** 依存関係の変更 (`utils` から `huggingface_hub` など) に伴う予期せぬ問題。
    -   **対策:** 依存ライブラリの変更は影響範囲を慎重に評価し、互換性を確認してから実施する。必要に応じて段階的に導入する。
-   **リスク3:** リファクタリング過程で不必要な複雑化を招く可能性。
    -   **対策:** 方針6「シンプルさの維持」を常に意識し、過剰な共通化や抽象化を避ける。

## 7. リファクタリングフェーズとタスクチェックリスト

### フェーズ 1: `ModelLoad` 内部ヘルパーの準備

-   [ ] `ModelLoad` 内にサイズ管理用の内部ヘルパーメソッド/クラス (`_get_or_calculate_size` 等) を定義する。
-   [ ] `ModelLoad` 内にキャッシュ/状態管理用の内部ヘルパーメソッド/クラス (`_update_model_state`, `_clear_cache_internal` 等) を定義する。
-   [ ] `ModelLoad` 内にデバイス移動用の内部ヘルパーメソッド (`_move_components_to_device`) を定義する。
-   [ ] `BaseModelLoader` のロジックを `ModelLoad` の内部ヘルパーに移管するか、内部クラスとして再定義する。
-   [ ] `ModelLoad` の静的変数 (`_MODEL_STATES` 等) が新しい内部ヘルパー経由でのみアクセスされるようにする。

### フェーズ 2: ローダー実装の内部化とリファクタリング

-   [ ] `TransformersLoader` を `ModelLoad` の内部クラス `_TransformersLoader` として定義する。
-   [ ] `ONNXLoader` を `ModelLoad` の内部クラス `_ONNXLoader` として定義する。
-   [ ] `TensorFlowLoader` を `ModelLoad` の内部クラス `_TensorFlowLoader` として定義する。
-   [ ] `CLIPLoader` を `ModelLoad` の内部クラス `_CLIPLoader` として定義する。
-   [ ] `_TransformersLoader.load_components` を `ModelLoad` の内部ヘルパーを使用するようにリファクタリングする。
-   [ ] `_ONNXLoader.load_components` を同様にリファクタリングする。
-   [ ] `_TensorFlowLoader.load_components` を同様にリファクタリングする。
-   [ ] `_CLIPLoader.load_components` を同様にリファクタリングする。
-   [ ] 共通のエラーハンドリングロジック (`_handle_load_error` など) を内部ヘルパーとして実装し、各ローダーから呼び出す。

### フェーズ 3: `ModelLoad` 静的メソッドのインターフェース維持と内部呼び出し修正

-   [ ] 静的メソッド `ModelLoad.load_transformers_components` の内部実装を `_TransformersLoader` を呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.load_onnx_components` の内部実装を `_ONNXLoader` を呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.load_tensorflow_components` の内部実装を `_TensorFlowLoader` を呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.load_clip_components` の内部実装を `_CLIPLoader` を呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.cache_to_main_memory` の内部実装を内部ヘルパー (`_move_components_to_device`, `_update_model_state` 等) を呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.restore_model_to_cuda` の内部実装を内部ヘルパーを呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.release_model` の内部実装を内部状態ヘルパーを呼び出すように修正する。
-   [ ] 静的メソッド `ModelLoad.release_model_components` の内部実装を内部状態/リソースヘルパーを呼び出すように修正する。

### フェーズ 4: CLIP関連、utils、Linter対応、最終調整

-   [x] `create_clip_model` 関数をリファクタリングする (構造推測を排除し、設定ファイル利用を検討。`ModelLoad` の内部ヘルパー `_create_clip_model_internal` とし、さらに内部ヘルパーメソッドに分割して複雑度を削減)。
-   [ ] `utils` モジュールの関数 (`download_onnx_tagger_model`, `load_file` 等) を `huggingface_hub` 等で代替可能か調査し、可能であれば `ModelLoad` 内部で直接ライブラリを使用する。
-   [ ] `model_factory.py` 内の Linter エラー (例: MyPy 型互換性エラー、TypedDict キー削除エラー) を修正する。
-   [ ] `model_factory.py` 全体の内部クラス/メソッドの命名規則を確認・統一する。
-   [ ] `model_factory.py` 内のログ出力メッセージを見直し、レベルや内容を改善する。
-   [ ] `model_factory.py` 内の型ヒントを見直し、整合性を確保する。
-   [ ] `ModelLoad` クラスおよび内部ヘルパー/クラスの Docstring を更新する。 