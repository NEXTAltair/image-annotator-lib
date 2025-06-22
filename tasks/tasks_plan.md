# タスク計画と進捗トラッカー

## 1. 現状のフォーカスとサマリー (Current Focus and Summary)
- 現在はユニットテストの安定化と主要な型エラーの解消に注力しており、その後BDDステップの再実装に着手予定です。
- ルールとドキュメントの整合性維持も継続的に行っています。

## 2. 詳細タスクリスト･進行状況 (Detailed Task List and Progress)

### 2.1. アクティブなタスク / 残存構築作業 (Active Tasks / Remaining Work)
- [ ] **`src/image_annotator_lib/core/base.py` の分割リファクタリング (詳細手順更新版)**
    - [ ] **目的:** 可読性向上、保守性向上、関心事の分離 (RFC: `tasks/rfc/base_py_split_plan.md` 参照)。
    - [ ] **手順:**
        1.  **準備:**
            - [ ] 新しいGitブランチを作成 (例: `refactor/split-base-py`)。
            - [x] 設計ドキュメント `tasks/rfc/base_py_split_plan.md` を最終確認。
            - [ ] **`src/image_annotator_lib/core/base.py` の既存機能に対するユニットテストを作成・拡充し、カバレッジを向上させる (リファクタリング前の動作保証)。**
                - [ ] **テストファイル構成案:**
                    - [ ] `tests/unit/core/` ディレクトリを作成 (存在しない場合)。
                    - [ ] `tests/unit/core/test_base_annotator.py` を作成。
                    - [ ] 各フレームワーク基底クラスに対応するテストファイルを作成 (例: `tests/unit/core/test_transformers_base.py`, `tests/unit/core/test_onnx_base.py` など)。
                - [ ] **(A) `BaseAnnotator` クラス (`tests/unit/core/test_base_annotator.py`):**
                    - [ ] **初期化 (`__init__`):**
                        - [ ] 必須引数・オプション引数が正しく設定されること。
                        - [ ] `config` 未指定時のデフォルト動作。
                        - [ ] `components` 未指定時のデフォルト動作。
                    - [ ] **抽象メソッドの呼び出し:**
                        - [ ] `_load_model()`, `_run_inference()`, `_format_predictions()`, `_get_model_info()` が直接呼び出された場合に `NotImplementedError` を送出すること。
                    - [ ] **具象メソッド:**
                        - [ ] **`get_model_name()`:** 正しいモデル名を返すこと。
                        - [ ] **`_validate_input()` (静的メソッド):**
                            - [ ] サポートされる全入力パターン (str, list[str], PIL.Image, list[PIL.Image]) のテスト。
                            - [ ] サポートされない型入力時の `TypeError`。
                            - [ ] 空リスト入力時の動作。
                        - [ ] **`predict()`:**
                            - [ ] `_validate_input` が呼び出されること (モックで確認)。
                            - [ ] `_load_model` が呼び出されること (モックで確認、`self.components` が `None` の場合)。
                            - [ ] `_run_inference` が呼び出されること (モックで確認)。
                            - [ ] `_format_predictions` が呼び出されること (モックで確認)。
                            - [ ] `_run_inference` 例外時の適切な処理。
                            - [ ] `_format_predictions` の戻り値の検証 (モック)。
                            - [ ] 空の画像リスト入力時の動作。
                    - [ ] **プロパティ:** 各プロパティが正しい値を返すこと。
                - [ ] **(B) 各フレームワーク基底クラス (例: `TransformersBaseAnnotator` in `tests/unit/core/test_transformers_base.py`):**
                    - [ ] **初期化 (`__init__`):**
                        - [ ] 親クラス `__init__` 呼び出し確認。
                        - [ ] フレームワーク固有引数の設定確認。
                    - [ ] **`_load_model()` (および `__enter__` / `__exit__` でのcomponents管理):**
                        - [ ] モデルロード処理呼び出し確認 (関連ヘルパーやライブラリ呼び出しをモック)。
                        - [ ] ロード成功時に `self.components` が適切な型・内容で設定されること。
                        - [ ] モデルロード失敗時 (ファイル欠損、不正なファイル) のエラー送出。
                    - [ ] **`_run_inference()`:**
                        - [ ] `self.components` が `None` の場合の適切な処理 (エラー or `_load_model` 呼び出し)。
                        - [ ] フレームワーク固有推論関数呼び出し確認 (モック)。
                        - [ ] 入力データの前処理確認。
                        - [ ] 推論結果の型・構造確認。
                    - [ ] **`_format_predictions()`:**
                        - [ ] `self.components` が `None` の場合のエラー送出。
                        - [ ] `RawOutput` から期待される `FormattedOutput` への変換確認。
                        - [ ] エラーを含む推論結果の適切な処理。
                    - [ ] **`_get_model_info()`:**
                        - [ ] `self.components` が `None` の場合のエラー送出。
                        - [ ] フレームワーク固有モデル情報の正しい返却確認。
                - [ ] **(C) `PipelineBaseAnnotator` (特殊ケース):**
                    - [ ] `transformers.pipeline` セットアップ・利用テスト。
                    - [ ] `task` 引数処理テスト。
                - [ ] **(D) `WebApiBaseAnnotator` (特殊ケース):**
                    - [ ] `api_client` 初期化・利用テスト。
                    - [ ] `_run_inference` でのAPI呼び出しテスト (モック)。
                    - [ ] APIレスポンス処理テスト。
        2.  **`core/types.py` の更新:**
            - [ ] `base.py` から以下の型定義を `core/types.py` に移動または新規作成:
                - `AnnotationResult`
                - `TagConfidence`
                - `TransformersComponents`
                - `TransformersPipelineComponents`
                - `ONNXComponents`
                - `TensorFlowComponents`
                - `CLIPComponents`
                - `LoaderComponents` (上記Components型のUnion)
            - [ ] `ModelComponents` 型の定義を `base.py` から削除 (廃止)。
        3.  **新しい `base` ディレクトリと基底クラスファイルの作成:**
            - [ ] `src/image_annotator_lib/base/` ディレクトリを新規作成。
            - [ ] `src/image_annotator_lib/base/__init__.py` を作成し、RFC通りに各基底クラスを re-export。
            - [ ] `src/image_annotator_lib/base/annotator.py` を新規作成:
                - `BaseAnnotator` クラス定義を移動。
                - `self.components` の型ヒントを `LoaderComponents | None` (from `core.types`) に修正。
                - 必要なインポートを追加。
            - [ ] 以下各フレームワーク基底クラスを `src/image_annotator_lib/base/` 配下の新ファイルに移動・修正:
                - `transformers.py` -> `TransformersBaseAnnotator` (components: `TransformersComponents | None`)
                - `tensorflow.py` -> `TensorflowBaseAnnotator` (components: `TensorFlowComponents | None`)
                - `onnx.py` -> `ONNXBaseAnnotator` (components: `ONNXComponents | None`)
                - `clip.py` -> `ClipBaseAnnotator` (components: `CLIPComponents | None`)
                - `pipeline.py` -> `PipelineBaseAnnotator` (components: `TransformersPipelineComponents | None`)
                - `webapi.py` -> `WebApiBaseAnnotator` (components: `WebApiComponents | None`)
                - 各ファイルで必要なインポートを解決。
        4.  **`core/model_factory.py` の修正:**
            - [ ] `ModelLoad` クラスメソッドの戻り値型ヒントを `core/types.py` の型に確認・修正。
            - [ ] `restore_model_to_cuda`, `cache_to_main_memory` の `components` 引数型を検討。
        5.  **インポートパスの修正:**
            - [ ] `src/image_annotator_lib/model_class/` 配下の具象クラスのインポート文を修正。
            - [ ] `src/image_annotator_lib/api.py` のインポート文を修正。
            - [ ] `src/image_annotator_lib/__init__.py` のインポート文を修正。
            - [ ] その他、コードベース全体で古い `core.base` を参照している箇所を修正。
        6.  **旧 `core/base.py` のクリーンアップ:**
            - [ ] `src/image_annotator_lib/core/base.py` を削除または内容を空に。
        7.  **Linterチェックと型チェック:**
            - [ ] `ruff format .` を実行。
            - [ ] `ruff check . --fix` を実行。
            - [ ] `mypy src/` を実行し、型エラーを修正 (RFCのLinter対応方針参照)。
        8.  **テストの実行と修正:**
            - [ ] `pytest` を実行。
            - [ ] テストコード内の古いパス参照を修正 (必要な場合)。
            - [ ] **リファクタリング前に作成・拡充したユニットテストを含め、全てのテストがパスすることを確認 (互換性担保)。**
        9.  **ドキュメント更新:**
            - [ ] `docs/architecture.md`, `docs/technical.md` を更新。
            - [ ] `tasks/rfc/base_py_split_plan.md` のステータスを更新。
            - [ ] この `tasks/tasks_plan.md` のタスクステータスを更新。
- [ ] **ユニットテストの全パス:** BDDステップ定義再実装の前提として、既存の全ユニットテストをパスさせる。
- [ ] **BDD ステップ定義の再実装 (Featureファイルは残存)**
  - 既存のFeatureファイルに基づき、品質の高いステップ定義を一から再実装する。
- [ ] **型エラーの解決** - `core/base.py` (`WebApiBaseAnnotator`)
- [ ] **テストリソース クリーンアップ実装 (Phase 2-3)**
  - [ ] フェーズ 3: テストと検証
    - [ ] `uv run pytest` を実行し、テストが正常に完了することを確認。
    - [ ] クリーンアップ処理が意図通り機能しているか確認 (例: キャッシュファイルが適切に削除/復元されるか)。
    - [ ] テスト実行時間やリソース使用量に悪影響がないか確認。
- [ ] **ルール･ドキュメントの再読と更新 (進行中)**
- [ ] **WebAPIアノテーターのテスト安定化 (残作業)**
    - [ ] テスト全体の再実行･安定化確認
    - [ ] ドキュメント間の整合性最終チェック
- [ ] **Google Gemini API･テスト設計関連タスク (残作業)**
    - [ ] テストコードの設計･責務の再確認と修正
    - [ ] テスト･実装･設定ファイルの整合性確認
    - [ ] 必要に応じてdocs/architecture.md, docs/technical.mdも更新
    - [ ] 異常系テストの網羅性強化
- **[進行中] Web API アノテーター (`annotator_webapi`) のリファクタリング (PydanticAI 導入準備)**
    - [済] `core/types.py` を導入し、共通の型定義 (WebApiComponents, AnnotationSchema, RawOutput, WebApiFormattedOutput, WebApiInput など) を集約。
    - [済] `_run_inference` の戻り値を `list[RawOutput]` (`response: AnnotationSchema | None`, `error: str | None`) に統一。
    - [済] `_format_predictions` を `WebApiBaseAnnotator` に共通実装し、戻り値を `list[WebApiFormattedOutput]` (`annotation: dict | None`, `error: str | None`) に統一。
    - [済] 各サブクラス (`google_api.py`, `anthropic_api.py`) から `_format_predictions` を削除。
    - [済] 関連テストコード (`test_google_api.py`, `test_anthropic_api.py`) を修正し、パスを確認。
    - [済] `tasks/rfc/pydanticai_integration_plan.md` を更新。
    - [済] `tasks/active_context.md` を更新。
    - [TODO] PydanticAI 対応に向けた改修（進行中）
        - [TODO] LLMレスポンスの基本的なパース処理をPydanticAIベースで試行。
        - [TODO] 関連する一部データ構造のPydanticモデル定義に着手。
        - [TODO] ユニットテストの一部をPydanticAI対応に合わせて更新開始。
    - [TODO] `core/base.py` の `self.components` 周りの型エラーを解消する。
    - [TODO] PydanticAI の Agent/tool 等の導入を本格的に検討･実装する。
    - [TODO] 未対応箇所のPydanticAIへの移行。
    - [TODO] 関連するユニットテストの拡充と全体的な検証。
    - [TODO] 関連ドキュメント (`architecture.md`, `technical.md`, `lessons-learned.mdc` 等) にPydanticAI導入の進捗と変更内容を反映させる。

### 2.2. バックログ / 今後の展望 (Backlog / Future Outlook)
- **ドキュメント･コード整理:**
    - [ ] ドキュメント整合性チェック･更新:
        - [ ] 過去指摘された矛盾点 (`error-documentation.mdc` セクション2) の解消状況確認
        - [ ] 全ドキュメントのリンク切れ･古情報の確認･修正
        - [ ] `available_api_models.toml` のパス表記統一
        - [ ] 全体的なドキュメントレビューと必要に応じた更新 (継続)
    - [ ] コーディングルール遵守確認･修正:
        - [ ] モダンな型ヒント使用状況の確認･修正
        - [ ] カプセル化原則遵守状況の確認･修正 (特に他クラス内部変数アクセス)
        - [ ] エラーハンドリング方針遵守状況の確認･修正
        - [ ] 半角文字ルール遵守状況の確認･修正
        - [ ] 命名規則 (`*er` 回避など) 遵守状況の確認･修正
        - [ ] `Any` 型使用状況の確認･理由コメント追加 or 型特定
        - [ ] `predict()` オーバーライド禁止ルールの徹底確認
        - [ ] 違反箇所の特定とリファクタリング計画 (必要に応じて)
    - [ ] テストコード整理:
        - [ ] pytest-bdd 日本語+`<param>` 問題の該当箇所特定と対応 (無視/英語化/回避)
- **既存タスク (継続):**
    - [ ] テスト拡充 (BDDステップ再実装後):
        - **単体テスト:**
            - [ ] `ModelLoad` のキャッシュ戦略(LRU、CPU 退避、メモリ逼迫時の挙動)を検証するテストケースを追加。
            - [ ] 各フレームワーク別ローダー (`TransformersLoader`, `ONNXLoader` 等) の正常系･異常系テストを追加。
            - [ ] `registry.py` の動的クラス検出･登録ロジックのテストを追加。
            - [ ] 各具象モデルクラス (`models/` 配下) の主要メソッドに対するテストを追加･拡充。
        - **結合テスト:**
            - [ ] BDDステップ再実装後、`api.annotate` 関数のテストシナリオを拡充 (複数モデル同時実行、pHash 生成失敗ケース、エラーハンドリング、Web API モデルなど)。
            - [ ] 各モデルのエラーハンドリング (特に OOM, APIエラー) が `annotate` 関数レベルで適切に処理され、結果に反映されるか検証するシナリオを追加。
            - [ ] 設定ファイル (`annotator_config.toml`) の様々なパターン (オプション指定有無など) に対するテストを追加。
        - **カバレッジ:** `pytest --cov` を実行し、テストカバレッジを確認･向上させる。
    - [ ] ドキュメント最終化:
        - [ ] `docs/` 内の各ドキュメント (API リファレンス、設定ガイド、モデル説明など) の内容を、最新のコード実装と完全に一致するように最終レビュー。
        - [ ] 全公開クラス･メソッド･関数の Docstring の網羅性と正確性を確認･修正。
        - [ ] 必要に応じてアーキテクチャ図などを `docs/` 内に追記･更新。
        - [ ] 日本語 README (`README-JP.md`) を作成または更新。
    - [ ] 依存関係の最終整理:
        - [ ] `pyproject.toml` の依存関係 (`dependencies`, `dev-dependencies`) を最終確認し、不要なライブラリがあれば削除。バージョン指定が適切か確認。
    - [ ] 実環境での動作確認:
        - [ ] ライブラリを実際に使用する環境 (例: LoRAIro) で `annotate` 関数を呼び出し、主要なモデルが問題なく動作するか確認。パフォーマンスやメモリ使用量についても簡易的に確認。
    - [ ] 静的解析エラーの完全解消:
        - [ ] `ruff check` および `mypy` を実行し、報告されるエラーや警告が完全に解消されていることを確認。(# type: ignore や # noqa が残っていないか確認)
    - [ ] 設定ファイル (`annotator_config.toml`) の構造改善検討 (将来対応):
        - [ ] 現在の `annotator_config.toml` の構造は読みにくいため、将来的に改善を検討する。具体的な改善案については別途相談する。

## 3. 完了事項 (Completed Tasks)
- [x] ModelLoad のリファクタリングとメモリ管理ロジックの改善 (初期実装完了, 2024-04-02 -> 2025-04-05)
- [x] Memory Bank 更新 (当時の状況記録, 2025-04-05)
- [x] 旧メモリファイルから推奨コンテキストファイルへの情報移行 (2025-04-29)
- [x] 設定ドキュメント (V2) に基づくファイル構造･ドキュメント整理 (主要部分完了, 旧コンテキスト構造削除, `docs/` 内不要ファイル削除, 2025-04-30)
- [x] docs/ディレクトリのドキュメント整理･.gitignoreの見直し･models/ディレクトリ運用方針の明確化(2025-04-30)
- [x] BDDステップ定義ファイルおよび関連conftest.pyの削除 (2025-05-XX)
- [x] ドキュメント / ルールファイルの整合性チェック (関連部分完了 2025-05-07)
- [x] テストクリーンナップタスク完了 (2025-05-04)
- [x] WebAPIアノテーターのテスト安定化タスク (初期対応完了 2025-05-08: 初期化ロジック修正、TOML反映処理追加、記述ミス修正、型･属性アクセス誤り修正)
- [x] Google Gemini API･テスト設計関連タスク (初期対応完了 2025-05-08: google-genai バージョンアップ対応、エラーハンドリング･テスト設計見直し、SDK起因問題解消確認、関連ドキュメント更新)
- [x] テストリソースクリーンアップ戦略の策定と実装 (フェーズ 1 および フェーズ 2 完了)
    - [x] フェーズ 1: 副作用特定と要件定義
    - [x] フェーズ 2: クリーンアップ実装 (`reset_model_load_state`, `manage_api_cache_file` フィクスチャ実装)

## 4. 既知の問題点 (Known Issues)
- (ここに、純粋な未解決の問題点やバグを記載)
- (例: WebAPIアノテーターのテストで稀にタイムアウトが発生する (原因調査中))

## [2025-05-10] Google Gemini annotator レスポンス型･エラーハンドリング設計変更

### 完了タスク
- google_api.py のレスポンス型を WebApiFormattedOutput (annotation: dict[str, Any] | None, error: str | None) に統一
- スキーマ不一致･APIエラー時のエラー格納設計実装
- テスト･型定義の修正
- ドキュメント(technical.md, architecture.md, active_context.md)への記録

### 今後のタスク
- 他WebAPIアノテーターへの同様の設計適用(必要に応じて)
- 設計方針･ドキュメントの定期的な見直し

## 2025-05-10 OpenAIApiAnnotator関連タスク

- OpenAI API画像入力(base64)はimage_url: dict型で渡すよう修正
- 型エラー(ImageURL型)を辞書型指定で解消
- 構造化出力モデルをAnnotationSchema(webapi_shared.py)に統一
- _run_inference/_format_predictionsの型安全･エラーハンドリングを整理
- ユニットテスト(test_openai_api_response.py)を追加し、正常系･異常系･API例外を網羅

## 2025-05-10 AnthropicApiAnnotator関連タスク(テスト用ToolUseBlockクラス名修正･型判定整理)

- テスト用ダミークラスのクラス名をToolUseBlockに合わせ、type(obj).__name__ == "ToolUseBlock" の判定に合致させることでテストがパス。
- _format_predictionsでAnnotationSchema型を許容し、APIレスポンスの型安全性･一貫性を向上。
- これにより、Anthropic/Claude系APIの構造化出力テストが全てパス。

## 2025-05-10 annotator_webapi.py から OpenAIApiAnnotator･AnthropicApiAnnotator 分離タスク

- annotator_webapi.py から OpenAIApiAnnotator を openai_api_response.py へ、AnthropicApiAnnotator を anthropic_api.py へ分離
- 分離に伴い、型定義･エラーハンドリング･テストを整理
- 共通スキーマ(AnnotationSchema)は webapi_shared.py に集約
- テスト用ダミークラスのクラス名･型判定ロジックを実装と一致させ、テストの信頼性を担保
- テスト全パスを確認

### 今後のタスク
- 他API(Google, OpenRouter等)も同様の分離･整理を検討
- ドキュメント･設計方針の定期的な見直し

### 2024/05/10 OpenRouterApiAnnotator テスト･設計･ドキュメント反映
- テストカバレッジ向上･異常系網羅のためのテスト設計･修正を実施。
- テスト通過後、設計･技術ドキュメントも最新状態に更新。
- AnnotationSchemaによる型安全化･エラーハンドリング明確化を反映。

### BDDテストスイートの安定化とPydantic導入準備 (2025-05-13)

- **完了:**
    - Web API連携テストにおけるタイムアウト処理のシミュレーション方法を改善 (`webapi_annotate_steps.py`)
        - `google-genai` SDK利用時のタイムアウトを `RuntimeError` でシミュレートするように修正。
    - APIエラーレスポンス時のテストステップにおけるアサーションを強化 (`webapi_annotate_steps.py`)
        - エラーメッセージの検証を小文字化比較に変更し、安定性を向上。
    - Linter (Mypy/Ruff) による型エラーおよびコーディングスタイル違反の修正 (`webapi_annotate_steps.py`)
        - ロガーの型不整合を解消。
        - 不要なインポートの整理。
    - `tasks/rfc/pydanticai_integration_plan.md` の更新。
        - `types.py` 作成経緯を追記。
        - BDDテスト修正(APIキー未設定、タイムアウト、エラーレスポンス)の記録を追記。
- **進行中:**
    - `pydanticai_integration_plan.md` に基づくPydanticモデルの設計と既存コードへの適用。
    - Web APIクライアントライブラリ (`annotator_webapi/`) のリファクタリング(PydanticAI適用を見据えた準備)。
- **残課題:**
    - (特になし。Pydanticモデル導入後に新たな課題が発生する可能性あり)

---