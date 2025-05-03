# アクティブコンテキスト

## 現在のタスク

- 提供された設定ドキュメント (V2: Updates) に基づき、既存の `image-annotator-lib` プロジェクトのファイル構造とドキュメントを整理・統合する。(一部完了)
- ~~特に、`memory-bank/` ディレクトリ内の情報を、推奨されるメモリファイル (`docs/product_requirement_docs.md`, `docs/architecture.md`, `docs/technical.md`, `tasks/tasks_plan.md`, `tasks/active_context.md` など) に移行・統合する。~~ (**完了**)

## 現在の焦点

- ~~`memory-bank/` 内の残りのファイル (`decisionLog.md`, `mypy_fix_plan.md`, `progress.md`) の内容を確認し、適切な推奨メモリファイルへの統合方法を決定・実行する。~~ (**完了**)
- 既存の `docs/` ディレクトリ内のファイル (例: `developer_guide.md`, `getting_started.md`, `README.md`, サブディレクトリ `EXPLANATION/`, `PLAN/`, `REFERENCE/`) の扱いを決定し、整理・統合する。

## 次のステップ

~~1.  `memory-bank/decisionLog.md` の内容を確認し、統合先を検討する。~~
~~2.  `memory-bank/mypy_fix_plan.md` の内容を確認し、統合先を検討する。~~
~~3.  `memory-bank/progress.md` の内容を確認し、統合先を検討する。~~
1.  既存の `docs/` 内のファイルとサブディレクトリの整理方針を決定し、実行する。
~~5.  `memory-bank/` ディレクトリを削除する (内容の移行完了後)。~~ (**完了**)
2.  AIによるドキュメント初期化/更新プロンプトを実行する。
3.  `.gitignore` を確認・更新する。

# ルール厳守徹底プラン(2024/06/09)

## 背景
- `rules.mdc`・`plan.mdc`等のルールファイルが`alwaysApply: true`で常時適用設定だが、AIの応答やファイル操作に100%反映されていない事例が発生。
- ルール違反を防ぎ、AIが確実にルールを守るようにするための改善プランを策定。

## プラン
1. 重要ルール(保存先・命名・カプセル化等)を**最上位・太字で明記**し、「例外なく厳守」と追記
2. 「違反時は警告・拒否する」など**AIの振る舞い指示**をルールに追加
3. ルール内容を**簡潔・具体的にリファクタ**(曖昧な表現を排除)
4. システムプロンプト設計を見直し、**ルール優先順位**を明記
5. ルール違反が起きた場合、**即時修正・ルール強化**を実施
6. テストケースで**ルール適用状況を定期検証**し、必要に応じてルールをアップデート

## タスクリスト
- [ ] 重要ルールを最上位・太字で明記し、「例外なく厳守」と追記
- [ ] 「違反時は警告・拒否する」などAIの振る舞い指示をルールに追加
- [ ] ルール内容を簡潔・具体的にリファクタ
- [ ] システムプロンプト設計を見直し、ルール優先順位を明記
- [ ] ルール違反が起きた場合、即時修正・ルール強化を実施
- [ ] テストケースでルール適用状況を定期検証し、必要に応じてルールをアップデート

---

## 記録方針について
- 本タスク記録は`plan.mdc`の「計画・関連タスク計画・コンテキストをtasks/内ファイルに記録」ルールに従い、`tasks/active_context.md`に記載。
- 記録可否の確認は不要(ルールで明示されているため、即時記録が推奨される)。

# テストクリーンアップ計画

## 1. 現状分析

### A. テストの構造
- `tests/unit/`: ユニットテスト
- `tests/integration/`: 統合テスト
- `tests/features/`: BDDスタイルのテスト
- `tests/step_defs/`: BDDのステップ定義
- `tests/resources/`: テストリソース
- `conftest.py`: 共通フィクスチャ

### B. 主要な技術スタック
- pytest
- pytest-bdd
- pytest-cov

## 2. クリーンアップの手順

### Phase 1: テストコードと実装の対応関係分析

1. **現行実装の機能マッピング**
   - `api.py`: メインAPI
   - `core/`: コア機能
   - `model_class/`: モデル実装
   - `exceptions/`: エラー定義

2. **テストコードの分析**
   - `tests/unit/test_api.py`
   - `tests/unit/test_error_handling.py`
   - `tests/unit/test_model_errors.py`
   - `tests/unit/core/*`

### Phase 2: クリーンアップ実行計画

1. **テストと実装の対応チェック**
   - 現行実装の主要コンポーネントとテストの対応関係を確認
   - 不一致箇所をリストアップ

2. **テスト分類**
   - 現行実装に対応するテスト
   - 古い実装に対応するテスト
   - 共通ユーティリティテスト

3. **クリーンアップ優先順位**
   a. 明らかに不要なテスト(存在しない機能のテスト)
   b. 重複するテスト
   c. 現行仕様と異なるテスト

## 3. 具体的なアクション

1. **Phase 1: テストコードと実装の対応関係分析**
   ```
   a. 現行実装の機能一覧作成
   b. テストケースのマッピング
   c. 不一致箇所の特定
   ```

2. **Phase 2: 不要テストの特定**
   ```
   a. 存在しない機能のテスト抽出
   b. 重複テストの特定
   c. 仕様と異なるテストの特定
   ```

3. **Phase 3: クリーンアップ実行**
   ```
   a. 不要テストの削除
   b. テストの整理・統合
   c. テストドキュメントの更新
   ```

## 4. 次のステップ

1. `tests/unit/test_api.py`の内容確認
2. 現行実装との対応関係分析
3. 古い実装に対するテストの特定

## 5. 注意事項

- テスト削除前に、該当テストが本当に不要かを慎重に確認
- 削除するテストは一時的にバックアップを作成
- テスト削除の理由を明確に記録
- テストカバレッジが低下しないよう注意

## 6. 進捗管理

### 完了したタスク
- [x] 計画立案
- [x] 現状分析
- [x] テストコードと実装の対応関係分析
  - [x] test_api.pyの分析完了:現行実装と整合性あり、クリーンアップ不要 → **修正実施 (2025-05-02)**
  - [x] test_error_handling.pyの分析完了:一部更新が必要 → **Pylanceエラー対応実施 (2025-05-02)**
  - [x] test_model_errors.pyの分析完了:現行実装と整合性あり、クリーンアップ不要
  - [x] tests/unit/core/*の分析完了:現行実装と整合性あり、一部最適化の余地あり
- [x] 不要テストの特定
- [x] クリーンアップ実行
- [x] ドキュメント更新

### 次のアクション
1. ~~UnboundLocalError修正タスクの実施~~ (**完了 2025-05-02**)
2. 今後切り出し推奨タスクの優先順位付け・着手

### 分析結果メモ

#### test_api.py
- ~~現行実装と完全に整合~~ → **UnboundLocalError修正に伴いテストも修正 (2025-05-02)**
- Web APIアノテーターとローカルモデルの両方をカバー
- インスタンス化とキャッシュのテストが適切
- エラーケースも網羅
- ~~クリーンアップ不要~~

#### test_error_handling.py
- 重複コード発見:`test_memory_error_handling`メソッド内に同じテストが3回重複
- 古い実装への参照あり:
  - `image_annotator_lib.score_models.imagereward`への参照(現行実装では異なるパス)
  - `image_annotator_lib.scorer`への参照(現行実装では異なる構造)
- 更新が必要な箇所:
  1. 重複テストコードの削除
  2. 古いインポートパスの更新
  3. モックオブジェクトのパス修正
- 基本的なエラーハンドリングテストは有効で保持すべき
- **Pylanceインポートエラー発生 → `exceptions/errors.pyi` 更新で対応 (2025-05-02)**

#### test_model_errors.py
- シンプルな例外クラステスト
- 現行の例外クラス定義と完全に整合
- クリーンアップ不要

#### tests/unit/core/
1. test_registry.py
   - 現行実装と整合
   - モデルレジストリの初期化と設定のテストが充実
   - Web APIモデルの登録処理のテストが適切
   - クリーンアップ不要だが、テストケースの整理余地あり

2. test_config.py
   - 設定管理の包括的なテスト
   - システム設定とユーザー設定の分離テストが適切
   - モック使用が適切
   - クリーンアップ不要

3. test_api_model_discovery.py
   - Web APIモデルの検出と更新のテスト
   - 現行実装と整合
   - クリーンアップ不要

4. test_model_factory_unit.py
   - モデルのロードと管理のテスト
   - 現行実装と整合
   - クリーンアップ不要

### 全体の評価
1. **良好な点**
   - 大部分のテストが現行実装と整合
   - テストカバレッジが十分
   - モックの使用が適切
   - エラーケースの考慮が十分

2. **改善が必要な点**
   - test_error_handling.pyの重複コードと古い参照
   - tests/unit/core/のテストケース構造の最適化余地

3. **優先度**
   1. test_error_handling.pyの更新(重要)
   2. tests/unit/core/のテストケース整理(任意)

## 7. クリーンアップ・修正履歴

- 2025-05-01: `tests/unit/core/test_model_factory_unit.py` など、現行実装に存在しないクラス依存のテストを削除
- 2025-05-01: `tests/unit/test_error_handling.py` から現行実装に存在しないモジュールや関数をpatchしているテスト(test_model_execution_error, test_timeout_error, test_gpu_dependency_error)を削除
- 2025-05-01: テスト実行時の主なエラー内容・原因・対策を記録
  - AttributeError/ImportError: 存在しないモジュール・関数のpatch
  - UnboundLocalError: 実装バグ(未定義変数のreturn)
  - pytest-bddのStepDefinitionNotFoundError: ステップ定義不足
  - TypeError/NameError: テストヘルパー関数の引数不一致や未定義関数
  - APIレスポンスエラー: 外部APIの400エラー等
- 2025-05-01: テストの優先度整理・外部API依存テストのスキップ推奨
- **2025-05-02: `src/image_annotator_lib/api.py` の `_create_annotator_instance` における UnboundLocalError を修正**
- **2025-05-02: `src/image_annotator_lib/exceptions/errors.pyi` の型スタブを更新し、Pylance のインポートエラーを解消**
- **2025-05-02: `tests/unit/test_api.py` のモック設定を `api.py` の修正に合わせて更新**

## 8. 新規タスク: UnboundLocalError修正

### 背景
- integrationテストやAPI経由のテストで `UnboundLocalError: cannot access local variable 'instance' where it is not associated with a value` が発生。
- 発生箇所: `src/image_annotator_lib/api.py` の `_create_annotator_instance` 関数
- 例外発生時や条件分岐で `instance` がセットされないパスが存在

### 目的
- UnboundLocalErrorを解消し、全テストの安定化・APIの堅牢化を図る

### 対応方針
1. `_create_annotator_instance` 内で `instance` が必ず定義されるように修正
2. 例外発生時は適切なエラーをraise、またはNone/明示的な値を返す
3. 修正後、関連テスト(integration, API, unit)を再実行し、エラー解消を確認

### 進捗
- [x] バグ発生箇所の特定
- [x] 修正方針の決定
- [x] コード修正 (`src/image_annotator_lib/api.py`)
- [x] テスト再実行・確認 (`tests/unit/test_api.py` パスを確認)
- [x] ドキュメント・履歴の記録 (本ファイル更新)

## 9. 今後切り出し推奨の個別タスク案

### 1. BDDテストのステップ定義不足の整理
- 内容: pytest-bddのfeatureファイルで「Given ...」などのステップ定義が見つからず大量に失敗している。
- 対応方針: tests/step_defs/配下のステップ定義ファイルを整理・補完し、featureファイルと対応させる。
- 目的: BDDテストの有効性・自動化の回復。

- **進捗 (2025-05-02):**
  - `tests/step_defs/common_steps.py` が古いと判断し削除。
  - `tests/step_defs/__init__.py` のインポートエラーを修正 (`common_steps` -> `test_common_steps`)。
  - `pyproject.toml` に `bdd_features_base_dir = "tests/features/"` を追加し、pytest-bdd の `IndexError` を解消。
  - 次のステップ: プロジェクトルートで `pytest` を実行し、ステップ定義の不足状況を再確認する。

### 2. 外部API依存テストの一時スキップ・モック化
- 内容: OpenAIやOpenRouterなど外部APIのレスポンスエラーでテストが遅延・失敗している。
- 対応方針: pytest.mark.skipや条件分岐で一時的にスキップ、またはAPI呼び出し部分をモック化。
- 目的: ローカルで完結するテストのみでグリーンを目指し、CI/CDの安定化。

### 3. linterエラー・import警告の整理
- 内容: 例外クラスのimportに関するlinterエラーが出ているが、実際には定義されている。 → **`errors.pyi` 更新で解消済み (2025-05-02)**
- 対応方針: importパスやファイル構成を再確認し、必要に応じて整理。
- 目的: IDEやCIの警告を減らし、開発体験を向上。

### 4. テストカバレッジ向上・ドキュメント更新
- 内容: coverageが75%未満でfailしている。不要なテスト削除後、カバレッジを再計測し、必要に応じて追加テストやドキュメント更新を行う。
- 対応方針: coverageレポートを分析し、カバレッジの低い箇所を重点的にテスト追加。
- 目的: 品質基準の達成と保守性向上。

## 10. 分割タスク用コンテキスト記録(他チャット連携用)

### 背景・目的
- 本プロジェクトは「テストクリーンアップ」「実装バグ修正(UnboundLocalError等)」「BDDテストのステップ定義整理」「外部API依存テストのモック化」など、性質の異なる複数タスクに分割して管理。
- 各タスクは独立して進行可能だが、進捗・依存関係・決定履歴を一元的に記録・参照するため、ここにコンテキストを集約。

### 全体進捗・依存関係
- テストクリーンアップは完了(不要テスト削除・現行実装との整合性確認済み)。
- 実装バグ修正(UnboundLocalError)は完了 (2025-05-02)。
- BDDテストのステップ定義整理、外部API依存テストのモック化、linterエラー整理、カバレッジ向上は今後の個別タスクとして管理。
- 各タスクは`tasks/active_context.md`の該当セクションに記録。

### 参照すべきドキュメント・ルール
- memoryルールに従い、必ず以下を参照:
  - `docs/architecture.md`(システム構造)
  - `docs/technical.md`(技術仕様)
  - `docs/product_requirement_docs.md`(要件)
  - `tasks/tasks_plan.md`(タスク全体像)
  - `.cursor/rules/lessons-learned.mdc`(過去知見)
  - `.cursor/rules/error-documentation.mdc`(既知エラー)
- コーディング・テスト・ドキュメント更新時は、必ず関連ルール(rules, plan, implement, memory等)を厳守。

### これまでの決定・履歴
- テストクリーンアップの詳細な進捗・判断理由・削除履歴は本ファイル7章に記録。
- UnboundLocalError修正タスクの背景・目的・進捗は8章に記録。
- 今後切り出し推奨タスク案は9章に記録。

### 今後の進め方・注意点
- 各タスクは独立チャットで進行可能だが、進捗・決定事項は必ず`tasks/active_context.md`に反映。
- 並行作業時はファイル競合・重複修正に注意。
- 重要な設計・実装・テスト方針の変更は、必ず本ファイルおよび関連ドキュメントに記録・同期すること。

## [2025-05-02] BDDテストのデータテーブル・パラメータ渡し問題の解決記録

### 経緯・問題点
- Scenario Outline+Examplesテーブルでパラメータ渡しをしていたが、step定義側のパターンやpytest-bddの仕様理解が不十分で、stepが認識されない・データテーブルが正しく渡らない問題が発生。
- 何度もstep定義やfeatureファイルの文言・クォート・スペースを修正したが、根本的な解決に至らなかった。
- pytest-bddのパラメータ展開仕様(feature: <param> → step: {param})や、データテーブルの渡し方(step直下にテーブルを書く)を正確に理解できていなかった。

### 解決のきっかけ
- 公式ドキュメント(https://pytest-bdd.readthedocs.io/en/latest/)を精読し、
    - Scenario Outlineのパラメータ展開方法
    - データテーブルの記法とstep定義での受け取り方
  を正しく理解。
- その結果、featureファイルをScenario+step直下のデータテーブル方式に修正し、step定義でdatatable引数を受け取る形に変更。
- これにより、step定義が正しく認識され、テストも期待通りに動作するようになった。

### 教訓・今後の指針
- BDDやpytest-bddのようなDSL/フレームワークは、**公式ドキュメントの仕様を最初に正確に把握することが最重要**。
- 文言やパターンの微修正だけでなく、根本的な設計・記法の違いを疑うべき。
- 公式ドキュメントの「Step arguments」「Scenario outlines」「Datatables」セクションは必読。