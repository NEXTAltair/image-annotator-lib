# アクティブコンテキスト

## 1. 現作業焦点 (Current Focus)
- Linterエラーおよび型エラーの完全解消（特に `annotator_webapi.py` 周辺）。
- ユニットテストの全パス達成とテストカバレッジ75%以上維持・向上。

## 2. 進行中の主要課題・決定事項 (Ongoing Key Issues and Decisions)
- **BDDテスト戦略:**
    - 現状: ステップ定義ファイルおよび関連 `conftest.py` は削除済み。Featureファイルのみ再実装前提で残存。
    - 方針: ユニットテストとインテグレーションテストに注力。BDDステップは将来的に高品質なものを再実装。
- **テスト方針:**
    - ユニットテストおよびインテグレーションテストを中心に品質を確保 (pytest使用)。
    - モックやダミーの使用は、ユニットテストの範囲に限定する。
- **コアロジック・API設計:**
    - `docs/architecture.md` および `docs/technical.md` に準拠。
- **設計・ルール準拠:**
    - `.cursor/rules/` 配下、および `docs/rules.md` に記載のルール・設計方針を厳密に遵守。
- **ドキュメント管理:**
    - 全てのルールファイルおよびドキュメントは、常に最新の状態を反映するように維持・管理する。

## 3. 最近の主要な変更点 (Recent Key Changes)
- テスト戦略の大幅見直し：BDDステップ定義を全て削除し、Featureファイルのみ残存。
- テストのクリーンアップ作業完了。
- 主要な実装バグ（UnboundLocalError等）の修正完了。
- `tasks/tasks_plan.md` のフォーマットを `memory.mdc` ルールに基づき修正。

## 4. 次のステップ (Next Steps)
- Linterエラーおよび型エラーの完全な解消。
- 全てのユニットテストをパスさせる。
- テストカバレッジの目標値達成と維持。
- (将来的に) Featureファイルに基づき、品質の高いBDDステップ定義を再実装する。
- 全てのタスク進捗および意思決定は、本ファイルを含む関連ドキュメントに即時反映する。
- 複数人での並行作業を想定し、ファイル競合や修正内容の重複に注意する。
- 重要な方針変更時は、影響範囲を考慮し、関連ドキュメント全てを更新する。

## 5. 参照すべき主要ドキュメント (Key Documents to Refer To)
- `docs/architecture.md` (システムアーキテクチャ)
- `docs/technical.md` (技術仕様、開発環境)
- `docs/product_requirement_docs.md` (製品要求仕様)
- `tasks/tasks_plan.md` (全体タスク計画、バックログ)
- `.cursor/rules/lessons-learned.mdc` (過去の教訓、知見)
- `.cursor/rules/error-documentation.mdc` (既知のエラーと解決策)

## [2025-05-10] Google Gemini annotator レスポンス型・エラーハンドリング設計変更

### 現状
- google_api.py のレスポンス型を WebApiFormattedOutput (annotation: dict[str, Any] | None, error: str | None) に統一済み。
- スキーマ不一致・APIエラー時も error に詳細を格納し、annotationはNoneで返す設計。
- テスト・型定義もこの設計に合わせて修正済み。

### 決定事項
- annotation/errorペア型の全WebAPIアノテーターでの統一運用。
- 型重複(Responsedict等)の排除。
- annotationはdict型、_format_predictionsでAnnotationSchemaへ変換。

### 次ステップ
- 他WebAPIアノテーターへの同様の設計適用(必要に応じて)
- ドキュメント・設計方針の定期的な見直し

## 2025-05-10 OpenAIApiAnnotator変更経緯

- OpenAI API画像入力（base64）はimage_url: dict型で渡す必要があることを公式ドキュメント・SDK型定義で再確認。
- 型エラー（ImageURL型）を辞書型指定で解消。
- 構造化出力モデルをAnnotationSchema（webapi_shared.py）に統一。
- _run_inference/_format_predictionsの型安全・エラーハンドリングを整理。
- ユニットテスト（test_openai_api_response.py）を追加し、正常系・異常系・API例外を網羅。

## 2025-05-10 AnthropicApiAnnotator変更経緯（テスト用ToolUseBlockクラス名修正・型判定整理）

- テスト用ダミークラスのクラス名をToolUseBlockに合わせ、type(obj).__name__ == "ToolUseBlock" の判定に合致させることでテストがパス。
- _format_predictionsでAnnotationSchema型を許容し、APIレスポンスの型安全性・一貫性を向上。
- これにより、Anthropic/Claude系APIの構造化出力テストが全てパス。

## 2025-05-10 annotator_webapi.py から OpenAIApiAnnotator・AnthropicApiAnnotator 分離の作業内容・進捗

- annotator_webapi.py から OpenAIApiAnnotator を openai_api_response.py へ、AnthropicApiAnnotator を anthropic_api.py へ分離。
- 分離に伴い、型定義・エラーハンドリング・テストを整理。
- 共通スキーマ（AnnotationSchema）は webapi_shared.py に集約。
- テスト用ダミークラスのクラス名・型判定ロジックを実装と一致させ、テストの信頼性を担保。
- テスト全パスを確認。

### 今後のタスク
- 他API（Google, OpenRouter等）も同様の分離・整理を検討。
- ドキュメント・設計方針の定期的な見直し。

### 2024/05/10 OpenRouterApiAnnotator テスト・型・設計最新状況
- OpenRouterApiAnnotatorのユニットテストを追加・修正し、全ケースでパスを確認。
- テスト失敗の主因（クライアント型チェック）を特定し、OpenAIインスタンス利用で解決。
- AnnotationSchemaによる型安全な注釈生成・バリデーションを徹底。
- レスポンス異常系（content空、choices空、JSON不正、API例外）も網羅的にテスト。

# Active Context (2025-08-07)

## 現在のフォーカス
- Web API アノテーター (`annotator_webapi` 関連) のリファクタリング。
- Pydantic および PydanticAI の導入による型安全性と保守性の向上。
- `_run_inference` と `_format_predictions` のインターフェース（型シグネチャ）統一。

## 最近の主な変更・決定事項
- **`core/types.py` の導入:** 共通の型定義 (TypedDict, Pydanticモデル) を一元管理。
- **`_run_inference` の戻り値統一:** `list[RawOutput]` (`response: AnnotationSchema | None`, `error: str | None`) に統一。
- **`_format_predictions` の共通化:** `WebApiBaseAnnotator` に共通ロジックを実装。
    - 戻り値は `list[WebApiFormattedOutput]` (`annotation: dict | None`, `error: str | None`) に統一。
    - 各サブクラス (Google, Anthropic) からメソッドを削除。
- **テストコードの修正:** 上記変更に伴い、`test_google_api.py` と `test_anthropic_api.py` を修正し、パスすることを確認。
- **ドキュメント更新:** `pydanticai_integration_plan.md` を最新の状態に更新。

## 現在の課題・ペンディング事項
- **`src/image_annotator_lib/core/base.py` の型エラー:** `_format_predictions` を移動した際に発生した `self.components` 周りの型エラーが未解決。（リンターエラーは無視する指示あり）

## 次のステップ
1.  **`src/image_annotator_lib/core/base.py` の型エラー解消:** `self.components` の型ヒント問題を解決する。（一時的に `Any` にしている箇所）
2.  **リファクタリング内容のドキュメント反映:** `architecture.md`, `technical.md`, `lessons-learned.mdc` など、関連する他のドキュメントに今回の変更内容（`types.py` の導入、`_format_predictions` の共通化など）を反映させる。
3.  **PydanticAI の Agent/tool 等の導入検討:** 依存性注入やエージェント設計をさらに進めるか検討。

## 関連 RFC/ドキュメント
- [tasks/rfc/pydanticai_integration_plan.md](mdc:tasks/rfc/pydanticai_integration_plan.md)
- [src/image_annotator_lib/core/types.py](mdc:src/image_annotator_lib/core/types.py)
- [src/image_annotator_lib/core/base.py](mdc:src/image_annotator_lib/core/base.py)
- [docs/architecture.md](mdc:docs/architecture.md)

## 開発現状 (2025-05-13)

### 現作業焦点
- Web API アノテーター周りのリファクタリングと、それに伴うBDDテストの修正・安定化。
- 特にAPIキー未設定、タイムアウト、APIエラーレスポンスなどの異常系シナリオのテスト動作確認と修正。

### 進行中決定・検討事項
- `src/image_annotator_lib/api.py` の `annotate` 関数の戻り値型ヒントのリンターエラー対応。（`PHashAnnotationResults` と `dict[str, dict[str, AnnotationResult]]` の不一致）

### 最近変更
- **`src/image_annotator_lib/api.py` の `_handle_error` 関数を修正 (2025-05-13):**
  - エラーメッセージを生成する際に、元の例外の型名 (e.g., `type(e).__name__`) をメッセージに含めるように変更した。
  - これにより、BDDテスト (`test_apiキーが未設定の場合は認証エラーが発生する`) で、エラーメッセージ内に期待するエラータイプ名 (`ApiAuthenticationError`) が含まれるようになり、テストがパスするようになった。
- **BDDステップ定義 (`tests/features/step_definitions/webapi_annotate_steps.py`) の修正 (2025-05-13):**
  - APIキー未設定シナリオの `@when` ステップから `pytest.raises` を削除し、結果辞書を返すように変更。
  - 対応する `@then` ステップで、結果辞書内のエラーメッセージに期待されるエラータイプ名が含まれていることを検証するように修正。
- **フィーチャーファイル (`tests/features/webapi_annotate.feature`) の修正 (2025-05-13):**
  - APIキー未設定シナリオの `@then` ステップを、エラーメッセージの内容を検証するように変更 (`Then "ApiAuthenticationError" のエラーメッセージが返される`)。

### 次ステップ
- `src/image_annotator_lib/api.py` の `annotate` 関数の戻り値型ヒントに関するリンターエラーを修正する。
- 他のエラー関連BDDシナリオ（タイムアウト、APIエラーレスポンス）が、今回の `api.py` の `_handle_error` 関数の変更によって影響を受けていないか（意図せず失敗するようになっていないか等）、念のためテストを実行して確認する。
- 上記確認後、問題がなければ、今回のリファクタリングとテスト修正に関する一連の作業を完了とする。

### 既知問題点
- (特になし)

## 現在の状況 (2025-05-13)

- PydanticAI 関連ドキュメント ([tasks/rfc/pydanticai_integration_plan.md](mdc:tasks/rfc/pydanticai_integration_plan.md)) の読解と、それに基づく型定義・API利用方法の理解を進めている。
- 既存BDDテストスイートの安定化作業を実施。
    - Web API 関連のテストにおけるタイムアウト処理のシミュレーション方法を改善。
    - APIからのエラーレスポンスに対するテストステップのアサーションを修正し、より堅牢な検証ロジックを導入。
    - Linter (Mypy/Ruff) による型エラーやコーディングスタイル違反を修正し、コード品質を向上。
- 特に `google-genai` SDK利用時のタイムアウト処理、エラーハンドリングについて調査・実装し、テストカバレッジを改善した。
- `tasks/rfc/pydanticai_integration_plan.md` の更新を実施し、`types.py` の作成経緯やBDDテスト修正の記録を反映した。

## 次のステップ

- 引き続き `pydanticai_integration_plan.md` に基づき、Pydanticモデルの導入とリファクタリング作業を進める。
- 未解決のテスト失敗があれば、原因調査と修正を行う。
- 関連ドキュメント（`error-documentation.mdc`, `lessons-learned.mdc`, `technical.md` 等）に必要な情報を追記・更新する。
