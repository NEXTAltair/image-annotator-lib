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
