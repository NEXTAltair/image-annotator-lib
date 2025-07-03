# アクティブコンテキスト

## 1. 現作業焦点 (Current Focus)
- **PydanticAI統合の堅牢性強化 (2025-07-02完了)**: Event loop管理、テスト環境検出、エラーハンドリングの改善が完了。
- **BDDテスト修正 (2025-07-02完了)**: ステップ定義の追加、期待値の統一、スキップ条件の強化が完了。
- **統合テスト実装**: RFC 005に基づく包括的統合テストスイートの構築。Provider-level PydanticAIアーキテクチャの信頼性と安定性を保証する3段階ハイブリッド戦略を実装中。
    - **Phase 1 (第1週)**: Provider Manager、PydanticAI Factory、Cross-Provider統合テストの実装
    - **Phase 2 (第2週)**: メモリ管理・設定統合テストの実装
    - **Phase 3 (第3週)**: エンドツーエンドワークフロー・エラー伝播統合テストの実装

## 2. 進行中の主要課題･決定事項 (Ongoing Key Issues and Decisions)
- **3層ハイブリッド戦略の採用**: CI対応高速テスト、実API検証テスト、システム統合テストの組み合わせ
- **テストカバレッジ向上**: 31.56% → 85%への向上を統合テストで実現
- **品質保証指標の確立**: 実行時間、成功率、メモリ使用量等の明確な指標設定
- **リスク管理戦略**: API仕様変更、リソース競合、テスト不安定性への対策

## 3. 最近の主要な変更点 (Recent Key Changes)
- **PydanticAI Event loop管理改善**: `_run_agent_safely()` メソッドによる堅牢な非同期処理制御を実装
- **テスト環境検出強化**: BDDテスト、スタックフレーム検査による確実なテスト環境認識を追加
- **BDDテストステップ定義完備**: `"モデルクラスレジストリが初期化されている"` 等の不足ステップ定義を追加
- **統合テストmockパターン修正**: `ProviderManager.get_provider_instance` mockによる実装フロー準拠テストに変更
- **TypedDict対応統一**: 全テストで `AnnotationResult` への辞書アクセス（`.get()`）を統一
- **アーキテクチャドキュメント更新**: エラーハンドリング、堅牢性改善の詳細を追記

## 4. 次のステップ (Next Steps)
1. **Phase 1実装開始**: Provider Manager統合テスト (`test_provider_manager_integration.py`) の作成
2. **テストインフラ構築**: 共通フィクスチャ、ユーティリティ、モック戦略の実装
3. **CI統合準備**: 高速統合テストのCI環境組み込み設定

## 5. 参照すべき主要ドキュメント (Key Documents to Refer To)
- `tasks/rfc/integration_test_implementation_plan.md` (RFC 005: 統合テスト実装計画)
- `tasks/tasks_plan.md` (全体タスク計画、バックログ)
- `tests/integration/test_openrouter_pydanticai_integration.py` (テスト実装の参考)
- `tests/integration/test_unified_provider_level_integration.py` (統合テスト構造の参考)
- `docs/architecture.md` (テスト対象のアーキテクチャ理解)

## 6. 重要な設計決定事項 (Key Design Decisions)
- **ハイブリッドテスト戦略**: モック中心のCI適用可能テストと、実API検証の組み合わせ
- **段階的実装**: 3フェーズによるリスク最小化と段階的価値提供
- **標準化パターン**: 再利用可能なテストユーティリティとフィクスチャの確立
- **性能重視**: CI実行時間<5分、高い成功率の維持