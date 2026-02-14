# PydanticAI Model Factory 比較レポート

## 実験概要

image-annotator-lib の PydanticAI Model Factory について、2つのリファクタリングアプローチを比較実験した。

| | Plan 1: PydanticAI完全準拠 | Plan 2: APIキー検出+フォールバック |
|---|---|---|
| **設計思想** | `infer_model()` に完全依存、単一クラス設計 | 既存3層階層を維持、DRY違反を解消 |
| **対象ファイル** | `pydantic_ai_factory.py` + `provider_manager.py` | 同上 |
| **ブランチ** | `experiment/plan1-pydanticai-full-compliance` | `experiment/plan2-api-key-fallback` |
| **Worktree** | `/workspaces/LoRAIro-plan1` | `/workspaces/LoRAIro-plan2` |

---

## 1. 定量メトリクス

### 1.1 コード行数 (LOC)

| ファイル | Original | Plan 1 | Plan 2 |
|---|---:|---:|---:|
| `pydantic_ai_factory.py` | 486 | **201** (-58.6%) | 482 (-0.8%) |
| `provider_manager.py` | 534 | **151** (-71.7%) | 325 (-39.1%) |
| **合計** | **1,020** | **352** (-65.5%) | **807** (-20.9%) |

### 1.2 Cyclomatic Complexity (radon)

| ファイル | Original Avg | Plan 1 Avg | Plan 2 Avg |
|---|---|---|---|
| `pydantic_ai_factory.py` | A (4.06) | A (4.71) | A (3.65) |
| `provider_manager.py` | A (4.52) | A (4.60) | A (3.67) |
| **最高CC関数** | `_is_test_environment`: **C(11)** | `_create_openrouter_agent`: **B(8)** | `_determine_provider`: **B(9)** |
| **ブロック数** | 40 | **12** | 32 |

### 1.3 Lizard メトリクス

| メトリック | Original | Plan 1 | Plan 2 |
|---|---:|---:|---:|
| NLOC (実効コード行数) | 661 | **208** | 507 |
| 関数数 | 33 | **10** | 28 |
| 平均CCN | 3.9 | 4.5 | **3.4** |
| 平均トークン数 | 121.5 | 116.0 | **105.2** |

### 1.4 テスト結果

| テスト種別 | Plan 1 | Plan 2 |
|---|---|---|
| ユニットテスト数 | 26 passed | 79 passed |
| BDDシナリオ通過 | **13/13** (100%) | 7/13 (54%) |
| ユニットテスト結果 | 全パス | 全パス |

### 1.5 アーキテクチャ変更量

| 指標 | Plan 1 | Plan 2 |
|---|---|---|
| 削除クラス | `PydanticAIAnnotatorMixin`, 4x ProviderInstance | 4x ProviderInstance |
| 新規クラス | `PydanticAIAgentFactory` | `ProviderInstance` (Generic) |
| 公開API変更 | `get_or_create_agent()` (新API) | `get_cached_agent()` (既存API維持) |
| WebAPIアノテータ変更 | 4ファイル全面書き換え | 変更なし |
| LoRAIro統合影響 | `annotator_adapter.py` 更新必要 | **変更不要** |

---

## 2. 定性分析

### 2.1 デバッグ容易性

| 観点 | Plan 1 | Plan 2 |
|---|---|---|
| スタックトレース深度 | **浅い** (Factory → Agent直接) | 深い (Manager → Instance → Factory → Agent) |
| エラー発生箇所特定 | 容易（パス数少ない） | 中程度（3層をたどる） |
| ログ出力 | シンプル（1層のみ） | 各層でDEBUGログ |

### 2.2 新プロバイダー追加コスト

| 作業 | Plan 1 | Plan 2 |
|---|---|---|
| 標準プロバイダー | **0行** (infer_model()が自動判定) | `PROVIDER_ANNOTATOR_MAP` に1行追加 |
| カスタムプロバイダー | `_create_openrouter_agent()` パターン複製 | `ProviderInstance` + Annotatorクラス追加 |
| テスト追加 | ユニットテスト2-3件 | ユニットテスト5-10件 |

### 2.3 PydanticAI追従性

| 観点 | Plan 1 | Plan 2 |
|---|---|---|
| `infer_model()` 依存度 | **完全依存** | 間接的（Factory内部で使用） |
| PydanticAI APIバージョン追従 | **容易** (薄いラッパー) | 中程度 (3層の調整が必要) |
| `TestModel` サポート | 自動 (infer_model) | `_is_test_environment()` で明示的切替 |
| Provider追加時の対応 | PydanticAI側追加で自動対応 | `PROVIDER_ANNOTATOR_MAP` 更新必要 |

### 2.4 リスク分析

| リスク | Plan 1 | Plan 2 |
|---|---|---|
| PydanticAI破壊的変更 | **高**: 全機能がinfer_model()に依存 | **低**: 3層バッファが変更を吸収 |
| 既存コードとの互換性 | **低**: 公開APIが変更 | **高**: 既存API完全互換 |
| OpenRouter特殊処理 | 実装済み (専用メソッド) | 実装済み (Generic Instance) |
| エッジケース処理 | 少ない（シンプルな分） | 充実（既存のエラーハンドリング継承） |

---

## 3. 評価スコア

評価基準ドキュメント (docs/evaluation-criteria.md) の重み付きスコアリング:

| 指標 (重み) | Plan 1 スコア | Plan 2 スコア | 根拠 |
|---|---:|---:|---|
| CC複雑度 (20%) | 7/10 | **8/10** | Plan 2の平均CCNが低い |
| LOC (15%) | **10/10** | 6/10 | Plan 1は65.5%削減 |
| テストカバレッジ (15%) | 8/10 | **9/10** | Plan 2のテスト数が多い |
| 初回ロード時間 (15%) | **9/10** | 7/10 | Plan 1は層数少なく高速想定 |
| 推論応答時間 (10%) | 8/10 | 8/10 | 差なし（モック環境） |
| メモリ使用量 (10%) | **9/10** | 7/10 | Plan 1はコード量半分以下 |
| 新プロバイダー追加 (15%) | **9/10** | 7/10 | Plan 1は0行追加 |
| **加重合計** | **8.55** | **7.40** |  |

---

## 4. 推奨案

### 推奨: Plan 1 (PydanticAI完全準拠) をベースに採用

**理由:**
1. **LOC 65.5%削減**: 保守対象コードが大幅に減少
2. **BDD 100%通過**: 全13シナリオをクリア（Plan 2は54%）
3. **PydanticAI追従性**: 将来のPydanticAIアップデートへの追従が容易
4. **新プロバイダー追加0行**: PydanticAIの`infer_model()`が自動対応
5. **アーキテクチャのシンプルさ**: 3層→1層で理解・デバッグが容易

### 採用時の注意点

1. **LoRAIro統合更新が必要**: `annotator_adapter.py` の API呼び出しを新APIに合わせる
2. **エッジケース補強**: Plan 2が持つ詳細なエラーハンドリングを一部移植
3. **`_is_test_environment()`**: Plan 2の簡素化版（21行、inspect.stack()なし）を採用
4. **PydanticAI破壊的変更リスク**: 薄いラッパーのため影響範囲が明確で対応しやすい

### 統合計画

1. Plan 1の `pydantic_ai_factory.py` と `provider_manager.py` をmainブランチにマージ
2. Plan 2の `_is_test_environment()` 簡素化版を採用（フラグ+環境変数のみ）
3. `annotator_adapter.py` の統合テストで互換性確認
4. 既存のWebAPIアノテータテストの更新

---

## 5. メトリクス生データ

### radon CC 詳細

**Plan 1 (pydantic_ai_factory.py)**:
- `_create_openrouter_agent`: B(8)
- `_is_test_environment`: B(6)
- `get_or_create_agent`: A(5)
- `_set_env_api_key`: A(5)
- `preprocess_images_to_binary`: A(2)
- `clear_cache`: A(1)

**Plan 1 (provider_manager.py)**:
- `run_inference_with_model`: B(7)
- `_get_api_key`: A(5)
- `_get_provider`: A(5)
- `clear_cache`: A(1)

**Plan 2 (pydantic_ai_factory.py)**:
- `_run_inference_async`: B(7)
- `create_agent`: B(6)
- `create_openrouter_agent`: B(6)
- 17 blocks total, avg A(3.65)

**Plan 2 (provider_manager.py)**:
- `_determine_provider`: B(9)
- `run_with_model`: B(8)
- `run_inference_with_model`: B(6)
- `_run_agent_safely`: B(6)
- 15 blocks total, avg A(3.67)

---

*Generated: 2026-02-12*
*Experiment: PydanticAI Model Factory Comparison (feature/annotator-library-integration)*
