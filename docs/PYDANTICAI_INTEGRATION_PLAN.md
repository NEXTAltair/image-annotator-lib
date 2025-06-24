# PydanticAI統合計画書

## 統合ステータス: Phase 2 - 段階的拡張実装

**作成日:** 2025-06-24  
**バージョン:** v1.0  
**ステータス:** 概念実装完了 → 本格統合実装中

---

## 1. 現在の統合状況分析

### ✅ 完了済み（Phase 1）
- **Agentキャッシュシステム**: `src/image_annotator_lib/core/webapi_agent_cache.py`
- **OpenAI PydanticAI統合**: `src/image_annotator_lib/model_class/annotator_webapi/openai_api_response.py`
- **概念実装**: `prototypes/pydanticai_integration/` 
- **テスト基盤**: 直下テストファイル群

### 🔄 Phase 2 実装対象
- Google API PydanticAI化
- Anthropic API PydanticAI化 
- 統一Agent管理システム
- 性能最適化と安定性向上

---

## 2. アーキテクチャ分析

### 現在の実装パターン（成功例：OpenAI）

```python
# openai_api_response.py の設計パターン
class OpenAIApiAnnotator(WebApiBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.agent: Agent | None = None
    
    def __enter__(self) -> Self:
        """Agent作成とキャッシュ管理"""
        self._load_configuration()
        self.agent = self._create_agent()
        return self
    
    def _create_agent(self) -> Agent:
        """キャッシュからAgent取得または新規作成"""
        cache_key = create_cache_key(...)
        config_hash = create_config_hash(...)
        return WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
```

### 基盤システム設計

```
WebApiBaseAnnotator (継承)
├── Agent Cache System (webapi_agent_cache.py)
│   ├── LRU戦略 (最大50インスタンス)
│   ├── 設定変更検出
│   └── メモリ効率管理
├── Configuration System (config.py)
│   ├── 階層設定マージ
│   ├── 動的API発見統合
│   └── プロバイダー固有パラメータ
└── Registry System (registry.py)
    ├── 自動クラス登録
    ├── API Model Discovery
    └── 設定ベース初期化
```

---

## 3. 段階的統合実装戦略

### Phase 2A: Google API PydanticAI化

**対象ファイル:** `src/image_annotator_lib/model_class/annotator_webapi/google_api.py`

**実装方針:**
1. **既存互換性保持**: 現在の`GoogleApiAnnotator`インターフェース維持
2. **Agent統合**: `pydantic_ai.models.google.GoogleModel`使用
3. **構造化出力**: Gemini nativeサポート活用
4. **キャッシュ統合**: `WebApiAgentCache`による効率化

**技術実装:**
```python
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

class GoogleApiAnnotator(WebApiBaseAnnotator):
    def _create_agent(self) -> Agent:
        provider = GoogleProvider(api_key=self.api_key)
        model = GoogleModel(model_name=self.api_model_id, provider=provider)
        return Agent(
            model=model,
            output_type=AnnotationSchema,
            system_prompt=BASE_PROMPT
        )
```

### Phase 2B: Anthropic API PydanticAI化  

**対象ファイル:** `src/image_annotator_lib/model_class/annotator_webapi/anthropic_api.py`

**実装方針:**
1. **Tool Use Block廃止**: PydanticAIの構造化出力に統一
2. **Claude最適化**: モデル固有特性の活用
3. **エラーハンドリング**: 既存404→ModelNotFoundError変換維持

**技術実装:**
```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

class AnthropicApiAnnotator(WebApiBaseAnnotator):
    def _create_agent(self) -> Agent:
        provider = AnthropicProvider(api_key=self.api_key)
        model = AnthropicModel(model_name=self.api_model_id, provider=provider)
        return Agent(
            model=model, 
            output_type=AnnotationSchema,
            system_prompt=BASE_PROMPT
        )
```

### Phase 2C: 統合最適化

**実装項目:**
1. **共通Agent Factory**: プロバイダー抽象化
2. **統一エラーハンドリング**: PydanticAI例外の標準化
3. **性能最適化**: 並列処理とキャッシュ戦略
4. **設定統合**: API固有パラメータの自動マッピング

---

## 4. 実装優先度とロードマップ

### 🟢 高優先度 (1-2週間)
1. **Google API統合** - 最大使用量プロバイダー
2. **統一テストスイート** - 回帰防止
3. **パフォーマンス測定** - Agent vs 既存比較

### 🟡 中優先度 (2-4週間)  
1. **Anthropic API統合** - Claude特性活用
2. **エラーハンドリング統一** - 例外体系整理
3. **設定システム拡張** - PydanticAI固有パラメータ

### 🔵 低優先度 (1-2ヶ月)
1. **OpenRouter統合検討** - Agent対応状況次第
2. **独自プロバイダー追加** - 拡張性向上
3. **メトリクス・監視** - 運用品質向上

---

## 5. 技術要件と制約

### 必須要件
- **後方互換性**: 既存WebApiBaseAnnotatorインターフェース維持
- **設定継続性**: 現在のTOML設定フォーマット互換
- **性能向上**: Agent Cache活用による高速化
- **テスト保証**: 既存テストケース全通

### 技術制約
- **PydanticAI依存**: v0.3.2以上必須
- **メモリ制約**: Agent Cache最大50インスタンス
- **API制限**: プロバイダーごとのレート制限対応
- **非同期対応**: 既存同期インターフェース保持

---

## 6. 品質保証戦略

### テスト戦略
1. **単体テスト**: 各プロバイダーAgent作成・実行
2. **統合テスト**: 既存アノテーションワークフロー 
3. **性能テスト**: Agent Cache効果測定
4. **回帰テスト**: 既存実装との結果比較

### 監視項目
- **レスポンス時間**: Agent初期化・推論実行時間
- **メモリ使用量**: キャッシュ効率性
- **エラー率**: API呼び出し成功率
- **キャッシュヒット率**: Agent再利用効率

---

## 7. リスク管理

### 高リスク項目
1. **API互換性変更**: PydanticAI更新による破壊的変更
2. **性能劣化**: Agent初期化オーバーヘッド
3. **メモリリーク**: 不適切なAgent管理

### 軽減策
1. **段階的移行**: プロバイダーごと個別実装・検証
2. **フォールバック**: 既存実装への切り戻し機能
3. **監視強化**: メトリクス収集とアラート設定

---

## 8. 実装チェックリスト

### Phase 2A: Google API統合
- [ ] `GoogleApiAnnotator` PydanticAI化実装
- [ ] Gemini固有パラメータ対応
- [ ] Agent Cache統合
- [ ] 単体テスト作成
- [ ] 性能ベンチマーク
- [ ] ドキュメント更新

### Phase 2B: Anthropic API統合  
- [ ] `AnthropicApiAnnotator` PydanticAI化実装
- [ ] Claude最適化プロンプト調整
- [ ] Tool Use Block廃止
- [ ] エラーハンドリング改善
- [ ] 統合テスト実行
- [ ] ユーザーガイド更新

### Phase 2C: 統合最適化
- [ ] 共通Agent Factory実装
- [ ] 統一エラーハンドリング
- [ ] 設定システム拡張
- [ ] 総合性能テスト
- [ ] 本番環境検証
- [ ] リリースノート作成

---

## 9. 成功指標

### 技術指標
- **レスポンス時間**: 20%以上改善（Agent Cache効果）
- **メモリ効率**: 30%以上改善（Agent再利用）
- **開発効率**: 新プロバイダー追加時間50%短縮

### 品質指標  
- **テストカバレッジ**: 90%以上維持
- **エラー率**: 1%以下
- **API互換性**: 100%（既存設定ファイル）

---

**責任者:** PydanticAI統合チーム  
**レビュー:** 2025-07-01予定  
**更新:** 必要に応じて随時

---

*このドキュメントは統合進捗に応じて継続的に更新されます。*