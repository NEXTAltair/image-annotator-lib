# 効率的なWebApiBaseAnnotator統合計画

## 1. 統合方針: 直接置き換えアプローチ

### 基本戦略
既存の `WebApiBaseAnnotator` を `PydanticAIWebApiAnnotator` に直接置き換え、価値のある機能を移植しつつPydanticAIの恩恵を最大化する。

### 実装状況 (2025-06-24 更新)
✅ **概念実装完了**: `prototypes/pydanticai_integration/` で概念検証済み
✅ **依存性モデル実装**: WebAPI向けPydantic依存性注入モデル完成
✅ **基底クラス実装**: PydanticAI Agent ベースの新基底クラス完成
✅ **OpenAI実装例**: OpenAI Agent アノテーターの概念実装完成
✅ **構造検証完了**: インターフェース互換性確認済み
✅ **後方互換性確認**: `from_model_name()` クラスメソッドで既存設定システム対応
✅ **設計方針確定**: Vision+構造化出力対応モデルに限定、不要な複雑性排除

## 2. 新アーキテクチャ設計

### A. 依存性モデル階層
```python
# 基底依存性モデル
class WebApiDependencies(BaseModel):
    """全WebAPIアノテーター共通の依存性"""
    api_key: SecretStr
    model_name: str 
    api_model_id: str
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    min_request_interval: float = 1.0
    max_output_tokens: int = 1800

# プロバイダー固有依存性
class OpenAIDependencies(WebApiDependencies):
    temperature: float = 0.7
    # OpenAI固有設定

class GoogleDependencies(WebApiDependencies):
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 32
    # Google固有設定

class AnthropicDependencies(WebApiDependencies):
    temperature: float = 0.7
    # Anthropic固有設定
```

### B. 新基底クラス構造
```python
class PydanticAIWebApiAnnotator(BaseAnnotator):
    """PydanticAI Agent ベースのWebAPIアノテーター基底クラス"""
    
    def __init__(self, dependencies: WebApiDependencies):
        """依存性注入による初期化"""
        super().__init__(dependencies.model_name)
        self.deps = dependencies
        self.agent = self._create_agent()
        
        # 既存の価値ある機能を保持
        self.last_request_time = 0.0
        
    def _create_agent(self) -> Agent:
        """PydanticAI Agentの作成（サブクラスで実装）"""
        raise NotImplementedError
        
    # 保持する価値ある機能
    def _wait_for_rate_limit(self) -> None:
        """レート制限機能（既存ロジック移植）"""
        # 既存の実装をそのまま移植
        
    def _handle_api_error(self, e: Exception) -> NoReturn:
        """エラーハンドリング（既存ロジック移植）"""
        # 既存の包括的エラーマッピングを移植
        
    def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
        """画像前処理（既存ロジック改良）"""
        # より効率的な前処理実装
        
    async def _run_inference(self, processed: list[bytes]) -> list[RawOutput]:
        """PydanticAI Agentによる推論実行"""
        results = []
        for image_data in processed:
            try:
                self._wait_for_rate_limit()
                
                # PydanticAI Agent呼び出し
                response = await self.agent.run(
                    user_prompt=self._create_prompt(), 
                    deps=self.deps,
                    # 画像データの渡し方は調整が必要
                )
                
                # AnnotationSchemaに変換
                annotation = AnnotationSchema.model_validate(response.data)
                results.append(RawOutput(response=annotation, error=None))
                
            except Exception as e:
                error_msg = self._handle_api_error(e)
                results.append(RawOutput(response=None, error=error_msg))
                
        return results
```

## 3. 段階的実装ステップ

### Phase 1: 基盤準備 ✅ **完了**
1. ✅ **依存性モデル定義**: `prototypes/pydanticai_integration/dependencies.py` 実装完了
2. ✅ **新基底クラス作成**: `PydanticAIWebApiAnnotator`の基本構造実装完了
3. ✅ **概念実装検証**: OpenAI Agent アノテーターの動作確認完了

### Phase 2: 価値ある機能の移植 ✅ **完了**
1. ✅ **レート制限機能**: `_wait_for_rate_limit`の移植完了
2. ✅ **エラーハンドリング**: `_handle_api_error`の移植・改良完了
3. ✅ **画像前処理**: `_preprocess_images`の移植・改良完了

### Phase 3: 本番実装移行 (2-3日) 🔄 **次のステップ**
1. **基底クラス移行**: `src/image_annotator_lib/core/base/webapi.py` の直接置き換え
2. **依存性モデル統合**: `src/image_annotator_lib/core/types.py` への依存性モデル追加
3. **OpenAI Agent実装**: `src/image_annotator_lib/model_class/annotator_webapi/` の更新

### Phase 4: プロバイダー別実装 (2-3日)
1. **Google Agent**: Gemini Vision Agent実装
2. **Anthropic Agent**: Claude Vision Agent実装  
3. **OpenRouter Agent**: OpenRouter Vision Agent実装

### Phase 5: 統合テスト・検証 (1-2日)
1. **BDDテスト実行**: 全シナリオの動作確認
2. **パフォーマンス検証**: レスポンス時間・精度確認
3. **エラーハンドリング検証**: 異常系シナリオ確認

## 4. リスク軽減策

### A. 段階的テスト戦略
1. **単体テスト**: 各Phase完了時にユニットテスト実行
2. **統合テスト**: Phase 3完了時に簡易統合テスト
3. **BDDテスト**: Phase 5でフルテスト実行
4. **ロールバック準備**: gitブランチでの確実な退避

### B. 設定移行の自動化 ✅ **実装完了**
```python
# 実装済み: prototypes/pydanticai_integration/pydanticai_webapi_annotator.py

@classmethod
def from_model_name(cls, model_name: str) -> Self:
    """Create annotator from model_name using existing config system."""
    dependencies = cls._create_dependencies_from_config(model_name)
    return cls(dependencies)

@classmethod
def _create_dependencies_from_config(cls, model_name: str) -> WebApiDependencies:
    """Convert existing config system to dependency model."""
    # config_registry からの設定読み込み・変換ロジック実装済み
```

## 5. 期待される効果

### A. 開発効率向上
- **型安全性**: Pydanticモデルによる完全な型安全性
- **デバッグ容易性**: PydanticAIの統合ツールチェーン活用
- **保守性**: 統一されたAgent パターンによる一貫性

### B. 機能強化
- **構造化出力**: JSONスキーマによる確実なレスポンス構造
- **エラー処理**: PydanticAI + 既存エラーハンドリングのハイブリッド
- **監視・ログ**: Logfireによる詳細なトレーシング

### C. 拡張性
- **新プロバイダー追加**: Agent パターンによる簡単な拡張
- **ツール統合**: PydanticAI toolsによる機能拡張
- **マルチモーダル**: 画像以外の入力への対応準備

## 6. 成功指標

1. **機能性**: 全BDDテストがパス
2. **パフォーマンス**: 既存実装と同等以上のレスポンス時間
3. **安定性**: エラー率の維持または改善
4. **拡張性**: 新プロバイダー追加が1日以内で完了

## 7. 実装優先度 (2025-06-24 更新)

**完了済み (Phase 1-2)**:
- ✅ 基盤・価値ある機能移植
- ✅ 概念実装・構造検証

**高優先度 (現在のフォーカス)**:
- 🔄 Phase 3: 本番実装移行
- 🔄 Phase 5: BDDテスト検証

**中優先度**:
- Phase 4: 各プロバイダー実装・移行

**低優先度 (将来拡張)**:
- Logfire統合
- 高度なツール機能
- マルチモーダル対応

**対象外 (別ブランチ・別タスク)**:
- Vision+構造化出力非対応モデルの対応
- モデル探索・発見機能の改善

## 8. 概念実装からの移行準備

### 準備完了事項
- `prototypes/pydanticai_integration/` での概念検証完了
- 依存性モデル、基底クラス、OpenAI実装例の動作確認済み
- 既存インターフェース互換性確認済み
- 設定システム移行方法確立済み

### 次のアクション
Phase 3の本番実装移行により、概念実装から実際の統合への移行を開始する準備が整いました。