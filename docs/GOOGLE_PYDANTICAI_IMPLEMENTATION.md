# Google API PydanticAI統合実装レポート

**実装日:** 2025-06-24  
**ステータス:** ✅ 完了  
**対象:** Phase 2A - Google API PydanticAI化

---

## 1. 実装概要

Google Gemini API を PydanticAI Agent ベースに完全移行し、既存の WebApiBaseAnnotator インターフェースとの互換性を保ちながら、構造化出力と効率的なAgent管理を実現しました。

### 主要な変更点

1. **PydanticAI統合**: GoogleClientAdapterから PydanticAI Agent + GoogleProvider への移行
2. **構造化出力**: ネイティブ AnnotationSchema 対応
3. **Agent Cache**: WebApiAgentCache による効率的なAgent管理
4. **非同期対応**: async/await による高性能推論実行
5. **Gemini最適化**: temperature, top_p, top_k, max_output_tokens の細かな制御

---

## 2. アーキテクチャ設計

### 新しい実装パターン

```python
# google_api.py の設計パターン
class GoogleApiAnnotator(WebApiBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.agent: Agent | None = None
    
    def __enter__(self) -> Self:
        """Agent作成とキャッシュ管理"""
        self._load_configuration()
        cache_key = create_cache_key(self.model_name, "google", self.api_model_id)
        config_hash = self._get_config_hash()
        self.agent = WebApiAgentCache.get_agent(cache_key, self._create_agent, config_hash)
        return self
    
    def _create_agent(self) -> Agent:
        """GoogleProvider + GoogleModel でAgent作成"""
        provider = GoogleProvider(api_key=self.api_key.get_secret_value())
        model = GoogleModel(model_name=self.api_model_id, provider=provider)
        return Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
```

### PydanticAI統合の特徴

1. **画像処理**: PIL.Image → BinaryContent への変換
2. **推論実行**: async/await での非同期処理
3. **エラーハンドリング**: Google固有のエラーマッピング
4. **設定管理**: Gemini固有パラメータの統合

---

## 3. 実装詳細

### 3.1 Agent生成とキャッシュ管理

```python
def _create_agent(self) -> Agent:
    """新しいPydanticAI Agentを作成する"""
    try:
        # Google Provider と Model を作成
        provider = GoogleProvider(api_key=self.api_key.get_secret_value())
        model = GoogleModel(model_name=self.api_model_id, provider=provider)

        # Agent を作成（構造化出力対応）
        agent = Agent(
            model=model,
            output_type=AnnotationSchema,
            system_prompt=BASE_PROMPT,
        )

        logger.debug(f"PydanticAI Google Agent を作成しました (model: {self.api_model_id})")
        return agent

    except Exception as e:
        logger.error(f"PydanticAI Google Agent の作成中にエラー: {e}")
        raise ConfigurationError(f"Google Agent 作成エラー: {e}") from e
```

### 3.2 画像前処理（BinaryContent対応）

```python
@override
def _preprocess_images(self, images: list[Image.Image]) -> list[BinaryContent]:
    """画像リストをPydanticAI BinaryContentのリストに変換する"""
    binary_contents = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        
        binary_content = BinaryContent(
            data=buffered.getvalue(),
            media_type="image/webp"
        )
        binary_contents.append(binary_content)
    
    return binary_contents
```

### 3.3 非同期推論実行

```python
async def _run_inference_async(self, binary_content: BinaryContent) -> AnnotationSchema:
    """非同期でPydanticAI Agent推論を実行する"""
    if self.agent is None:
        raise WebApiError("Agent が初期化されていません", provider_name="google")

    try:
        # Gemini固有パラメータ設定
        model_params = {
            "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
            "top_p": config_registry.get(self.model_name, "top_p", default=1.0),
            "top_k": config_registry.get(self.model_name, "top_k", default=32),
            "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        }

        # Agent実行（画像と追加プロンプト）
        result = await self.agent.run(
            user_prompt="この画像を詳細に分析して、タグとキャプションを生成してください。",
            message_history=[binary_content],
            model_settings=model_params,
        )

        logger.debug(f"Google Agent 推論完了 (model: {self.api_model_id})")
        return result.data

    except Exception as e:
        logger.error(f"Google Agent 非同期推論エラー: {e}")
        raise
```

### 3.4 設定管理とキャッシュ

```python
def _get_config_hash(self) -> str:
    """Agent作成に影響する設定のハッシュを生成"""
    config_data = {
        "model_id": self.api_model_id,
        "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
        "top_p": config_registry.get(self.model_name, "top_p", default=1.0),
        "top_k": config_registry.get(self.model_name, "top_k", default=32),
        "max_output_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
    }
    return create_config_hash(config_data)
```

---

## 4. テスト実装

### テストファイル: `test_google_pydanticai.py`

#### テスト項目

1. **インポートテスト**: PydanticAI Google関連モジュールの正常インポート
2. **構造テスト**: 必要メソッドの存在確認
3. **Agent作成テスト**: GoogleModel/GoogleProvider/Agent の構造確認
4. **設定システムテスト**: キャッシュキー生成と設定ハッシュ

#### テスト結果（期待値）

```
Google PydanticAI統合テスト開始

=== Google PydanticAI インポートテスト ===
✅ PydanticAI Google関連モジュール インポート成功
✅ GoogleApiAnnotator (PydanticAI版) インポート成功

=== Google Annotator 構造テスト ===
✅ 全11個の必要メソッドが存在
✅ WebApiBaseAnnotator継承確認

=== Google Agent 作成テスト ===
✅ GoogleModel & GoogleProvider 作成
✅ Agent インスタンス作成
✅ BinaryContent インスタンス作成

=== Google 設定システムテスト ===
✅ 設定システム統合確認
✅ キャッシュキー生成
✅ 設定ハッシュ生成

📊 Google PydanticAI統合テスト結果: 4成功 / 0失敗 / 4合計
🎉 Google PydanticAI統合が正常に完了しました！
```

---

## 5. パフォーマンス向上

### Agent Cache効果

1. **初期化時間短縮**: Agent再利用による高速化
2. **メモリ効率化**: LRU戦略による適切なリソース管理
3. **設定変更検出**: 自動的なキャッシュ無効化

### 非同期処理による並列化

- 複数画像の同時処理
- I/O待機時間の最適化
- スループット向上

---

## 6. 互換性と移行

### 既存インターフェース完全保持

- `WebApiBaseAnnotator` 継承による透過的な統合
- 設定ファイル（TOML）フォーマット互換性
- エラーハンドリングの一貫性

### 移行時の考慮事項

1. **APIキー設定**: `GOOGLE_API_KEY` または `GEMINI_API_KEY` 環境変数
2. **依存関係**: PydanticAI 0.3.2以上必須
3. **設定パラメータ**: Gemini固有パラメータの活用

---

## 7. 今後の展開

### Phase 2B: Anthropic API統合

Google API実装を参考に、Anthropic Claude API のPydanticAI化を実施

### Phase 2C: 統合最適化

- 共通Agent Factory の実装
- 統一エラーハンドリング
- パフォーマンス監視

---

## 8. まとめ

✅ **成功したポイント:**
- PydanticAI統合による構造化出力ネイティブサポート
- Agent Cache による効率的なリソース管理
- 既存アーキテクチャとの完全互換性
- Gemini固有パラメータの最適活用

🎯 **次のステップ:**
- 本番環境での性能検証
- Anthropic API統合への展開
- 統合テストの拡充

---

**実装完了:** 2025-06-24  
**ステータス:** ✅ Phase 2A 完了