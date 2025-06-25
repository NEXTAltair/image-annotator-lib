# OpenRouter API PydanticAI統合実装ガイド

**作成日**: 2025-06-24  
**ステータス**: ✅ 実装完了 - Phase 2D  
**対象**: OpenRouter API PydanticAI化

---

## 1. 実装概要

### 🔄 実装の変更点

**従来の実装**:
- OpenAI Python SDK の直接利用
- JSON Schema サポート有無での分岐処理
- 複雑なレスポンス解析とコードブロック除去

**新実装**:
- PydanticAI Agent + OpenAI Provider (base_url変更) 統合
- OpenRouter固有ヘッダー対応
- ネイティブ構造化出力対応

### 📋 主要技術特徴

- **OpenAI互換アーキテクチャ**: OpenAI Provider の base_url を OpenRouter に設定
- **カスタムヘッダー対応**: HTTP-Referer, X-Title の自動設定
- **Agent Cache 統合**: LRU戦略による効率的リソース管理
- **既存互換性保持**: WebApiBaseAnnotator インターフェース継承

---

## 2. アーキテクチャ設計

### コア実装パターン

```python
# OpenRouter PydanticAI統合の基本構造
class OpenRouterApiAnnotator(WebApiBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.agent: Agent | None = None
        self.api_key: SecretStr | None = None
        self.api_model_id: str | None = None

    def __enter__(self):
        """コンテキストマネージャーでAgent初期化"""
        self._load_configuration()
        cache_key = create_cache_key(self.model_name, "openrouter", self.api_model_id)
        config_hash = self._get_config_hash()
        
        def creator_func() -> Agent:
            return self._create_agent()
        
        self.agent = WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
        return self
```

### OpenRouter固有Agent作成

```python
def _create_agent(self) -> Agent:
    """新しいPydanticAI Agentを作成する"""
    # OpenRouter固有ヘッダー設定
    default_headers = {}
    referer = config_registry.get(self.model_name, "referer")
    app_name = config_registry.get(self.model_name, "app_name")
    
    if referer and isinstance(referer, str):
        default_headers["HTTP-Referer"] = referer
    if app_name and isinstance(app_name, str):
        default_headers["X-Title"] = app_name

    provider = OpenAIProvider(
        api_key=self.api_key.get_secret_value(),
        base_url="https://openrouter.ai/api/v1",  # OpenRouter エンドポイント
        default_headers=default_headers
    )
    model = OpenAIModel(model_name=self.api_model_id, provider=provider)
    agent = Agent(
        model=model, 
        output_type=AnnotationSchema, 
        system_prompt=BASE_PROMPT
    )
    return agent
```

### 画像処理フロー

```python
def _preprocess_images(self, images: list[Image.Image]) -> list[BinaryContent]:
    """PIL.Image → BinaryContent変換"""
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

---

## 3. 推論実行システム

### 非同期推論実行

```python
async def _run_inference_async(self, binary_content: BinaryContent) -> AnnotationSchema:
    """非同期推論実行"""
    # OpenRouter固有パラメータ
    temperature = config_registry.get(self.model_name, "temperature", default=0.7)
    max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
    timeout = config_registry.get(self.model_name, "timeout", default=120)
    
    model_params = {
        "temperature": float(temperature) if temperature is not None else 0.7,
        "max_tokens": int(max_tokens) if max_tokens is not None else 1800,
        "timeout": float(timeout) if timeout is not None else 120.0,
    }
    
    result = await self.agent.run(
        user_prompt="この画像を詳細に分析して、タグとキャプションを生成してください。",
        message_history=[binary_content],
        model_settings=model_params,
    )
    return result.data
```

### 同期ラッパー

```python
def _run_inference_sync(self, binary_content: BinaryContent) -> AnnotationSchema:
    """同期版推論実行"""
    return asyncio.run(self._run_inference_async(binary_content))
```

---

## 4. エラーハンドリング

### OpenAI互換エラー処理

```python
def _handle_api_error(self, error: Exception) -> str:
    """API エラーを適切な例外に変換"""
    error_str = str(error)
    
    # OpenRouter/OpenAI 互換エラーパターン
    if "401" in error_str or "authentication" in error_str.lower():
        raise ApiAuthenticationError(f"OpenRouter API 認証エラー: {error_str}")
    elif "429" in error_str or "rate limit" in error_str.lower():
        raise ApiRateLimitError(f"OpenRouter API レート制限: {error_str}")
    elif "timeout" in error_str.lower():
        raise ApiTimeoutError(f"OpenRouter API タイムアウト: {error_str}")
    elif "500" in error_str or "server error" in error_str.lower():
        raise ApiServerError(f"OpenRouter API サーバーエラー: {error_str}")
    
    # 一般エラー
    return f"OpenRouter API Error: {error_str}"
```

---

## 5. Agent Cache統合

### キャッシュキー生成

```python
def _get_config_hash(self) -> str:
    """設定のハッシュ値を生成する"""
    config_data = {
        "model_id": self.api_model_id,
        "temperature": config_registry.get(self.model_name, "temperature", default=0.7),
        "max_tokens": config_registry.get(self.model_name, "max_output_tokens", default=1800),
        "json_schema_supported": config_registry.get(self.model_name, "json_schema_supported", default=False),
        "referer": config_registry.get(self.model_name, "referer"),
        "app_name": config_registry.get(self.model_name, "app_name"),
    }
    return create_config_hash(config_data)
```

### Cache統合パターン

```python
# Agent Cacheからの取得/作成
cache_key = create_cache_key(self.model_name, "openrouter", self.api_model_id)
config_hash = self._get_config_hash()

def creator_func() -> Agent:
    return self._create_agent()

self.agent = WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
```

---

## 6. 設定システム統合

### 必須設定項目

```toml
[model_name]
class = "OpenRouterApiAnnotator"
api_key = "your-openrouter-api-key"
api_model_id = "anthropic/claude-3.5-sonnet"
temperature = 0.7
max_output_tokens = 1800
timeout = 120
json_schema_supported = true  # モデル依存

# OpenRouter固有設定
referer = "https://your-app.com"
app_name = "YourAppName"
```

### 設定読み込み

```python
def _load_configuration(self):
    """設定を読み込む"""
    self.api_key = SecretStr(config_registry.get(self.model_name, "api_key", default=""))
    self.api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
    
    if not self.api_key.get_secret_value():
        raise WebApiError("OpenRouter API キーが設定されていません", provider_name="OpenRouter")
    if not self.api_model_id:
        raise WebApiError("OpenRouter API モデルIDが設定されていません", provider_name="OpenRouter")
```

---

## 7. 移行ガイドライン

### 従来実装からの主要変更点

| 機能 | 従来実装 | 新実装 | 変更理由 |
|------|----------|--------|----------|
| API呼び出し | OpenAI SDK直接 | PydanticAI Agent | 統一アーキテクチャ |
| JSON Schema | 分岐処理 | ネイティブ構造化出力 | 簡素化 |
| レスポンス解析 | 手動パース + コードブロック除去 | 自動変換 | 信頼性向上 |
| ヘッダー管理 | 個別設定 | Provider統合 | 管理性向上 |

### 既存コードからの移行

**従来のJSON Schema分岐処理**:
```python
# 廃止された実装
if json_schema_supported:
    response = self._call_openrouter_with_json_schema(...)
else:
    response = self._call_openrouter_without_json_schema(...)

# 手動レスポンス解析
if content_text.startswith("```json"):
    # コードブロック除去処理
    ...
```

**新しいPydanticAI実装**:
```python
# 新実装
agent = Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
result = await agent.run(user_prompt="...", message_history=[binary_content])
annotation = result.data  # 直接AnnotationSchema取得
```

---

## 8. OpenRouter固有機能

### 対応モデル

OpenRouter で利用可能な主要モデル:
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-haiku`
- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`
- **Google**: `google/gemini-pro-vision`, `google/gemini-flash`
- **Meta**: `meta-llama/llama-3.1-8b-instruct`
- **その他**: 100+ モデル対応

### カスタムヘッダー

```python
# OpenRouter推奨ヘッダー設定
default_headers = {
    "HTTP-Referer": "https://your-app.com",  # アプリURL
    "X-Title": "YourAppName"                 # アプリ名
}
```

### レート制限とコスト管理

- **レート制限**: モデル・プランに依存
- **コスト追跡**: OpenRouterダッシュボードで確認
- **クレジット管理**: 残高監視とアラート設定推奨

---

## 9. 性能最適化

### Agent Cache効果

- **初期化時間短縮**: Agent再利用による高速化
- **メモリ効率化**: LRU戦略による適切なリソース管理
- **設定変更検出**: 自動的なキャッシュ無効化

### OpenRouter固有最適化

- **モデル選択**: コスト・性能・速度のバランス
- **タイムアウト設定**: モデル特性に合わせた適切な値
- **バッチ処理**: 複数画像の効率的処理

---

## 10. テスト戦略

### 統合テストカバレッジ

1. **構造テスト**: 必要メソッド11個の存在確認
2. **画像前処理テスト**: PIL.Image → BinaryContent変換
3. **設定ハッシュ生成テスト**: Agent Cache用設定管理（OpenRouter固有項目含む）
4. **Agent作成モックテスト**: OpenAI Provider + カスタムヘッダー設定
5. **エラーハンドリングテスト**: OpenAI互換エラーパターン処理
6. **推論パイプラインテスト**: 全体フロー統合動作

### テストパターン

```python
# Agent作成テスト例（OpenRouter固有）
with patch("...OpenAIProvider") as mock_provider_class:
    # Agent作成実行
    agent = annotator._create_agent()
    
    # OpenRouter固有設定確認
    call_kwargs = mock_provider_class.call_args[1]
    assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert "HTTP-Referer" in call_kwargs["default_headers"]
    assert "X-Title" in call_kwargs["default_headers"]
```

---

## 11. 運用考慮事項

### モニタリング

- **レスポンス時間**: Agent初期化・推論実行時間
- **Agent Cache ヒット率**: 再利用効率
- **API呼び出し成功率**: エラー発生頻度
- **コスト追跡**: OpenRouterダッシュボード監視

### トラブルシューティング

**よくある問題**:
1. **Agent初期化失敗**: API キー・モデルID設定確認
2. **認証エラー**: OpenRouterアカウント・クレジット確認
3. **タイムアウト**: モデル特性に合わせた timeout 調整

**デバッグ手順**:
1. ログレベルをDEBUGに設定
2. Agent作成プロセスの確認
3. OpenRouterダッシュボードでAPI使用状況確認
4. モデル可用性とレート制限確認

---

## 12. 他プロバイダーとの統合

### プロバイダー間統合状況

| プロバイダー | 統合状況 | 特徴 |
|-------------|----------|------|
| **OpenAI** | ✅ 完了 | Response parsing 形式 |
| **Google** | ✅ 完了 | Gemini ネイティブ API |
| **Anthropic** | ✅ 完了 | Tool Use Block 廃止 |
| **OpenRouter** | ✅ 完了 | OpenAI互換 + カスタムヘッダー |

### 統一パターンの確立

```python
# 全プロバイダー共通パターン
class AnyProviderAnnotator(WebApiBaseAnnotator):
    def __enter__(self):
        # Agent Cache統合
        cache_key = create_cache_key(...)
        config_hash = self._get_config_hash()
        self.agent = WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
        return self

    async def _run_inference_async(self, binary_content: BinaryContent):
        # 統一推論実行
        result = await self.agent.run(...)
        return result.data
```

---

## 13. 今後の拡張計画

### Phase 2D完了効果

- **4大プロバイダー統合**: OpenAI, Google, Anthropic, OpenRouter
- **統一Agent Factory準備**: 完全プロバイダー抽象化基盤
- **OpenRouter生態系**: 100+モデルへのアクセス

### 次期拡張

- **統一Agent Factory**: プロバイダー抽象化層
- **動的モデル選択**: コスト・性能・可用性に基づく自動選択
- **マルチプロバイダー**: 単一APIで複数プロバイダー利用

---

## 14. まとめ

### ✅ 実装成果

- **完全PydanticAI統合**: OpenAI互換アーキテクチャによる統一実装
- **OpenRouter固有機能**: カスタムヘッダー・モデル管理対応
- **Agent Cache統合**: 効率的リソース管理システム
- **既存互換性100%**: WebApiBaseAnnotator インターフェース維持

### 🎯 技術的価値

- **プロバイダー多様性**: 100+モデルへのアクセス提供
- **コスト最適化**: モデル比較・選択の柔軟性
- **OpenAI互換性**: 既存OpenAIコードの移植容易性
- **統一アーキテクチャ**: 4プロバイダー統一管理

### 🚀 **Phase 2D完了**

**4大プロバイダー統合達成**:
- OpenAI, Google, Anthropic, OpenRouter すべてPydanticAI統合完了
- 統一Agent Factory実装準備完了
- 次世代画像アノテーションプラットフォーム基盤確立

---

**実装責任者**: PydanticAI統合チーム  
**レビュー完了**: 2025-06-24  
**ステータス**: ✅ Phase 2D完了 → 統一Agent Factory準備完了