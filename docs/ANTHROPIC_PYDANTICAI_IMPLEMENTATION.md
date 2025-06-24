# Anthropic API PydanticAI統合実装ガイド

**作成日**: 2025-06-24  
**ステータス**: ✅ 実装完了 - Phase 2B  
**対象**: Claude API PydanticAI化

---

## 1. 実装概要

### 🔄 実装の変更点

**従来の実装**:
- Anthropic Python SDK の直接利用
- Tool Use Block による構造化出力
- Base64画像処理とメッセージ構築

**新実装**:
- PydanticAI Agent + AnthropicProvider/AnthropicModel 統合
- ネイティブ構造化出力対応
- Agent Cache システム統合

### 📋 主要技術特徴

- **Agent ベースアーキテクチャ**: 統一的なPydanticAI インターフェース
- **ネイティブ構造化出力**: Tool Use Block 廃止によるシンプル化
- **Agent Cache 統合**: LRU戦略による効率的リソース管理
- **既存互換性保持**: WebApiBaseAnnotator インターフェース継承

---

## 2. アーキテクチャ設計

### コア実装パターン

```python
# Anthropic PydanticAI統合の基本構造
class AnthropicApiAnnotator(WebApiBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.agent: Agent | None = None
        self.api_key: SecretStr | None = None
        self.api_model_id: str | None = None

    def __enter__(self):
        """コンテキストマネージャーでAgent初期化"""
        self._load_configuration()
        cache_key = create_cache_key(self.model_name, "anthropic", self.api_model_id)
        config_hash = self._get_config_hash()
        
        def creator_func() -> Agent:
            return self._create_agent()
        
        self.agent = WebApiAgentCache.get_agent(cache_key, creator_func, config_hash)
        return self
```

### Agent作成プロセス

```python
def _create_agent(self) -> Agent:
    """新しいPydanticAI Agentを作成する"""
    provider = AnthropicProvider(api_key=self.api_key.get_secret_value())
    model = AnthropicModel(model_name=self.api_model_id, provider=provider)
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
    # Claude固有パラメータ
    temperature = config_registry.get(self.model_name, "temperature", default=0.7)
    max_tokens = config_registry.get(self.model_name, "max_output_tokens", default=1800)
    
    model_params = {
        "temperature": float(temperature) if temperature is not None else 0.7,
        "max_tokens": int(max_tokens) if max_tokens is not None else 1800,
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

### 包括的エラー処理

```python
def _handle_api_error(self, error: Exception) -> str:
    """API エラーを適切な例外に変換"""
    error_str = str(error)
    
    # 404エラー → ModelNotFoundError
    if "404" in error_str or "not_found_error" in error_str:
        import re
        m = re.search(r"model: ([\w\.\-\:]+)", error_str)
        model_name = m.group(1) if m else "不明"
        raise ModelNotFoundError(model_name)
    
    # その他のエラーパターン
    if "authentication" in error_str.lower():
        raise ApiAuthenticationError(f"Anthropic API 認証エラー: {error_str}")
    elif "rate limit" in error_str.lower():
        raise ApiRateLimitError(f"Anthropic API レート制限: {error_str}")
    elif "timeout" in error_str.lower():
        raise ApiTimeoutError(f"Anthropic API タイムアウト: {error_str}")
    elif "500" in error_str or "server error" in error_str.lower():
        raise ApiServerError(f"Anthropic API サーバーエラー: {error_str}")
    
    # 一般エラー
    return f"Anthropic API Error: {error_str}"
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
    }
    return create_config_hash(config_data)
```

### Cache統合パターン

```python
# Agent Cacheからの取得/作成
cache_key = create_cache_key(self.model_name, "anthropic", self.api_model_id)
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
class = "AnthropicApiAnnotator"
api_key = "your-anthropic-api-key"
api_model_id = "claude-3-5-sonnet-20241022"
temperature = 0.7
max_output_tokens = 1800
```

### 設定読み込み

```python
def _load_configuration(self):
    """設定を読み込む"""
    self.api_key = SecretStr(config_registry.get(self.model_name, "api_key", default=""))
    self.api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
    
    if not self.api_key.get_secret_value():
        raise WebApiError("Anthropic API キーが設定されていません", provider_name="Anthropic")
    if not self.api_model_id:
        raise WebApiError("Anthropic API モデルIDが設定されていません", provider_name="Anthropic")
```

---

## 7. 移行ガイドライン

### Google API統合との整合性

| 機能 | Google実装 | Anthropic実装 | 備考 |
|------|------------|---------------|------|
| Agent作成 | GoogleProvider | AnthropicProvider | プロバイダー差分のみ |
| 画像処理 | BinaryContent | BinaryContent | 統一フォーマット |
| Cache統合 | WebApiAgentCache | WebApiAgentCache | 完全共通 |
| エラー処理 | 統一例外体系 | 統一例外体系 | 同一パターン |

### 既存コードからの移行

**従来のTool Use Block**:
```python
# 廃止された実装
tools = [{
    "name": "Annotatejson",
    "description": "Parsing image annotation results to JSON",
    "input_schema": JSON_SCHEMA,
}]
response = client.messages.create(..., tools=tools)
```

**新しいPydanticAI実装**:
```python
# 新実装
agent = Agent(model=model, output_type=AnnotationSchema, system_prompt=BASE_PROMPT)
result = await agent.run(user_prompt="...", message_history=[binary_content])
annotation = result.data  # 直接AnnotationSchema取得
```

---

## 8. 性能最適化

### Claude固有最適化

- **温度設定**: Claude特性に合わせた適切な temperature 値
- **最大トークン**: 長文出力対応の max_tokens 設定
- **システムプロンプト**: Claude専用最適化プロンプト

### Agent Cache効果

- **初期化時間短縮**: Agent再利用による高速化
- **メモリ効率化**: LRU戦略による適切なリソース管理
- **設定変更検出**: 自動的なキャッシュ無効化

---

## 9. テスト戦略

### 統合テストカバレッジ

1. **構造テスト**: 必要メソッド11個の存在確認
2. **画像前処理テスト**: PIL.Image → BinaryContent変換
3. **設定ハッシュ生成テスト**: Agent Cache用設定管理
4. **Agent作成モックテスト**: AnthropicProvider/AnthropicModel/Agent作成
5. **エラーハンドリングテスト**: 6種類のAPIエラー適切処理
6. **推論パイプラインテスト**: 全体フロー統合動作

### テストパターン

```python
# Agent作成テスト例
with patch("...AnthropicProvider") as mock_provider_class, \
     patch("...AnthropicModel") as mock_model_class, \
     patch("...Agent") as mock_agent_class:
    
    # モック設定
    mock_provider = MagicMock()
    mock_provider_class.return_value = mock_provider
    
    # Agent作成実行
    agent = annotator._create_agent()
    
    # 呼び出し確認
    mock_provider_class.assert_called_once_with(api_key="test-api-key")
    mock_model_class.assert_called_once_with(
        model_name="claude-3-5-sonnet", 
        provider=mock_provider
    )
```

---

## 10. 運用考慮事項

### モニタリング

- **レスポンス時間**: Agent初期化・推論実行時間
- **Agent Cache ヒット率**: 再利用効率
- **API呼び出し成功率**: エラー発生頻度
- **メモリ使用量**: キャッシュ効率性

### トラブルシューティング

**よくある問題**:
1. **Agent初期化失敗**: API キー設定確認
2. **モデル未検出エラー**: model_id設定確認  
3. **キャッシュ無効化**: 設定変更時の自動更新確認

**デバッグ手順**:
1. ログレベルをDEBUGに設定
2. Agent作成プロセスの確認
3. 設定ハッシュ値の検証
4. API呼び出しパラメータの確認

---

## 11. 今後の拡張計画

### Phase 2C統合準備

- **統一Agent Factory**: プロバイダー抽象化層
- **共通エラーハンドリング**: PydanticAI例外の標準化  
- **性能ベンチマーク**: 実API呼び出し性能測定
- **統一設定システム**: API固有パラメータの自動マッピング

### 長期拡張性

- **Tool使用対応**: PydanticAI Tool機能統合
- **ストリーミング対応**: リアルタイム推論処理
- **マルチモーダル拡張**: 動画・音声対応検討

---

## 12. まとめ

### ✅ 実装成果

- **完全PydanticAI統合**: Tool Use Block廃止による構造化出力ネイティブサポート
- **Agent Cache統合**: 効率的リソース管理システム
- **既存互換性100%**: WebApiBaseAnnotator インターフェース維持
- **包括的テスト**: 6項目全テスト成功確認

### 🎯 次のステップ

- **Phase 2C**: 統一Agent Factory実装
- **実API統合テスト**: 実際のClaude API呼び出し検証
- **性能最適化**: Agent Cache効果測定とチューニング

---

**実装責任者**: PydanticAI統合チーム  
**レビュー完了**: 2025-06-24  
**ステータス**: ✅ Phase 2B完了 → Phase 2C準備完了