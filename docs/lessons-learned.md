# Lessons Learned

## 2025-07-01: Integration Test Design and PydanticAI Testing Patterns

### 問題の概要
Memory management integration testでテスト失敗が多発。主な問題：
1. 複雑すぎるモック設定
2. PydanticAIテストパターンの誤解
3. 実装の詳細に過度に依存したテスト設計

### 学んだ教訓

#### 1. PydanticAIテストのベストプラクティス
**正しいアプローチ:**
- `TestModel`または`FunctionModel`を使用してLLMコールをモック
- `Agent.override()`でAgentのモデルを置き換え
- `models.ALLOW_MODEL_REQUESTS=False`で実APIコールを防止

**間違ったアプローチ:**
- 内部のPydanticAI Agentを直接モック
- 複雑な`side_effect`でAsyncMockを設定
- Provider factoryの内部実装詳細をモック

#### 2. テスト設計の原則
**堅牢なテスト設計:**
```python
# 良い例：本質的な機能をテスト
def test_cache_functionality(self):
    cache_info = WebApiAgentCache.get_cache_info()
    assert cache_info["cache_size"] == 0
```

**脆弱なテスト設計:**
```python
# 悪い例：実装詳細に依存
def mock_agent_creation(model_name, api_model_id, api_key, config_data=None):
    # 複雑な内部状態シミュレーション
```

#### 3. 段階的テスト開発
**推奨アプローチ:**
1. 最初に単純で本質的な機能をテスト
2. 段階的に複雑さを追加
3. 実装詳細ではなく公開インターフェースをテスト

**事例:**
- `test_memory_management_simple.py`を作成
- 基本的なキャッシュ操作とメモリクリーンアップをテスト
- 複雑なE2Eテストは後回し

#### 4. モック設定のベストプラクティス
**設定不備の回避:**
```python
# 必要な設定項目を事前に確認
"memory_local_small": {
    "class": "ImprovedAesthetic",
    "model_path": "test/small/model",
    "base_model": "improved-aesthetic",  # 必須項目追加
    "device": "cpu",
    "estimated_size_gb": 0.5,
}
```

**クラス登録の確実な実行:**
```python
# conftest.pyでテスト用クラスマッピングを確実に設定
def _ensure_test_class_mapping(model_name: str, config: dict):
    # 実際のクラス名との対応を正確に設定
    if class_name == "OpenAIApiChatAnnotator":
        from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import OpenRouterApiAnnotator
        registry[model_name] = OpenRouterApiAnnotator
```

#### 5. エラーパターンの分類と対策

**設定エラー:**
- 症状：`base_model が設定されていません`
- 対策：テスト設定でモデル固有の必須パラメータを確認

**モックエラー:**
- 症状：`object MagicMock can't be used in 'await' expression`
- 対策：PydanticAIの正しいテストパターンを使用

**クラス登録エラー:**
- 症状：`Model 'X' not found in class registry`
- 対策：テスト用のクラスマッピングを事前に設定

### 技術的な解決策

#### 1. テスト環境でのPydanticAI設定
```python
@pytest.fixture(autouse=True)
def setup_pydantic_ai_testing(self):
    # APIコールを無効化
    models.ALLOW_MODEL_REQUESTS = False
    yield
```

#### 2. 段階的モック戦略
```python
# Level 1: 基本的なキャッシュ機能テスト
def test_cache_basic_operations():
    # 内部実装に依存しないテスト
    
# Level 2: 設定とクラス登録テスト  
def test_model_configuration():
    # 設定が正しく読み込まれることを確認

# Level 3: E2Eテスト（慎重に設計）
def test_full_annotation_workflow():
    # TestModelを使用した実際のワークフローテスト
```

#### 3. 堅牢なテストfixtureパターン
```python
@pytest.fixture
def memory_test_configs(self, managed_config_registry):
    configs = {...}  # 必須パラメータを全て含む
    
    for model_name, config in configs.items():
        managed_config_registry.set(model_name, config)
        _ensure_test_class_mapping(model_name, config)  # 確実に登録
    
    return configs
```

### 今後の開発指針

1. **テストファーストではなく、理解ファースト**: 複雑なシステムでは、まず動作原理を理解してからテストを書く

2. **段階的複雑化**: 単純なテストから始めて、段階的に複雑さを追加

3. **公開インターフェースのテスト**: 実装詳細ではなく、公開されたAPIをテスト

4. **フレームワーク固有のベストプラクティスを尊重**: PydanticAIのようなフレームワークには推奨テストパターンがある

5. **エラードキュメント化**: 遭遇したエラーパターンと解決策を記録

### 参考リソース

- [PydanticAI Unit Testing Documentation](https://ai.pydantic.dev/unit-testing/)
- プロジェクト内の`test_memory_management_simple.py`（段階的テスト設計の例）
- `tests/integration/conftest.py`（テスト用クラス登録パターン）

この経験により、複雑なシステムのintegration testでは「動作する最小版から始める」アプローチが重要であることを再確認。