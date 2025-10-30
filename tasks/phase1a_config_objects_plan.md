# Phase 1A: Config Objects導入 - 詳細実装計画

## 目的
グローバル config_registry への依存を減らし、型安全な設定オブジェクトを導入する

## 背景分析

### 現在のconfig_registry使用パターン

#### 1. 使用箇所の分類（48箇所の使用を確認）

**A. モデル設定の読み取り（最も多い）**
- `config_registry.get(model_name, key, default)` - 38箇所
- 主な使用場所:
  - `core/base/annotator.py` - model_path, device
  - `core/base/webapi.py` - timeout, retry_count, retry_delay, min_request_interval, max_output_tokens, prompt_template
  - `core/base/pydantic_ai_annotator.py` - api_model_id, temperature, max_tokens, timeout, top_p, seed, retries, etc.
  - `core/pydantic_ai_factory.py` - api_key, api_model_id, temperature, max_output_tokens
  - `core/provider_manager.py` - provider, api_key (複数プロバイダー)
  - `model_class/*.py` - tag_threshold, batch_size, max_length, num_beams, etc.

**B. システム設定の書き込み**
- `config_registry.set_system_value()` - 1箇所（model_factory.py: estimated_size_gb）
- `config_registry.save_system_config()` - 1箇所
- `config_registry.add_default_setting()` - 3箇所（registry.py: WebAPIモデルの自動設定）

**C. 全設定の取得**
- `config_registry.get_all_config()` - 3箇所（registry.py, list_available_annotators_with_metadata）

**D. 設定の更新（主にテスト）**
- `config_registry.set()` - テストのみで使用（managed_config_registry fixture）

### 設定項目の種類

#### 1. ローカルMLモデル設定（例: WDTagger, ImprovedAesthetic）
```python
model_path: str
class: str
device: str = "cuda"
estimated_size_gb: float
type: str  # "scorer", "tagger", "captioner"
# オプショナル
tag_threshold: float = 0.35
batch_size: int = 8
max_length: int = 75
processor_path: str | None = None
base_model: str | None = None  # CLIP系
activation_type: str | None = None
final_activation_type: str | None = None
model_format: str = "h5"  # TensorFlow
```

#### 2. WebAPIモデル設定（例: OpenAI, Anthropic, Google）
```python
class: str = "PydanticAIWebAPIAnnotator"
api_model_id: str  # "openai/gpt-4o", "anthropic/claude-3-5-sonnet"
provider: str  # "OpenAI", "Anthropic", "Google", "OpenRouter"
max_output_tokens: int = 1800
# オプショナル
api_key: str = ""
temperature: float = 0.7
timeout: int = 60
retry_count: int = 3
retry_delay: float = 1.0
min_request_interval: float = 1.0
prompt_template: str | None = None
# PydanticAI固有
retries: int = 3
output_retries: int = 2
enable_streaming: bool = False
instrument: bool = True
focus_quality: bool = False
analyze_style: bool = False
custom_instructions: str = ""
top_p: float | None = None
seed: int | None = None
# OpenRouter固有
referer: str | None = None
app_name: str | None = None
```

#### 3. 動的API設定（available_api_models.toml）
```python
provider: str
model_name_short: str
display_name: str
created: str
modality: str
input_modalities: list[str]
last_seen: str
deprecated_on: str | None = None
```

## 設計案

### Phase 1A-1: 型安全な設定クラスの導入

#### 1. 基本設計クラス

```python
# src/image_annotator_lib/core/config_models.py

from typing import Literal
from pydantic import BaseModel, Field

class BaseModelConfig(BaseModel):
    """全モデル共通の基本設定"""
    class_: str = Field(alias="class")
    estimated_size_gb: float | None = None
    type: Literal["scorer", "tagger", "captioner", "webapi"] | None = None
    
    class Config:
        populate_by_name = True  # 'class'エイリアスを許可

class LocalMLModelConfig(BaseModelConfig):
    """ローカルMLモデルの設定"""
    model_path: str
    device: str = "cuda"
    
    # オプショナル設定（サブクラスで使用）
    tag_threshold: float | None = None
    batch_size: int | None = None
    max_length: int | None = None
    processor_path: str | None = None
    base_model: str | None = None
    activation_type: str | None = None
    final_activation_type: str | None = None
    model_format: str = "h5"
    num_beams: int | None = None
    task: str | None = None

class WebAPIModelConfig(BaseModelConfig):
    """WebAPIモデルの設定"""
    api_model_id: str
    provider: str
    max_output_tokens: int = 1800
    
    # APIキー関連
    api_key: str = ""
    
    # 推論パラメータ
    temperature: float = 0.7
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    min_request_interval: float = 1.0
    
    # プロンプト
    prompt_template: str | None = None
    
    # PydanticAI固有
    retries: int = 3
    output_retries: int = 2
    enable_streaming: bool = False
    instrument: bool = True
    focus_quality: bool = False
    analyze_style: bool = False
    custom_instructions: str = ""
    top_p: float | None = None
    seed: int | None = None
    
    # OpenRouter固有
    referer: str | None = None
    app_name: str | None = None

class ModelConfigFactory:
    """設定辞書から適切な設定オブジェクトを生成"""
    
    @staticmethod
    def create(model_name: str, config_dict: dict) -> BaseModelConfig:
        """設定辞書から適切な設定オブジェクトを生成
        
        Args:
            model_name: モデル名
            config_dict: config_registryから取得した設定辞書
            
        Returns:
            型安全な設定オブジェクト
        """
        # api_model_idがあればWebAPI、なければローカルML
        if "api_model_id" in config_dict:
            return WebAPIModelConfig(**config_dict)
        else:
            return LocalMLModelConfig(**config_dict)
```

#### 2. ConfigRegistry の拡張

```python
# src/image_annotator_lib/core/config.py に追加

class ModelConfigRegistry:
    # 既存のメソッドはそのまま維持
    
    def get_model_config(self, model_name: str) -> BaseModelConfig:
        """型安全な設定オブジェクトを取得
        
        Args:
            model_name: モデル名
            
        Returns:
            型安全な設定オブジェクト
            
        Raises:
            KeyError: モデルが存在しない場合
        """
        if model_name not in self._merged_config_data:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        
        config_dict = self._merged_config_data[model_name]
        return ModelConfigFactory.create(model_name, config_dict)
    
    def get_model_config_safe(self, model_name: str) -> BaseModelConfig | None:
        """型安全な設定オブジェクトを取得（エラーセーフ）
        
        Args:
            model_name: モデル名
            
        Returns:
            型安全な設定オブジェクト。存在しない場合はNone
        """
        try:
            return self.get_model_config(model_name)
        except KeyError:
            return None
```

### Phase 1A-2: 段階的移行戦略

#### ステップ1: 新APIの追加（既存コード影響なし）
- `config_models.py` の追加
- `ModelConfigRegistry.get_model_config()` の追加
- 既存の `get()` メソッドは維持
- **テストポイント**: 新APIが正しく動作すること

#### ステップ2: 高頻度使用箇所の移行
**対象ファイル（優先順位順）:**

1. `core/base/annotator.py` - 全アノテーターの基底クラス
   ```python
   # Before
   self.model_path = config_registry.get(model_name, "model_path")
   self.device = config_registry.get(model_name, "device", "cpu")
   
   # After
   config = config_registry.get_model_config(model_name)
   self.model_path = config.model_path
   self.device = config.device
   ```

2. `core/base/webapi.py` - WebAPI系の基底クラス
   ```python
   # Before
   self.timeout = int(config_registry.get(self.model_name, "timeout", 60))
   self.retry_count = int(config_registry.get(self.model_name, "retry_count", 3))
   
   # After
   config = config_registry.get_model_config(self.model_name)
   assert isinstance(config, WebAPIModelConfig)
   self.timeout = config.timeout
   self.retry_count = config.retry_count
   ```

3. `core/pydantic_ai_factory.py` - PydanticAI統合
   ```python
   # Before
   self.api_key = SecretStr(config_registry.get(self.model_name, "api_key", default=""))
   self.api_model_id = config_registry.get(self.model_name, "api_model_id", default=None)
   
   # After
   config = config_registry.get_model_config(self.model_name)
   assert isinstance(config, WebAPIModelConfig)
   self.api_key = SecretStr(config.api_key)
   self.api_model_id = config.api_model_id
   ```

**テストポイント**: 各ファイル移行後、対応するテストが全て通ること

#### ステップ3: 特殊なケースの移行
1. `core/model_factory.py` - estimated_size_gbの動的更新
   - 書き込み処理は既存のまま維持（後方互換性）
   - 読み取りは新APIに移行

2. `core/registry.py` - get_all_config()の使用
   - list_available_annotators_with_metadata()で使用
   - 段階的に新APIに移行

3. `core/provider_manager.py` - 複数プロバイダーのキー取得
   - provider判定ロジックを型安全に

**テストポイント**: 特殊処理が正しく動作すること

#### ステップ4: テストコードの移行
1. `tests/integration/conftest.py` - managed_config_registry fixture
   - 新APIをサポート
   - 既存のset()互換性維持

2. 各テストファイル
   - 新APIを使用するように段階的に移行

### Phase 1A-3: 後方互換性の維持

#### 移行期間中の戦略
```python
# core/config.py

class ModelConfigRegistry:
    def get(self, model_name: str, key: str, default: Any = None) -> Any | None:
        """後方互換性のための既存API
        
        Note:
            このメソッドは非推奨です。get_model_config()を使用してください。
        """
        # 既存の実装を維持
        if model_name not in self._merged_config_data:
            return default
        model_config = self._merged_config_data[model_name]
        return model_config.get(key, default)
```

#### Deprecation Warning（Phase 2で導入予定）
```python
import warnings

def get(self, model_name: str, key: str, default: Any = None) -> Any | None:
    warnings.warn(
        "config_registry.get() is deprecated. Use get_model_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # 実装...
```

### Phase 1A-4: テスト戦略

#### 1. 新API単体テスト
```python
# tests/unit/core/test_config_models.py

def test_local_ml_model_config_creation():
    config_dict = {
        "class": "WDTagger",
        "model_path": "deepghs/wd-vit-tagger-v3",
        "device": "cuda",
        "estimated_size_gb": 0.529,
        "type": "tagger",
        "tag_threshold": 0.35
    }
    config = ModelConfigFactory.create("test_model", config_dict)
    assert isinstance(config, LocalMLModelConfig)
    assert config.class_ == "WDTagger"
    assert config.tag_threshold == 0.35

def test_webapi_model_config_creation():
    config_dict = {
        "class": "PydanticAIWebAPIAnnotator",
        "api_model_id": "openai/gpt-4o",
        "provider": "OpenAI",
        "max_output_tokens": 1800,
        "temperature": 0.7
    }
    config = ModelConfigFactory.create("test_model", config_dict)
    assert isinstance(config, WebAPIModelConfig)
    assert config.api_model_id == "openai/gpt-4o"
    assert config.temperature == 0.7
```

#### 2. 統合テスト
- 既存の全統合テストを実行
- 新API使用後も同じ結果になることを確認

#### 3. 回帰テスト
- Phase 2のテストスイートを実行
- エラーハンドリングが正しく動作すること

## 実装順序

### Week 1: 基礎実装
- [ ] Day 1: `config_models.py` 実装
- [ ] Day 2: `ModelConfigRegistry.get_model_config()` 実装
- [ ] Day 3: 単体テスト作成・実行

### Week 2: 基底クラス移行
- [ ] Day 1: `core/base/annotator.py` 移行
- [ ] Day 2: `core/base/webapi.py` 移行
- [ ] Day 3: 統合テスト実行・修正

### Week 3: 高頻度使用箇所移行
- [ ] Day 1: `core/pydantic_ai_factory.py` 移行
- [ ] Day 2: `core/provider_manager.py` 移行
- [ ] Day 3: 統合テスト実行・修正

### Week 4: 特殊ケース・テスト移行
- [ ] Day 1: `core/model_factory.py`, `core/registry.py` 移行
- [ ] Day 2: テストコード移行
- [ ] Day 3: 全テスト実行・回帰テスト

## 成功基準

1. **型安全性**: 全ての設定アクセスが型チェック可能
2. **後方互換性**: 既存のテストが全て通ること
3. **テストカバレッジ**: 75%以上を維持
4. **パフォーマンス**: 設定読み取り速度が5%以内の変動

## リスクと緩和策

### リスク1: 予期しない設定項目の発見
**緩和策**: 
- Pydantic の `Extra.allow` を使用して未知のフィールドを許容
- ログで警告を出力

### リスク2: テスト環境での設定管理の複雑化
**緩和策**:
- `managed_config_registry` fixture を強化
- テスト用のヘルパー関数を提供

### リスク3: パフォーマンス低下
**緩和策**:
- 設定オブジェクトのキャッシング
- Pydantic の最適化オプション使用

## 次のPhase（Phase 1B: API Keys Management）への接続

Phase 1Aで型安全な設定オブジェクトが導入されることで、Phase 1BでのAPIキー管理が容易になります：

1. `WebAPIModelConfig` にキー管理ロジックを追加
2. 環境変数からの読み取りを統合
3. テスト時のモック化が容易に

## 参考資料

- Pydantic V2 Documentation: https://docs.pydantic.dev/latest/
- Python Type Hints: https://docs.python.org/3/library/typing.html
- 既存のPhase 2実装: `/workspaces/LoRAIro/local_packages/image-annotator-lib/src/image_annotator_lib/core/types.py`
