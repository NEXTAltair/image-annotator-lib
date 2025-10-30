# Phase 1A Investigation Summary - Config Registry Usage Analysis

## 調査概要

**調査日**: 2025-10-29
**対象**: image-annotator-lib の config_registry 使用状況
**目的**: Phase 1A（Config Objects導入）の詳細計画策定

## 主要な発見

### 1. config_registry使用箇所の統計

- **総使用箇所**: 48箇所（Grepで確認）
- **使用パターン別分類**:
  - 設定読み取り（get）: 38箇所（79%）
  - 設定書き込み（set_system_value, add_default_setting）: 4箇所（8%）
  - 全設定取得（get_all_config）: 3箇所（6%）
  - テスト用（set）: 3箇所（6%）

### 2. 最も頻繁に使用されるファイル（Top 10）

1. `core/base/pydantic_ai_annotator.py` - 12箇所（PydanticAI設定）
2. `core/base/webapi.py` - 8箇所（WebAPI基底クラス）
3. `core/provider_manager.py` - 7箇所（プロバイダー管理）
4. `core/base/annotator.py` - 2箇所（基底クラス）
5. `core/pydantic_ai_factory.py` - 4箇所（Agent Factory）
6. `tests/integration/conftest.py` - 3箇所（テストfixture）
7. `model_class/tagger_onnx.py` - 2箇所（タグ閾値）
8. `model_class/annotator_webapi/openai_api_chat.py` - 4箇所（OpenRouter設定）
9. `core/registry.py` - 5箇所（モデル登録・メタデータ）
10. `core/model_factory.py` - 3箇所（メモリ管理）

### 3. 設定項目の種類

#### A. ローカルMLモデル設定（16項目）
```
必須: model_path, class, device, estimated_size_gb, type
オプショナル: tag_threshold, batch_size, max_length, processor_path, 
              base_model, activation_type, final_activation_type, 
              model_format, num_beams, task
```

#### B. WebAPIモデル設定（24項目）
```
必須: class, api_model_id, provider, max_output_tokens
APIキー: api_key
推論: temperature, timeout, retry_count, retry_delay, min_request_interval
プロンプト: prompt_template
PydanticAI: retries, output_retries, enable_streaming, instrument, 
            focus_quality, analyze_style, custom_instructions, 
            top_p, seed
OpenRouter: referer, app_name
```

#### C. 動的API設定（7項目）
```
provider, model_name_short, display_name, created, modality, 
input_modalities, last_seen, deprecated_on
```

### 4. 重要なアーキテクチャパターン

#### パターン1: 階層的設定読み取り
```python
# core/base/annotator.py (基底クラス)
self.model_path = config_registry.get(model_name, "model_path")
self.device = config_registry.get(model_name, "device", "cpu")

# core/base/webapi.py (WebAPI基底クラス)
self.timeout = int(config_registry.get(self.model_name, "timeout", 60))

# model_class/tagger_onnx.py (具体実装)
self.tag_threshold = config_registry.get(self.model_name, "tag_threshold", 0.35)
```

**インサイト**: 基底クラスでの移行が下流の全実装に影響

#### パターン2: プロバイダー判定ロジック
```python
# core/provider_manager.py
provider = config_registry.get(model_name, "provider")
if config_registry.get(model_name, "anthropic_api_key"):
    api_key = ...
elif config_registry.get(model_name, "openai_api_key"):
    api_key = ...
```

**インサイト**: 型安全な設定クラスで条件分岐を簡素化可能

#### パターン3: メタデータ生成
```python
# core/registry.py - list_available_annotators_with_metadata()
all_config = config_registry.get_all_config()
for model_name in _MODEL_CLASS_OBJ_REGISTRY.keys():
    model_config = all_config.get(model_name, {})
    metadata = {
        "class": model_class.__name__,
        "provider": model_config.get("provider"),
        "api_model_id": model_config.get("api_model_id"),
        ...
    }
```

**インサイト**: 設定オブジェクトから直接メタデータを生成可能

### 5. テスト環境での使用パターン

#### managed_config_registry fixture
```python
# tests/integration/conftest.py
@pytest.fixture(scope="function")
def managed_config_registry():
    # グローバル状態を保存
    original_system = copy.deepcopy(config_registry._system_config_data)
    original_user = copy.deepcopy(config_registry._user_config_data)
    original_merged = copy.deepcopy(config_registry._merged_config_data)
    
    # クリーンな状態を提供
    config_registry._system_config_data.clear()
    config_registry._user_config_data.clear()
    config_registry._merged_config_data.clear()
    
    def _set_config(model_name: str, config: dict):
        config_registry._merged_config_data[model_name] = config
        config_registry._user_config_data[model_name] = config
    
    yield config_registry
    
    # 状態を復元
    config_registry._system_config_data = original_system
    ...
```

**インサイト**: 新API導入時も同様のパターンをサポート必要

### 6. 特殊なケース

#### ケース1: 動的なestimated_size_gb更新
```python
# core/model_factory.py
estimated_size_gb_any = config_registry.get(model_name, "estimated_size_gb")
# ... サイズ計算 ...
config_registry.set_system_value(model_name, "estimated_size_gb", round(size_gb, 3))
config_registry.save_system_config()
```

**インサイト**: 書き込み処理は既存APIを維持、読み取りのみ新APIへ

#### ケース2: API自動検出での設定追加
```python
# core/registry.py - _update_config_with_api_models()
for model_id, model_info in api_models.items():
    model_name_short = model_info.get("model_name_short")
    config_registry.add_default_setting(model_name_short, "class", target_class_name)
    config_registry.add_default_setting(model_name_short, "max_output_tokens", 1800)
    config_registry.add_default_setting(model_name_short, "api_model_id", model_id)
```

**インサイト**: 動的モデル追加のため、柔軟な設計が必要

## 推奨事項

### 優先度1: 基底クラスの早期移行
- `core/base/annotator.py` - 全モデルに影響
- `core/base/webapi.py` - 全WebAPIモデルに影響
- **理由**: 下流への波及効果が大きい

### 優先度2: Provider-level統合
- `core/provider_manager.py` - プロバイダー判定の型安全化
- `core/pydantic_ai_factory.py` - Agent設定の統合
- **理由**: WebAPI統合の中核

### 優先度3: メタデータ生成
- `core/registry.py` - list_available_annotators_with_metadata()
- **理由**: LoRAIro統合での使用頻度が高い

## リスク評価

### 高リスク領域
1. **テスト環境管理** - managed_config_registry fixtureの複雑性
2. **動的設定更新** - estimated_size_gbの書き込み処理
3. **API自動検出** - add_default_settingの使用

### 低リスク領域
1. **読み取り専用箇所** - 大部分のget()呼び出し
2. **基底クラス** - 変更の影響範囲が明確
3. **型チェック** - Pydanticによる自動検証

## 次のステップ

1. **Week 1**: `config_models.py` と新APIの実装
2. **Week 2**: 基底クラス（annotator.py, webapi.py）の移行
3. **Week 3**: Provider-level統合の移行
4. **Week 4**: テストコードの移行と最終検証

## 参考データ

### ファイル別使用頻度（詳細）
```
12: core/base/pydantic_ai_annotator.py
 8: core/base/webapi.py
 7: core/provider_manager.py
 5: core/registry.py
 4: core/pydantic_ai_factory.py
 4: model_class/annotator_webapi/openai_api_chat.py
 3: core/model_factory.py
 3: tests/integration/conftest.py
 2: core/base/annotator.py
 2: model_class/tagger_onnx.py
 2: core/base/transformers.py
 2: core/base/clip.py
 2: model_class/tagger_tensorflow.py
 2: core/base/pipeline.py
 2: core/base/captioner.py
 1: api.py
 1: core/utils.py
 1: core/base/tensorflow.py
```

### 設定キーの使用頻度（Top 15）
```
api_model_id: 6回
temperature: 5回
max_output_tokens: 5回
timeout: 4回
api_key: 4回
device: 3回
model_path: 2回
retry_count: 2回
retry_delay: 2回
min_request_interval: 2回
tag_threshold: 2回
provider: 2回
capabilities: 1回
estimated_size_gb: 1回
prompt_template: 1回
```

## 結論

Phase 1Aの実装は、段階的な移行戦略により、**後方互換性を維持しながら型安全性を導入可能**です。

重要な成功要因:
1. 基底クラスの早期移行（Week 2）
2. managed_config_registry fixtureの強化（Week 4）
3. 既存テストの継続的実行（各Week）

予想される効果:
- **型安全性**: 100%のカバレッジ（新API使用箇所）
- **保守性**: 設定エラーの早期発見
- **テスト容易性**: モック化の簡素化
