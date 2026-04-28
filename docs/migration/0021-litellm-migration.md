# 移行ガイド: ADR 0021 LiteLLM-Driven WebAPI Model Registry

## 概要

ADR 0021 により、WebAPI モデル（OpenAI / Anthropic / Google など）の管理方法が変更されました。
`annotator_config.toml` に手動定義していた WebAPI モデルエントリは不要になり、LiteLLM が自動的に検出・管理します。

## 影響を受けるユーザー

以下のような設定を `config/annotator_config.toml` に記述していた場合に影響があります:

```toml
# 旧設定（削除対象）
["GPT-4o"]
class = "OpenAIApiAnnotator"
api_model_id = "gpt-4o"

["Claude 3.5 Sonnet"]
class = "AnthropicApiAnnotator"
api_model_id = "claude-3-5-sonnet-20241022"

["Gemini 2.0 Flash"]
class = "GoogleApiAnnotator"
api_model_id = "gemini-2.0-flash-exp"
```

## 移行手順

### 1. `annotator_config.toml` から WebAPI エントリを削除

WebAPI モデル（`class` が `*ApiAnnotator` 系のエントリ）を削除してください。ローカル ML モデル（ONNX / Transformers / CLIP など）のエントリは**そのまま維持**します。

**削除するエントリの判断基準:**

- `class = "OpenAIApiAnnotator"` / `"AnthropicApiAnnotator"` / `"GoogleApiAnnotator"` / `"OpenRouterAnnotator"` → **削除**
- `class = "ONNXBaseAnnotator"` / `"TransformersBaseAnnotator"` / `"TensorflowBaseAnnotator"` / `"ClipBaseAnnotator"` → **維持**

### 2. 自動検出の動作確認

ライブラリ（またはそれを組み込んだアプリケーション）を起動すると、初回起動時に LiteLLM DB から Vision 対応モデルが自動的に検出され、`config/available_api_models.toml` に保存されます。

```bash
# 手動で動作確認（プロジェクトルートから）
uv run python tools/check_api_model_discovery.py
```

### 3. モデルリストの確認

```python
from image_annotator_lib.core.simplified_agent_factory import (
    get_available_models,
    list_all_models,
)

# アクティブなモデルのみ（廃止モデルを除く）
active_models = get_available_models()
print("利用可能なモデル:", active_models)

# 全モデル（廃止モデルを含む）
all_models = list_all_models()
print("全モデル:", all_models)
```

## 変更点の詳細

### `annotator_config.toml`

| 変更前 | 変更後 |
|---|---|
| WebAPI モデルを手動定義 | WebAPI モデルは LiteLLM が自動管理 |
| OpenAI / Anthropic / Google のモデルを個別に列挙 | 不要（`config/available_api_models.toml` に自動生成） |
| 廃止モデルは手動削除が必要 | `deprecated_on` フィールドで自動的にフィルタ |

### `available_api_models.toml`

LiteLLM が自動生成・更新するキャッシュファイルです。手動編集は原則不要です。

```toml
# 自動生成される [meta] セクション
[meta]
last_refresh = "2026-04-28T09:30:00+00:00"
schema_version = 1

# モデルエントリも自動生成
[available_vision_models."openai/gpt-4o"]
provider = "OpenAI"
display_name = "OpenAI: gpt-4o"
...
```

## オフライン環境での動作

ネットワーク接続なしで起動する場合は、環境変数で discovery をスキップできます:

```bash
IMAGE_ANNOTATOR_SKIP_API_DISCOVERY=true uv run lorairo
```

この場合、既存の `config/available_api_models.toml` がそのまま使用されます。ファイルが存在しない場合は WebAPI モデルなしで起動します。

## TTL の調整

デフォルトでは 7 日間隔でバックグラウンド refresh が実行されます。間隔を変更するには:

```bash
# TTL を 14 日に変更
IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS=14 uv run lorairo

# CI / 自動化環境で即時 refresh したい場合（TTL を 0 に近い値に設定）
IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS=0 uv run python tools/check_api_model_discovery.py
```

## トラブルシューティング

### モデルが検出されない

1. LiteLLM のインストールを確認: `uv run python -c "import litellm; print(litellm.__version__)"`
2. キャッシュをリセット: `available_api_models.toml` を削除して再起動
3. 強制再取得: `uv run python tools/check_api_model_discovery.py`

### 廃止モデルが表示される

`get_available_models()` ではなく `list_all_models()` を使用している可能性があります。
`get_available_models()` は `deprecated_on` が設定されたモデルを自動的に除外します。
