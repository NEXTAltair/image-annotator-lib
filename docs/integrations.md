# LiteLLM 統合ガイド

本ライブラリは [LiteLLM](https://github.com/BerriAI/litellm) を使用して WebAPI モデルを自動的に検出・管理します。

## 概要

LiteLLM は OpenAI / Anthropic / Google など 100 以上のプロバイダーの 2600 以上のモデルを pip パッケージに同梱しています。`litellm.supports_vision(model_id)` と `litellm.get_model_info(model_id)` を使い、Vision 対応かつ structured output 対応のモデルだけをアノテーション対象として登録します。

## インストール

`litellm` は本ライブラリの依存関係として自動的にインストールされます。

```bash
uv sync
```

## 公開 API

### `discover_available_vision_models(force_refresh: bool = False) -> dict[str, Any]`

利用可能な Vision 対応 WebAPI モデルの一覧を返します。

```python
from image_annotator_lib import discover_available_vision_models

# キャッシュから読み込み（available_api_models.toml が存在すれば再取得しない）
result = discover_available_vision_models()

if "models" in result:
    print(f"{len(result['models'])} 件のモデルが利用可能")
    for model_id in result["models"][:5]:
        print(f"  {model_id}")
elif "error" in result:
    print(f"取得失敗: {result['error']}")

# 強制再取得（LiteLLM DB を再スキャン）
result = discover_available_vision_models(force_refresh=True)
```

**戻り値:**

| キー | 型 | 説明 |
|---|---|---|
| `models` | `list[str]` | 全モデル ID（廃止モデルを含む） |
| `toml_data` | `dict[str, Any]` | 全モデルのメタデータ辞書 |
| `error` | `str` | エラー時のみ存在。`models` / `toml_data` は存在しない |

---

### `should_refresh(ttl_days: int | None = None) -> bool`

`config/available_api_models.toml` の `[meta] last_refresh` を確認して TTL 超過を判定します。

```python
from image_annotator_lib.core.api_model_discovery import should_refresh

if should_refresh():
    print("モデルリストの更新が必要です")

# TTL を明示的に指定（日数）
if should_refresh(ttl_days=1):
    print("1日以上経過しています")
```

`last_refresh` が記録されていない場合は常に `True` を返します。

---

### `trigger_background_refresh() -> threading.Thread`

バックグラウンドスレッドで `available_api_models.toml` を非同期更新します。呼び出し元をブロックしません。

```python
from image_annotator_lib.core.api_model_discovery import (
    should_refresh,
    trigger_background_refresh,
)

if should_refresh():
    thread = trigger_background_refresh()
    # thread は daemon スレッド。プロセス終了時に自動的に停止します。
    # 完了を待つ場合: thread.join(timeout=30)
```

同時に複数回呼ばれても 1 回だけ実行されます（内部でロックを使用）。

---

## `available_api_models.toml` のスキーマ

```toml
# === メタデータセクション ===
[meta]
last_refresh = "2026-04-28T09:30:00+00:00"  # ISO 8601 with UTC offset
schema_version = 1

# === モデルエントリ ===
[available_vision_models."google/gemini-2.5-pro"]
provider = "Google"
model_name_short = "Gemini 2.5 Pro"
display_name = "Google: Gemini 2.5 Pro"
mode = "chat"
max_input_tokens = 1048576
max_output_tokens = 8192
supports_vision = true
supports_response_schema = true
supports_function_calling = true
supports_tool_choice = true
created = "2025-01-01T00:00:00Z"
modality = "text+image->text"
input_modalities = ["text", "image"]
last_seen = "2026-04-28T09:30:00+00:00Z"
deprecated_on = None  # 廃止された場合は ISO 8601 タイムスタンプ
```

### フィールド説明

| フィールド | 説明 |
|---|---|
| `provider` | プロバイダー名（OpenAI / Anthropic / Google など） |
| `display_name` | UI 表示用の名前 |
| `supports_vision` | 画像入力対応。`true` のモデルだけ登録対象 |
| `supports_response_schema` | structured output 対応。`true` のモデルだけ登録対象 |
| `max_input_tokens` / `max_output_tokens` | LiteLLM DB 由来のトークン上限 |
| `last_seen` | 最後に API レスポンスに現れた時刻 |
| `deprecated_on` | 廃止が確認された時刻（`None` = アクティブ） |

---

## 環境変数

| 変数名 | 既定値 | 説明 |
|---|---|---|
| `IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS` | `7` | モデルリストの有効期間（日数）。この期間を超えると起動時にバックグラウンド refresh が実行される |
| `IMAGE_ANNOTATOR_SKIP_API_DISCOVERY` | `false` | `true` に設定すると起動時の API discovery を完全スキップ。CI / オフライン環境で有用 |
| `IMAGE_ANNOTATOR_ENABLE_OPENROUTER_FALLBACK` | `false` | `true` に設定すると LiteLLM 未収録モデル補完用の OpenRouter API fallback を有効化 |

```bash
# TTL を 14 日に変更
IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS=14 uv run lorairo

# Discovery を無効化してオフラインで起動
IMAGE_ANNOTATOR_SKIP_API_DISCOVERY=true uv run lorairo
```

---

## CI 週次更新

`.github/workflows/refresh-models.yml` が毎週月曜 09:00 UTC に以下を実行します:

1. LiteLLM を最新版にアップデート（`uv lock --upgrade-package litellm`）
2. `tools/check_api_model_discovery.py` を実行して `config/available_api_models.toml` を再生成
3. 変更がある場合は `chore/refresh-model-registry` ブランチに PR を自動作成

手動実行:

```bash
# GitHub Actions 手動トリガー
gh workflow run refresh-models.yml --repo NEXTAltair/image-annotator-lib

# ローカルで実行（プロジェクトルートから）
uv run python tools/check_api_model_discovery.py
```

---

## OpenRouter フォールバック

LiteLLM のローカル DB に未収録のモデル（OpenRouter 限定の free tier モデルなど）は、必要な場合のみ OpenRouter API から補完取得できます。既定では無効です。有効化した場合も、Vision・structured output・tool use 対応モデルだけを追加します。
