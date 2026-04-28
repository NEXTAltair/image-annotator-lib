# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — ADR 0021: LiteLLM-Driven WebAPI Model Registry

この変更セットは [ADR 0021](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0021-litellm-driven-model-registry.md) に基づく WebAPI モデルレジストリの刷新です。

### Added

- **LiteLLM 統合** (ISSUE B): `core/api_model_discovery.py` に LiteLLM ローカル DB を使用した Vision モデル自動検出機能を追加
- **`deprecated_on` フィルタ** (ISSUE C): `simplified_agent_factory.py` に `get_available_models()` / `list_all_models()` / `is_model_deprecated()` を追加
- **TTL ベース自動 refresh** (ISSUE E): `core/api_model_discovery.py` に `should_refresh()` / `trigger_background_refresh()` を追加
- **`[meta] last_refresh`**: `config/available_api_models.toml` に refresh タイムスタンプのメタデータセクションを導入
- **アトミック書き込み**: `save_available_api_models()` を `tempfile + os.replace()` によるアトミック書き込みに変更
- **CI 週次 refresh** (ISSUE G): `.github/workflows/refresh-models.yml` による週次自動更新と PR 自動作成
- **環境変数サポート**: `IMAGE_ANNOTATOR_API_MODELS_TTL_DAYS` / `IMAGE_ANNOTATOR_SKIP_API_DISCOVERY` で動作を制御可能

### Changed

- **`annotator_config.toml` の WebAPI セクション削除** (ISSUE D): WebAPI モデルの手動定義を廃止。モデルリストは LiteLLM により動的に管理される
  - **破壊的変更**: 詳細は [`docs/migration/0021-litellm-migration.md`](docs/migration/0021-litellm-migration.md) を参照
- **`core/registry.py` の起動シーケンス**: TTL 判定に基づいて同期 fetch / バックグラウンド refresh / スキップを自動切り替え
- **`discover_available_vision_models()` の戻り値**: `{"models": list[str], "toml_data": dict}` 形式に統一

### Deprecated

- `annotator_config.toml` への WebAPI モデルの手動定義（既存設定は無視される）

### Fixed

- OpenRouter API フォールバックが LiteLLM 未収録モデルを補完するよう改善
- 起動時の同期 API fetch が UI/CLI 起動をブロックしていた問題を修正（バックグラウンド化）
