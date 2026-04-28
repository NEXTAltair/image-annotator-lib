## モデルレジストリ更新 PR

### 変更内容
<!-- 追加・削除・変更されたモデルを記載してください -->

- 新規モデル:
- 廃止モデル:
- LiteLLM バージョン変更: (例: 1.x.y → 1.x.z)

### 確認手順

- [ ] `config/available_api_models.toml` の `[meta] last_refresh` が更新されている
- [ ] 新規モデルの `provider` / `display_name` が正しい
- [ ] 廃止モデルに `deprecated_on` タイムスタンプが設定されている
- [ ] CI (lint / typecheck / unit tests) が通過している
