# ログ出力多重化バグの原因と解決策

## 概要
image-annotator-lib でログ出力が多重（同じ内容が2回以上出力）される問題が発生した。

## 原因
- `src/image_annotator_lib/core/utils.py` で loguru logger の初期化（logger.remove(), logger.add(...)）が、
  複数回importされることで何度も実行され、logger.addが多重に呼ばれていた。
- Pythonのimportは通常1回だが、相対importや絶対importの混在、
  モジュールのimport経路の違いで初期化処理が複数回走ることがある。
- その結果、loggerのsink（出力先）が重複登録され、同じログが複数回出力された。

## 設計変更（2025-04-19）
- loggerの初期化（`init_logger()`）およびレジストリ初期化（`register_annotators()`）を、
  import時の即時実行から**明示的な初期化関数でのみ実行**する設計に変更。
- `core/registry.py` の末尾で行っていた loggerによるログ出力・register_annotators()の即時実行を廃止。
- logger/レジストリ初期化はエントリーポイントやAPI利用時に明示的に呼ぶ。

## メリット
- loggerのsink設定前にログ出力が発生し、ログが出力されない問題を根本解決。
- import時の副作用（グローバル状態変更や重い処理）を排除し、テスト・再利用性・保守性が向上。
- Pythonのベストプラクティス（import時は副作用を持たせない）に準拠。
- ログ出力の信頼性・初期化順序の一貫性が向上。

## 影響
- logger/レジストリ初期化はエントリーポイントやAPI利用時に明示的に呼ぶ必要がある。
- 既存のimport時自動初期化に依存したコードは修正が必要。
- 既存のlogger利用コードには影響なし。

## 解決策
1. logger初期化処理を `init_logger` 関数として分離し、多重初期化を防ぐガードを追加。
2. loggerの初期化は `__init__.py` でプロセス起動時に1回だけ明示的に呼ぶ。
3. loggerのimport経路を統一し、絶対/相対importの混在を避ける。

## 効果
- ログ出力が1回だけとなり、可読性・運用性が向上。
- 不要なI/O負荷が削減される。
- 既存のlogger利用コードには影響なし。

--- 