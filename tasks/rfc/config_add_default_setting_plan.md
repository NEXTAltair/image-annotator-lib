# config.py への add_default_setting メソッド追加計画

## 1. 目的

`src/image_annotator_lib/core/config.py` の `ModelConfigRegistry` クラスに、既存の設定を上書きせずに、指定されたセクションにキーと値のペアを追加する新しいメソッド `add_default_setting` を実装する。このメソッドは、設定が実際に追加された場合に、変更をシステム設定ファイルに自動的に保存する。**(当面の主な目的は、Web API モデルのデフォルト設定を自動追加することである。)**

## 2. 背景

*   `webapi_model_discovery_integration_plan.md` で定義されている通り、**現在の主な動機として、** `available_api_models.toml` から動的に取得した Web API モデル情報に基づいて、`annotator_config.toml` (システム設定) にデフォルト設定 (例: `class` 名、`max_output_tokens`) を自動追加する必要がある。
*   この自動追加処理は、ユーザーが既に `annotator_config.toml` で行っているかもしれないカスタム設定や、既存のモデル設定を意図せず上書きしてはならない。
*   そのため、指定されたキーがセクション内に存在しない場合にのみ値を追加し、変更があった場合は自動でファイルに保存する、非破壊的な設定追加メソッドが必要となる。**このメソッドは汎用的に設計されるが、当面は Web API モデルのデフォルト設定追加に使用される。**

## 3. 仕様詳細

### 3.1. メソッドシグネチャ

```python
def add_default_setting(self, section_name: str, key: str, value: Any) -> None:
    """
    システム設定データにデフォルト値を設定し、変更があればファイルに保存する。

    指定されたセクションが存在しない場合は作成する。
    指定されたキーがセクション内に存在しない場合のみ、値を設定する。
    既存のキーの値は上書きしない。
    値を設定した場合（データが変更された場合）のみ、システム設定ファイルを保存する。

    Args:
        section_name: 設定を追加するセクション名 (モデル名など)。
        key: 追加する設定のキー。
        value: 追加する設定の値。
    """
```

### 3.2. 対象データ

*   このメソッドは、`ModelConfigRegistry` インスタンスの `self._system_config_data` (システム設定データ) を変更対象とする。

### 3.3. 動作詳細

1.  `setting_added = False` フラグを初期化する。
2.  `self._system_config_data` 内に `section_name` のキーが存在するか確認する。
3.  `section_name` が存在しない場合:
    *   `self._system_config_data[section_name] = {}` として新しい空の辞書を作成する。
    *   `logger.debug` などで新しいセクションが作成されたことを記録する (任意)。
4.  `self._system_config_data[section_name]` 内に `key` が存在するか確認する。
5.  `key` が **存在しない** 場合:
    *   `self._system_config_data[section_name][key] = value` として値を設定する。
    *   `logger.info` などで、デフォルト設定が追加されたこと (セクション、キー、値) を記録する。
    *   `setting_added = True` に設定する。
6.  `key` が **既に存在する場合**:
    *   **何もしない** (値の上書きは行わない)。
    *   `logger.debug` などで、キーが既に存在するためスキップしたことを記録する。
7.  メソッドの最後に `self._merge_configs()` を呼び出し、`self._merged_config_data` を更新する。
8.  **`setting_added` が `True` の場合のみ、`self.save_system_config()` を呼び出して変更をファイルに保存する。**

### 3.4. ファイル保存

*   このメソッドはメモリ上の `_system_config_data` を変更し、**変更があった場合のみ、自動的にシステム設定ファイル (`annotator_config.toml`) に保存する。**
*   外部の呼び出し元コードで `save_system_config()` を再度呼び出す必要は通常ない。

## 4. 実装タスク

*   [X] `src/image_annotator_lib/core/config.py` の `ModelConfigRegistry` クラス内に `add_default_setting` メソッドを上記の仕様通りに実装する。
*   [X] メソッド内に適切なログ出力 (追加時、スキップ時) を追加する。
*   [X] メソッドの最後に `self._merge_configs()` の呼び出しが含まれていることを確認する。
*   [X] **設定が追加された場合にのみ `self.save_system_config()` が呼び出されるロジックを実装する。**

## 5. テストタスク

*   [X] `tests/unit/image_annotator_lib/core/test_config.py` に `add_default_setting` 用の新しいテストを追加する。
    *   [X] テストケース1: 新しいセクションに新しいキーと値が正しく追加されること。
    *   [X] テストケース2: 既存のセクションに新しいキーと値が正しく追加されること。
    *   [X] テストケース3: 既存のセクションの既存のキーに対して呼び出された場合、値が**上書きされない**こと。
    *   [X] テストケース4: `_merge_configs` が呼び出されることの確認 (Mock を使用)。 -> **修正:** `_merge_configs` はモックせず、直接呼び出しを確認（あるいはテスト不要と判断）
    *   [X] テストケース5: デフォルト値追加後、`get` メソッドでその値が取得できること (ユーザー設定で上書きされていない場合)。
    *   [X] テストケース6: 設定が追加された場合に `save_system_config` が呼び出されることの確認 (Mock を使用)。
    *   [X] テストケース7: キーが既に存在して設定がスキップされた場合に `save_system_config` が呼び出されないことの確認 (Mock を使用)。

## 6. ドキュメント更新

*   [X] `src/image_annotator_lib/core/config.py` の `add_default_setting` メソッドに Docstring を更新する (自動保存について言及)。
*   [X] `docs/REFERENCE/configuration.md` を確認し、この内部的な変更によって更新が必要か検討する (不要と判断)。

## 7. 改訂履歴

*   **1.3.0 (2024-20-20T19:50): 実装完了。試行錯誤の結果を追記。チェックリストを更新。**
*   1.2.0 (YYYY-MM-DD): デフォルト値が当面は Web API モデル向けであることを明確化。
*   1.1.0 (YYYY-MM-DD): 設定追加時に自動でファイル保存するよう仕様変更。
*   1.0.0 (YYYY-MM-DD): 初版作成

## 8. 実装結果と試行錯誤のまとめ

*   計画に基づき、`add_default_setting` メソッドを `src/image_annotator_lib/core/config.py` の `ModelConfigRegistry` クラスに実装した。仕様通りの動作（セクション/キー存在確認、非上書き追加、条件付き自動保存）を確認した。
*   対応するユニットテストを `tests/unit/core/test_config.py` に追加した。
*   **試行錯誤:**
    *   当初、テストコードで `_merge_configs` メソッドもモックしていたため、`add_default_setting` で追加された値がマージ済みデータ (`_merged_config_data`) に反映されず、`get` メソッドによる検証で `AssertionError` が発生した。
    *   **解決策:** `add_default_setting` のテストでは `_merge_configs` のモックを解除し、実際のメソッドを実行させるように修正した。ファイルへの書き込み副作用を防ぐため、`save_system_config` のモックは維持した。
*   上記の修正により、すべてのテストがパスすることを確認した。