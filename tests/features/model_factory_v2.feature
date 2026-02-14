@bdd
@bdd_core
Feature: ModelLoad Factory Pattern
  ModelLoadはモデルのロード、キャッシュ、メモリ管理を提供する

  Background:
    Given ModelLoad factory が初期化されている
    And annotator_config.toml が読み込まれている

  Scenario: モデルロード成功
    Given モデル "wd-swinv2-tagger-v3" の設定が存在する
    And 十分な空きメモリが存在する
    When load_model メソッドを呼び出す
    Then モデルインスタンスが返される
    And モデルがLRUキャッシュに保存される

  Scenario: LRUキャッシュヒット
    Given モデル "wd-swinv2-tagger-v3" が既にキャッシュされている
    When 同じモデルをload_model で要求する
    Then キャッシュされたインスタンスが返される
    And 新たなロード処理は実行されない

  Scenario: メモリ不足時のLRU退避
    Given 3つのモデルが既にキャッシュされている (cache size = 3)
    And メモリ使用量が上限に近い
    When 4つ目のモデルをload_modelで要求する
    Then 最も使用されていないモデルがキャッシュから削除される
    And 新しいモデルがロードされる

  Scenario: デバイス配置の尊重
    Given モデル設定で device = "cuda" が指定されている
    And CUDAが利用可能である
    When load_model メソッドを呼び出す
    Then モデルがCUDAデバイスに配置される

  Scenario: モデルパス不正時のエラー
    Given 存在しないmodel_pathが設定されている
    When load_model メソッドを呼び出す
    Then FileNotFoundError または ModelLoadError が発生する

  Scenario: メモリサイズ事前計算
    Given モデル設定で estimated_size_gb = 1.5 が指定されている
    When _estimate_model_size メソッドを呼び出す
    Then 1.5 GB に相当するバイト数が返される

  Scenario: キャッシュクリア操作
    Given 複数のモデルがキャッシュされている
    When clear_cache メソッドを呼び出す
    Then 全てのモデルがキャッシュから削除される
    And メモリが解放される
