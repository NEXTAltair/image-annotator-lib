@bdd
@bdd_core
Feature: SimplifiedAgentWrapper PydanticAI Integration
  SimplifiedAgentWrapperはPydanticAI Agentをラップし、統一インターフェースを提供する

  Background:
    Given PydanticAI環境が設定されている

  Scenario: Agent初期化成功
    Given モデルID "gpt-4o-mini" が指定される
    When SimplifiedAgentWrapperを初期化する
    Then Agentが正常にキャッシュから取得される
    And ログに "SimplifiedAgentWrapper initialized for model" が記録される

  Scenario: Agent初期化失敗時の例外処理
    Given 不正なモデルID "invalid-model-id" が指定される
    When SimplifiedAgentWrapperを初期化する
    Then Exception が発生する
    And ログに "Failed to setup agent" エラーが記録される

  Scenario: 画像のBinaryContent変換
    Given SimplifiedAgentWrapperインスタンスが初期化されている
    And PIL Image形式の画像が1つ準備されている
    When _preprocess_images メソッドを呼び出す
    Then BinaryContentリストが返される
    And BinaryContentのmedia_typeが "image/png" である

  Scenario: 正常な推論実行
    Given SimplifiedAgentWrapperインスタンスが初期化されている
    And BinaryContent形式の画像が準備されている
    When _run_inference メソッドを呼び出す
    Then 推論結果リストが返される
    And 結果にtagsが含まれる

  Scenario: Agent未初期化時のRuntimeError
    Given SimplifiedAgentWrapperの_agentがNoneに設定されている
    When _run_inference メソッドを呼び出す
    Then RuntimeError "Agent not initialized" が発生する

  Scenario: Event loop衝突時のAsync fallback
    Given 既存のevent loopが実行中である
    And SimplifiedAgentWrapperインスタンスが初期化されている
    When run_syncがRuntimeError (Event loop関連) を発生させる
    Then _run_async_with_new_loop が自動的に呼び出される
    And 新しいevent loopが作成される
    And ThreadPoolExecutorが使用される
    And 推論が正常に完了する

  Scenario: 推論実行時の例外処理
    Given SimplifiedAgentWrapperインスタンスが初期化されている
    And 推論中に予期せぬ例外が発生する状況をシミュレートする
    When run_inference メソッドを呼び出す
    Then AnnotationResultにerrorフィールドが設定される
    And ログに "Inference failed" エラーが記録される
    And tagsは空リストである

  Scenario: Tags欠損時の空リスト返却
    Given SimplifiedAgentWrapperインスタンスが初期化されている
    And Agent結果にtagsフィールドが存在しない
    When _generate_tags メソッドを呼び出す
    Then 空のtagsリストが返される
