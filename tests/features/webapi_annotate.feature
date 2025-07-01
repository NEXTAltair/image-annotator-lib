@webapi
Feature: WebAPIアノテーターによる画像アノテーション
    AIモデルWebAPIを使用した画像アノテーション機能が
    正しく動作し、タグ・キャプション・スコアを返すことを検証する。

    Background:
        Given アノテーション環境が設定されている
        And モデル設定ファイルが正しく構成されている
        And 1つの有効な画像ファイルが準備されている

    Scenario Outline: 各種AIモデルで画像を分析しタグキャプションスコアを取得できる
        Given <model_name>が利用可能になっている
        When 画像を指定して<model_name>でアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

        Examples:
            | model_name      |
            | gpt-4o-mini     |
            | claude-3-5-haiku |
            | gemini-1.5-flash |

    Scenario: APIキーが未設定の場合は適切なエラーを返す
        Given 全てのAPIキーが未設定になっている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then 認証エラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: 存在しないモデルを指定した場合は適切なエラーを返す
        Given モデルが利用不可になっている
        When 画像を指定してinvalid-modelでアノテーションを実行する
        Then モデル利用不可エラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: APIがタイムアウトした場合は適切なエラーを返す
        Given APIがタイムアウトするよう設定されている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then タイムアウトエラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: 主要プロバイダーのAPIキーがない場合はOpenRouterにフォールバックする
        Given OpenAI、Anthropic、GoogleのAPIキーが未設定になっている
        And OpenRouterのAPIキーが設定されている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

    Scenario: 複数モデルで同時にアノテーションを実行できる
        Given 複数のモデルが利用可能になっている
        When 画像を指定して複数モデルでアノテーションを実行する
        Then 全てのモデルからアノテーション結果が返される
        And エラーは発生していない