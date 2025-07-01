@webapi @pydanticai
Feature: PydanticAI WebAPIアノテーターによる画像アノテーション
    PydanticAI フレームワークを使用したWebAPI OpenAI, Anthropic, Google による
    アノテーターが画像に対して正しく動作し、タグ・キャプション・スコアを返すことを検証する。

    Background:
        Given PydanticAI テスト環境が設定されている
        And モデル設定ファイルが正しく構成されている
        And 1つの有効な画像ファイルが準備されている

    Scenario Outline: 各種PydanticAI WebAPIで画像を分析しタグキャプションスコアを取得できる
        Given <provider> プロバイダーが利用可能になっている
        When 画像を指定して<model_name>でアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

        Examples:
            | provider   | model_name      |
            | OpenAI     | gpt-4o-mini     |
            | Anthropic  | claude-3-5-haiku |
            | Google     | gemini-1.5-flash |

    Scenario: PydanticAI認証エラーの場合は適切なエラーを返す
        Given PydanticAI認証が失敗するよう設定されている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then 認証エラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: PydanticAIモデルが利用不可の場合は適切なエラーを返す
        Given PydanticAIモデルが利用不可になっている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then モデル利用不可エラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: PydanticAI TestModelを使用したモック実行ができる
        Given PydanticAI TestModelが設定されている
        When 画像を指定してgpt-4o-miniでアノテーションを実行する
        Then TestModelによるモック結果が返される
        And エラーは発生していない