@webapi
Feature: WebAPIアノテーターによる画像アノテーション
    外部Web API (Google Gemini, OpenAI, Anthropic, OpenRouter) を用いたアノテーターが
    画像に対して正しく動作し、タグキャプションスコアを返すことを検証する。

    Background:
        Given APIキーが環境変数に設定されている
        And モデル設定ファイルが正しく構成されている
        And 1つの有効な画像ファイルが準備されている

    Scenario Outline: 各種WebAPIで画像を分析しタグキャプションスコアを取得できる
        Given <provider> APIが利用可能な状態になっている
        When 画像を指定して<model_alias>モデルでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

        Examples:
            | provider   | model_alias                         |
            | Google     | Gemini 2.5 Pro Preview              |
            | OpenAI     | GPT-4.1 Mini                        |
            | Anthropic  | Claude 3 Haiku (self-moderated)     |
            | OpenRouter | Qwen2.5 VL 72B Instruct             |

    Scenario: APIキーが未設定の場合は認証エラーが発生する
        Given APIキーが未設定の状態になっている
        When 画像を指定してアノテーションを実行する
        Then "ApiAuthenticationError" のエラーメッセージが返される

    Scenario: APIリクエストがタイムアウトした場合は適切なエラーを返す
        Given APIがタイムアウトするよう設定されている
        When 画像を指定してアノテーションを実行する
        Then タイムアウトエラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: APIからエラーレスポンスが返された場合は適切に処理する
        Given APIからエラーレスポンスが返されるよう設定されている
        When 画像を指定してアノテーションを実行する
        Then APIエラーメッセージが結果に含まれる
        And 結果のタグリストは空である