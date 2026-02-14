@bdd
@pydantic_ai_factory
@evaluator_unified
Feature: PydanticAI Model Factory 統一テスト

    PydanticAI Model Factory の共通機能をテストする。
    本フィーチャーは両プロバイダー実装プラン共通で通過すべきシナリオを定義する。

    Background:
        Given PydanticAI環境が設定されている
        And テストモデル（TestModel）を使用している

    @provider_single
    Scenario: 単一プロバイダーでのアノテーション実行
        Given OpenAIプロバイダーが設定されている
        And モデルID "gpt-4o-mini" が指定される
        And テスト用の画像1枚が準備されている
        When PydanticAIAnnotatorでアノテーションを実行する
        Then AnnotationSchemaに準拠した結果が返される
        And 結果のtagsフィールドが存在する
        And エラーフィールドがNoneである

    @provider_multiple_parallel
    Scenario: 複数プロバイダーの並行実行
        Given OpenAIプロバイダーが設定されている
        And Anthropicプロバイダーが設定されている
        And モデルID "gpt-4o-mini" と "claude-3-5-sonnet" が指定される
        And テスト用の画像1枚が準備されている
        When 両方のプロバイダーで同時にアノテーションを実行する
        Then 両方のプロバイダーから結果が返される
        And 各結果がAnnotationSchemaに準拠している
        And 結果の統合に成功する

    @api_key_missing
    Scenario: APIキー未設定時のエラーハンドリング
        Given OpenAIプロバイダーが設定されている
        And APIキーが設定されていない
        And モデルID "gpt-4o-mini" が指定される
        When アノテーションを実行する
        Then ApiAuthenticationError が発生する
        And エラーメッセージに "API key" が含まれる

    @openrouter_headers
    Scenario: OpenRouterカスタムヘッダー送信
        Given OpenRouterプロバイダーが設定されている
        And モデルID "openrouter:meta-llama/llama-2-70b-chat" が指定される
        And OpenRouterカスタムヘッダー（HTTP-Referer, X-Title）が設定されている
        When アノテーションを実行する
        Then HTTP-RefererヘッダーがHTTPリクエストに含まれる
        And X-TitleヘッダーがHTTPリクエストに含まれる
        And AnnotationSchemaに準拠した結果が返される

    @model_id_auto_detect
    Scenario Outline: モデルID文字列からのプロバイダー自動判定
        Given モデルID "<model_id>" が指定される
        When プロバイダーを自動判定する
        Then "<provider>" プロバイダーが選択される
        And 選択されたプロバイダーの設定が正しく適用される

        Examples:
            | model_id                         | provider     |
            | gpt-4o-mini                      | openai       |
            | claude-3-5-sonnet                | anthropic    |
            | gemini-2.0-flash                 | google       |
            | openrouter:meta-llama/llama-2    | openrouter   |

    @cache_efficiency
    Scenario: Agent/Providerキャッシュの効率性
        Given OpenAIプロバイダーが設定されている
        And モデルID "gpt-4o-mini" が指定される
        And テスト用の画像1枚が準備されている
        When 同一プロバイダーで2回目のアノテーションを実行する
        Then 1回目と2回目でProvider インスタンスが同じオブジェクトである
        And Agentインスタンスがキャッシュから再利用される
        And 2回目の実行時間が1回目より短い

    @auth_error_graceful
    Scenario: 認証エラー時のグレースフル処理
        Given OpenAIプロバイダーが設定されている
        And 無効なAPIキーが設定されている
        And モデルID "gpt-4o-mini" が指定される
        And テスト用の画像1枚が準備されている
        When アノテーションを実行する
        Then ApiAuthenticationError が発生する
        And エラーメッセージが明確である
        And アプリケーションが適切にエラーを処理する

    @provider_caching_across_models
    Scenario: 複数モデルでのプロバイダー共有
        Given OpenAIプロバイダーが設定されている
        And モデルID "gpt-4o-mini" と "gpt-4o" が指定される
        And テスト用の画像1枚が準備されている
        When 異なるモデルIDで連続してアノテーションを実行する
        Then 両方のモデルが同じプロバイダーインスタンスを使用している
        And プロバイダーが複数回の生成を避けている

    @factory_model_override
    Scenario: Factory の model override 機能
        Given OpenAIプロバイダーが初期化されている
        And 初期モデルID "gpt-4o-mini" が設定されている
        When run_with_model("gpt-4o") を実行する
        Then 推論に "gpt-4o" が使用される
        And プロバイダーは再利用される
        And 結果がAnnotationSchemaに準拠している

    @test_model_support
    Scenario: TestModel対応（テスト環境）
        Given PydanticAI TestModel が使用可能である
        And モデルID "test" が指定される
        When TestModel でアノテーションを実行する
        Then 常に決定的な結果が返される
        And AnnotationSchemaに準拠している
