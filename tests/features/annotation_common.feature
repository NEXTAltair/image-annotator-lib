Feature: アノテーション共通機能
    モデルタイプに依存しないアノテーションライブラリの共通的な振る舞いをテストする

Background:
    Given モデルクラスレジストリが初期化されている

Scenario: レジストリに登録された利用可能なモデルをインスタンス化できる
    Given モデルクラスレジストリが初期化されている
    # Scorer/Taggerタイプに限定しないため "any" を想定
    Given anyタイプのモデルを利用する
    And 1つの利用可能なモデルが指定されている
    When アノテーションアクション "single_image_single_model" を実行する
    Then (any) アノテーション結果タイプ "single_image_single_model_meta_only" の検証が成功する

Scenario: 既にインスタンス化済みのモデルを再度インスタンス化すると、同じインスタンスが返される (共通キャッシュ確認)
    Given anyタイプのモデルを利用する
    And 1つの有効な画像ファイルが準備されている
    And 1つの利用可能なモデルが指定されている
    When 同じモデルクラスを再度インスタンス化する
    Then キャッシュされた同一のモデルインスタンスが返される

Scenario Outline: pHashリストの提供有無による挙動確認
    Given モデルクラスレジストリが初期化されている
    Given anyタイプのモデルを利用する
    Given 1つの利用可能なモデルが指定されている
    Given 1つの有効な画像ファイルが準備されている
    When pHash提供 "<provide_phash>" でアノテーションを実行する
    # ステップ定義側でprovide_phashに応じて検証
    Then "pHashキーが期待通りに処理される"

    Examples:
    | provide_phash |
    | true          |
    | false         |

Scenario: 存在しないモデル名でアノテーションを実行すると結果にエラーが含まれる
    Given 1つの有効な画像ファイルが準備されている
    When 存在しないモデル名 "non_existent_model_dummy_name" でアノテーションを実行する
    Then モデル "non_existent_model_dummy_name" の結果にエラー情報が含まれる

Scenario: 不正な形式のモデル名でアノテーションを実行すると結果にエラーが含まれる (例: 空文字)
    Given モデルクラスレジストリが初期化されている
    Given 1つの有効な画像ファイルが準備されている
    When 存在しないモデル名 "" でアノテーションを実行する
    Then モデル "" の結果にエラー情報が含まれる

# TODO: さらに以下のような共通シナリオも追加検討
# - 画像リストが空の場合の挙動
# - モデルリストが空の場合の挙動
# - annotate関数呼び出し時の直接的な例外発生ケース (より低レベルなモックが必要か) 