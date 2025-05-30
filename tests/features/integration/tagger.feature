Feature: 画像アノテーション機能
    これにより画像の選別や分類が効率化される

    Background:
        Given モデルクラスレジストリが初期化されている

    # モデルの初期化に関するシナリオ
    Scenario: レジストリに登録された利用可能なモデルをインスタンス化できる
        Given レジストリに登録されたモデルのリストを取得する
        When これらのモデルをそれぞれインスタンス化する
        Then 各モデルが正常にインスタンス化される

    Scenario: 既にインスタンス化済みのモデルを再度インスタンス化すると、同じインスタンスが返される
        Given インスタンス化済みのモデルクラスが存在する
        When 同じモデルクラスを再度インスタンス化する
        Then キャッシュされた同一のモデルインスタンスが返される

    # 単一画像の評価に関するシナリオ
    Scenario: 単一の画像を評価してタグを取得できる
        Given 有効な画像ファイルが準備されている
        And タガーがインスタンス化されている
        When この画像をタグ付けする
        Then 画像に対するモデルの処理結果が返される

    # 複数画像の評価に関するシナリオ
    Scenario: 複数の画像を一括で評価できる
        Given 複数の有効な画像ファイルが準備されている
        And タガーがインスタンス化されている
        When これらの画像を一括アノテーションを実行
        Then 各画像に対するモデルの処理結果が返される

    # 複数モデルによる評価に関するシナリオ
    Scenario: 同じ画像を複数のモデルで評価できる
        Given 有効な画像ファイルが準備されている
        And 複数のモデルが指定されている
        When この画像を複数のモデルでアノテーションを実行
        Then 画像に対する各モデルの処理結果が返される

    Scenario: 複数の画像を複数のモデルで一括評価できる
        Given 複数の有効な画像ファイルが準備されている
        And 複数のモデルが指定されている
        When これらの画像を複数のモデルで一括アノテーションを実行
        Then 各画像に対する各モデルの処理結果が返される

    @skip
    Scenario: 大量の画像を複数モデルで繰り返しアノテーションを実行ストレステスト
        Given 30枚の有効な画像ファイルが準備されている
        And すべての利用可能なモデルが指定されている
        When これらの画像を複数回連続でアノテーションを実行
        Then 全ての評価が正常に完了している
        And リソースリークが発生していない

    @skip
    Scenario: モデルの頻繁な切り替えによるメモリ管理テスト
        Given 有効な画像ファイルが準備されている
        And すべての利用可能なモデルが指定されている
        When 各モデルを交互に100回切り替えながら画像をアノテーションを実行
        Then モデル切り替えが正常に動作している
        And リソースリークが発生していない