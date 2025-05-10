@scorer
Feature: Scorer アノテーション機能
    画像のスコアリングに関する振る舞いをテストする

Background:
    Given モデルクラスレジストリが初期化されている
    # And テストにはモックやダミーを使用しない
    And "scorer" タイプのモデルを利用する

Scenario Outline: 様々な条件下でのScorerアノテーション
    Given <num_images>つの有効な画像ファイルが準備されている
    And <num_models>つのスコアラーモデルが指定されている
    When アノテーションアクション "<action_key>" を実行する
    Then (Scorer) アノテーション結果タイプ "<result_type_key>" の検証が成功する

    Examples: 画像数・モデル数・アクション・結果タイプ
        | num_images | num_models | action_key                       | result_type_key                |
        | 1          | 1          | single_image_single_model        | single_image_single_model_eval |
        | 5          | 1          | multiple_images_single_model     | multiple_images_single_model_eval|
        | 1          | 3          | single_image_multiple_models     | single_image_multiple_models_eval|
        | 5          | 3          | multiple_images_multiple_models  | multiple_images_multiple_models_eval|

@skip
Scenario Outline: Scorerのストレステスト
    Given <num_images>枚の有効な画像ファイルが準備されている
    And すべての利用可能なスコアラーモデルが指定されている
    When これらの画像を複数回連続でアノテーションを実行する
    Then 全ての評価が正常に完了している
    And GPU・CPUメモリの使用状況が許容範囲内である

    Examples: 大量画像
        | num_images |
        | 50         |

@skip
Scenario Outline: Scorerのモデル切り替えテスト
    Given 1つの有効な画像ファイルが準備されている
    And すべての利用可能なスコアラーモデルが指定されている
    When 各モデルを交互に100回切り替えながら画像のアノテーションを実行する
    Then モデル切り替えが正常に動作している
    And リソースリークが発生していない 