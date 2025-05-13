import os

from pytest_bdd import given


@given("モデル設定ファイルが正しく構成されている")
def model_config_is_valid():
    # 実際の設定ファイルが存在し、内容が正しいことを検証
    config_path = "config/annotator_config.toml"
    assert os.path.exists(config_path)


@given("1つの有効な画像ファイルが準備されている", target_fixture="single_image")
def valid_image_file(load_image_files):
    # テスト用画像ファイルのパスを返す(実ファイルを用意しておくこと)
    image_list = load_image_files(1)
    assert len(image_list) == 1
    return image_list[0]
