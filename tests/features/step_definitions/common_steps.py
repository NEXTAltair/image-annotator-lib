import os

from pytest_bdd import given

from image_annotator_lib.core.registry import initialize_registry


@given("モデルクラスレジストリが初期化されている")
def model_class_registry_initialized():
    """モデルクラスレジストリを初期化"""
    initialize_registry()


@given("モデル設定ファイルが正しく構成されている")
def model_config_is_valid():
    # 実際の設定ファイルが存在し、内容が正しいことを検証
    config_path = "config/annotator_config.toml"
    assert os.path.exists(config_path)

    # レジストリを初期化してモデルが利用可能になるようにする
    initialize_registry()


@given("1つの有効な画像ファイルが準備されている", target_fixture="single_image")
def valid_image_file(load_image_files):
    # テスト用画像ファイルのパスを返す(実ファイルを用意しておくこ��)
    image_list = load_image_files(1)
    assert len(image_list) == 1
    return image_list[0]


@given('"any"タイプのモデルを利用する', target_fixture="context")
def any_model_type(context):
    """任意のモデルタイプとして、高速なWeb APIモデルを利用する設定"""
    context["model_name"] = "gpt-4o-mini"
    return context
