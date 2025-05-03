from pytest_bdd import scenarios, given, when, then, parsers
from image_annotator_lib.core import registry # Added import

# 他のステップ定義ファイルと区別するため、特定のfeatureファイルを指定しない
# scenarios('../features/') # この行はコメントアウトまたは削除

@given("アプリケーションを起動している")
def given_app_is_running(mocker):
    """
    アプリケーションが起動している状態を準備する（仮実装）。
    必要に応じてモックなどを設定する。
    """
    # TODO: アプリケーションの初期化や必要なモックの設定をここに追加
    print("ステップ定義: アプリケーションを起動している")
    pass

@given("モデルクラスレジストリが初期化されている")
def given_model_registry_initialized():
    """
    モデルクラスレジストリを初期化する。
    """
    registry.initialize_registry() # Corrected function call
    # 必要に応じて初期化を確認するアサーションを追加
    # assert 'wd-vit-tagger-v3' in registry.CLASS_REGISTRY
    print("ステップ定義: モデルクラスレジストリが初期化されている")
    pass

# --- BDD未定義Given句のダミー実装（テスト用） ---
from pytest_bdd import given

@given("APIキーが環境変数に設定されている")
def given_api_key_is_set(monkeypatch):
    # 必要に応じて環境変数をセットする処理を追加
    pass

@given("有効な画像ファイルが準備されている")
def given_valid_image_file(tmp_path):
    # テスト用のダミー画像ファイルを生成する処理を追加
    pass

@given("複数の有効な画像ファイルが準備されている")
def given_multiple_valid_image_files(tmp_path):
    # テスト用の複数画像ファイルを生成する処理を追加
    pass

@given("レジストリに登録されたモデルのリストを取得する")
def given_model_list_in_registry():
    # モデルリストの初期化処理を追加
    pass

@given("インスタンス化済みのモデルクラスが存在する")
def given_model_class_is_instantiated():
    # モデルクラスのインスタンス化処理を追加
    pass