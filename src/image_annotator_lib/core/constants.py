import importlib.resources
from pathlib import Path

# パッケージ名
PACKAGE_NAME = "image_annotator_lib"

# --- パッケージ内部リソースパス (読み取り専用テンプレート用) ---
_PACKAGE_RESOURCES_PATH = importlib.resources.files(PACKAGE_NAME).joinpath("resources")
_PACKAGE_SYSTEM_RESOURCES_PATH = _PACKAGE_RESOURCES_PATH.joinpath("system")
TEMPLATE_SYSTEM_CONFIG_PATH = _PACKAGE_SYSTEM_RESOURCES_PATH.joinpath("annotator_config.toml")

# --- プロジェクトルート基準のパス設定 (書き込み可能) ---
# ライブラリを利用するプロジェクトの実行時カレントディレクトリを取得
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "models"

# システム設定ファイルのパス (プロジェクトルート/config/annotator_config.toml)
SYSTEM_CONFIG_PATH = CONFIG_DIR / "annotator_config.toml"
# ユーザー設定ファイルのパス (プロジェクトルート/config/user_config.toml)
USER_CONFIG_PATH = CONFIG_DIR / "user_config.toml"
# 動的 API モデル情報ファイルのパス (プロジェクトルート/config/available_api_models.toml)
AVAILABLE_API_MODELS_CONFIG_PATH = CONFIG_DIR / "available_api_models.toml"

DEFAULT_PATHS = {
    "config_toml": SYSTEM_CONFIG_PATH,  # プロジェクト config
    "user_config_toml": USER_CONFIG_PATH,  # プロジェクト config
    "available_api_models_toml": AVAILABLE_API_MODELS_CONFIG_PATH,  # プロジェクト config
    "log_file": LOG_DIR / "image-annotator-lib.log",  # プロジェクト logs
    "cache_dir": CACHE_DIR,  # プロジェクト models
}

# テンプレートパスも必要に応じてエクスポート (config.py で使う)
# (直接定数を使うのでエクスポート不要かも)
