import importlib.resources
from pathlib import Path

# パッケージ名
PACKAGE_NAME = "image_annotator_lib"

# システム設定ファイルのパス
SYSTEM_CONFIG_PATH = importlib.resources.files(PACKAGE_NAME).joinpath(
    "resources", "system", "annotator_config.toml"
)
# --- プロジェクトルート基準のパス設定 ---
# ライブラリを利用するプロジェクトの実行時カレントディレクトリを取得
PROJECT_ROOT = Path.cwd()

# ユーザー設定ファイルのパス (プロジェクトルート/config/user_config.toml)
USER_CONFIG_PATH = PROJECT_ROOT / "config" / "user_config.toml"

DEFAULT_PATHS = {
    "config_toml": SYSTEM_CONFIG_PATH,
    "user_config_toml": USER_CONFIG_PATH,
    # ログファイルのパス (プロジェクトルート/logs/image-annotator-lib.log)
    "log_file": PROJECT_ROOT / "logs" / "image-annotator-lib.log",
    # キャッシュディレクトリのパス (プロジェクトルート/models)
    "cache_dir": PROJECT_ROOT / "models",
}
