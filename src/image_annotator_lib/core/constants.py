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
# Runtime-derived model metadata cache (deletable local state)
MODEL_RUNTIME_CACHE_PATH = CONFIG_DIR / "model_runtime_cache.toml"

# ADR 0023 Phase 1 (Issue #35, PR #40): `AVAILABLE_API_MODELS_CONFIG_PATH` 定数は廃止。
# WebAPI モデル一覧は `webapi/api_model_discovery.py` で LiteLLM 同梱 DB から runtime
# 取得する (TOML キャッシュなし)。

DEFAULT_PATHS = {
    "config_toml": SYSTEM_CONFIG_PATH,  # プロジェクト config
    "user_config_toml": USER_CONFIG_PATH,  # プロジェクト config
    "model_runtime_cache_toml": MODEL_RUNTIME_CACHE_PATH,  # プロジェクト runtime state
    "log_file": LOG_DIR / "image-annotator-lib.log",  # プロジェクト logs
    "cache_dir": CACHE_DIR,  # プロジェクト models
}

# ADR 0023 Phase 1: TOML cache / TTL refresh / OpenRouter fallback は廃止された。
# 旧 `DEFAULT_API_MODELS_TTL_DAYS` / `ENV_API_MODELS_TTL_DAYS` /
# `ENV_ENABLE_OPENROUTER_FALLBACK` / `AVAILABLE_API_MODELS_CONFIG_PATH` 定数は
# 本ファイルから削除済。

# テンプレートパスも必要に応じてエクスポート (config.py で使う)
# (直接定数を使うのでエクスポート不要かも)
