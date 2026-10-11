import importlib.resources
import os
from pathlib import Path

from platformdirs import user_cache_path, user_config_path, user_log_path

# パッケージ名
PACKAGE_NAME = "image_annotator_lib"

# --- パッケージ内部リソースパス (読み取り専用テンプレート用) ---
_PACKAGE_RESOURCES_PATH = importlib.resources.files(PACKAGE_NAME).joinpath("resources")
_PACKAGE_SYSTEM_RESOURCES_PATH = _PACKAGE_RESOURCES_PATH.joinpath("system")
TEMPLATE_SYSTEM_CONFIG_PATH = _PACKAGE_SYSTEM_RESOURCES_PATH.joinpath("annotator_config.toml")

# --- Shared portable catalog and model weights ---
# ライブラリを利用するプロジェクトの実行時カレントディレクトリを取得
PROJECT_ROOT = Path.cwd()
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "models"


def _local_directory(variable: str, default: Path) -> Path:
    """Resolve an explicit local override without importing shared project paths."""
    configured = os.environ.get(variable)
    return Path(configured).expanduser() if configured else default


# Mutable preferences, derived metadata, install markers and logs are owned by
# the current OS user. Do not copy legacy project files into another OS's home.
USER_CONFIG_DIR = _local_directory(
    "IMAGE_ANNOTATOR_LIB_CONFIG_HOME", user_config_path("image-annotator-lib", appauthor=False)
)
RUNTIME_CACHE_DIR = _local_directory(
    "IMAGE_ANNOTATOR_LIB_CACHE_HOME", user_cache_path("image-annotator-lib", appauthor=False)
)
LOG_DIR = _local_directory(
    "IMAGE_ANNOTATOR_LIB_LOG_HOME", user_log_path("image-annotator-lib", appauthor=False)
)

# システム設定ファイルのパス (プロジェクトルート/config/annotator_config.toml)
SYSTEM_CONFIG_PATH = CONFIG_DIR / "annotator_config.toml"
# OS user model overrides (explicit user_config_path remains supported)
USER_CONFIG_PATH = USER_CONFIG_DIR / "user_config.toml"
# Runtime-derived model metadata cache (deletable local state)
MODEL_RUNTIME_CACHE_PATH = RUNTIME_CACHE_DIR / "model_runtime_cache.toml"

# ADR 0023 Phase 1 (Issue #35, PR #40): `AVAILABLE_API_MODELS_CONFIG_PATH` 定数は廃止。
# WebAPI モデル一覧は `webapi/api_model_discovery.py` で LiteLLM 同梱 DB から runtime
# 取得する (TOML キャッシュなし)。

DEFAULT_PATHS = {
    "config_toml": SYSTEM_CONFIG_PATH,  # プロジェクト config
    "user_config_toml": USER_CONFIG_PATH,  # OS user preferences
    "model_runtime_cache_toml": MODEL_RUNTIME_CACHE_PATH,  # OS user runtime state
    "install_marker_dir": RUNTIME_CACHE_DIR / "install_markers",
    "log_file": LOG_DIR / "image-annotator-lib.log",  # OS user logs
    "cache_dir": CACHE_DIR,  # プロジェクト models
}

# ADR 0023 Phase 1: TOML cache / TTL refresh / OpenRouter fallback は廃止された。
# 旧 `DEFAULT_API_MODELS_TTL_DAYS` / `ENV_API_MODELS_TTL_DAYS` /
# `ENV_ENABLE_OPENROUTER_FALLBACK` / `AVAILABLE_API_MODELS_CONFIG_PATH` 定数は
# 本ファイルから削除済。

# テンプレートパスも必要に応じてエクスポート (config.py で使う)
# (直接定数を使うのでエクスポート不要かも)
