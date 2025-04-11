import importlib.resources
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

# --- 定数定義 ---
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
DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"

# ロガーの初期化
logger = logging.getLogger(__name__)


@lru_cache
def _load_config_from_file(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """設定ファイルをTOML形式で読み込む内部ヘルパー関数。"""
    try:
        logger.debug(f"構成ファイルを読み込みます: {config_path}")
        config_data = toml.load(config_path)
        if not isinstance(config_data, dict):
            logger.error(f"設定ファイル {config_path} の形式が不正です。辞書形式である必要があります。")
            raise TypeError("構成データは辞書である必要があります")
        logger.debug(f"構成ファイルを正常に読み込みました: {config_path}")
        return dict(config_data)
    except FileNotFoundError:
        logger.error(f"設定ファイル {config_path} が見つかりません。デフォルト設定で続行します。")
        return {}  # ファイルが見つからない場合は空の辞書を返す
    except Exception as e:
        logger.exception(f"設定ファイル {config_path} の読み込み中に予期せぬエラーが発生しました: {e}")
        raise  # 予期せぬエラーは再送出


def save_model_size(model_name: str, size_mb: float, config_path: str | Path | None = None) -> None:
    """モデルのサイズ推定値を保存します。"""
    if config_path is None:
        config_path = DEFAULT_PATHS["config_toml"]
    try:
        size_gb = size_mb / 1024

        config_path_obj = Path(config_path) if isinstance(config_path, str) else config_path
        if config_path_obj.exists():
            config_data = toml.load(config_path)
        else:
            logger.error(f"設定ファイル {config_path} が見つかりません")
            return

        if model_name not in config_data:
            logger.warning(f"モデル '{model_name}' の設定が見つかりません")
            return

        config_data[model_name]["estimated_size_gb"] = round(size_gb, 3)

        with open(config_path, "w") as f:
            toml.dump(config_data, f)

        logger.debug(f"モデル '{model_name}' の推定サイズ ({size_gb:.3f}GB) を保存しました")
    except Exception as e:
        logger.error(f"モデルサイズの保存に失敗しました: {e}")


class ModelConfigRegistry:
    """設定ファイル全体を管理し、設定値へのアクセスを提供します。"""

    def __init__(self) -> None:
        """初期化時に設定データを空の辞書で初期化します。"""
        self._config_data: dict[str, dict[str, Any]] = {}
        self._is_loaded = False

    def load(self, config_path: str | Path | None = None, user_config_path: str | Path | None = None) -> None:
        """システム設定ファイルとユーザー設定ファイルを読み込み、内部データ (_config_data) を更新します。
        ユーザー設定はシステム設定を上書きします。ユーザー設定ファイルが存在しない場合は無視されます。
        """
        if config_path is None:
            config_path = DEFAULT_PATHS["config_toml"]

        if user_config_path is None:
            user_config_path = DEFAULT_PATHS["user_config_toml"]

        # システム設定の読み込み
        self._config_data = _load_config_from_file(config_path)
        logger.info(f"設定マネージャーがシステム設定を {config_path} から読み込みました。")

        # ユーザー設定の読み込み（存在する場合のみ）
        if user_config_path is not None:
            user_config_path_obj = Path(user_config_path) if isinstance(user_config_path, str) else user_config_path
            if user_config_path_obj.exists():
                try:
                    user_config = _load_config_from_file(user_config_path)
                    # ユーザー設定でシステム設定を更新（ユーザー設定が優先）
                    for model_name, model_config in user_config.items():
                        if model_name in self._config_data:
                            # 既存のモデル設定をユーザー設定で上書き
                            self._config_data[model_name].update(model_config)
                        else:
                            # 新しいモデル設定を追加
                            self._config_data[model_name] = model_config
                    logger.info(f"ユーザー設定を {user_config_path} から読み込み、システム設定を上書きしました。")
                except Exception as e:
                    logger.warning(f"ユーザー設定ファイル {user_config_path} の読み込み中にエラーが発生しました: {e}")
            else:
                logger.debug(f"ユーザー設定ファイル {user_config_path} が存在しないため、システム設定のみ使用します。")

    def get(self, model_name: str, key: str, default: Any = None) -> Any | None:
        """指定されたモデルとキーに対応する設定値を取得します。

        前提: `model_name` は設定データ内に必ず存在します。

        - `key` がモデルの設定内に存在する場合: その値を返します。
        - `key` がモデルの設定内に存在しない場合: `default` 引数で指定された値を返します。

        Args:
            model_name: 設定値を取得したいモデルの名前 (設定内に存在することが前提)。
            key: 取得したい設定のキー。
            default: `key` が見つからなかった場合に返すデフォルト値 (デフォルト: None)。

        Returns:
            取得した設定値。`key` が見つからなければ `default` 引数の値。
        """
        model_config = self._config_data[model_name]
        return model_config.get(key, default)

    def get_all_config(self) -> dict[str, Any]:
        """ロード済みの設定データ全体を辞書として返します。"""
        return self._config_data.copy()  # 内部データのコピーを返す


# --- 共有インスタンスの作成と初期ロード --- #
config_registry = ModelConfigRegistry()
try:
    config_registry.load()
except Exception:
    logger.exception("共有設定レジストリの初期ロード中にエラーが発生しました。")
