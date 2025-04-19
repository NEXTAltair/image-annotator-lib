from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

from .constants import DEFAULT_PATHS
from .utils import logger


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


# save_model_size 関数は ModelConfigRegistry に統合するため削除またはコメントアウトを検討
# def save_model_size(model_name: str, size_mb: float, config_path: str | Path | None = None) -> None:
#     ...


class ModelConfigRegistry:
    """設定ファイル全体を管理し、設定値へのアクセスを提供します。"""

    def __init__(self) -> None:
        """初期化時に設定データを空の辞書で初期化します。"""
        self._system_config_data: dict[str, dict[str, Any]] = {}  # System config
        self._user_config_data: dict[str, dict[str, Any]] = {}  # User overrides/additions
        self._merged_config_data: dict[str, dict[str, Any]] = {}  # Merged view
        self._system_config_path: Path | None = None
        self._user_config_path: Path | None = None

    def load(
        self, config_path: str | Path | None = None, user_config_path: str | Path | None = None
    ) -> None:
        """システム設定ファイルとユーザー設定ファイルを読み込み、内部データを更新します。
        ユーザー設定はシステム設定を上書きします。
        """
        # Determine paths and ensure they are Path objects
        # Cast Traversable from DEFAULT_PATHS to str before creating Path
        sys_path_arg = config_path if config_path else str(DEFAULT_PATHS["config_toml"])
        user_path_arg = user_config_path if user_config_path else str(DEFAULT_PATHS["user_config_toml"])

        self._system_config_path = Path(sys_path_arg) if sys_path_arg else None
        self._user_config_path = Path(user_path_arg) if user_path_arg else None

        # Load system config
        if self._system_config_path:
            try:
                self._system_config_data = _load_config_from_file(self._system_config_path)
                logger.info(
                    f"設定マネージャーがシステム設定を {self._system_config_path} から読み込みました。"
                )
            except Exception as e:
                logger.error(
                    f"システム設定ファイル {self._system_config_path} の読み込みに失敗しました: {e}"
                )
                self._system_config_data = {}
        else:
            logger.error("システム設定ファイルのパスが決定できませんでした。")
            self._system_config_data = {}

        # Load user config (if exists)
        self._user_config_data = {}
        if self._user_config_path is not None and self._user_config_path.exists():
            try:
                self._user_config_data = _load_config_from_file(self._user_config_path)
                logger.info(f"ユーザー設定を {self._user_config_path} から読み込みました。")
            except Exception as e:
                logger.warning(
                    f"ユーザー設定ファイル {self._user_config_path} の読み込み中にエラーが発生しました: {e}"
                )
                self._user_config_data = {}
        else:
            logger.debug(
                f"ユーザー設定ファイル {self._user_config_path} が存在しないかパスが指定されていません。"
            )

        # Merge configs (user overrides system)
        self._merge_configs()

    def _merge_configs(self) -> None:
        """システム設定とユーザー設定をマージして _merged_config_data を作成"""
        self._merged_config_data = self._system_config_data.copy()
        for model_name, user_model_config in self._user_config_data.items():
            if model_name in self._merged_config_data:
                self._merged_config_data[model_name].update(user_model_config)
            else:
                self._merged_config_data[model_name] = user_model_config
        logger.debug("システム設定とユーザー設定をマージしました。")

    def get(self, model_name: str, key: str, default: Any = None) -> Any | None:
        """指定されたモデルとキーに対応するマージ済みの設定値を取得します。"""
        if model_name not in self._merged_config_data:
            # モデル自体が存在しない場合は警告を出し、デフォルト値を返す
            # logger.warning(f"設定内にモデル '{model_name}' が見つかりません。") # 必要に応じてログ出力
            return default
        model_config = self._merged_config_data[model_name]
        return model_config.get(key, default)

    def set(self, model_name: str, key: str, value: Any) -> None:
        """指定されたモデルとキーに対応する設定値をユーザー設定として更新します。"""
        logger.debug(f"ユーザー設定を更新: モデル '{model_name}', キー '{key}', 値 '{value}'")
        if model_name not in self._user_config_data:
            self._user_config_data[model_name] = {}
        self._user_config_data[model_name][key] = value
        # Immediately update the merged view as well
        self._merge_configs()

    def set_system_value(self, model_name: str, key: str, value: Any) -> None:
        """指定されたモデルとキーに対応する設定値をシステム設定として更新します。"""
        logger.debug(f"システム設定を更新: モデル '{model_name}', キー '{key}', 値 '{value}'")
        if model_name not in self._system_config_data:
            self._system_config_data[model_name] = {}
        self._system_config_data[model_name][key] = value
        # Immediately update the merged view as well
        self._merge_configs()

    def save_user_config(self, user_config_path: str | Path | None = None) -> None:
        """現在のユーザー設定 (_user_config_data) を指定されたファイルパスに保存します。"""
        save_path = Path(user_config_path) if user_config_path else self._user_config_path

        if save_path is None:
            logger.error("ユーザー設定の保存パスが指定されていません。保存をスキップします。")
            return

        if not self._user_config_data:
            logger.debug("保存すべきユーザー設定がありません。保存をスキップします。")
            return

        try:
            # Ensure the parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the user config data to the file
            with open(save_path, "w", encoding="utf-8") as f:
                toml.dump(self._user_config_data, f)
            logger.info(f"ユーザー設定を {save_path} に保存しました。")

        except Exception as e:
            logger.error(
                f"ユーザー設定ファイル {save_path} の保存中にエラーが発生しました: {e}", exc_info=True
            )

    def save_system_config(self, system_config_path: str | Path | None = None) -> None:
        """現在のシステム設定 (_system_config_data) を指定されたファイルパスに保存します。"""
        save_path = Path(system_config_path) if system_config_path else self._system_config_path

        if save_path is None:
            logger.error("システム設定の保存パスが指定されていません。保存をスキップします。")
            return

        if not self._system_config_data:
            # 通常、システム設定が空になることはないはずだが念のため
            logger.warning("保存すべきシステム設定がありません。保存をスキップします。")
            return

        try:
            # Ensure the parent directory exists (though usually not needed for system config)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the system config data to the file
            with open(save_path, "w", encoding="utf-8") as f:
                toml.dump(self._system_config_data, f)
            logger.info(f"システム設定を {save_path} に保存しました。")

        except Exception as e:
            logger.error(
                f"システム設定ファイル {save_path} の保存中にエラーが発生しました: {e}", exc_info=True
            )

    def get_all_config(self) -> dict[str, Any]:
        """ロード済みのマージされた設定データ全体を辞書として返します。"""
        return self._merged_config_data.copy()  # 内部データのコピーを返す


# --- 共有インスタンスの作成と初期ロード --- #
config_registry = ModelConfigRegistry()
try:
    config_registry.load()
except Exception:
    logger.exception("共有設定レジストリの初期ロード中にエラーが発生しました。")
