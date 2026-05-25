import copy
import importlib.resources
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

from .constants import (
    DEFAULT_PATHS,
    TEMPLATE_SYSTEM_CONFIG_PATH,
)
from .utils import logger


@lru_cache
def _load_config_from_file(config_path: Path) -> dict[str, dict[str, Any]]:
    """設定ファイルをTOML形式で読み込む内部ヘルパー関数。"""
    try:
        logger.debug(f"構成ファイルを読み込みます: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config_data = toml.load(f)
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


class ModelConfigRegistry:
    """設定ファイル全体を管理し、設定値へのアクセスを提供します。"""

    def __init__(self) -> None:
        """初期化時に設定データを空の辞書で初期化します。"""
        self._system_config_data: dict[str, dict[str, Any]] = {}
        self._user_config_data: dict[str, dict[str, Any]] = {}
        self._merged_config_data: dict[str, dict[str, Any]] = {}
        self._system_config_path: Path | None = None
        self._user_config_path: Path | None = None

    def _determine_config_paths(
        self,
        config_path: str | Path | None = None,
        user_config_path: str | Path | None = None,
    ) -> None:
        """システム設定とユーザー設定のパスを決定し、インスタンス変数に格納する。"""
        self._system_config_path = Path(config_path if config_path else DEFAULT_PATHS["config_toml"])
        self._user_config_path = Path(
            user_config_path if user_config_path else DEFAULT_PATHS["user_config_toml"]
        )
        logger.debug(f"システム設定パスを決定: {self._system_config_path}")
        logger.debug(f"ユーザー設定パスを決定: {self._user_config_path}")

    def _ensure_system_config_exists(self) -> None:
        """システム設定ファイルが存在しない場合、テンプレートからコピーする。"""
        if not self._system_config_path or not self._system_config_path.exists():
            logger.info(
                f"システム設定ファイルが見つかりません: {self._system_config_path}。テンプレートからのコピーを試みます。"
            )
            try:
                # TEMPLATE_SYSTEM_CONFIG_PATH を config モジュール経由で参照
                with importlib.resources.as_file(TEMPLATE_SYSTEM_CONFIG_PATH) as template_path:
                    if template_path.is_file():
                        if self._system_config_path:
                            self._system_config_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(template_path, self._system_config_path)
                            logger.info(
                                f"テンプレートから設定ファイルをコピーしました: {self._system_config_path}"
                            )
                        else:
                            logger.error(
                                f"テンプレートから設定ファイルをコピーしましたが、システム設定パスが指定されていません: {self._system_config_path}"
                            )
                    else:
                        logger.error(f"テンプレートパスが無効です: {template_path}")
            except FileNotFoundError:
                logger.error(
                    f"パッケージ内のテンプレート設定ファイルが見つかりません: {TEMPLATE_SYSTEM_CONFIG_PATH}"
                )
            except Exception as e:
                logger.error(f"設定ファイルの自動コピー中にエラー: {e}", exc_info=True)
        else:
            logger.debug(f"システム設定ファイルが存在します: {self._system_config_path}")

    def _load_and_set_system_config(self) -> None:
        """システム設定ファイルを読み込み、内部状態を更新する。"""
        if self._system_config_path and self._system_config_path.is_file():
            try:
                self._system_config_data = _load_config_from_file(self._system_config_path)
                logger.info(f"システム設定を {self._system_config_path} から読み込みました。")
            except Exception as e:
                logger.error(
                    f"システム設定ファイル {self._system_config_path} の読み込みに失敗しました: {e}"
                )
                self._system_config_data = {}
        else:
            logger.error(
                f"システム設定ファイル {self._system_config_path} が見つからないか、ファイルではありません。読み込みをスキップします。"
            )
            self._system_config_data = {}

    def _load_and_set_user_config(self) -> None:
        """ユーザー設定ファイルを読み込み、内部状態を更新する。"""
        self._user_config_data = {}
        if self._user_config_path is not None and self._user_config_path.exists():
            # is_file チェックを追加
            if self._user_config_path.is_file():
                try:
                    self._user_config_data = _load_config_from_file(self._user_config_path)
                    logger.info(f"ユーザー設定を {self._user_config_path} から読み込みました。")
                except Exception as e:
                    logger.warning(
                        f"ユーザー設定ファイル {self._user_config_path} の読み込み中にエラー: {e}"
                    )
                    self._user_config_data = {}
            else:
                logger.warning(
                    f"ユーザー設定パス {self._user_config_path} はファイルではありません。読み込みをスキップします。"
                )
        else:
            logger.debug(
                f"ユーザー設定ファイル {self._user_config_path} が存在しないかパスが指定されていません。"
            )

    def load(
        self,
        config_path: str | Path | None = None,
        user_config_path: str | Path | None = None,
    ) -> None:
        """システム設定ファイルとユーザー設定ファイルを読み込み、内部データを更新します。"""
        logger.info("設定ファイルの読み込みを開始します...")
        self._determine_config_paths(config_path, user_config_path)
        self._ensure_system_config_exists()
        self._load_and_set_system_config()
        self._load_and_set_user_config()
        self._merge_configs()
        logger.info("設定ファイルの読み込みが完了しました。")

    def _merge_configs(self) -> None:
        """システム設定とユーザー設定をマージして _merged_config_data を作成 (Deep Copyを使用)"""
        self._merged_config_data = copy.deepcopy(self._system_config_data)
        for model_name, user_model_config in self._user_config_data.items():
            if model_name in self._merged_config_data:
                merged_model_config = self._merged_config_data[model_name]
                system_capabilities = merged_model_config.get("capabilities")
                user_capabilities = user_model_config.get("capabilities")

                merged_model_config.update(copy.deepcopy(user_model_config))
                if system_capabilities is not None or user_capabilities is not None:
                    merged_model_config["capabilities"] = self._merge_capabilities(
                        system_capabilities,
                        user_capabilities,
                    )
            else:
                self._merged_config_data[model_name] = copy.deepcopy(user_model_config)
        logger.debug("システム設定とユーザー設定をディープコピーでマージしました。")

    @staticmethod
    def _merge_capabilities(system_capabilities: Any, user_capabilities: Any) -> list[Any]:
        """Merge capability declarations without letting user config remove system capabilities."""
        merged: list[Any] = []
        for capability_source in (system_capabilities, user_capabilities):
            if capability_source is None:
                continue
            capabilities = capability_source if isinstance(capability_source, list) else [capability_source]
            for capability in capabilities:
                if capability not in merged:
                    merged.append(copy.deepcopy(capability))
        return merged

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

    def add_default_setting(self, section_name: str, key: str, value: Any) -> None:
        """
        システム設定データにデフォルト値を設定し、変更があればファイルに保存する。

        指定されたセクションが存在しない場合は作成する。
        指定されたキーがセクション内に存在しない場合のみ、値を設定する。
        既存のキーの値は上書きしない。
        値を設定した場合（データが変更された場合）のみ、システム設定ファイルを保存する。

        Args:
            section_name: 設定を追加するセクション名 (モデル名など)。
            key: 追加する設定のキー。
            value: 追加する設定の値。
        """
        setting_added = False
        # Ensure the section exists
        if section_name not in self._system_config_data:
            self._system_config_data[section_name] = {}
            logger.debug(f"システム設定に新しいセクションを追加: [{section_name}]")
            # Technically, creating a section might be considered an addition
            # But we only trigger save if a key/value is added.

        # Add the key/value only if the key does not exist
        if key not in self._system_config_data[section_name]:
            self._system_config_data[section_name][key] = value
            logger.info(f"システム設定 [{section_name}] にデフォルト値を追加: {key} = {value}")
            setting_added = True
        else:
            logger.debug(
                f"システム設定 [{section_name}] のキー '{key}' は既に存在するため、デフォルト値の追加をスキップしました。"
            )

        # Update the merged config regardless of whether a setting was added
        self._merge_configs()

        # Save the system config file only if a setting was actually added
        if setting_added:
            logger.debug("デフォルト設定が追加されたため、システム設定ファイルを保存します。")
            self.save_system_config()
        else:
            logger.debug(
                "デフォルト設定の追加はスキップされたため、システム設定ファイルの保存は行いません。"
            )

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


# --- 共有インスタンスの作成と遅延ロード --- #
_config_registry: ModelConfigRegistry | None = None


def get_config_registry() -> ModelConfigRegistry:
    """設定レジストリのシングルトンインスタンスを取得（遅延初期化）"""
    global _config_registry
    if _config_registry is None:
        _config_registry = ModelConfigRegistry()
        # テスト環境以外では自動ロード
        import os

        if not os.getenv("PYTEST_CURRENT_TEST"):
            try:
                _config_registry.load()
            except Exception:
                logger.exception("共有設定レジストリの初期ロード中にエラーが発生しました。")
    return _config_registry


# 後方互換性のための遅延プロパティ
class _ConfigRegistryProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_config_registry(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_config_registry(), name, value)


config_registry = _ConfigRegistryProxy()

# ADR 0023 Phase 1 (Issue #35 PR #40): `available_api_models.toml` cache は **明示的に廃止**。
# 旧 `_load_full_api_models_file` / `load_available_api_models` / `load_api_models_meta` /
# `load_last_refresh` / `save_available_api_models` は本ファイルから削除された。
# WebAPI モデル一覧と capability metadata は `core/api_model_discovery.py` で
# LiteLLM 同梱 DB から runtime 取得する (TOML キャッシュなし)。
