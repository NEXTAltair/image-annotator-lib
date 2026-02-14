"""モデルローダー基底クラスとキャッシュ/状態管理。

LoaderBase は全てのフレームワーク固有ローダーの抽象基底クラスであり、
モデル状態管理・キャッシュ・メモリ管理の共通ロジックを提供する。

Dependencies:
    - psutil: メモリ使用量の監視
    - image_annotator_lib.core.config: モデルサイズ設定の読み書き
    - image_annotator_lib.core.utils: デバイス判定、ログ
"""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import psutil

from ...exceptions.errors import OutOfMemoryError
from ..config import config_registry
from ..types import LoaderComponents
from ..utils import logger

if TYPE_CHECKING:
    import torch


class LoaderBase(ABC):
    """全フレームワーク固有ローダーの抽象基底クラス。

    ClassVar による共有状態を一元管理し、ロード・キャッシュ・解放の
    共通フローを提供する。サブクラスは `_calculate_specific_size` と
    `_load_components_internal` を実装する必要がある。
    """

    # --- 共有状態 (全ローダーで共通) ---
    _MODEL_STATES: ClassVar[dict[str, str]] = {}
    _MEMORY_USAGE: ClassVar[dict[str, float]] = {}
    _MODEL_LAST_USED: ClassVar[dict[str, float]] = {}
    _CACHE_RATIO: ClassVar[float] = 0.5
    _MODEL_SIZES: ClassVar[dict[str, float]] = {}

    def __init__(self, model_name: str, device: str) -> None:
        """ベースローダーを初期化する。

        Args:
            model_name: モデルインスタンスの名前。
            device: ターゲットデバイス (例: "cuda", "cpu")。
        """
        from .. import utils

        self.model_name = model_name
        # CUDA利用可否を検証し、利用不可ならCPUにフォールバック
        self.device = utils.determine_effective_device(device, model_name)

    # --- 抽象メソッド ---

    @abstractmethod
    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """特定のモデルタイプの推定サイズ (MB) を計算する。

        Args:
            model_path: モデルのパスまたは識別子。
            **kwargs: フレームワーク固有の引数。

        Returns:
            推定サイズ (MB)。計算不能時は 0.0。
        """
        raise NotImplementedError

    @abstractmethod
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> LoaderComponents:
        """特定のフレームワーク用モデルコンポーネントをロードする。

        Args:
            model_path: モデルのパスまたは識別子。
            **kwargs: フレームワーク固有の引数。

        Returns:
            ロードされたコンポーネントを含む TypedDict。

        Raises:
            Exception: ロード失敗時。
        """
        raise NotImplementedError

    # --- ロードオーケストレーション ---

    def load_components(self, model_path: str, **kwargs: Any) -> LoaderComponents | None:
        """モデルのロードプロセスを調整する。

        標準シーケンス:
        1. 既存状態を確認
        2. モデルサイズを取得/計算
        3. メモリチェック
        4. 必要に応じてキャッシュクリア
        5. コンポーネントをロード
        6. 成功時に状態更新
        7. エラー処理

        Args:
            model_path: モデルのパスまたは識別子。
            **kwargs: サイズ計算・ロードメソッドに渡す追加引数。

        Returns:
            ロード成功時はコンポーネント TypedDict、
            既にロード済みまたは失敗時は None。
        """
        model_type = self.__class__.__name__.replace("Loader", "").lower()

        # 0. 状態チェック
        if LoaderBase._get_model_state(self.model_name):
            logger.debug(f"モデル '{self.model_name}' ({model_type}) は既にロード/キャッシュ済み。")
            LoaderBase._update_model_state(self.model_name)
            return None

        # 1. サイズ取得/計算
        model_size_mb = LoaderBase._get_or_calculate_size(
            self.model_name, model_path, model_type, self, **kwargs
        )

        # 2. メモリチェック
        if not LoaderBase._check_memory_before_load(model_size_mb, self.model_name):
            LoaderBase._handle_load_error(
                self.model_name, MemoryError(f"Pre-load memory check failed for {self.model_name}")
            )
            return None

        # 3. キャッシュクリア
        if model_size_mb > 0:
            if not LoaderBase._clear_cache_internal(self.model_name, model_size_mb):
                LoaderBase._handle_load_error(
                    self.model_name,
                    MemoryError(f"Failed to clear sufficient cache for {self.model_name}"),
                )
                return None
        else:
            logger.warning(f"モデル '{self.model_name}' サイズ不明/0、キャッシュクリアはベストエフォート。")

        # 4. コンポーネントロード
        try:
            logger.info(
                f"モデル '{self.model_name}' ({model_type}) ロード開始 (デバイス: {self.device})..."
            )
            components = self._load_components_internal(model_path=model_path, **kwargs)

            # 5. 成功時の状態更新
            LoaderBase._update_model_state(self.model_name, self.device, "loaded", model_size_mb)
            logger.info(f"モデル '{self.model_name}' ({model_type}) ロード成功 (デバイス: {self.device})。")
            return components

        except Exception as e:
            LoaderBase._handle_load_error(self.model_name, e)
            return None

    # --- サイズ管理ヘルパー ---

    @staticmethod
    def _get_model_size_from_config(model_name: str) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB) を取得する。"""
        try:
            estimated_size_gb_any = config_registry.get(model_name, "estimated_size_gb")
            if estimated_size_gb_any is None:
                return None
            try:
                estimated_size_gb = float(estimated_size_gb_any)
                size_mb = estimated_size_gb * 1024
                logger.debug(f"モデル '{model_name}' サイズを config から読み込み: {size_mb / 1024:.3f}GB")
                return size_mb
            except (ValueError, TypeError):
                logger.error(
                    f"モデル '{model_name}' config 内 estimated_size_gb"
                    f" ('{estimated_size_gb_any}') を float に変換できません。"
                )
                return None
        except KeyError:
            return None
        except Exception as e:
            logger.error(f"モデル '{model_name}' config サイズ取得中に予期せぬエラー: {e}", exc_info=True)
            return None

    @staticmethod
    def _calculate_file_size_mb(file_path: Path) -> float:
        """ファイルサイズを MB 単位で取得する。"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            logger.error(f"サイズ計算エラー: ファイルが見つかりません {file_path}")
            return 0.0
        except Exception as e:
            logger.error(f"ファイルサイズ取得エラー ({file_path}): {e}")
            return 0.0

    @staticmethod
    def _calculate_dir_size_mb(dir_path: Path) -> float:
        """ディレクトリサイズを MB 単位で計算する。"""
        total_size = 0
        try:
            for item in dir_path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
            return total_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"ディレクトリサイズ計算エラー ({dir_path}): {e}", exc_info=True)
            return 0.0

    @staticmethod
    def _calculate_transformer_size_mb(model: torch.nn.Module) -> float:
        """Transformer モデルのパラメータ/バッファからメモリ使用量 (MB) を計算する。"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)
        except Exception as e:
            logger.error(f"Transformerモデルサイズ計算エラー: {e}", exc_info=True)
            return 0.0

    @staticmethod
    def _save_size_to_config(model_name: str, size_mb: float) -> None:
        """計算されたサイズを MB 単位で Config に保存する。"""
        if size_mb <= 0:
            return
        try:
            size_gb = size_mb / 1024
            config_registry.set_system_value(model_name, "estimated_size_gb", round(size_gb, 3))
            config_registry.save_system_config()
            logger.debug(f"モデル '{model_name}' 計算サイズ ({size_gb:.3f}GB) をシステム設定に保存。")
        except Exception as e:
            logger.error(f"モデル '{model_name}' サイズのシステム設定保存中にエラー: {e}", exc_info=True)

    @classmethod
    def _get_or_calculate_size(
        cls,
        model_name: str,
        model_path: str,
        model_type: str,
        loader_instance: LoaderBase,
        **kwargs: Any,
    ) -> float:
        """モデルサイズを取得または計算する (MB)。

        静的キャッシュ → Config → ローダー計算の順に確認する。

        Args:
            model_name: モデル名。
            model_path: モデルパスまたは識別子。
            model_type: モデルタイプ (例: 'transformers')。
            loader_instance: 具象ローダーインスタンス。
            **kwargs: ローダーのサイズ計算メソッドに渡す追加引数。

        Returns:
            推定/計算されたモデルサイズ (MB)。失敗時は 0.0。
        """
        # 1. 静的キャッシュ
        if model_name in cls._MODEL_SIZES and cls._MODEL_SIZES[model_name] > 0:
            logger.debug(
                f"モデル '{model_name}' サイズキャッシュ取得: {cls._MODEL_SIZES[model_name]:.2f} MB"
            )
            return cls._MODEL_SIZES[model_name]

        # 2. Config
        config_size_mb = cls._get_model_size_from_config(model_name)
        if config_size_mb is not None and config_size_mb > 0:
            cls._MODEL_SIZES[model_name] = config_size_mb
            return config_size_mb

        # 3. ローダー経由で計算
        logger.info(f"モデル '{model_name}' サイズ不明 ({model_type})。計算試行...")
        calculated_size_mb = 0.0
        try:
            calculated_size_mb = loader_instance._calculate_specific_size(model_path=model_path, **kwargs)
            if calculated_size_mb > 0:
                logger.info(f"モデル '{model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB")
            else:
                logger.warning(f"モデル '{model_name}' サイズ計算失敗または結果が0。")
        except Exception as e:
            logger.warning(f"モデル '{model_name}' サイズ計算中にエラー: {e}", exc_info=False)
            calculated_size_mb = 0.0

        # 4. キャッシュと Config に保存
        cls._MODEL_SIZES[model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            cls._save_size_to_config(model_name, calculated_size_mb)

        return calculated_size_mb

    # --- キャッシュ/状態管理ヘルパー ---

    @classmethod
    def _get_current_cache_usage(cls) -> float:
        """現在のキャッシュ使用量合計 (MB) を返す。"""
        return sum(cls._MEMORY_USAGE.values())

    @classmethod
    def _get_models_sorted_by_last_used(cls) -> list[tuple[str, float]]:
        """最終使用時刻でソートされた (モデル名, 時刻) のリストを返す (古い順)。"""
        return sorted(cls._MODEL_LAST_USED.items(), key=lambda x: x[1])

    @classmethod
    def _get_model_memory_usage(cls, model_name: str) -> float:
        """指定モデルのメモリ使用量 (MB) を返す。"""
        return cls._MEMORY_USAGE.get(model_name, 0.0)

    @staticmethod
    def _check_memory_before_load(model_size_mb: float, model_name: str) -> bool:
        """モデルロード前に十分な仮想メモリが利用可能か確認する。

        Args:
            model_size_mb: ロードするモデルの推定サイズ (MB)。
            model_name: モデル名。

        Returns:
            十分なメモリがあれば True、なければ False。
        """
        if model_size_mb <= 0:
            logger.debug(
                f"モデル '{model_name}' サイズ不明/無効 ({model_size_mb})、事前メモリチェックをスキップ。"
            )
            return True

        available_memory_bytes = psutil.virtual_memory().available
        required_memory_bytes = model_size_mb * 1024 * 1024
        available_memory_gb = available_memory_bytes / (1024**3)
        required_memory_gb = required_memory_bytes / (1024**3)

        logger.debug(
            f"メモリチェック ({model_name}): 必要={required_memory_gb:.3f}GB,"
            f" 利用可能={available_memory_gb:.3f}GB"
        )

        if available_memory_bytes < required_memory_bytes:
            logger.warning(
                f"メモリ不足警告: モデル '{model_name}' ({required_memory_gb:.3f}GB) ロード不可。"
                f"空きメモリ ({available_memory_gb:.3f}GB) 不足。"
            )
            return False
        logger.debug(f"モデル '{model_name}' ロードに必要なメモリ確保済み。")
        return True

    @classmethod
    def _get_max_cache_size(cls) -> float:
        """システムの総メモリとキャッシュ比率に基づいて最大キャッシュサイズを計算する。"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        return float(total_memory * cls._CACHE_RATIO)

    @classmethod
    def _clear_cache_internal(cls, model_name_to_load: str, required_size_mb: float) -> bool:
        """必要に応じて LRU でモデルをキャッシュから削除する。

        Args:
            model_name_to_load: スペースが必要なモデル名。
            required_size_mb: 必要なメモリ量 (MB)。

        Returns:
            十分なスペースがあるか確保できた場合 True、そうでなければ False。
        """
        max_cache = cls._get_max_cache_size()
        initial_cache_size = cls._get_current_cache_usage()

        if initial_cache_size + required_size_mb <= max_cache:
            return True

        max_cache_gb = max_cache / 1024
        current_cache_gb = initial_cache_size / 1024
        required_gb = required_size_mb / 1024
        logger.warning(
            f"キャッシュ容量({max_cache_gb:.3f}GB)超過。解放試行..."
            f"(現: {current_cache_gb:.3f}GB + 新: {required_gb:.3f}GB)"
        )

        models_by_age = cls._get_models_sorted_by_last_used()
        released_something = False

        for old_model_name, last_used in models_by_age:
            current_cache_size = cls._get_current_cache_usage()
            if current_cache_size + required_size_mb <= max_cache:
                logger.info("キャッシュ解放停止: 必要容量確保完了。")
                break

            if old_model_name == model_name_to_load:
                continue
            if old_model_name not in cls._MODEL_STATES:
                continue

            freed_memory = cls._get_model_memory_usage(old_model_name)
            logger.info(
                f"モデル '{old_model_name}' を解放"
                f" (最終使用: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_used))},"
                f" 解放メモリ: {freed_memory / 1024:.3f}GB)"
            )
            cls._release_model_state(old_model_name)
            released_something = True

        final_cache_size = cls._get_current_cache_usage()
        if final_cache_size + required_size_mb > max_cache:
            final_cache_gb = final_cache_size / 1024
            logger.error(
                f"キャッシュ解放後もモデル '{model_name_to_load}'"
                f" ({required_gb:.3f}GB) の容量確保失敗。"
                f"(最大: {max_cache_gb:.3f}GB, 最終使用量: {final_cache_gb:.3f}GB)"
            )
            return False
        if released_something:
            logger.info("キャッシュの解放処理完了。")
        return True

    @classmethod
    def _update_model_state(
        cls,
        model_name: str,
        device: str | None = None,
        status: str | None = None,
        size_mb: float | None = None,
    ) -> None:
        """モデルの状態、メモリ使用量、最終使用時刻を一元的に更新する。

        Args:
            model_name: モデル名。
            device: モデルのデバイス (例: "cuda", "cpu")。
            status: 新しいステータス ("loaded", "cached_cpu", "released")。
                    None の場合は最終使用時刻のみ更新。
            size_mb: メモリ使用量 (MB)。None または <= 0 の場合は使用量エントリを削除。
        """
        current_time = time.time()

        if status == "released":
            cls._MODEL_STATES.pop(model_name, None)
            cls._MEMORY_USAGE.pop(model_name, None)
            cls._MODEL_LAST_USED.pop(model_name, None)
            logger.debug(f"モデル '{model_name}' 状態情報解放。")
            return

        if model_name in cls._MODEL_STATES or status:
            cls._MODEL_LAST_USED[model_name] = current_time

        if status and device:
            new_state = f"on_{device}" if status == "loaded" else "on_cpu"
            cls._MODEL_STATES[model_name] = new_state
            logger.debug(f"モデル '{model_name}' 状態 -> {new_state}")

        if size_mb is not None:
            if size_mb > 0:
                cls._MEMORY_USAGE[model_name] = size_mb
                logger.debug(f"モデル '{model_name}' メモリ使用量 -> {size_mb / 1024:.3f} GB")
            elif model_name in cls._MEMORY_USAGE:
                del cls._MEMORY_USAGE[model_name]
                logger.debug(f"モデル '{model_name}' メモリ使用量クリア (サイズ0または無効)")

    @classmethod
    def _get_model_state(cls, model_name: str) -> str | None:
        """モデルの現在の状態を取得する (例: "on_cuda", "on_cpu")。"""
        return cls._MODEL_STATES.get(model_name)

    @staticmethod
    def _move_components_to_device(components: dict[str, Any], target_device: str) -> None:
        """モデルコンポーネント (PyTorch Module/Tensor) を指定デバイスに移動する。

        Args:
            components: モデルコンポーネント辞書。
            target_device: ターゲットデバイス文字列 (例: "cuda", "cpu")。
        """
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        logger.debug(f"コンポーネントを {target_device} に移動中...")
        for component_name, component in list(components.items()):
            moved = False
            current_device_str = "unknown"
            try:
                if (
                    component_name == "pipeline"
                    and hasattr(component, "model")
                    and hasattr(component.model, "to")
                ):
                    current_device_str = str(getattr(component.model, "device", "unknown"))
                    if current_device_str != target_device:
                        component.model.to(target_device)
                        moved = True
                elif hasattr(component, "to") and callable(component.to) and hasattr(component, "device"):
                    current_device_str = str(getattr(component, "device", "unknown"))
                    if (
                        torch is not None
                        and current_device_str != target_device
                        and isinstance(component, torch.Tensor | torch.nn.Module)
                    ):
                        component.to(target_device)
                        moved = True

                if moved:
                    logger.debug(
                        f"  - '{component_name}' を {current_device_str} -> {target_device} へ移動完了。"
                    )
                elif current_device_str == target_device:
                    logger.debug(f"  - '{component_name}' は既に {target_device} にあります。")

            except Exception as e:
                logger.warning(
                    f"コンポーネント '{component_name}' デバイス移動中にエラー ({target_device}): {e}",
                    exc_info=False,
                )

    @classmethod
    def _release_model_state(cls, model_name: str) -> None:
        """モデルの状態情報のみを解放する。"""
        cls._update_model_state(model_name, status="released")

    @classmethod
    def _release_model_internal(cls, model_name: str, components: dict[str, Any] | None = None) -> None:
        """モデルの状態と関連コンポーネントを解放する。

        Args:
            model_name: 解放するモデル名。
            components: 削除を試みるモデルコンポーネント辞書 (省略可)。
        """
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        logger.info(f"モデル '{model_name}' 解放処理開始...")
        if components:
            try:
                logger.debug("コンポーネント削除試行...")
                for component_name in list(components.keys()):
                    component = components[component_name]
                    if component_name in ("model", "pipeline", "session", "clip_model", "processor"):
                        logger.debug(f"  - Deleting component: {component_name}")
                        del components[component_name]
                        del component
                logger.debug("コンポーネント削除完了。")
            except Exception as e:
                logger.error(f"コンポーネント削除中にエラー ({model_name}): {e}", exc_info=True)

        cls._release_model_state(model_name)

        try:
            logger.debug("ガベージコレクションとCUDAキャッシュクリア実行...")
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("クリーンアップ完了。")
        except Exception as e:
            logger.error(f"GC/CUDAキャッシュクリア中にエラー: {e}", exc_info=True)
        logger.info(f"モデル '{model_name}' 解放処理完了。")

    @classmethod
    def _handle_load_error(cls, model_name: str, error: Exception) -> None:
        """ロードエラーを処理し、モデル状態をクリーンアップする。

        メモリエラーとその他のエラーを区別してログ出力する。

        Args:
            model_name: ロードに失敗したモデル名。
            error: ロード中に発生した例外。
        """
        torch_cuda_oom_error = None
        tf_resource_error = None
        torch_module = None

        try:
            import torch

            torch_module = torch
            torch_cuda_oom_error = torch.cuda.OutOfMemoryError
        except ImportError:
            pass

        try:
            import tensorflow as tf

            tf_resource_error = tf.errors.ResourceExhaustedError
        except ImportError:
            pass

        error_msg = str(error)
        is_memory_error = _check_is_memory_error(error, error_msg, torch_cuda_oom_error, tf_resource_error)

        if is_memory_error:
            logger.error(f"メモリ不足エラー: モデル '{model_name}' ロード中。詳細: {error_msg}")
            if (
                torch_cuda_oom_error is not None
                and isinstance(error, torch_cuda_oom_error)
                and torch_module is not None
                and torch_module.cuda.is_available()
            ):
                try:
                    device_name = str(error.device) if hasattr(error, "device") else "cuda"
                    logger.error(f"CUDA メモリサマリー ({device_name}):")
                    logger.error(torch_module.cuda.memory_summary(device=device_name, abbreviated=True))
                except Exception as mem_e:
                    logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
        elif isinstance(error, FileNotFoundError):
            logger.error(f"ファイル未検出: モデル '{model_name}' ロード中: {error_msg}", exc_info=False)
        else:
            logger.error(f"予期せぬロードエラー ({model_name}): {error_msg}", exc_info=True)

        # 失敗した計算キャッシュをクリア
        if model_name in cls._MODEL_SIZES and cls._MODEL_SIZES[model_name] == 0.0:
            del cls._MODEL_SIZES[model_name]
        cls._release_model_state(model_name)


def _check_is_memory_error(
    error: Exception,
    error_msg: str,
    torch_cuda_oom_error: type | None,
    tf_resource_error: type | None,
) -> bool:
    """エラーがメモリ不足関連かどうかを判定する。"""
    if isinstance(error, MemoryError | OutOfMemoryError):
        return True
    if torch_cuda_oom_error is not None and isinstance(error, torch_cuda_oom_error):
        return True
    if isinstance(error, OSError) and "allocate memory" in error_msg.lower():
        return True
    if "onnxruntime" in str(type(error)).lower() and (
        "Failed to allocate memory" in error_msg or "AllocateRawInternal" in error_msg
    ):
        return True
    if tf_resource_error is not None and isinstance(error, tf_resource_error):
        return True
    return False
