import gc
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast, override

import onnxruntime as ort
import psutil
import tensorflow as tf
import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.clip import CLIPModel, CLIPProcessor
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline  # Import Pipeline base class

from ..exceptions.errors import ModelLoadError, OutOfMemoryError
from . import utils
from .config import config_registry
from .utils import logger


# --- TypedDict Definitions ---
class TransformersComponents(TypedDict):
    model: AutoModelForVision2Seq
    processor: AutoProcessor


class TransformersPipelineComponents(TypedDict):
    pipeline: Pipeline


class ONNXComponents(TypedDict):
    session: ort.InferenceSession
    csv_path: Path


class TensorFlowComponents(TypedDict):
    model_dir: Path
    model: tf.Module | tf.keras.Model


class CLIPComponents(TypedDict):
    model: nn.Module  # Classifier head
    processor: CLIPProcessor
    clip_model: CLIPModel


# --- ModelLoad Refactoring ---
class ModelLoad:
    """
    ModelLoadクラスの実装

    モデルのロード、メモリ管理、キャッシュ制御を担当します。
    内部ヘルパーメソッドと内部ローダークラスを用いて、責務分離と共通化を図ります。
    外部インターフェース (静的メソッド) は後方互換性のために維持されます。
    """

    # --- Class Variables ---
    _MODEL_STATES: ClassVar[dict[str, str]] = {}
    _MEMORY_USAGE: ClassVar[dict[str, float]] = {}
    _MODEL_LAST_USED: ClassVar[dict[str, float]] = {}
    _CACHE_RATIO: ClassVar[float] = 0.5
    _MODEL_SIZES: ClassVar[dict[str, float]] = {}  # Central static cache

    # --- Internal Helper Methods: Size Management ---

    @staticmethod
    def _get_model_size_from_config(model_name: str) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB単位) を取得。"""
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
                    f"モデル '{model_name}' config 内 estimated_size_gb ('{estimated_size_gb_any}') を float に変換できません。"
                )
                return None
        except KeyError:
            return None  # Model not in config
        except Exception as e:
            logger.error(f"モデル '{model_name}' config サイズ取得中に予期せぬエラー: {e}", exc_info=True)
            return None

    @staticmethod
    def _calculate_file_size_mb(file_path: Path) -> float:
        """ファイルサイズをMB単位で取得"""
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
        """ディレクトリサイズをMB単位で計算"""
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
        """Transformerモデルのパラメータ/バッファサイズからメモリ使用量(MB)を計算"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)
        except Exception as e:
            logger.error(f"Transformerモデルサイズ計算エラー: {e}", exc_info=True)
            return 0.0

    @staticmethod
    def _save_size_to_config(model_name: str, size_mb: float) -> None:
        """計算されたサイズをMB単位でConfigに保存"""
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
        model_path: str,  # Path or identifier
        model_type: str,  # 'transformers', 'pipeline', 'onnx', 'tensorflow', 'clip'
        loader_instance: Any,  # Instance of the internal loader for calculation method
        **kwargs: Any,  # Additional args for specific calculators (task, format, etc.)
    ) -> float:
        """モデルサイズを取得または計算 (MB単位、内部ヘルパー)"""
        # 1. Check static cache
        if model_name in cls._MODEL_SIZES and cls._MODEL_SIZES[model_name] > 0:
            logger.debug(
                f"モデル '{model_name}' サイズキャッシュ取得: {cls._MODEL_SIZES[model_name]:.2f} MB"
            )
            return cls._MODEL_SIZES[model_name]

        # 2. Check config
        config_size_mb = cls._get_model_size_from_config(model_name)
        if config_size_mb is not None and config_size_mb > 0:
            cls._MODEL_SIZES[model_name] = config_size_mb  # Cache config value
            return config_size_mb

        # 3. Calculate size using loader-specific method
        logger.info(f"モデル '{model_name}' サイズ不明 ({model_type})。計算試行...")
        calculated_size_mb = 0.0
        try:
            # Delegate to the specific loader's calculation method
            if hasattr(loader_instance, "_calculate_specific_size"):
                calculated_size_mb = loader_instance._calculate_specific_size(
                    model_path=model_path, **kwargs
                )
            else:
                logger.warning(
                    f"ローダー ({type(loader_instance).__name__}) にサイズ計算メソッド _calculate_specific_size がありません。"
                )

            if calculated_size_mb > 0:
                logger.info(f"モデル '{model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB")
            else:
                logger.warning(f"モデル '{model_name}' サイズ計算失敗または結果が0。")

        except Exception as e:
            logger.error(f"モデル '{model_name}' サイズ計算中にエラー: {e}", exc_info=True)
            calculated_size_mb = 0.0  # Treat as unknown on error

        # 4. Cache and save calculated size (even if 0.0, indicates calculation was attempted)
        cls._MODEL_SIZES[model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            cls._save_size_to_config(model_name, calculated_size_mb)

        return calculated_size_mb

    # --- Internal Helper Methods: Cache/State Management ---

    @staticmethod
    def _check_memory_before_load(model_size_mb: float, model_name: str) -> bool:
        """モデルロード前に利用可能なメモリを確認"""
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
            f"メモリチェック ({model_name}): 必要={required_memory_gb:.3f}GB, 利用可能={available_memory_gb:.3f}GB"
        )

        if available_memory_bytes < required_memory_bytes:
            error_detail = f"メモリ不足警告: モデル '{model_name}' ({required_memory_gb:.3f}GB) ロード不可。空きメモリ ({available_memory_gb:.3f}GB) 不足。"
            logger.warning(error_detail)
            return False  # Indicate failure
        else:
            logger.debug(f"モデル '{model_name}' ロードに必要なメモリ確保済み。")
            return True

    @classmethod
    def _get_max_cache_size(cls) -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        return float(total_memory * cls._CACHE_RATIO)

    @classmethod
    def _clear_cache_internal(cls, model_name_to_load: str, required_size_mb: float) -> bool:
        """必要に応じて古いモデルをキャッシュから削除 (内部ヘルパー)。成功/失敗を返す。"""
        max_cache = cls._get_max_cache_size()
        initial_cache_size = sum(cls._MEMORY_USAGE.values())

        if initial_cache_size + required_size_mb <= max_cache:
            return True  # Enough space

        max_cache_gb = max_cache / 1024
        current_cache_gb = initial_cache_size / 1024
        required_gb = required_size_mb / 1024
        logger.warning(
            f"キャッシュ容量({max_cache_gb:.3f}GB)超過。解放試行..."
            f"(現: {current_cache_gb:.3f}GB + 新: {required_gb:.3f}GB)"
        )

        models_by_age = sorted(cls._MODEL_LAST_USED.items(), key=lambda x: x[1])
        released_something = False

        for old_model_name, last_used in models_by_age:
            current_cache_size = sum(cls._MEMORY_USAGE.values())
            if current_cache_size + required_size_mb <= max_cache:
                logger.info("キャッシュ解放停止: 必要容量確保完了。")
                break

            if old_model_name == model_name_to_load:
                continue
            if old_model_name not in cls._MODEL_STATES:
                continue

            freed_memory = cls._MEMORY_USAGE.get(old_model_name, 0)
            logger.info(
                f"モデル '{old_model_name}' を解放 (最終使用: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_used))}, "
                f"解放メモリ: {freed_memory / 1024:.3f}GB)"
            )
            cls._release_model_state(old_model_name)  # Release state only
            released_something = True

        final_cache_size = sum(cls._MEMORY_USAGE.values())
        if final_cache_size + required_size_mb > max_cache:
            final_cache_gb = final_cache_size / 1024
            logger.error(
                f"キャッシュ解放後もモデル '{model_name_to_load}' ({required_gb:.3f}GB) の容量確保失敗。"
                f"(最大: {max_cache_gb:.3f}GB, 最終使用量: {final_cache_gb:.3f}GB)"
            )
            return False  # Failed to make enough space
        else:
            if released_something:
                logger.info("キャッシュの解放処理完了。")
            return True  # Enough space now

    @classmethod
    def _update_model_state(
        cls,
        model_name: str,
        device: str | None = None,
        status: str | None = None,  # "loaded", "cached_cpu", "released", or None to just update time
        size_mb: float | None = None,
    ) -> None:
        """モデルの状態、メモリ使用量、最終使用時刻を一元的に更新"""
        current_time = time.time()

        if status == "released":
            if model_name in cls._MODEL_STATES:
                del cls._MODEL_STATES[model_name]
            if model_name in cls._MEMORY_USAGE:
                del cls._MEMORY_USAGE[model_name]
            if model_name in cls._MODEL_LAST_USED:
                del cls._MODEL_LAST_USED[model_name]
            # Don't clear _MODEL_SIZES on release, keep calculated value. Clear on load error.
            # if model_name in cls._MODEL_SIZES: del cls._MODEL_SIZES[model_name]
            logger.debug(f"モデル '{model_name}' 状態情報解放。")
            return

        if model_name in cls._MODEL_STATES or status:  # Update if exists or status is changing
            cls._MODEL_LAST_USED[model_name] = current_time

        if status and device:
            new_state = f"on_{device}" if status == "loaded" else "on_cpu"
            cls._MODEL_STATES[model_name] = new_state
            logger.debug(f"モデル '{model_name}' 状態 -> {new_state}")

        if size_mb is not None:
            if size_mb > 0:
                cls._MEMORY_USAGE[model_name] = size_mb
                logger.debug(f"モデル '{model_name}' メモリ使用量 -> {size_mb / 1024:.3f} GB")
            elif model_name in cls._MEMORY_USAGE:  # size is 0 or invalid
                del cls._MEMORY_USAGE[model_name]
                logger.debug(f"モデル '{model_name}' メモリ使用量クリア (サイズ0または無効)")

    @classmethod
    def _get_model_state(cls, model_name: str) -> str | None:
        """モデルの現在の状態を取得"""
        return cls._MODEL_STATES.get(model_name)

    @staticmethod
    def _move_components_to_device(components: dict[str, Any], target_device: str) -> None:
        """コンポーネントを指定デバイスに移動 (共通ロジック)"""
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
                    if current_device_str != target_device and isinstance(
                        component, torch.Tensor | torch.nn.Module
                    ):
                        component.to(target_device)
                        moved = True
                # Add specific handling for ONNX/TF if needed (likely not standard .to method)

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
        """モデルの状態情報のみを解放"""
        cls._update_model_state(model_name, status="released")

    @classmethod
    def _release_model_internal(cls, model_name: str, components: dict[str, Any] | None = None) -> None:
        """モデルの状態と関連コンポーネントを解放 (内部ヘルパー)"""
        logger.info(f"モデル '{model_name}' 解放処理開始...")
        if components:
            try:
                logger.debug("コンポーネント削除試行...")
                for component_name in list(components.keys()):
                    component = components[component_name]
                    if component_name in (
                        "model",
                        "pipeline",
                        "session",
                        "clip_model",
                        "processor",
                    ):  # Add processor
                        logger.debug(f"  - Deleting component: {component_name}")
                        del components[component_name]
                        del component
                logger.debug("コンポーネント削除完了。")
            except Exception as e:
                logger.error(f"コンポーネント削除中にエラー ({model_name}): {e}", exc_info=True)

        cls._release_model_state(model_name)  # Release state regardless

        try:
            logger.debug("ガベージコレクションとCUDAキャッシュクリア実行...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("クリーンアップ完了。")
        except Exception as e:
            logger.error(f"GC/CUDAキャッシュクリア中にエラー: {e}", exc_info=True)
        logger.info(f"モデル '{model_name}' 解放処理完了。")

    @classmethod
    def _handle_load_error(cls, model_name: str, error: Exception) -> None:
        """ロードエラー処理 (ログ出力と状態クリーンアップ)"""
        error_msg = str(error)
        is_memory_error = False
        if isinstance(error, torch.cuda.OutOfMemoryError | MemoryError | OutOfMemoryError):
            is_memory_error = True
        elif isinstance(error, OSError) and "allocate memory" in error_msg.lower():
            is_memory_error = True
        elif "onnxruntime" in str(type(error)).lower() and (
            "Failed to allocate memory" in error_msg or "AllocateRawInternal" in error_msg
        ):
            is_memory_error = True
        elif isinstance(error, tf.errors.ResourceExhaustedError):
            is_memory_error = True

        if is_memory_error:
            logger.error(f"メモリ不足エラー: モデル '{model_name}' ロード中。詳細: {error_msg}")
            if isinstance(error, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
                try:
                    device_name = str(error.device) if hasattr(error, "device") else "cuda"
                    logger.error(f"CUDA メモリサマリー ({device_name}):")
                    logger.error(torch.cuda.memory_summary(device=device_name, abbreviated=True))
                except Exception as mem_e:
                    logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
        elif isinstance(error, FileNotFoundError):
            logger.error(f"ファイル未検出: モデル '{model_name}' ロード中: {error_msg}", exc_info=False)
        else:
            logger.error(f"予期せぬロードエラー ({model_name}): {error_msg}", exc_info=True)

        # Clean up state and potentially calculated size cache on any load error
        if model_name in cls._MODEL_SIZES and cls._MODEL_SIZES[model_name] == 0.0:
            del cls._MODEL_SIZES[model_name]  # Clear failed calculation attempt
        cls._release_model_state(model_name)

    # --- Internal Loader Base Class ---
    class _BaseLoaderInternal(ABC):
        """Internal base loader."""

        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        @abstractmethod
        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            """Abstract: Calculate size for this loader type."""
            raise NotImplementedError

        @abstractmethod
        def _load_components_internal(self, model_path: str, **kwargs) -> dict[str, Any]:
            """Abstract: Load actual components for this loader type."""
            raise NotImplementedError

        def load_components(self, model_path: str, **kwargs) -> dict[str, Any] | None:
            """Generic load sequence using ModelLoad helpers."""
            model_type = self.__class__.__name__.replace("_LoaderInternal", "").lower()

            # 0. Check state
            if ModelLoad._get_model_state(self.model_name):
                logger.debug(f"モデル '{self.model_name}' ({model_type}) は既にロード/キャッシュ済み。")
                ModelLoad._update_model_state(self.model_name)  # Update last used
                return None  # Indicate no *new* load occurred

            # 1. Get/Calculate Size
            model_size_mb = ModelLoad._get_or_calculate_size(
                self.model_name, model_path, model_type, self, **kwargs
            )

            # 2. Memory Check
            if not ModelLoad._check_memory_before_load(model_size_mb, self.model_name):
                ModelLoad._handle_load_error(
                    self.model_name, MemoryError(f"Pre-load memory check failed for {self.model_name}")
                )
                return None

            # 3. Clear Cache
            if model_size_mb > 0:
                if not ModelLoad._clear_cache_internal(self.model_name, model_size_mb):
                    # Failed to clear enough space, should we still try loading?
                    # Let's prevent loading if cache clear fails explicitly.
                    ModelLoad._handle_load_error(
                        self.model_name,
                        MemoryError(f"Failed to clear sufficient cache for {self.model_name}"),
                    )
                    return None
            else:
                logger.warning(
                    f"モデル '{self.model_name}' サイズ不明/0、キャッシュクリアはベストエフォート。"
                )

            # 4. Load Components
            components: dict[str, Any] | None = None
            try:
                logger.info(
                    f"モデル '{self.model_name}' ({model_type}) ロード開始 (デバイス: {self.device})..."
                )
                components = self._load_components_internal(model_path=model_path, **kwargs)

                # 5. Handle Success
                ModelLoad._update_model_state(self.model_name, self.device, "loaded", model_size_mb)
                logger.info(
                    f"モデル '{self.model_name}' ({model_type}) ロード成功 (デバイス: {self.device})。"
                )
                return components

            except Exception as e:
                ModelLoad._handle_load_error(self.model_name, e)
                return None

    # --- Internal Loader Implementations ---

    class _TransformersLoader(_BaseLoaderInternal):
        """Internal loader for Transformers models."""

        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            logger.debug(f"一時ロードによる Transformer サイズ計算開始: {model_path}")
            calculated_size_mb = 0.0
            temp_model = None
            try:
                with utils.suppress_logging(level="WARNING"):
                    temp_model = AutoModelForVision2Seq.from_pretrained(model_path).to("cpu")
                if isinstance(temp_model, torch.nn.Module):
                    calculated_size_mb = ModelLoad._calculate_transformer_size_mb(temp_model)
            except Exception as e:
                logger.error(f"一時ロード計算エラー ({self.model_name}): {e}", exc_info=True)
            finally:
                if temp_model:
                    del temp_model
                    gc.collect()
            logger.debug(f"一時ロード Transformer サイズ計算完了: {calculated_size_mb:.2f} MB")
            return calculated_size_mb

        @override
        def _load_components_internal(self, model_path: str, **kwargs) -> TransformersComponents:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
            return {"model": model, "processor": processor}

    class _TransformersPipelineLoader(_BaseLoaderInternal):
        """Internal loader for Transformers Pipelines."""

        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            task = cast(str, kwargs.get("task"))
            if not task:
                return 0.0  # Task is required

            logger.debug(f"一時ロードによる Pipeline サイズ計算開始: task={task}, path={model_path}")
            calculated_size_mb = 0.0
            temp_pipeline = None
            try:
                with utils.suppress_logging(level="WARNING"):
                    temp_pipeline = pipeline(task, model=model_path, device="cpu", batch_size=1)
                if hasattr(temp_pipeline, "model") and isinstance(temp_pipeline.model, torch.nn.Module):
                    calculated_size_mb = ModelLoad._calculate_transformer_size_mb(temp_pipeline.model)
            except Exception as e:
                logger.error(f"一時ロード Pipeline 計算エラー ({self.model_name}): {e}", exc_info=True)
            finally:
                if temp_pipeline:
                    del temp_pipeline
                    gc.collect()
            logger.debug(f"一時ロード Pipeline サイズ計算完了: {calculated_size_mb:.2f} MB")
            return calculated_size_mb

        @override
        def _load_components_internal(self, model_path: str, **kwargs) -> TransformersPipelineComponents:
            task = cast(str, kwargs.get("task"))
            batch_size = cast(int, kwargs.get("batch_size"))
            if not task or not batch_size:
                raise ValueError("Pipeline loader requires 'task' and 'batch_size' kwargs.")

            pipeline_obj: Pipeline = pipeline(
                task, model=model_path, device=self.device, batch_size=batch_size
            )
            return {"pipeline": pipeline_obj}

    class _ONNXLoader(_BaseLoaderInternal):
        """Internal loader for ONNX models."""

        def _resolve_model_path_internal(self, model_path: str) -> tuple[Path | None, Path | None]:
            """Resolves ONNX model path and associated CSV path."""
            try:
                # TODO: Replace utils.download_onnx_tagger_model if possible
                csv_path, model_repo_or_path_obj = utils.download_onnx_tagger_model(model_path)
                if model_repo_or_path_obj is None:
                    logger.error(f"ONNX モデルパス/リポジトリ解決失敗: {model_path}")
                    return None, None
                logger.debug(f"ONNXモデルパス解決: {model_repo_or_path_obj}")
                return csv_path, model_repo_or_path_obj
            except Exception as e:
                logger.error(f"ONNXモデルパス解決中にエラー ({model_path}): {e}", exc_info=True)
                return None, None

        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            _, resolved_path = self._resolve_model_path_internal(model_path)
            if resolved_path and resolved_path.is_file():
                # Use multiplier? Original used 1.5. Keep for now.
                return ModelLoad._calculate_file_size_mb(resolved_path) * 1.5
            elif resolved_path and resolved_path.is_dir():
                logger.warning(
                    f"ONNX パス {resolved_path} はディレクトリです。サイズ計算はベストエフォート。"
                )
                # Calculate directory size * multiplier? Or just file size?
                # Let's try finding the largest .onnx file in the dir.
                onnx_files = list(resolved_path.glob("*.onnx"))
                if onnx_files:
                    largest_onnx = max(onnx_files, key=lambda p: p.stat().st_size)
                    return ModelLoad._calculate_file_size_mb(largest_onnx) * 1.5
                else:
                    return ModelLoad._calculate_dir_size_mb(resolved_path) * 1.5  # Fallback to dir size
            else:
                logger.warning(f"ONNXモデル有効パス見つからず ({resolved_path})。サイズ計算スキップ。")
                return 0.0

        @override
        def _load_components_internal(self, model_path: str, **kwargs) -> ONNXComponents:
            csv_path, resolved_model_path = self._resolve_model_path_internal(model_path)
            if resolved_model_path is None or csv_path is None:
                raise FileNotFoundError(f"ONNXモデルパス解決失敗: {model_path}")

            logger.debug("ONNXキャッシュクリア試行...")
            gc.collect()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            logger.debug(f"ONNX実行プロバイダー: {providers}")

            logger.info(
                f"ONNXモデル '{self.model_name}' ロード中: '{resolved_model_path}' on {providers}... "
            )
            session = ort.InferenceSession(str(resolved_model_path), providers=providers)
            logger.info(f"ONNXモデル '{self.model_name}' ロード成功。")

            return {"session": session, "csv_path": csv_path}

    class _TensorFlowLoader(_BaseLoaderInternal):
        """Internal loader for TensorFlow models."""

        def _resolve_model_dir_internal(self, model_path: str) -> Path | None:
            try:
                # TODO: Replace utils.load_file if possible
                model_dir_obj = utils.load_file(model_path)
                if model_dir_obj is None or not model_dir_obj.is_dir():
                    logger.error(f"有効な TensorFlow モデルディレクトリが見つかりません: {model_path}")
                    return None
                return model_dir_obj
            except Exception as e:
                logger.error(f"TFモデルディレクトリ解決中にエラー ({model_path}): {e}", exc_info=True)
                return None

        def _get_tf_calc_params(self, model_dir: Path, model_format: str) -> tuple[Path, float]:
            """Get path and multiplier for TF size calculation."""
            if model_format == "h5":
                h5_files = list(model_dir.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError(f"H5ファイルが見つかりません: {model_dir}")
                return h5_files[0], 1.2  # Path to .h5 file
            elif model_format == "saved_model":
                if (
                    not (model_dir / "saved_model.pb").exists()
                    and not (model_dir / "saved_model.pbtxt").exists()
                ):
                    raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {model_dir}")
                return model_dir, 1.3  # Path to directory
            elif model_format == "pb":
                raise NotImplementedError(".pb 単体フォーマットのサイズ計算は未サポート。")
            else:
                raise ValueError(f"未対応のTFフォーマット: {model_format}")

        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            model_format = cast(str, kwargs.get("model_format"))
            if not model_format:
                return 0.0

            model_dir = self._resolve_model_dir_internal(model_path)
            if not model_dir:
                return 0.0

            try:
                target_path, multiplier = self._get_tf_calc_params(model_dir, model_format)
                if target_path.is_file():
                    return ModelLoad._calculate_file_size_mb(target_path) * multiplier
                elif target_path.is_dir():
                    # Calculate based on directory size
                    return ModelLoad._calculate_dir_size_mb(target_path) * multiplier
                else:
                    return 0.0
            except (FileNotFoundError, NotImplementedError, ValueError) as e:
                logger.error(f"TFモデル '{self.model_name}' サイズ計算エラー: {e}", exc_info=False)
                return 0.0
            except Exception as e:
                logger.error(
                    f"TFモデル '{self.model_name}' サイズ計算中に予期せぬエラー: {e}", exc_info=True
                )
                return 0.0

        @override
        def _load_components_internal(self, model_path: str, **kwargs) -> TensorFlowComponents:
            model_format = cast(str, kwargs.get("model_format"))
            if not model_format:
                raise ValueError("TensorFlow loader requires 'model_format' kwarg.")

            model_dir = self._resolve_model_dir_internal(model_path)
            if model_dir is None:
                raise FileNotFoundError(f"TensorFlow モデルディレクトリ解決失敗: {model_path}")

            model_instance: tf.Module | tf.keras.Model | None = None
            if model_format == "h5":
                h5_files = list(model_dir.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError(f"H5ファイルが見つかりません: {model_dir}")
                target_path = h5_files[0]
                logger.info(f"H5モデルロード中: {target_path}")
                model_instance = tf.keras.models.load_model(target_path, compile=False)
            elif model_format == "saved_model":
                target_path = model_dir
                logger.info(f"SavedModelロード中: {target_path}")
                if (
                    not (target_path / "saved_model.pb").exists()
                    and not (target_path / "saved_model.pbtxt").exists()
                ):
                    raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {target_path}")
                model_instance = tf.saved_model.load(str(target_path))
            elif model_format == "pb":
                raise NotImplementedError("Direct loading from .pb is not supported.")
            else:
                raise ValueError(f"未対応のTensorFlowモデルフォーマット: {model_format}")

            if model_instance is None:
                raise ModelLoadError("TensorFlowモデルインスタンスのロード失敗。")

            return {"model_dir": model_dir, "model": model_instance}

    class _CLIPLoader(_BaseLoaderInternal):
        """Internal loader for CLIP-based models."""

        def _infer_classifier_structure(self, state_dict: dict[str, Any]) -> list[int]:
            """state_dictから分類器ヘッドの隠れ層サイズを推測する。"""
            hidden_features = []
            current_layer = 0
            while True:
                weight_key = f"layers.{current_layer}.weight"
                bias_key = f"layers.{current_layer}.bias"  # bias_keyも確認した方が良い
                # Look for the next potential layer if current is missing
                if weight_key not in state_dict or bias_key not in state_dict:
                    found_next = False
                    # Look ahead a few layers in case numbering is sparse
                    for lookahead in range(1, 5):
                        next_weight_key = f"layers.{current_layer + lookahead}.weight"
                        if next_weight_key in state_dict:
                            # Found a subsequent layer, update current_layer index
                            current_layer += lookahead
                            weight_key = next_weight_key
                            found_next = True
                            break
                    if not found_next:
                        # No more layers found within lookahead range
                        break

                # Check again if the (potentially updated) weight_key exists
                if weight_key in state_dict:
                    # Assuming the shape[0] of the weight tensor gives the output size of that layer
                    hidden_features.append(state_dict[weight_key].shape[0])
                    current_layer += 1  # Move to check the next sequential layer number
                else:
                    # Should not happen if found_next logic worked, but as safety break
                    break

            # The last feature size is the output layer, hidden sizes are all before that
            hidden_sizes = hidden_features[:-1] if hidden_features else []
            if not hidden_sizes:  # Use default if inference failed
                logger.warning(
                    f"CLIP分類器 '{self.model_name}' 構造推測失敗。デフォルト [1024, 128, 64, 16] 使用。"
                )
                hidden_sizes = [1024, 128, 64, 16]
            logger.info(f"推測された隠れ層サイズ: {hidden_sizes}")
            return hidden_sizes

        def _load_base_clip_components(self, base_model: str) -> tuple[CLIPProcessor, CLIPModel, int]:
            """CLIP プロセッサとベースモデルをロードし、特徴量次元を返す"""
            logger.debug(f"CLIPプロセッサロード中: {base_model}")
            clip_processor = CLIPProcessor.from_pretrained(base_model)
            logger.debug(f"CLIPモデルロード中: {base_model} on {self.device}")
            clip_model = CLIPModel.from_pretrained(base_model).to(self.device).eval()  # type: ignore
            input_size = clip_model.config.projection_dim
            logger.debug(f"CLIPモデル {base_model} 特徴量次元: {input_size}")
            return clip_processor, clip_model, input_size

        def _create_and_load_classifier_head(
            self,
            input_size: int,
            hidden_sizes: list[int],
            state_dict: dict[str, Any],
            activation_type: str | None,
            final_activation_type: str | None,
            model_path_for_log: str,  # Logging purpose
        ) -> nn.Module:
            """分類器ヘッドを作成し、重みをロードして返す"""
            activation_map = {"ReLU": nn.ReLU, "GELU": nn.GELU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}
            use_activation = activation_type is not None
            activation_func = (
                activation_map.get(activation_type, nn.ReLU)
                if use_activation and activation_type in activation_map
                else nn.ReLU
            )
            use_final_activation = final_activation_type is not None
            final_activation_func = (
                activation_map.get(final_activation_type, nn.Sigmoid)
                if use_final_activation and final_activation_type in activation_map
                else nn.Sigmoid
            )

            logger.info("分類器モデル初期化...")
            classifier_head = Classifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=1,
                use_activation=use_activation,
                activation=activation_func,
                use_final_activation=use_final_activation,
                final_activation=final_activation_func,
            )
            classifier_head.load_state_dict(state_dict, strict=False)
            classifier_head = classifier_head.to(self.device).eval()
            logger.debug(f"CLIP分類器ヘッド '{model_path_for_log}' ロード完了 (デバイス: {self.device})")
            return classifier_head

        # TODO: Refactor create_clip_model structure inference logic (use config).
        def _create_clip_model_internal(
            self,
            base_model: str,
            model_path: str,  # Path to classifier head weights
            activation_type: str | None,
            final_activation_type: str | None,
        ) -> CLIPComponents | None:
            """Creates the CLIP processor, base model, and classifier head."""
            try:
                # 1. Load Base CLIP Model and Processor
                clip_processor, clip_model, input_size = self._load_base_clip_components(base_model)

                # 2. Load Classifier Head Weights
                logger.debug(f"分類器ヘッド重みロード中: {model_path}")
                local_path = utils.load_file(model_path)
                if local_path is None:
                    logger.error(f"分類器ヘッドパス '{model_path}' 解決失敗。")
                    return None
                state_dict = torch.load(local_path, map_location=self.device)
                logger.debug("重みロード完了、構造推測開始...")

                # 3. Infer Classifier Structure
                hidden_sizes_for_classifier = self._infer_classifier_structure(state_dict)

                # 4. Create and Load Classifier Head
                classifier_head = self._create_and_load_classifier_head(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes_for_classifier,
                    state_dict=state_dict,
                    activation_type=activation_type,
                    final_activation_type=final_activation_type,
                    model_path_for_log=model_path,
                )

                # 5. Type safety checks
                if not isinstance(classifier_head, nn.Module):
                    raise TypeError("Classifier head not a Module")
                if not isinstance(clip_processor, CLIPProcessor):
                    raise TypeError("Processor not a CLIPProcessor")
                if not isinstance(clip_model, CLIPModel):
                    raise TypeError("Base model not a CLIPModel")

                result: CLIPComponents = {
                    "model": classifier_head,
                    "processor": clip_processor,
                    "clip_model": clip_model,
                }
                return result

            except FileNotFoundError as e:
                logger.error(f"CLIPモデル作成エラー: ファイル未検出: {e}")
                return None
            except KeyError as e:
                logger.error(f"CLIPモデル作成エラー: state_dict キーエラー: {e}")
                return None
            except Exception as e:
                logger.error(
                    f"CLIPモデル作成中に予期せぬエラー ({base_model}/{model_path}): {e}", exc_info=True
                )
                return None

        def _calculate_specific_size(self, model_path: str, **kwargs) -> float:
            """Calculate size by temporarily loading CLIP model + head."""
            base_model = cast(str, kwargs.get("base_model"))
            activation_type = cast(str | None, kwargs.get("activation_type"))
            final_activation_type = cast(str | None, kwargs.get("final_activation_type"))
            if not base_model:
                return 0.0

            logger.debug(f"一時ロードによる CLIP サイズ計算開始: base={base_model}, head={model_path}")
            calculated_size_mb = 0.0
            temp_components: CLIPComponents | None = None
            try:
                with utils.suppress_logging(level="WARNING"):
                    # Temporarily create on CPU
                    temp_components = self._create_clip_model_internal(
                        base_model,
                        model_path,
                        activation_type,
                        final_activation_type,
                        # Override device to CPU for calculation
                        # Need to modify _create_clip_model_internal to accept device override or create on CPU first
                        # For now, let's assume _create_clip_model_internal handles CPU creation if device='cpu'
                        # self.device = 'cpu' # This would modify the instance state - bad.
                        # --> Let's call helper directly with device='cpu'
                    )
                    # Need a way to call the helper with device='cpu' without modifying self.device
                    # Simplest for now: create a temporary instance or pass device explicitly if helper allows
                    temp_helper_instance = ModelLoad._CLIPLoader(self.model_name, "cpu")  # Temp instance
                    temp_components = temp_helper_instance._create_clip_model_internal(
                        base_model, model_path, activation_type, final_activation_type
                    )

                if temp_components and isinstance(temp_components.get("model"), torch.nn.Module):
                    # Calculate size of the classifier head only? Or base + head?
                    # Original calculation seemed to be based on the classifier head. Let's stick to that.
                    classifier_model = temp_components["model"]
                    calculated_size_mb = ModelLoad._calculate_transformer_size_mb(classifier_model)
                else:
                    logger.warning(f"一時ロード CLIP '{self.model_name}' から有効な分類器取得失敗。")
            except Exception as e:
                logger.error(f"一時ロード CLIP 計算エラー ({self.model_name}): {e}", exc_info=True)
            finally:
                if temp_components:
                    # Manual cleanup of components in the temporary dict
                    if "model" in temp_components:
                        del temp_components["model"]
                    if "processor" in temp_components:
                        del temp_components["processor"]
                    if "clip_model" in temp_components:
                        del temp_components["clip_model"]
                    del temp_components
                    del temp_helper_instance  # Delete the temporary instance
                    gc.collect()
            logger.debug(f"一時ロード CLIP サイズ計算完了: {calculated_size_mb:.2f} MB")
            return calculated_size_mb

        @override
        def _load_components_internal(self, model_path: str, **kwargs) -> CLIPComponents:
            base_model = cast(str, kwargs.get("base_model"))
            activation_type = cast(str | None, kwargs.get("activation_type"))
            final_activation_type = cast(str | None, kwargs.get("final_activation_type"))
            if not base_model:
                raise ValueError("CLIP loader requires 'base_model' kwarg.")

            components = self._create_clip_model_internal(
                base_model, model_path, activation_type, final_activation_type
            )
            if components is None:
                raise ModelLoadError(f"CLIPモデル '{self.model_name}' のコンポーネント作成失敗。")
            return components

    # --- Public Static Methods (Interface Preservation) ---

    @staticmethod
    def get_model_size(model_name: str) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB単位) を取得。"""
        return ModelLoad._get_model_size_from_config(model_name)

    @staticmethod
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        return ModelLoad._get_max_cache_size()

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントをメインメモリ (CPU) にキャッシュ。"""
        state = ModelLoad._get_model_state(model_name)
        if state == "on_cpu":
            logger.debug(f"モデル '{model_name}' は既にCPUキャッシュにあります。")
            ModelLoad._update_model_state(model_name)  # Update last used time
            return components

        model_size = ModelLoad._MODEL_SIZES.get(model_name, 0.0)
        can_cache = True
        if model_size <= 0:
            logger.warning(f"モデル '{model_name}' サイズ不明/0、CPUキャッシュ前の容量確認スキップ。")
        else:
            # Check if space is available BEFORE moving
            # Use a different check: Does current cache + this model exceed limit?
            # _clear_cache_internal checks and clears, but here we only want to check.
            max_cache = ModelLoad._get_max_cache_size()
            current_usage = sum(ModelLoad._MEMORY_USAGE.values())
            # Temporarily remove the model's usage if it was loaded on GPU
            usage_without_model = current_usage - ModelLoad._MEMORY_USAGE.get(model_name, 0.0)
            if usage_without_model + model_size > max_cache:
                logger.warning(
                    f"CPUキャッシュ不可: モデル '{model_name}' ({model_size / 1024:.3f}GB) を追加するとキャッシュ容量超過。解放試行..."
                )
                # Try clearing space for it
                if not ModelLoad._clear_cache_internal(model_name, model_size):
                    can_cache = False  # Still not enough space after clearing

        if not can_cache:
            logger.error(f"モデル '{model_name}' CPUキャッシュ失敗: 十分な空き容量なし。")
            # Should we release the model state if caching fails? Yes.
            ModelLoad._release_model_state(model_name)
            return {}  # Return empty dict on failure?

        try:
            ModelLoad._move_components_to_device(components, "cpu")
            ModelLoad._update_model_state(model_name, "cpu", "cached_cpu", model_size)
            logger.info(f"モデル '{model_name}' をCPUにキャッシュしました。")
            return components
        except Exception as e:
            logger.error(f"モデル '{model_name}' CPUキャッシュ中にエラー: {e}", exc_info=True)
            ModelLoad._release_model_state(model_name)
            return {}

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> TransformersComponents | None:
        """Transformersモデルをロード (静的インターフェース)"""
        loader = ModelLoad._TransformersLoader(model_name, device)
        # Cast the result to the specific TypedDict or None
        result = loader.load_components(model_path)
        return cast(TransformersComponents | None, result)

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> TransformersPipelineComponents | None:
        """TransformersPipelineモデルをロード"""
        loader = ModelLoad._TransformersPipelineLoader(model_name, device)
        result = loader.load_components(model_path, task=task, batch_size=batch_size)
        return cast(TransformersPipelineComponents | None, result)

    @staticmethod
    def load_onnx_components(model_name: str, model_path: str, device: str) -> ONNXComponents | None:
        """ONNXモデルをロード"""
        loader = ModelLoad._ONNXLoader(model_name, device)
        result = loader.load_components(model_path)
        return cast(ONNXComponents | None, result)

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,  # TF device handling is implicit usually
        model_format: str,
    ) -> TensorFlowComponents | None:
        """TensorFlowモデルをロード"""
        loader = ModelLoad._TensorFlowLoader(model_name, device)
        result = loader.load_components(model_path, model_format=model_format)
        return cast(TensorFlowComponents | None, result)

    @staticmethod
    def load_clip_components(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> CLIPComponents | None:
        """CLIPモデルをロード"""
        loader = ModelLoad._CLIPLoader(model_name, device)
        result = loader.load_components(
            model_path,  # model_path is head weights path
            base_model=base_model,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )
        return cast(CLIPComponents | None, result)

    @staticmethod
    def restore_model_to_cuda(
        model_name: str, device: str, components: dict[str, Any]
    ) -> dict[str, Any] | None:
        """モデルをCUDAデバイスに復元 (インターフェース維持)"""
        state = ModelLoad._get_model_state(model_name)
        if not state:
            logger.warning(f"モデル '{model_name}' 状態不明。CUDA復元スキップ。")
            return None
        if state == f"on_{device}":
            logger.debug(f"モデル '{model_name}' は既に {device} にあります。")
            ModelLoad._update_model_state(model_name)  # Update last used
            return components

        logger.info(f"モデル '{model_name}' を {device} に復元中...")
        model_size = ModelLoad._MODEL_SIZES.get(model_name, 0.0)

        # Check memory BEFORE attempting move
        if not ModelLoad._check_memory_before_load(model_size, model_name):
            logger.error(f"CUDA復元失敗: メモリ不足 ({model_name} -> {device})")
            return None  # Keep state as is (likely on_cpu)

        # Try clearing cache needed for this model size on the target device
        if model_size > 0:
            if not ModelLoad._clear_cache_internal(model_name, model_size):
                logger.error(f"CUDA復元失敗: キャッシュ解放失敗 ({model_name} -> {device})")
                return None  # Keep state as is

        try:
            ModelLoad._move_components_to_device(components, device)
            ModelLoad._update_model_state(model_name, device, "loaded", model_size)
            logger.info(f"モデル '{model_name}' を {device} に復元完了。")
            return components
        except Exception as e:
            logger.error(f"モデル '{model_name}' の {device} への復元中にエラー: {e}", exc_info=True)
            try:  # Attempt fallback to CPU
                logger.warning(f"CUDA復元エラー後、CPUへのフォールバック試行 ({model_name})...")
                ModelLoad._move_components_to_device(components, "cpu")
                ModelLoad._update_model_state(model_name, "cpu", "cached_cpu", model_size)
                logger.warning(f"CUDA復元エラーのため、モデル '{model_name}' をCPUに戻しました。")
            except Exception as fallback_e:
                logger.error(f"CPUフォールバック中にエラー ({model_name}): {fallback_e}", exc_info=True)
                ModelLoad._release_model_state(model_name)  # Release if fallback fails
            return None  # Indicate restore failed

    @staticmethod
    def release_model(model_name: str) -> None:
        """指定されたモデルをキャッシュから解放 (インターフェース維持)"""
        ModelLoad._release_model_internal(model_name, components=None)

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントリソースを解放 (インターフェース維持)"""
        ModelLoad._release_model_internal(model_name, components)
        return {}  # Return empty dict


# --- Classifier and create_clip_model (Keep at module level for now) ---
class Classifier(nn.Module):
    """画像特徴量を入力として、分類スコアを出力する柔軟な分類器。"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        dropout_rates: list[float] | None = None,
        use_activation: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        use_final_activation: bool = False,
        final_activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [1024, 128, 64, 16]
        dropout_rates = dropout_rates if dropout_rates is not None else [0.2, 0.2, 0.1, 0.0]
        if len(dropout_rates) < len(hidden_sizes):
            dropout_rates.extend([0.0] * (len(hidden_sizes) - len(dropout_rates)))

        layers: list[nn.Module] = []
        prev_size = input_size
        for size, drop in zip(hidden_sizes, dropout_rates, strict=False):
            layers.append(nn.Linear(prev_size, size))
            if use_activation:
                layers.append(activation())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        if use_final_activation:
            layers.append(final_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Note: create_clip_model is now effectively handled within ModelLoad._CLIPLoader._create_clip_model_internal
# We keep the Classifier class definition here as it's used by the CLIPLoader.
