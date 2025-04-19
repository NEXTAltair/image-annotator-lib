import gc
import time
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast

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
    model: tf.Module | tf.keras.Model  # tf.Module covers SavedModel and Keras models


class CLIPComponents(TypedDict):
    model: nn.Module  # This is the Classifier
    processor: CLIPProcessor
    clip_model: CLIPModel


# --- BaseModelLoader ---
class BaseModelLoader:
    """モデルローダーの基底クラス"""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

        # --- 既存の初期化処理---
        self._MODEL_STATES = ModelLoad._MODEL_STATES
        self._MEMORY_USAGE = ModelLoad._MEMORY_USAGE
        self._MODEL_LAST_USED = ModelLoad._MODEL_LAST_USED
        self._CACHE_RATIO = 0.5
        # Remove _MODEL_SIZES instance variable, it will be managed by the static cache
        # self._MODEL_SIZES: dict[str, float] = {}

    def _check_memory_before_load(self, model_size_mb: float | None) -> bool:
        """モデルロード前に利用可能なメモリを確認する"""
        # model_size_mb = self.get_model_size()
        if model_size_mb is None or model_size_mb <= 0:
            logger.debug(
                f"モデル '{self.model_name}' のサイズが不明または無効 ({model_size_mb}) なため、"
                f"事前メモリチェックをスキップします。"
            )
            return True  # Allow loading if size is unknown

        available_memory_bytes = psutil.virtual_memory().available
        required_memory_bytes = model_size_mb * 1024 * 1024
        available_memory_gb = available_memory_bytes / (1024**3)
        required_memory_gb = required_memory_bytes / (1024**3)

        logger.debug(
            f"メモリチェック ({self.model_name}): 必要={required_memory_gb:.3f}GB, 利用可能={available_memory_gb:.3f}GB"
        )

        if available_memory_bytes < required_memory_bytes:
            error_detail = f"メモリ不足警告: モデル '{self.model_name}' ({required_memory_gb:.3f}GB) のロードをスキップします。利用可能なシステムメモリ ({available_memory_gb:.3f}GB) が不足しています。"
            logger.warning(error_detail)
            return False
        else:
            logger.debug(f"モデル '{self.model_name}' のロードに必要なメモリは確保されています。")
            return True

    def get_model_size(self) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB単位) を取得。なければ None を返す。"""
        # This method now ONLY checks the config via the registry.
        # Calculation is handled elsewhere.
        try:
            estimated_size_gb_any = config_registry.get(self.model_name, "estimated_size_gb")

            if estimated_size_gb_any is None:
                # logger.warning( # No warning here, expected case
                #     f"モデル '{self.model_name}' の estimated_size_gb が config に見つかりません。"
                # )
                return None  # Return None if not found in config

            # Attempt conversion, handle potential error
            try:
                estimated_size_gb = float(estimated_size_gb_any)
                size_mb = estimated_size_gb * 1024
                # No need to cache here (_MODEL_SIZES is static, handled elsewhere)
                # self._MODEL_SIZES[self.model_name] = size_mb
                logger.debug(
                    f"モデル '{self.model_name}' のサイズを config から読み込みました: {size_mb / 1024:.3f}GB"
                )
                return size_mb
            except (ValueError, TypeError):
                logger.error(
                    f"モデル '{self.model_name}' の config 内の estimated_size_gb 値 '{estimated_size_gb_any}' を float に変換できません。"
                )
                return None  # Return None if value is invalid

        except KeyError:
            # Model might not be in the config at all
            # logger.debug(f"モデル '{self.model_name}' が config に存在しません。")
            return None
        except Exception as e:
            logger.error(
                f"モデル '{self.model_name}' の config サイズ取得中に予期せぬエラー: {e}", exc_info=True
            )
            return None

    def get_max_cache_size(self) -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        cache_size = total_memory * self._CACHE_RATIO
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        logger.debug(
            f"システム全体のメモリ: {total_memory:.1f}MB, "
            f"現在の空きメモリ: {available_memory:.1f}MB, "
            f"設定キャッシュ容量: {cache_size:.1f}MB"
        )
        return float(cache_size)

    def _clear_cache_if_needed(self, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除"""
        max_cache = self.get_max_cache_size()
        initial_cache_size = sum(self._MEMORY_USAGE.values())

        if initial_cache_size + model_size <= max_cache:
            return

        max_cache_gb = max_cache / 1024
        current_cache_gb = initial_cache_size / 1024
        model_size_gb = model_size / 1024
        logger.warning(
            f"キャッシュ容量({max_cache_gb:.3f}GB)を超過します。"
            f"現在の使用量: {current_cache_gb:.3f}GB + 新規: {model_size_gb:.3f}GB"
        )

        models_by_age = sorted(self._MODEL_LAST_USED.items(), key=lambda x: x[1])

        for old_model_name, last_used in models_by_age:
            current_cache_size = sum(self._MEMORY_USAGE.values())
            if current_cache_size + model_size <= max_cache:
                logger.info("必要なキャッシュ容量が確保されたため、解放処理を停止します。")
                break

            if old_model_name == self.model_name:
                continue

            freed_memory = self._MEMORY_USAGE.get(old_model_name, 0)
            logger.info(
                f"モデル '{old_model_name}' を解放します"
                f"(最終使用: {time.strftime('%H:%M:%S', time.localtime(last_used))}, "
                f"解放メモリ: {freed_memory:.1f}MB)"
            )
            self.release_model(old_model_name)

        final_cache_size = sum(self._MEMORY_USAGE.values())
        if final_cache_size + model_size > max_cache:
            final_cache_gb = final_cache_size / 1024
            logger.error(
                f"古いモデルを解放しても、モデル '{self.model_name}' ({model_size_gb:.3f}GB) "
                f"のための十分なキャッシュ容量 ({max_cache_gb:.3f}GB) を確保できませんでした。"
                f"現在の使用量: {final_cache_gb:.3f}GB"
            )

    def release_model(self, model_name: str) -> None:
        """モデルの状態とメモリ使用量の記録を削除"""
        if model_name in self._MODEL_STATES:
            del self._MODEL_STATES[model_name]
        if model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[model_name]
        if model_name in self._MODEL_LAST_USED:
            del self._MODEL_LAST_USED[model_name]

    def _calculate_model_size(self, model_file_path: Path, multiplier: float) -> float:
        """モデルファイルサイズからメモリ使用量を推定(MB単位)"""
        try:
            file_size_mb = model_file_path.stat().st_size / (1024 * 1024)
            return file_size_mb * multiplier
        except FileNotFoundError:
            logger.error(f"モデルサイズ計算エラー: ファイルが見つかりません {model_file_path}")
            return 0.0
        except Exception as e:
            logger.error(f"モデルサイズ計算エラー ({model_file_path}): {e}")
            return 0.0


class TransformersLoader(BaseModelLoader):
    """Transformersモデルのローダー"""

    def _get_or_calculate_model_size(self, model_path: str) -> float:
        """モデルサイズをキャッシュ、Config、または計算によって取得・保存する。"""
        # 1. Check static cache
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] > 0:
            logger.debug(
                f"モデル '{self.model_name}' サイズをキャッシュから取得: {ModelLoad._MODEL_SIZES[self.model_name]:.2f} MB"
            )
            return ModelLoad._MODEL_SIZES[self.model_name]

        # 2. Check config
        config_size_mb = self.get_model_size()  # Use the updated method
        if config_size_mb is not None and config_size_mb > 0:
            ModelLoad._MODEL_SIZES[self.model_name] = config_size_mb  # Cache the value from config
            return config_size_mb

        # 3. Calculate size (if not found in cache or config)
        logger.info(f"モデル '{self.model_name}' サイズ不明。計算を試行します... (path: {model_path})")
        calculated_size_mb = 0.0
        temp_model = None
        try:
            # Temporarily load the model structure on CPU to calculate size
            logger.debug("サイズ計算のために一時的に モデル を CPU にロードします...")
            # Consider adding trust_remote_code=True if needed
            with utils.suppress_logging(level="WARNING"):  # Suppress verbose loading logs
                temp_model = AutoModelForVision2Seq.from_pretrained(model_path).to("cpu")

            if isinstance(temp_model, torch.nn.Module):
                calculated_size_mb = self._calculate_transformer_size(temp_model)
                logger.info(f"モデル '{self.model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB")
            else:
                logger.warning(
                    f"一時ロードした モデル '{self.model_name}' から有効なモデルオブジェクトを取得できませんでした。"
                )
        except Exception as e:
            logger.error(f"モデル '{self.model_name}' のサイズ計算中にエラー: {e}", exc_info=True)
            calculated_size_mb = 0.0  # Treat as unknown size on error
        finally:
            # Clean up the temporary model and release memory
            if temp_model:
                del temp_model
                gc.collect()
                if torch.cuda.is_available():  # Just in case something moved to GPU
                    torch.cuda.empty_cache()
            logger.debug("一時 モデル のクリーンアップ完了。")

        # 4. Cache and save the calculated size (even if 0.0)
        ModelLoad._MODEL_SIZES[self.model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            try:
                size_gb = calculated_size_mb / 1024
                config_registry.set_system_value(self.model_name, "estimated_size_gb", round(size_gb, 3))
                config_registry.save_system_config()
                logger.debug(
                    f"モデル '{self.model_name}' の計算サイズ ({size_gb:.3f}GB) をシステム設定に保存しました。"
                )
            except Exception as e:
                logger.error(
                    f"モデル '{self.model_name}' のサイズをシステム設定に保存中にエラー: {e}", exc_info=True
                )

        return calculated_size_mb

    def _load_model_and_processor(self, model_path: str) -> TransformersComponents:
        """実際のモデルとプロセッサのロード処理 (ターゲットデバイス上)"""
        logger.debug(f"モデル '{self.model_name}' のロード試行 (ターゲットデバイス: {self.device}) ... ")
        processor = AutoProcessor.from_pretrained(model_path)
        # Consider adding trust_remote_code=True if needed for some models
        model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
        logger.info(f"モデル '{self.model_name}' のロード完了 (ターゲットデバイス: {self.device})")
        # Ensure the return type matches the TypedDict definition
        return {"model": model, "processor": processor}

    def _handle_load_success(self, components: TransformersComponents, known_size_mb: float) -> None:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力 (サイズ計算は事前に行う)"""
        # Size calculation and saving is done beforehand
        final_model_size = known_size_mb if known_size_mb > 0 else 0.0

        # Update state, memory usage, and log
        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        # _MODEL_SIZES should already be updated by _get_or_calculate_model_size
        self._MEMORY_USAGE[self.model_name] = final_model_size
        self._MODEL_LAST_USED[self.model_name] = time.time()
        # Change log level to DEBUG
        logger.debug(
            f"モデル '{self.model_name}' のロード後状態更新完了 (デバイス: {self.device}, サイズ: {final_model_size / 1024:.3f}GB)"
        )
        # Remove config saving logic
        # # 4. Save calculated size ...
        # if calculated_size > 0:
        #    ...

    def _handle_memory_error(self, e: Exception) -> None:
        """メモリ関連エラー発生時の処理"""
        error_detail = f"モデル '{self.model_name}' のロード中にメモリ関連エラーが発生しました (デバイス: {self.device})。詳細: {e}"
        logger.error(f"メモリ不足エラー: {error_detail}")
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            try:
                logger.error(f"CUDA メモリサマリー ({self.device}):")
                # Limit summary length if it's too verbose
                logger.error(torch.cuda.memory_summary(device=self.device, abbreviated=True))
            except Exception as mem_e:
                logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]  # Also clear memory usage record
        # Also clear potential failed size cache entry if error occurred during calculation
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
            del ModelLoad._MODEL_SIZES[self.model_name]

    def _handle_generic_error(self, e: Exception) -> None:
        """その他の予期せぬエラー発生時の処理"""
        logger.error(
            f"モデル '{self.model_name}' のロード中に予期せぬエラーが発生しました: {e}", exc_info=True
        )
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]
        # Also clear potential failed size cache entry
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
            del ModelLoad._MODEL_SIZES[self.model_name]

    def load_components(self, model_path: str) -> TransformersComponents | None:
        """Transformersモデルをロード (早期サイズ計算・キャッシュ管理対応版)"""
        # --- 0. Check if already loaded --- #
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            # TODO: Return existing components if needed from a cache
            return None  # Indicate no load occurred

        # --- 1. Get or Calculate Model Size --- #
        model_size_mb = self._get_or_calculate_model_size(model_path)

        # --- 2. Memory Check --- #
        if not self._check_memory_before_load(model_size_mb):
            # Clean up potentially saved 0.0 size if memory check fails after calculation
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None  # Memory check failed

        # --- 3. Clear Cache if Needed --- #
        if model_size_mb > 0:
            logger.debug(
                f"モデル '{self.model_name}' ({model_size_mb:.2f} MB) ロード前にキャッシュクリア実行。"
            )
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"モデル '{self.model_name}' サイズ不明または0のため、キャッシュクリアはベストエフォートになります。"
            )

        # --- 4. Load Actual Components --- #
        components: TransformersComponents | None = None
        try:
            # Load model and processor onto the target device
            components = self._load_model_and_processor(model_path)

            # --- 5. Handle Load Success (State Update) --- #
            if components:
                self._handle_load_success(components, model_size_mb)
                return components
            else:
                # Should not happen if _load_model_and_processor raises errors
                logger.error(f"モデル '{self.model_name}' のロード内部処理で予期せず失敗しました。")
                self._handle_generic_error(
                    Exception(f"_load_model_and_processor for {self.model_name} returned None")
                )
                return None

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            # Check if OSError is likely a memory issue (e.g., "Cannot allocate memory")
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                self._handle_generic_error(e)  # Treat as generic if not clearly memory related
            else:
                self._handle_memory_error(e)
            return None
        except Exception as e:
            self._handle_generic_error(e)
            return None

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
            return param_size + buffer_size
        except Exception as e:
            logger.error(f"Transformerモデルサイズ計算エラー: {e}")
            return 0.0


class TransformersPipelineLoader(BaseModelLoader):
    """TransformersPipelineモデルのローダー"""

    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        # Add a static cache for model sizes if not already present in ModelLoad
        # Re-use ModelLoad's static cache directly
        # self._MODEL_SIZES = ModelLoad._MODEL_SIZES # This line caused errors and is removed

    def _get_or_calculate_model_size(self, task: str, model_path: str) -> float:
        """モデルサイズをキャッシュ、Config、または計算によって取得・保存する。"""
        # 1. Check static cache
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] > 0:
            logger.debug(
                f"Pipeline '{self.model_name}' サイズをキャッシュから取得: {ModelLoad._MODEL_SIZES[self.model_name]:.2f} MB"
            )
            return ModelLoad._MODEL_SIZES[self.model_name]

        # 2. Check config
        config_size_mb = self.get_model_size()  # Use the updated method
        if config_size_mb is not None and config_size_mb > 0:
            ModelLoad._MODEL_SIZES[self.model_name] = config_size_mb  # Cache the value from config
            return config_size_mb

        # 3. Calculate size (if not found in cache or config)
        logger.info(
            f"Pipeline '{self.model_name}' サイズ不明。計算を試行します... (task: {task}, path: {model_path})"
        )
        calculated_size_mb = 0.0
        temp_pipeline = None
        try:
            # Temporarily load the pipeline structure on CPU to calculate size
            # Avoid loading full weights if possible, but pipeline() might load them.
            # Use batch_size=1 for potentially lower memory usage during calculation.
            # Note: This still might be memory intensive.
            logger.debug("サイズ計算のために一時的に Pipeline を CPU にロードします...")
            with utils.suppress_logging(level="WARNING"):  # Suppress verbose loading logs
                temp_pipeline = pipeline(task, model=model_path, device="cpu", batch_size=1)

            if hasattr(temp_pipeline, "model") and isinstance(temp_pipeline.model, torch.nn.Module):
                calculated_size_mb = self._calculate_transformer_size(temp_pipeline.model)
                logger.info(f"Pipeline '{self.model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB")
            else:
                logger.warning(
                    f"一時ロードした Pipeline '{self.model_name}' から有効なモデルオブジェクトを取得できませんでした。"
                )
        except Exception as e:
            logger.error(f"Pipeline '{self.model_name}' のサイズ計算中にエラー: {e}", exc_info=True)
            calculated_size_mb = 0.0  # Treat as unknown size on error
        finally:
            # Clean up the temporary pipeline and release memory
            if temp_pipeline:
                del temp_pipeline
                gc.collect()
                if torch.cuda.is_available():  # Just in case something moved to GPU
                    torch.cuda.empty_cache()
            logger.debug("一時 Pipeline のクリーンアップ完了。")

        # 4. Cache and save the calculated size (even if 0.0)
        ModelLoad._MODEL_SIZES[self.model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            try:
                size_gb = calculated_size_mb / 1024
                config_registry.set_system_value(self.model_name, "estimated_size_gb", round(size_gb, 3))
                config_registry.save_system_config()
                logger.debug(
                    f"Pipeline '{self.model_name}' の計算サイズ ({size_gb:.3f}GB) をシステム設定に保存しました。"
                )
            except Exception as e:
                logger.error(
                    f"Pipeline '{self.model_name}' のサイズをシステム設定に保存中にエラー: {e}",
                    exc_info=True,
                )

        return calculated_size_mb

    def load_components(
        self, task: str, model_path: str, batch_size: int
    ) -> TransformersPipelineComponents | None:
        """Pipelineモデルをロード (早期サイズ計算・キャッシュ管理対応版)"""
        # --- 0. Check if already loaded --- #
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"Pipeline '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            # TODO: Should we return the existing components here? Need a component cache.
            # For now, return None to indicate no *new* load occurred.
            return None

        # --- 1. Get or Calculate Model Size --- #
        model_size_mb = self._get_or_calculate_model_size(task, model_path)

        # --- 2. Memory Check --- #
        if not self._check_memory_before_load(model_size_mb):
            return None  # Memory check failed

        # --- 3. Clear Cache if Needed --- #
        if model_size_mb > 0:
            logger.debug(
                f"Pipeline '{self.model_name}' ({model_size_mb:.2f} MB) ロード前にキャッシュクリア実行。"
            )
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            # This warning now only appears if size calculation failed or is genuinely 0
            logger.warning(
                f"Pipeline '{self.model_name}' サイズ不明または0のため、キャッシュクリアはベストエフォートになります。"
            )

        # --- 4. Load Actual Components --- #
        components: TransformersPipelineComponents | None = None
        try:
            # Load the pipeline on the target device
            components = self._load_pipeline(task, model_path, batch_size)

            # --- 5. Handle Load Success (State Update) --- #
            if components:
                # Pass the known size to handle_load_success
                self._handle_load_success(components, model_size_mb)
                return components
            else:
                # _load_pipeline itself might return None or raise error handled below
                logger.error(f"Pipeline '{self.model_name}' のロード内部処理で失敗しました。")
                return None

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                self._handle_generic_error(e)
            else:
                self._handle_memory_error(e)
            return None
        except Exception as e:
            self._handle_generic_error(e)
            return None

    def _load_pipeline(self, task: str, model_path: str, batch_size: int) -> TransformersPipelineComponents:
        """実際の Pipeline オブジェクトのロード処理 (ターゲットデバイス上)"""
        logger.debug(f"Pipeline '{self.model_name}' のロード試行 (ターゲットデバイス: {self.device}) ... ")
        pipeline_obj: Pipeline = pipeline(  # Add type hint for clarity
            task,
            model=model_path,
            device=self.device,  # Load directly on the target device
            batch_size=batch_size,
        )
        logger.info(f"Pipeline '{self.model_name}' のロード完了 (ターゲットデバイス: {self.device})")
        return {"pipeline": pipeline_obj}

    def _handle_load_success(
        self, components: TransformersPipelineComponents, known_size_mb: float
    ) -> None:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力 (サイズ計算は事前に行う)"""
        # Size calculation and saving is done beforehand in _get_or_calculate_model_size
        # We just need to update the state and memory usage based on the known size

        final_model_size = known_size_mb if known_size_mb > 0 else 0.0

        # Update state, memory usage, and log
        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        # _MODEL_SIZES should already be updated by _get_or_calculate_model_size
        self._MEMORY_USAGE[self.model_name] = final_model_size
        self._MODEL_LAST_USED[self.model_name] = time.time()
        # Change log level from INFO to DEBUG
        logger.debug(
            f"Pipeline '{self.model_name}' のロード後状態更新完了 (デバイス: {self.device}, サイズ: {final_model_size / 1024:.3f}GB)"
        )
        # Remove config saving logic from here
        # # 4. Save calculated size to user config if it was calculated
        # if calculated_size > 0:
        #    ...

    def _handle_memory_error(self, e: Exception) -> None:
        """メモリ関連エラー発生時の処理 (Pipeline用)"""
        error_detail = f"Pipeline '{self.model_name}' のロード中にメモリ関連エラーが発生しました (デバイス: {self.device})。詳細: {e}"
        logger.error(f"メモリ不足エラー: {error_detail}")
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            try:
                logger.error(f"CUDA メモリサマリー ({self.device}):")
                logger.error(torch.cuda.memory_summary(device=self.device, abbreviated=True))
            except Exception as mem_e:
                logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def _handle_generic_error(self, e: Exception) -> None:
        """その他の予期せぬエラー発生時の処理 (Pipeline用)"""
        logger.error(
            f"Pipeline '{self.model_name}' のロード中に予期せぬエラーが発生しました: {e}", exc_info=True
        )
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
            return param_size + buffer_size
        except Exception as e:
            logger.error(f"Transformer Pipeline モデルサイズ計算エラー: {e}")
            return 0.0


class ONNXLoader(BaseModelLoader):
    """ONNXモデルのローダー"""

    def _get_or_calculate_model_size(self, model_path: str) -> float:
        """モデルサイズをキャッシュ、Config、または計算によって取得・保存する。"""
        # 1. Check static cache
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] > 0:
            logger.debug(
                f"ONNXモデル '{self.model_name}' サイズをキャッシュから取得: {ModelLoad._MODEL_SIZES[self.model_name]:.2f} MB"
            )
            return ModelLoad._MODEL_SIZES[self.model_name]

        # 2. Check config
        config_size_mb = self.get_model_size()
        if config_size_mb is not None and config_size_mb > 0:
            ModelLoad._MODEL_SIZES[self.model_name] = config_size_mb  # Cache the value from config
            return config_size_mb

        # 3. Calculate size from file path (if not found in cache or config)
        logger.info(
            f"ONNXモデル '{self.model_name}' サイズ不明。ファイルパスから計算を試行します... (path: {model_path})"
        )
        calculated_size_mb = 0.0
        onnx_file_path: Path | None = None
        try:
            # Resolve path to get the actual .onnx file location
            _, resolved_path = self._resolve_model_path(model_path)
            if resolved_path and resolved_path.is_file():
                onnx_file_path = resolved_path
                logger.debug(
                    f"ONNXモデル '{self.model_name}' ファイル ({onnx_file_path}) からサイズ計算試行。"
                )
                # Use the base class method for file size calculation
                calculated_size_mb = self._calculate_model_size(
                    onnx_file_path, 1.5
                )  # Multiplier specific to ONNX?
                if calculated_size_mb > 0:
                    logger.info(
                        f"ONNXモデル '{self.model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB"
                    )
                else:
                    logger.warning(f"ONNXモデル '{self.model_name}' ファイルからのサイズ計算失敗。")
            else:
                logger.warning(
                    f"ONNXモデル '{self.model_name}' の有効な .onnx ファイルパスが見つかりません ({resolved_path})。サイズ計算スキップ。"
                )

        except Exception as e:
            logger.error(
                f"ONNXモデル '{self.model_name}' のサイズ計算 (パス解決含む) 中にエラー: {e}", exc_info=True
            )
            calculated_size_mb = 0.0  # Treat as unknown size on error
        # No finally block needed as we don't load the model here

        # 4. Cache and save the calculated size (even if 0.0)
        ModelLoad._MODEL_SIZES[self.model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            try:
                size_gb = calculated_size_mb / 1024
                config_registry.set_system_value(self.model_name, "estimated_size_gb", round(size_gb, 3))
                config_registry.save_system_config()
                logger.debug(
                    f"ONNXモデル '{self.model_name}' の計算サイズ ({size_gb:.3f}GB) をシステム設定に保存しました。"
                )
            except Exception as e:
                logger.error(
                    f"ONNXモデル '{self.model_name}' のサイズをシステム設定に保存中にエラー: {e}",
                    exc_info=True,
                )

        return calculated_size_mb

    def _resolve_model_path(self, model_path: str) -> tuple[Path | None, Path | None]:
        """モデルパスを解決し、関連ファイルパスを取得"""
        logger.debug(
            f"ONNXモデル '{self.model_name}' のダウンロード/検索開始 (パス/リポジトリ: {model_path})... "
        )
        try:
            # Ensure download function handles errors and returns appropriate values
            csv_path, model_repo_or_path_obj = utils.download_onnx_tagger_model(model_path)
            if model_repo_or_path_obj is None:
                logger.error(
                    f"ONNX モデル '{self.model_name}' のパス/リポジトリが見つかりません: {model_path}"
                )
                return None, None
            logger.debug(f"ONNXモデルパス解決: {model_repo_or_path_obj}")
            return csv_path, model_repo_or_path_obj
        except Exception as e:
            logger.error(f"モデルパス解決中にエラー ({model_path}): {e}", exc_info=True)
            return None, None

    def _prepare_onnx_session(self) -> list[str]:
        """ONNXセッション作成の準備 (キャッシュクリアとプロバイダー決定)"""
        logger.debug("ONNXロード前のキャッシュクリア試行...")
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("キャッシュクリア完了。")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        logger.debug(f"ONNX実行プロバイダー: {providers}")
        return providers

    def _create_onnx_session(self, model_repo_or_path: Path, providers: list[str]) -> ort.InferenceSession:
        """ONNX InferenceSession を作成"""
        logger.info(f"ONNXモデル '{self.model_name}' をロード中: '{model_repo_or_path}' on {providers}... ")
        # Ensure providers list matches expected type for InferenceSession
        session = ort.InferenceSession(str(model_repo_or_path), providers=providers)
        logger.info(f"ONNXモデル '{self.model_name}' のロード成功。")
        return session

    def _handle_load_success(
        self, session: ort.InferenceSession, csv_path: Path, model_repo_or_path: Path, known_size_mb: float
    ) -> ONNXComponents:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力 (サイズ計算は事前に行う)"""
        components: ONNXComponents = {"session": session, "csv_path": csv_path}
        # Size calculation and saving is done beforehand
        final_model_size = known_size_mb if known_size_mb > 0 else 0.0

        # Update state, memory usage, and log
        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        # _MODEL_SIZES should already be updated by _get_or_calculate_model_size
        self._MEMORY_USAGE[self.model_name] = final_model_size
        self._MODEL_LAST_USED[self.model_name] = time.time()
        # Change log level to DEBUG
        logger.debug(
            f"ONNXモデル '{self.model_name}' のロード後状態更新完了 (デバイス: {self.device}, サイズ: {final_model_size / 1024:.3f}GB)"
        )
        # Remove config saving logic
        # # 4. Save calculated size ...
        # if calculated_size > 0:
        #    ...

        return components

    def _handle_load_error(self, e: Exception) -> None:
        """ロード失敗時のエラーハンドリングと状態クリーンアップ"""
        is_memory_error = False
        if isinstance(e, MemoryError | OSError):
            # Further check OSError for memory allocation patterns
            if "allocate" in str(e).lower() or "memory" in str(e).lower():
                is_memory_error = True
        elif isinstance(e, Exception) and (
            "Failed to allocate memory" in str(e)
            or "CUDA error" in str(e)  # General CUDA errors
            or "AllocateRawInternal" in str(e)
            or "onnxruntime::BFCArena::AllocateRawInternal" in str(e)
        ):
            is_memory_error = True

        if is_memory_error:
            error_detail = (
                f"モデル '{self.model_name}' (ONNX) のロード中にメモリ関連エラーが発生しました。詳細: {e}"
            )
            logger.error(f"メモリ不足エラー: {error_detail}")
        elif "onnxruntime" in str(type(e)).lower():
            logger.error(
                f"ONNXモデル '{self.model_name}' のロード中にランタイムエラーが発生しました: {e}",
                exc_info=True,
            )
        elif isinstance(e, FileNotFoundError):
            logger.error(
                f"ONNXモデル '{self.model_name}' のロードに必要なファイルが見つかりません: {e}",
                exc_info=False,  # No need for stack trace if file not found
            )
        else:
            logger.error(
                f"ONNXモデル '{self.model_name}' のロード中に予期せぬ汎用エラーが発生しました: {e}",
                exc_info=True,
            )

        # Clean up state regardless of error type
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def load_components(self, model_path: str) -> ONNXComponents | None:
        """ONNXモデルをロード (早期サイズ計算・キャッシュ管理対応版)"""
        # --- 0. Check if already loaded --- #
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        # --- 1. Get or Calculate Model Size --- #
        model_size_mb = self._get_or_calculate_model_size(model_path)

        # --- 2. Memory Check --- #
        if not self._check_memory_before_load(model_size_mb):
            # Clean up potentially saved 0.0 size
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None  # Memory check failed

        # --- 3. Clear Cache if Needed --- #
        if model_size_mb > 0:
            logger.debug(
                f"ONNXモデル '{self.model_name}' ({model_size_mb:.2f} MB) ロード前にキャッシュクリア実行。"
            )
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"ONNXモデル '{self.model_name}' サイズ不明または0のため、キャッシュクリアはベストエフォートになります。"
            )

        # --- 4. Load Actual Components --- #
        csv_path: Path | None = None
        model_repo_or_path: Path | None = None
        session: ort.InferenceSession | None = None
        components: ONNXComponents | None = None

        try:
            # a. Resolve path (again, necessary for actual loading)
            csv_path, model_repo_or_path = self._resolve_model_path(model_path)
            if model_repo_or_path is None or csv_path is None:
                raise FileNotFoundError(f"モデルパス再解決失敗: {model_path}")

            # b. Prepare session (Cache clear is done above, just get providers)
            providers = self._prepare_onnx_session()  # Note: this also does gc.collect/empty_cache

            # c. Create session
            session = self._create_onnx_session(model_repo_or_path, providers)

            # d. Handle success (State update)
            components = self._handle_load_success(session, csv_path, model_repo_or_path, model_size_mb)
            return components

        except (MemoryError, OSError, FileNotFoundError, Exception) as e:
            self._handle_load_error(e)
            # Clean up potentially saved 0.0 size on load error
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None


class TensorFlowLoader(BaseModelLoader):
    """TensorFlowモデルのローダー"""

    def _get_or_calculate_model_size(self, model_path: str, model_format: str) -> float:
        """モデルサイズをキャッシュ、Config、またはファイル/ディレクトリパスから計算して取得・保存する。"""
        # 1. Check static cache
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] > 0:
            logger.debug(
                f"TensorFlowモデル '{self.model_name}' サイズをキャッシュから取得: {ModelLoad._MODEL_SIZES[self.model_name]:.2f} MB"
            )
            return ModelLoad._MODEL_SIZES[self.model_name]

        # 2. Check config
        config_size_mb = self.get_model_size()
        if config_size_mb is not None and config_size_mb > 0:
            ModelLoad._MODEL_SIZES[self.model_name] = config_size_mb  # Cache the value from config
            return config_size_mb

        # 3. Calculate size from file/directory path
        logger.info(
            f"TensorFlowモデル '{self.model_name}' サイズ不明。ファイル/ディレクトリパスから計算を試行します... (path: {model_path}, format: {model_format})"
        )
        calculated_size_mb = 0.0
        target_path_for_calc: Path | None = None
        multiplier_for_calc = 1.0
        try:
            # Resolve model directory first
            model_dir = self._resolve_model_dir(model_path)
            if model_dir is None:
                raise FileNotFoundError(f"サイズ計算のためのモデルディレクトリ解決失敗: {model_path}")

            # Determine target path and multiplier based on format for calculation
            if model_format == "h5":
                h5_files = list(model_dir.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError(f"サイズ計算用H5ファイルが見つかりません: {model_dir}")
                target_path_for_calc = h5_files[0]
                multiplier_for_calc = 1.2
            elif model_format == "saved_model":
                target_path_for_calc = model_dir
                if (
                    not (target_path_for_calc / "saved_model.pb").exists()
                    and not (target_path_for_calc / "saved_model.pbtxt").exists()
                ):
                    raise FileNotFoundError(
                        f"サイズ計算用の有効な SavedModel ディレクトリではありません: {target_path_for_calc}"
                    )
                multiplier_for_calc = 1.3
            elif model_format == "pb":
                raise NotImplementedError("サイズ計算は .pb フォーマット単体ではサポートされていません。")
            else:
                raise ValueError(f"サイズ計算で未対応のTensorFlowフォーマット: {model_format}")

            # Calculate size using the determined path and multiplier
            if target_path_for_calc:
                logger.debug(
                    f"TensorFlowモデル '{self.model_name}' パス ({target_path_for_calc}) からサイズ計算試行 (係数: {multiplier_for_calc:.1f})。"
                )
                calculated_size_mb = self._calculate_model_size(target_path_for_calc, multiplier_for_calc)
                if calculated_size_mb > 0:
                    logger.info(
                        f"TensorFlowモデル '{self.model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB"
                    )
                else:
                    logger.warning(f"TensorFlowモデル '{self.model_name}' パスからのサイズ計算失敗。")
            else:
                logger.warning(f"TensorFlowモデル '{self.model_name}' のサイズ計算用パス特定失敗。")

        except (FileNotFoundError, NotImplementedError, ValueError) as e:
            logger.error(
                f"TensorFlowモデル '{self.model_name}' のサイズ計算中にエラー: {e}", exc_info=False
            )
            calculated_size_mb = 0.0
        except Exception as e:
            logger.error(
                f"TensorFlowモデル '{self.model_name}' のサイズ計算中に予期せぬエラー: {e}", exc_info=True
            )
            calculated_size_mb = 0.0  # Treat as unknown size on error
        # No finally block needed as we don't load the model here

        # 4. Cache and save the calculated size (even if 0.0)
        ModelLoad._MODEL_SIZES[self.model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            try:
                size_gb = calculated_size_mb / 1024
                config_registry.set_system_value(self.model_name, "estimated_size_gb", round(size_gb, 3))
                config_registry.save_system_config()
                logger.debug(
                    f"TensorFlowモデル '{self.model_name}' の計算サイズ ({size_gb:.3f}GB) をシステム設定に保存しました。"
                )
            except Exception as e:
                logger.error(
                    f"TensorFlowモデル '{self.model_name}' のサイズをシステム設定に保存中にエラー: {e}",
                    exc_info=True,
                )

        return calculated_size_mb

    def _resolve_model_dir(self, model_path: str) -> Path | None:
        """モデルディレクトリパスを解決する"""
        logger.debug(f"TensorFlowモデル '{self.model_name}' の検索/ロード開始 (パス: {model_path})... ")
        try:
            model_dir_obj = utils.load_file(model_path)
            if model_dir_obj is None:
                logger.error(f"TensorFlow モデルディレクトリが見つかりません: {model_path}")
                return None
            # Ensure it's a directory
            if not model_dir_obj.is_dir():
                logger.error(f"指定されたパスはディレクトリではありません: {model_dir_obj}")
                return None
            return model_dir_obj
        except Exception as e:
            logger.error(f"モデルディレクトリ解決中にエラー ({model_path}): {e}", exc_info=True)
            return None

    def _load_model_by_format(
        self, model_dir: Path, model_format: str
    ) -> tuple[TensorFlowComponents, Path | None, float]:
        """指定されたフォーマットに基づいてTensorFlowモデルをロード"""
        components: TensorFlowComponents = {"model_dir": model_dir, "model": None}  # type: ignore # Initialize model as None temporarily
        target_path: Path | None = None
        model_instance: tf.Module | tf.keras.Model | None = None  # Use explicit type
        multiplier = 1.0

        if model_format == "h5":
            h5_files = list(model_dir.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"H5ファイルがディレクトリ内に見つかりません: {model_dir}")
            target_path = h5_files[0]
            logger.info(f"H5モデルをロード中: {target_path}")
            model_instance = tf.keras.models.load_model(
                target_path, compile=False
            )  # Assign to typed variable
            multiplier = 1.2

        elif model_format == "saved_model":
            target_path = model_dir
            logger.info(f"SavedModelをロード中: {target_path}")
            if (
                not (target_path / "saved_model.pb").exists()
                and not (target_path / "saved_model.pbtxt").exists()
            ):
                raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {target_path}")
            # tf.saved_model.load returns a Trackable object, which inherits from tf.Module
            model_instance = tf.saved_model.load(str(target_path))  # Assign to typed variable
            multiplier = 1.3

        elif model_format == "pb":
            # Currently unsupported direct PB loading
            pb_files = list(model_dir.glob("*.pb"))
            pb_path_str = pb_files[0] if pb_files else "(見つかりません)"
            logger.error(
                f"単体の .pb ファイル ({pb_path_str}) からの直接ロードは現在サポートされていません。SavedModel を使用してください。"
            )
            raise NotImplementedError("Direct loading from .pb is not supported. Use SavedModel.")

        else:
            raise ValueError(f"未対応のTensorFlowモデルフォーマットです: {model_format}")

        if model_instance is None:
            raise ValueError("モデルインスタンスのロードに失敗しました。")

        components["model"] = model_instance  # Assign loaded model to the dictionary

        return components, target_path, multiplier

    def _handle_load_success(
        self,
        components: TensorFlowComponents,
        target_path: Path | None,
        multiplier: float,
        known_size_mb: float,
    ) -> TensorFlowComponents:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力 (サイズ計算は事前に行う)"""
        # Size calculation and saving is done beforehand
        final_model_size = known_size_mb if known_size_mb > 0 else 0.0

        # Update state, memory usage, and log
        self._MODEL_STATES[self.model_name] = f"on_{self.device}"  # TF device mgmt is less explicit here
        self._MEMORY_USAGE[self.model_name] = final_model_size
        self._MODEL_LAST_USED[self.model_name] = time.time()
        # Change log level to DEBUG
        logger.debug(
            f"TensorFlowモデル '{self.model_name}' のロード後状態更新完了 (サイズ: {final_model_size / 1024:.3f}GB)"
        )
        # Remove config saving logic
        # # 4. Save calculated size ...
        # if calculated_size > 0:
        #    ...

        return components

    def _handle_load_error(self, e: Exception) -> None:
        """ロード失敗時のエラーハンドリングと状態クリーンアップ"""
        if isinstance(e, FileNotFoundError):
            logger.error(
                f"TensorFlowモデル '{self.model_name}' のロードに必要なファイルが見つかりません: {e}"
            )
        elif isinstance(e, NotImplementedError):
            logger.error(f"TensorFlowモデル '{self.model_name}' のロードに失敗 (未対応の操作): {e}")
        elif isinstance(e, ValueError):
            logger.error(f"TensorFlowモデル '{self.model_name}' のロードに失敗 (不正な値): {e}")
        else:
            # Check for memory errors (heuristic)
            is_memory_error = (
                "OOM" in str(e) or "memory" in str(e).lower() or "resource exhausted" in str(e).lower()
            )
            if is_memory_error:
                error_detail = f"TensorFlowモデル '{self.model_name}' のロード中にメモリ不足が発生した可能性があります: {e}"
                logger.error(f"メモリ不足エラー: {error_detail}")
            else:
                logger.error(
                    f"TensorFlowモデル '{self.model_name}' のロード中に予期せぬエラーが発生しました: {e}",
                    exc_info=True,
                )

        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def load_components(self, model_path: str, model_format: str) -> TensorFlowComponents | None:
        """TensorFlowモデルをロード (早期サイズ計算・キャッシュ管理対応版)"""
        # --- 0. Check if already loaded --- #
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        # --- 1. Get or Calculate Model Size --- #
        model_size_mb = self._get_or_calculate_model_size(model_path, model_format)

        # --- 2. Memory Check --- #
        if not self._check_memory_before_load(model_size_mb):
            # Clean up potentially saved 0.0 size
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None  # Memory check failed

        # --- 3. Clear Cache if Needed --- #
        if model_size_mb > 0:
            logger.debug(
                f"TensorFlowモデル '{self.model_name}' ({model_size_mb:.2f} MB) ロード前にキャッシュクリア実行。"
            )
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"TensorFlowモデル '{self.model_name}' サイズ不明または0のため、キャッシュクリアはベストエフォートになります。"
            )

        # --- 4. Load Actual Components --- #
        model_dir: Path | None = None
        components: TensorFlowComponents | None = None
        target_path: Path | None = None
        multiplier: float = 1.0

        try:
            # a. Resolve model directory (again, for actual loading)
            model_dir = self._resolve_model_dir(model_path)
            if model_dir is None:
                raise FileNotFoundError(f"モデルディレクトリ再解決失敗: {model_path}")

            # b. Load model based on format
            # Note: device parameter isn't explicitly used by TF loading funcs here
            components, target_path, multiplier = self._load_model_by_format(model_dir, model_format)

            # c. Handle success (State update)
            # Pass the target_path and multiplier determined during load, plus known size
            components = self._handle_load_success(components, target_path, multiplier, model_size_mb)
            return components

        except (FileNotFoundError, NotImplementedError, ValueError, MemoryError, Exception) as e:
            self._handle_load_error(e)
            # Clean up potentially saved 0.0 size on load error
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None


class CLIPLoader(BaseModelLoader):
    """CLIPモデルのローダー"""

    def _get_or_calculate_model_size(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None,
        final_activation_type: str | None,
    ) -> float:
        """モデルサイズをキャッシュ、Config、または計算によって取得・保存する。"""
        # 1. Check static cache
        if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] > 0:
            logger.debug(
                f"CLIPモデル '{self.model_name}' サイズをキャッシュから取得: {ModelLoad._MODEL_SIZES[self.model_name]:.2f} MB"
            )
            return ModelLoad._MODEL_SIZES[self.model_name]

        # 2. Check config
        config_size_mb = self.get_model_size()
        if config_size_mb is not None and config_size_mb > 0:
            ModelLoad._MODEL_SIZES[self.model_name] = config_size_mb  # Cache the value from config
            return config_size_mb

        # 3. Calculate size (if not found in cache or config)
        logger.info(
            f"CLIPモデル '{self.model_name}' サイズ不明。計算を試行します... (base: {base_model}, path: {model_path})"
        )
        calculated_size_mb = 0.0
        temp_model_dict: dict[str, Any] | None = None
        classifier_model: nn.Module | None = None
        try:
            # Temporarily load the model structure on CPU to calculate size
            # This requires calling create_clip_model
            logger.debug("サイズ計算のために一時的に CLIPモデル を CPU にロードします...")
            with utils.suppress_logging(level="WARNING"):  # Suppress verbose loading logs
                temp_model_dict = create_clip_model(
                    base_model=base_model,
                    model_path=model_path,
                    device="cpu",  # Load on CPU for calculation
                    activation_type=activation_type,
                    final_activation_type=final_activation_type,
                )

            if temp_model_dict and isinstance(temp_model_dict.get("model"), torch.nn.Module):
                classifier_model = temp_model_dict["model"]
                assert classifier_model is not None  # Add assertion for type checker
                calculated_size_mb = self._calculate_transformer_size(classifier_model)
                logger.info(f"CLIPモデル '{self.model_name}' サイズ計算成功: {calculated_size_mb:.2f} MB")
            else:
                logger.warning(
                    f"一時ロードした CLIPモデル '{self.model_name}' から有効な分類器オブジェクトを取得できませんでした。"
                )
        except Exception as e:
            logger.error(f"CLIPモデル '{self.model_name}' のサイズ計算中にエラー: {e}", exc_info=True)
            calculated_size_mb = 0.0  # Treat as unknown size on error
        finally:
            # Clean up the temporary models and release memory
            if temp_model_dict:
                if "model" in temp_model_dict:
                    del temp_model_dict["model"]
                if "clip_model" in temp_model_dict:
                    del temp_model_dict["clip_model"]
                if "processor" in temp_model_dict:
                    del temp_model_dict["processor"]
                del temp_model_dict
                del classifier_model  # Redundant but safe
                gc.collect()
                if torch.cuda.is_available():  # Just in case
                    torch.cuda.empty_cache()
            logger.debug("一時 CLIPモデル のクリーンアップ完了。")

        # 4. Cache and save the calculated size (even if 0.0)
        ModelLoad._MODEL_SIZES[self.model_name] = calculated_size_mb
        if calculated_size_mb > 0:
            try:
                size_gb = calculated_size_mb / 1024
                config_registry.set_system_value(self.model_name, "estimated_size_gb", round(size_gb, 3))
                config_registry.save_system_config()
                logger.debug(
                    f"CLIPモデル '{self.model_name}' の計算サイズ ({size_gb:.3f}GB) をシステム設定に保存しました。"
                )
            except Exception as e:
                logger.error(
                    f"CLIPモデル '{self.model_name}' のサイズをシステム設定に保存中にエラー: {e}",
                    exc_info=True,
                )

        return calculated_size_mb

    def _create_clip_model_internal(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None,
        final_activation_type: str | None,
    ) -> CLIPComponents | None:
        """外部の create_clip_model を呼び出し、基本的な検証を行う"""
        logger.info(
            f"CLIPモデル '{self.model_name}' のロードを開始します (ベース: {base_model}, パス: {model_path})... "
        )
        # Call create_clip_model which returns dict[str, Any] | None
        model_dict_any = create_clip_model(
            base_model=base_model,
            model_path=model_path,
            device=self.device,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )

        if model_dict_any is None:
            # Error logged within create_clip_model
            return None

        # --- Type Safety Check & Cast ---
        # Check essential keys and their types before casting to TypedDict
        if not isinstance(model_dict_any.get("model"), torch.nn.Module):
            logger.error(
                f"CLIP Loader: create_clip_model の戻り値の 'model' が torch.nn.Module ではありません (型: {type(model_dict_any.get('model'))})。"
            )
            return None
        if not isinstance(model_dict_any.get("processor"), CLIPProcessor):
            logger.error(
                f"CLIP Loader: create_clip_model の戻り値の 'processor' が CLIPProcessor ではありません (型: {type(model_dict_any.get('processor'))})。"
            )
            return None
        if not isinstance(model_dict_any.get("clip_model"), CLIPModel):
            logger.error(
                f"CLIP Loader: create_clip_model の戻り値の 'clip_model' が CLIPModel ではありません (型: {type(model_dict_any.get('clip_model'))})。"
            )
            return None

        # If checks pass, cast the dictionary to the specific TypedDict
        model_dict: CLIPComponents = cast(CLIPComponents, model_dict_any)
        return model_dict

    def _handle_load_success(self, model_dict: CLIPComponents, known_size_mb: float) -> CLIPComponents:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力 (サイズ計算は事前に行う)"""
        # Size calculation and saving is done beforehand
        final_model_size = known_size_mb if known_size_mb > 0 else 0.0

        # Update state, memory usage, and log
        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        self._MEMORY_USAGE[self.model_name] = final_model_size
        self._MODEL_LAST_USED[self.model_name] = time.time()
        # Change log level to DEBUG
        logger.debug(
            f"CLIPモデル '{self.model_name}' のロード後状態更新完了 (デバイス: {self.device}, サイズ: {final_model_size / 1024:.3f}GB)"
        )
        # Remove config saving logic
        # # 4. Save calculated size ...
        # if calculated_size > 0:
        #    ...

        return model_dict

    def _handle_memory_error(self, e: Exception) -> None:
        """メモリ関連エラー発生時の処理 (CLIP用)"""
        error_detail = f"CLIPモデル '{self.model_name}' のロード/作成中にメモリ関連エラーが発生しました (デバイス: {self.device})。詳細: {e}"
        logger.error(f"メモリ不足エラー: {error_detail}")
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            try:
                logger.error(f"CUDA メモリサマリー ({self.device}):")
                logger.error(torch.cuda.memory_summary(device=self.device, abbreviated=True))
            except Exception as mem_e:
                logger.error(f"CUDAメモリ情報取得失敗: {mem_e}")
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def _handle_generic_error(self, e: Exception) -> None:
        """その他の予期せぬエラー発生時の処理 (CLIP用)"""
        logger.error(
            f"CLIPモデル '{self.model_name}' のロード/作成中に予期せぬエラーが発生しました: {e}",
            exc_info=True,
        )
        # Clean up state
        if self.model_name in self._MODEL_STATES:
            del self._MODEL_STATES[self.model_name]
        if self.model_name in self._MEMORY_USAGE:
            del self._MEMORY_USAGE[self.model_name]

    def load_components(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> CLIPComponents | None:
        """CLIPモデルをロード (早期サイズ計算・キャッシュ管理対応版)"""
        # --- 0. Check if already loaded --- #
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        # --- 1. Get or Calculate Model Size --- #
        model_size_mb = self._get_or_calculate_model_size(
            base_model, model_path, activation_type, final_activation_type
        )

        # --- 2. Memory Check --- #
        if not self._check_memory_before_load(model_size_mb):
            # Clean up potentially saved 0.0 size
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None  # Memory check failed

        # --- 3. Clear Cache if Needed --- #
        if model_size_mb > 0:
            logger.debug(
                f"CLIPモデル '{self.model_name}' ({model_size_mb:.2f} MB) ロード前にキャッシュクリア実行。"
            )
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"CLIPモデル '{self.model_name}' サイズ不明または0のため、キャッシュクリアはベストエフォートになります。"
            )

        # --- 4. Load Actual Components --- #
        components: CLIPComponents | None = None
        try:
            # Create CLIP model components using the external function on the target device
            model_dict = self._create_clip_model_internal(
                base_model, model_path, activation_type, final_activation_type
            )
            if model_dict is None:
                # Error already logged in helper
                self._handle_generic_error(
                    Exception(f"_create_clip_model_internal for {self.model_name} returned None")
                )
                return None  # Creation failed

            # --- 5. Handle success (state update) --- #
            components = self._handle_load_success(model_dict, model_size_mb)
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                self._handle_generic_error(e)
            else:
                self._handle_memory_error(e)
            # Clean up potentially saved 0.0 size on load error
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None
        except Exception as e:
            self._handle_generic_error(e)
            # Clean up potentially saved 0.0 size on load error
            if self.model_name in ModelLoad._MODEL_SIZES and ModelLoad._MODEL_SIZES[self.model_name] == 0.0:
                del ModelLoad._MODEL_SIZES[self.model_name]
            return None

    def _calculate_transformer_size(self, model: torch.nn.Module) -> float:
        """Transformerモデルのメモリ使用量を計算(MB単位)"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
            return param_size + buffer_size
        except Exception as e:
            logger.error(f"CLIPモデルサイズ計算エラー: {e}")
            return 0.0


class ModelLoad:
    """
    ModelLoadクラスの実装

    このモジュールは、モデルのロード、メモリ管理、およびキャッシュ制御を担当します。
    二階層のローダー構造により、各モデルタイプの特性に応じた効率的なモデル管理を実現します。
    """

    _MODEL_STATES: ClassVar[dict[str, str]] = {}
    _MEMORY_USAGE: ClassVar[dict[str, float]] = {}
    _MODEL_LAST_USED: ClassVar[dict[str, float]] = {}
    _CACHE_RATIO: ClassVar[float] = 0.5
    # _MODEL_SIZES is now the single source of truth for cached sizes
    _MODEL_SIZES: ClassVar[dict[str, float]] = {}  # Defined static cache here

    @staticmethod
    def get_model_size(model_name: str) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB単位) を取得。なければ None を返す。"""
        # This method now ONLY checks the config via the registry.
        # Calculation is handled elsewhere.
        try:
            estimated_size_gb_any = config_registry.get(model_name, "estimated_size_gb")

            if estimated_size_gb_any is None:
                # logger.warning( # No warning here, expected case
                #     f"モデル '{model_name}' の estimated_size_gb が config に見つかりません。"
                # )
                return None  # Return None if not found in config

            # Attempt conversion, handle potential error
            try:
                estimated_size_gb = float(estimated_size_gb_any)
                size_mb = estimated_size_gb * 1024
                # No need to cache here (_MODEL_SIZES is static, handled elsewhere)
                # self._MODEL_SIZES[model_name] = size_mb
                logger.debug(
                    f"モデル '{model_name}' のサイズを config から読み込みました: {size_mb / 1024:.3f}GB"
                )
                return size_mb
            except (ValueError, TypeError):
                logger.error(
                    f"モデル '{model_name}' の config 内の estimated_size_gb 値 '{estimated_size_gb_any}' を float に変換できません。"
                )
                return None  # Return None if value is invalid

        except KeyError:
            # Model might not be in the config at all
            # logger.debug(f"モデル '{model_name}' が config に存在しません。")
            return None
        except Exception as e:
            logger.error(
                f"モデル '{model_name}' の config サイズ取得中に予期せぬエラー: {e}", exc_info=True
            )
            return None

    @staticmethod
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        base_loader = BaseModelLoader("", "cpu")
        return base_loader.get_max_cache_size()

    @staticmethod
    def _clear_cache_if_needed(model_name: str, model_size: float) -> None:
        """必要に応じて古いモデルをキャッシュから削除"""
        base_loader = BaseModelLoader(model_name, "cpu")
        base_loader._clear_cache_if_needed(model_size)

    @staticmethod
    def _prepare_cache_for_cpu(model_name: str) -> float | None:
        """CPUキャッシュの準備: サイズ取得と必要に応じたキャッシュクリア"""
        model_size = ModelLoad.get_model_size(model_name)
        if model_size is None or model_size <= 0:
            logger.warning(
                f"モデル '{model_name}' のサイズが不明または無効 ({model_size}) なため、"
                f"キャッシュサイズチェックをスキップしてキャッシュ試行します。"
            )
            # Return None to indicate unknown/invalid size
            return None
        # Now we know model_size is a valid float > 0
        else:
            # Proceed only if model_size is a valid float > 0
            model_size_gb = model_size / 1024
            logger.info(f"モデル '{model_name}' の推定サイズ: {model_size_gb:.3f}GB")
            # Pass the valid float to _clear_cache_if_needed
            ModelLoad._clear_cache_if_needed(model_name, model_size)
            return model_size

    @staticmethod
    def _move_components_to_cpu(components: dict[str, Any]) -> None:
        """コンポーネントをCPUに移動させる"""
        for component_name, component in components.items():
            if component_name == "pipeline":
                if hasattr(component, "model") and hasattr(component.model, "to"):
                    component.model.to("cpu")
            elif hasattr(component, "to") and not isinstance(
                component,
                str | Path | int | float | bool | None,  # Avoid simple types
            ):
                # Check if it's already on CPU to avoid unnecessary moves/warnings
                # This requires checking the device attribute if available
                current_device = getattr(component, "device", None)
                if current_device and str(current_device) == "cpu":
                    continue  # Already on CPU
                component.to("cpu")
            # Add specific handling for ONNX sessions or TF models if needed
            # (Currently, they don't have a standard `.to('cpu')` method)
            # For TF, models are often implicitly on CPU unless configured otherwise
            # For ONNX, the session provider determines the device

    @staticmethod
    def _update_state_after_cpu_cache(model_name: str, model_size: float | None) -> None:
        """CPUキャッシュ成功後の状態更新とログ出力"""
        ModelLoad._MODEL_STATES[model_name] = "on_cpu"
        known_size = model_size if model_size and model_size > 0 else 0.0
        if known_size > 0:
            ModelLoad._MEMORY_USAGE[model_name] = known_size
        else:
            # If size became known during load but wasn't passed, re-fetch?
            # Or simply don't record usage if size is unknown.
            # Current: Don't record if size is 0 or None.
            if model_name in ModelLoad._MEMORY_USAGE:
                # Remove potentially stale entry if size is now unknown
                del ModelLoad._MEMORY_USAGE[model_name]

        ModelLoad._MODEL_LAST_USED[model_name] = time.time()

        # Remove redundant max_cache calculation for logging
        # max_cache = ModelLoad.get_max_cache_size()
        current_usage = sum(ModelLoad._MEMORY_USAGE.values())
        size_log = f"{known_size / 1024:.3f}GB" if known_size > 0 else "不明"
        logger.info(
            f"モデル '{model_name}' をキャッシュしました "
            f"(サイズ: {size_log}, "
            f"現在のキャッシュ使用量: {current_usage / 1024:.3f}GB)"  # Removed max_cache display
            # f"現在のキャッシュ使用量: {current_usage / 1024:.3f}GB/{max_cache / 1024:.3f}GB)"
        )

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """メモリ管理を行いながらモデルをキャッシュ (複雑度削減版)"""
        if model_name in ModelLoad._MODEL_STATES and ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            logger.debug(f"モデル '{model_name}' は既にCPUにあります。")
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()
            return components  # Return unmodified components

        # Prepare cache (get size, clear if needed)
        model_size = ModelLoad._prepare_cache_for_cpu(model_name)
        # model_size can be float > 0 or 0.0 if unknown

        try:
            # Move components to CPU
            ModelLoad._move_components_to_cpu(components)

            # Update state and log
            ModelLoad._update_state_after_cpu_cache(model_name, model_size)

            return components

        except Exception as e:
            logger.error(f"モデル '{model_name}' のCPUへのキャッシュに失敗しました: {e!s}", exc_info=True)
            # Clean up potential inconsistent state if move failed partially
            if model_name in ModelLoad._MODEL_STATES and ModelLoad._MODEL_STATES[model_name] == "on_cpu":
                del ModelLoad._MODEL_STATES[model_name]
            if model_name in ModelLoad._MEMORY_USAGE:
                del ModelLoad._MEMORY_USAGE[model_name]
            # Return the original components dict as caching failed
            return components

    @staticmethod
    def _check_cuda_restore_preconditions(model_name: str, device: str) -> bool:
        """CUDA復元の事前条件を確認する"""
        if model_name not in ModelLoad._MODEL_STATES:
            logger.warning(f"モデル '{model_name}' の状態が不明です。CUDAへの復元をスキップします。")
            return False  # Cannot restore if state unknown

        if ModelLoad._MODEL_STATES[model_name] == f"on_{device}":
            logger.debug(f"モデル '{model_name}' は既に {device} にあります。")
            ModelLoad._MODEL_LAST_USED[model_name] = time.time()  # Update last used time
            return False  # Already on target device

        # If state is known and not on the target device, proceed
        return True

    @staticmethod
    def _move_components_to_cuda(device: str, components: dict[str, Any]) -> None:
        """コンポーネントをCUDAデバイスに移動させる"""
        logger.info(f"コンポーネントを {device} に移動中...")
        for component_name, component in components.items():
            if component_name == "pipeline":
                if hasattr(component, "model") and hasattr(component.model, "to"):
                    component.model.to(device)
            elif hasattr(component, "to") and not isinstance(
                component, str | Path | int | float | bool | None
            ):  # Avoid trying .to on simple types
                # Check device before moving if possible
                current_device = getattr(component, "device", None)
                if current_device and str(current_device) == device:
                    continue  # Already on target device
                component.to(device)
            # Add specific handling for ONNX/TF if needed for CUDA placement

    @staticmethod
    def _handle_cuda_restore_success(model_name: str, device: str) -> None:
        """CUDA復元成功時の状態更新とログ出力"""
        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        ModelLoad._MODEL_LAST_USED[model_name] = time.time()
        logger.info(f"モデル '{model_name}' を {device} に復元完了。")

    @staticmethod
    def _handle_cuda_restore_memory_error(model_name: str, device: str, e: Exception) -> None:
        """CUDA復元中のメモリ関連エラー処理"""
        error_detail = f"モデル '{model_name}' の CUDA デバイス '{device}' への復元中にメモリ関連エラーが発生しました。詳細: {e}"
        logger.error(f"メモリ不足エラー: {error_detail}")
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            and device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            try:
                logger.error(f"CUDA メモリサマリー ({device}):")
                logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
            except Exception as mem_e:
                logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
        # Revert state on failure - assume it ends up back on CPU or in an unusable state
        # Reverting to 'on_cpu' might be optimistic if the move failed badly.
        # Consider introducing an 'error' state or removing the state entry.
        # For now, revert to 'on_cpu' as per original logic.
        if model_name in ModelLoad._MODEL_STATES:  # Only update if state was previously known
            ModelLoad._MODEL_STATES[model_name] = "on_cpu"
            logger.warning(f"CUDA復元失敗のため、モデル '{model_name}' の状態を 'on_cpu' に戻しました。")

    @staticmethod
    def _handle_cuda_restore_generic_error(model_name: str, device: str, e: Exception) -> None:
        """CUDA復元中のその他のエラー処理"""
        logger.error(
            f"モデル '{model_name}' の {device} への復元中に予期せぬエラーが発生しました: {e}",
            exc_info=True,
        )
        # Revert state on failure, similar to memory error
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad._MODEL_STATES[model_name] = "on_cpu"
            logger.warning(
                f"CUDA復元中の予期せぬエラーのため、モデル '{model_name}' の状態を 'on_cpu' に戻しました。"
            )

    @staticmethod
    def restore_model_to_cuda(
        model_name: str, device: str, components: dict[str, Any]
    ) -> dict[str, Any] | None:
        """モデルをCUDAデバイスに復元 (複雑度削減版)"""
        if not ModelLoad._check_cuda_restore_preconditions(model_name, device):
            # If already on device, return original components
            # If state is unknown, return None (or components? Needs clarification)
            # Let's return `components` if already on device, `None` if state unknown, aligning with checks.
            if (
                model_name in ModelLoad._MODEL_STATES
                and ModelLoad._MODEL_STATES[model_name] == f"on_{device}"
            ):
                return components  # Already there
            else:
                return None  # State unknown or other precondition failed

        logger.info(f"モデル '{model_name}' を {device} に復元中...")
        try:
            # Move components
            ModelLoad._move_components_to_cuda(device, components)

            # Handle success
            ModelLoad._handle_cuda_restore_success(model_name, device)
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            # Check if OSError is memory related
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                ModelLoad._handle_cuda_restore_generic_error(model_name, device, e)
            else:
                ModelLoad._handle_cuda_restore_memory_error(model_name, device, e)
            # Return None as restoration failed
            return None
        except Exception as e:
            ModelLoad._handle_cuda_restore_generic_error(model_name, device, e)
            # Return None as restoration failed
            return None

    @staticmethod
    def release_model(model_name: str) -> None:
        """モデルの状態とメモリ使用量の記録を削除"""
        base_loader = BaseModelLoader(model_name, "cpu")
        base_loader.release_model(model_name)

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルのコンポーネントを解放"""
        logger.info(f"モデル '{model_name}' のコンポーネント解放試行...")
        try:
            for component_name, component in list(components.items()):  # Iterate over copy
                if (
                    component_name == "model"
                    or component_name == "pipeline"
                    or component_name == "session"
                    or component_name == "clip_model"
                ):
                    logger.debug(f"Deleting component: {component_name}")
                    del components[component_name]
                    del component  # Try deleting the reference too
                elif hasattr(component, "cpu"):
                    logger.debug(f"Moving component {component_name} to CPU (if applicable)")
                    component.cpu()
                    if hasattr(component, "to"):
                        component.to("cpu")
            # More aggressive memory release
            logger.debug("Running garbage collection and emptying CUDA cache...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Garbage collection and cache empty finished.")

            # Clear state and memory usage from manager
            ModelLoad.release_model(model_name)
            logger.info(f"モデル '{model_name}' のコンポーネント解放完了。")
            return components  # Return modified components dict (or empty?)

        except Exception as e:
            logger.error(
                f"モデル '{model_name}' のコンポーネント解放中にエラーが発生しました: {e!s}", exc_info=True
            )
            # Attempt to clear state even if release failed partially
            ModelLoad.release_model(model_name)
            return components

    # --- Static Load Methods --- #
    # These methods now correctly return Optional[TypedDict]

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> TransformersComponents | None:
        """Transformersモデルをロード"""
        loader = TransformersLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> TransformersPipelineComponents | None:
        """TransformersPipelineモデルをロード"""
        loader = TransformersPipelineLoader(model_name, device)
        return loader.load_components(task, model_path, batch_size)

    @staticmethod
    def load_onnx_components(model_name: str, model_path: str, device: str) -> ONNXComponents | None:
        """ONNXモデルをロード"""
        loader = ONNXLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,  # Device might not be directly used by TF loader but kept for consistency
        model_format: str,
    ) -> TensorFlowComponents | None:
        """TensorFlowモデルをロード"""
        loader = TensorFlowLoader(model_name, device)
        return loader.load_components(model_path, model_format)

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
        loader = CLIPLoader(model_name, device)
        return loader.load_components(base_model, model_path, activation_type, final_activation_type)


class Classifier(nn.Module):
    """画像特徴量を入力として、分類スコアを出力する柔軟な分類器。

    Args:
        input_size (int): 入力特徴量の次元数
        hidden_sizes (list[int], optional): 各隠れ層のユニット数のリスト
        output_size (int, optional): 出力層のユニット数 (通常は1)
        dropout_rates (list[float], optional): 各隠れ層のドロップアウト率
        use_activation (bool, optional): 活性化関数を使用するかどうか
        activation (Type[nn.Module], optional): 使用する活性化関数
        use_final_activation (bool, optional): 最終層に活性化関数を使用するかどうか
        final_activation (Type[nn.Module], optional): 最終層に使用する活性化関数
    """

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

        # Default values
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [1024, 128, 64, 16]
        dropout_rates = dropout_rates if dropout_rates is not None else [0.2, 0.2, 0.1, 0.0]

        # Adjust dropout list length
        if len(dropout_rates) < len(hidden_sizes):
            dropout_rates.extend([0.0] * (len(hidden_sizes) - len(dropout_rates)))

        # Build layers
        layers: list[nn.Module] = []
        prev_size = input_size

        for _, (size, drop) in enumerate(zip(hidden_sizes, dropout_rates, strict=False)):
            layers.append(nn.Linear(prev_size, size))

            if use_activation:
                layers.append(activation())

            if drop > 0:
                layers.append(nn.Dropout(drop))

            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Final activation
        if use_final_activation:
            layers.append(final_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ネットワークの順伝播を実行します。"""
        return self.layers(x)


def create_clip_model(  # noqa: C901
    base_model: str,
    model_path: str,
    device: str,
    activation_type: str | None = None,
    final_activation_type: str | None = None,
) -> dict[str, Any] | None:  # Allow returning None on failure
    """どの CLIP モデルでも使用可能なモデルを作成します。"""
    try:
        # Common CLIP model and processor initialization
        logger.debug(f"CLIPプロセッサロード中: {base_model}")
        clip_processor = CLIPProcessor.from_pretrained(base_model)
        logger.debug(f"CLIPモデルロード中: {base_model} on {device}")
        # Ignore potential incorrect type hint from transformers for this call
        clip_model = CLIPModel.from_pretrained(base_model).to(device).eval()  # type: ignore

        # Auto-detect input size
        input_size = clip_model.config.projection_dim
        logger.debug(f"CLIPモデル {base_model} の特徴量次元: {input_size}")

        # Load model weights
        logger.debug(f"モデル重みロード中: {model_path}")
        local_path = utils.load_file(model_path)
        if local_path is None:
            logger.error(f"モデルパス '{model_path}' の解決に失敗しました。")
            return None
        # Specify map_location to avoid issues if model was saved on GPU but loading on CPU
        state_dict = torch.load(local_path, map_location=device)
        logger.debug("重みロード完了、構造推測開始...")

        # Infer hidden_features structure from state_dict
        hidden_features = []
        current_layer = 0
        while True:
            # Try different naming conventions
            weight_key = f"layers.{current_layer}.weight"
            bias_key = f"layers.{current_layer}.bias"
            if weight_key not in state_dict or bias_key not in state_dict:
                # Try alternative naming (e.g., if layers list includes non-Linear)
                found_next = False
                for lookahead in range(1, 5):  # Look ahead a few steps
                    next_weight_key = f"layers.{current_layer + lookahead}.weight"
                    if next_weight_key in state_dict:
                        current_layer += lookahead
                        weight_key = next_weight_key
                        bias_key = f"layers.{current_layer}.bias"
                        found_next = True
                        break
                if not found_next:
                    break  # Stop if no linear layer found ahead

            if weight_key in state_dict:
                weight = state_dict[weight_key]
                hidden_features.append(weight.shape[0])
                current_layer += 1  # Move to next potential layer index
            else:
                break  # Should not happen if found_next logic is correct

        # Exclude the final output layer if it was captured
        # The last appended size corresponds to the output size of the *last hidden layer*
        # We need the sizes *leading up to* the final Linear layer
        # The final Linear layer's input size is the last element of hidden_features
        if len(hidden_features) > 0:
            # The features list contains the output dimensions of each linear layer
            # The Classifier expects sizes of hidden layers, not the final output layer
            hidden_sizes_for_classifier = hidden_features[
                :-1
            ]  # Exclude the size of the final layer's output
        else:
            hidden_sizes_for_classifier = []

        if not hidden_sizes_for_classifier:
            logger.warning(f"CLIP分類器 {base_model} の構造推測失敗。デフォルト値を使用。")
            hidden_sizes_for_classifier = [1024, 128, 64, 16]

        logger.info(f"推測された隠れ層サイズ: {hidden_sizes_for_classifier}")

        # Activation function mapping
        activation_map = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
        }

        # Get activation parameters from config
        use_activation = activation_type is not None
        activation_func = (
            activation_map.get(activation_type, nn.ReLU) if activation_type is not None else nn.ReLU
        )

        use_final_activation = final_activation_type is not None
        final_activation_func = (
            activation_map.get(final_activation_type, nn.Sigmoid)
            if final_activation_type is not None
            else nn.Sigmoid
        )

        # Initialize the Classifier model
        logger.info("分類器モデル初期化...")
        model = Classifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes_for_classifier,
            output_size=1,
            use_activation=use_activation,
            activation=activation_func,
            use_final_activation=use_final_activation,
            final_activation=final_activation_func,
        )
        logger.debug("分類器初期化完了、重みロード...")
        model.load_state_dict(state_dict, strict=False)  # Use strict=False for flexibility
        logger.debug("重みロード完了、デバイス転送...")
        model = model.to(device).eval()  # Set to eval mode
        logger.debug(f"CLIP分類器モデル '{model_path}' のロード完了 (デバイス: {device})")

        # Return type matches CLIPComponents structure, but keep Any annotation for the function itself
        return {"model": model, "processor": clip_processor, "clip_model": clip_model}

    except FileNotFoundError as e:
        logger.error(f"CLIPモデル作成エラー: ファイルが見つかりません: {e}")
        return None
    except KeyError as e:
        logger.error(
            f"CLIPモデル作成エラー: state_dict に予期しないキーまたは不足しているキーがあります: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"CLIPモデル '{base_model}' / '{model_path}' の作成中に予期せぬエラーが発生しました: {e}",
            exc_info=True,
        )
        return None
