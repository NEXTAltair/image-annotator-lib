import gc
import time
from pathlib import Path
from typing import Any, ClassVar

import onnxruntime as ort
import psutil
import tensorflow as tf
import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.clip import CLIPModel, CLIPProcessor
from transformers.pipelines import pipeline

from . import utils
from .config import config_registry
from .utils import logger


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
        self._MODEL_SIZES: dict[str, float] = {}

    def _check_memory_before_load(self) -> bool:
        """モデルロード前に利用可能なメモリを確認する"""
        model_size_mb = self.get_model_size()
        if model_size_mb <= 0:  # サイズ不明の場合はチェックをスキップ
            logger.debug(
                f"モデル '{self.model_name}' のサイズが不明なため、事前メモリチェックをスキップします。"
            )
            return True

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

    def get_model_size(self) -> float:
        """モデルの推定メモリ使用量を取得(MB単位)"""
        if self.model_name in self._MODEL_SIZES:
            return self._MODEL_SIZES[self.model_name]

        # config_registry から estimated_size_gb を取得
        try:
            estimated_size_gb_any = config_registry.get(self.model_name, "estimated_size_gb")
            # Handle potential None before converting to float
            if estimated_size_gb_any is None:
                logger.warning(
                    f"モデル '{self.model_name}' の estimated_size_gb が config に見つかりません。"
                )
                return 0.0
            # Attempt conversion, handle potential error
            try:
                estimated_size_gb = float(estimated_size_gb_any)
            except (ValueError, TypeError):
                logger.error(
                    f"モデル '{self.model_name}' の estimated_size_gb 値 '{estimated_size_gb_any}' を float に変換できません。"
                )
                return 0.0

            size_mb = estimated_size_gb * 1024
            self._MODEL_SIZES[self.model_name] = size_mb
            logger.debug(
                f"モデル '{self.model_name}' のサイズをキャッシュから読み込みました: {size_mb / 1024:.3f}GB"
            )
            return size_mb
        except Exception as e:
            logger.error(f"モデル '{self.model_name}' のサイズ取得中に予期せぬエラー: {e}", exc_info=True)
            return 0.0

    def get_max_cache_size(self) -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算"""
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        cache_size = total_memory * self._CACHE_RATIO
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(
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

    def _load_model_and_processor(self, model_path: str) -> dict[str, Any]:
        """実際のモデルとプロセッサのロード処理"""
        logger.debug(f"モデル '{self.model_name}' のロード試行 (デバイス: {self.device}) ... ")
        processor = AutoProcessor.from_pretrained(model_path)
        # Consider adding trust_remote_code=True if needed for some models
        model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
        return {"model": model, "processor": processor}

    def _handle_load_success(self, components: dict[str, Any]) -> None:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力"""
        model_size = self._MODEL_SIZES.get(self.model_name)
        if model_size is None or model_size <= 0:
            # Calculate size if not already cached
            if "model" in components and isinstance(components["model"], torch.nn.Module):
                calculated_size = self._calculate_transformer_size(components["model"])
                if calculated_size > 0:
                    self._MODEL_SIZES[self.model_name] = calculated_size
                    model_size = calculated_size
                    logger.debug(
                        f"モデル '{self.model_name}' のサイズを計算・キャッシュしました: {model_size:.2f} MB"
                    )
                else:
                    logger.warning(f"モデル '{self.model_name}' のサイズ計算に失敗しました。")
            else:
                logger.warning(f"モデル '{self.model_name}' のコンポーネントからサイズを計算できません。")

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        # Use calculated or existing size, default to 0 if still unknown
        self._MEMORY_USAGE[self.model_name] = model_size if model_size else 0.0
        self._MODEL_LAST_USED[self.model_name] = time.time()
        logger.info(
            f"モデル '{self.model_name}' のロード成功 (デバイス: {self.device}, サイズ: {(model_size or 0.0) / 1024:.3f}GB)"
        )  # Handle None case for logging

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

    def load_components(self, model_path: str) -> dict[str, Any] | None:
        """Transformersモデルをロード (複雑度削減版)"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            # Return None or existing components? Returning None to indicate no *new* load occurred.
            # If the components are needed, they should be retrieved from a cache/manager.
            # For now, align with original behavior: return None if already loaded.
            return None  # Indicates no load operation was performed now.

        if not self._check_memory_before_load():
            return None  # Memory check failed

        # Prepare for cache clearing before attempting load
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:
            # Use ModelLoad's static method for cache clearing as it manages shared state
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"モデル '{self.model_name}' サイズ不明のため、キャッシュクリアはベストエフォートになります。"
            )
            # Optionally, attempt cache clearing anyway based on available memory?
            # ModelLoad._clear_cache_if_needed(self.model_name, 0) # Or a small placeholder?

        try:
            components = self._load_model_and_processor(model_path)
            self._handle_load_success(components)
            return components

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

    def _load_pipeline(self, task: str, model_path: str, batch_size: int) -> dict[str, Any]:
        """実際の Pipeline オブジェクトのロード処理"""
        logger.debug(f"Pipeline '{self.model_name}' のロード試行 (デバイス: {self.device}) ... ")
        pipeline_obj = pipeline(
            task,
            model=model_path,
            device=self.device,
            batch_size=batch_size,
        )
        return {"pipeline": pipeline_obj}

    def _handle_load_success(self, components: dict[str, Any]) -> None:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力"""
        model_size = self._MODEL_SIZES.get(self.model_name)
        pipeline_obj = components.get("pipeline")

        if (model_size is None or model_size <= 0) and pipeline_obj is not None:
            # Calculate size if not already cached and pipeline exists
            if hasattr(pipeline_obj, "model") and isinstance(pipeline_obj.model, torch.nn.Module):
                calculated_size = self._calculate_transformer_size(pipeline_obj.model)
                if calculated_size > 0:
                    self._MODEL_SIZES[self.model_name] = calculated_size
                    model_size = calculated_size
                    logger.debug(
                        f"Pipeline '{self.model_name}' のサイズを計算・キャッシュしました: {model_size:.2f} MB"
                    )
                else:
                    logger.warning(f"Pipeline '{self.model_name}' のサイズ計算に失敗しました。")
            else:
                logger.warning(
                    f"Pipeline '{self.model_name}' が予期したモデル構造を持っていません。サイズを推定できません。"
                )

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        self._MEMORY_USAGE[self.model_name] = (
            model_size if model_size else 0.0
        )  # Use calculated/existing or 0
        self._MODEL_LAST_USED[self.model_name] = time.time()
        logger.info(
            f"Pipeline '{self.model_name}' のロード成功 (デバイス: {self.device}, サイズ: {(model_size or 0.0) / 1024:.3f}GB)"
        )

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

    def load_components(self, task: str, model_path: str, batch_size: int) -> dict[str, Any] | None:
        """Pipelineモデルをロード (複雑度削減版)"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"Pipeline '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        if not self._check_memory_before_load():
            return None  # Memory check failed

        # Prepare for cache clearing before attempting load
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"Pipeline '{self.model_name}' サイズ不明のため、キャッシュクリアはベストエフォートになります。"
            )

        try:
            components = self._load_pipeline(task, model_path, batch_size)
            self._handle_load_success(components)
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                self._handle_generic_error(e)
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
            logger.error(f"Transformer Pipeline モデルサイズ計算エラー: {e}")
            return 0.0


class ONNXLoader(BaseModelLoader):
    """ONNXモデルのローダー"""

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
        session = ort.InferenceSession(str(model_repo_or_path), providers=providers)
        logger.info(f"ONNXモデル '{self.model_name}' のロード成功。")
        return session

    def _handle_load_success(
        self, session: ort.InferenceSession, csv_path: Path, model_repo_or_path: Path
    ) -> dict[str, Any]:
        """ロード成功時の状態更新、メモリ使用量記録、ログ出力"""
        components = {"session": session, "csv_path": csv_path}
        model_size = self._MODEL_SIZES.get(self.model_name)

        if model_size is None or model_size <= 0:
            # Calculate size using the actual model file path
            calculated_size = self._calculate_model_size(model_repo_or_path, 1.5)  # Use actual path
            if calculated_size > 0:
                self._MODEL_SIZES[self.model_name] = calculated_size
                model_size = calculated_size
                logger.debug(
                    f"ONNXモデル '{self.model_name}' のサイズを計算・キャッシュしました: {model_size:.2f} MB"
                )
            else:
                logger.warning(f"ONNXモデル '{self.model_name}' のサイズ計算に失敗しました。")

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        self._MEMORY_USAGE[self.model_name] = model_size if model_size else 0.0
        self._MODEL_LAST_USED[self.model_name] = time.time()
        logger.info(
            f"ONNXモデル '{self.model_name}' のロード成功 (デバイス: {self.device}, サイズ: {(model_size or 0.0) / 1024:.3f}GB)"
        )
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

    def load_components(self, model_path: str) -> dict[str, Any] | None:
        """ONNXモデルをロード (複雑度削減版)"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        if not self._check_memory_before_load():
            return None  # Memory check failed

        # Prepare for cache clearing before attempting load
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"ONNXモデル '{self.model_name}' サイズ不明のため、キャッシュクリアはベストエフォートになります。"
            )

        csv_path: Path | None = None
        model_repo_or_path: Path | None = None
        session: ort.InferenceSession | None = None

        try:
            # 1. Resolve path
            csv_path, model_repo_or_path = self._resolve_model_path(model_path)
            if model_repo_or_path is None or csv_path is None:
                # Error already logged in _resolve_model_path
                raise FileNotFoundError(f"モデルパス解決失敗: {model_path}")  # Raise specific error

            # 2. Prepare session (Cache clear, Providers)
            providers = self._prepare_onnx_session()

            # 3. Create session
            session = self._create_onnx_session(model_repo_or_path, providers)

            # 4. Handle success (State update, Size calc, Logging)
            components = self._handle_load_success(session, csv_path, model_repo_or_path)
            return components

        except (MemoryError, OSError, FileNotFoundError, Exception) as e:
            self._handle_load_error(e)
            return None


class TensorFlowLoader(BaseModelLoader):
    """TensorFlowモデルのローダー"""

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
    ) -> tuple[dict[str, Any], Path | None, float]:
        """指定されたフォーマットに基づいてTensorFlowモデルをロード"""
        components: dict[str, Any] = {"model_dir": model_dir}
        target_path: Path | None = None
        multiplier = 1.0

        if model_format == "h5":
            h5_files = list(model_dir.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"H5ファイルがディレクトリ内に見つかりません: {model_dir}")
            target_path = h5_files[0]
            logger.info(f"H5モデルをロード中: {target_path}")
            components["model"] = tf.keras.models.load_model(target_path, compile=False)  # type: ignore
            multiplier = 1.2

        elif model_format == "saved_model":
            target_path = model_dir
            logger.info(f"SavedModelをロード中: {target_path}")
            if (
                not (target_path / "saved_model.pb").exists()
                and not (target_path / "saved_model.pbtxt").exists()
            ):
                raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {target_path}")
            components["model"] = tf.saved_model.load(str(target_path))
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

        return components, target_path, multiplier

    def _handle_load_success(
        self, components: dict[str, Any], target_path: Path | None, multiplier: float
    ) -> dict[str, Any]:
        """ロード成功時の状態更新、サイズ計算/記録、ログ出力"""
        model_size = self._MODEL_SIZES.get(self.model_name)

        if (model_size is None or model_size <= 0) and target_path is not None:
            # Calculate size if not cached and a relevant path exists
            calculated_size = self._calculate_model_size(target_path, multiplier)
            if calculated_size > 0:
                self._MODEL_SIZES[self.model_name] = calculated_size
                model_size = calculated_size
                logger.debug(
                    f"TensorFlowモデル '{self.model_name}' のサイズを計算・キャッシュしました: {model_size:.2f} MB"
                )
            else:
                logger.warning(f"TensorFlowモデル '{self.model_name}' のサイズ計算に失敗しました。")

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"  # TF device mgmt is less explicit here
        self._MEMORY_USAGE[self.model_name] = model_size if model_size else 0.0
        self._MODEL_LAST_USED[self.model_name] = time.time()
        logger.info(
            f"TensorFlowモデル '{self.model_name}' のロード成功 (サイズ: {(model_size or 0.0) / 1024:.3f}GB)"
        )
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

    def load_components(self, model_path: str, model_format: str) -> dict[str, Any] | None:
        """TensorFlowモデルをロード (複雑度削減版)"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        if not self._check_memory_before_load():
            return None  # Memory check failed

        # Prepare for cache clearing before attempting load
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"TensorFlowモデル '{self.model_name}' サイズ不明のため、キャッシュクリアはベストエフォートになります。"
            )

        model_dir: Path | None = None
        components: dict[str, Any] | None = None
        target_path: Path | None = None  # Path used for size calculation
        multiplier: float = 1.0

        try:
            # 1. Resolve model directory
            model_dir = self._resolve_model_dir(model_path)
            if model_dir is None:
                # Error logged in helper
                raise FileNotFoundError(f"モデルディレクトリ解決失敗: {model_path}")

            # 2. Load model based on format
            components, target_path, multiplier = self._load_model_by_format(model_dir, model_format)

            # 3. Handle success
            components = self._handle_load_success(components, target_path, multiplier)
            return components

        except (FileNotFoundError, NotImplementedError, ValueError, MemoryError, Exception) as e:
            self._handle_load_error(e)
            return None


class CLIPLoader(BaseModelLoader):
    """CLIPモデルのローダー"""

    def _create_clip_model_internal(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None,
        final_activation_type: str | None,
    ) -> dict[str, Any] | None:
        """外部の create_clip_model を呼び出し、基本的な検証を行う"""
        logger.info(
            f"CLIPモデル '{self.model_name}' のロードを開始します (ベース: {base_model}, パス: {model_path})... "
        )
        model_dict = create_clip_model(
            base_model=base_model,
            model_path=model_path,
            device=self.device,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )

        if model_dict is None or not isinstance(model_dict.get("model"), torch.nn.Module):
            logger.error(
                f"CLIPモデル '{self.model_name}' の作成に失敗しました。create_clip_model が None または不正な値を返しました。"
            )
            return None
        return model_dict

    def _handle_load_success(self, model_dict: dict[str, Any]) -> dict[str, Any]:
        """ロード成功時の状態更新、サイズ計算/記録、ログ出力"""
        model_size = self._MODEL_SIZES.get(self.model_name)
        clip_nn_model = model_dict.get("model")  # Classifier model

        if (model_size is None or model_size <= 0) and isinstance(clip_nn_model, torch.nn.Module):
            # Calculate size based on the Classifier part if not cached
            calculated_size = self._calculate_transformer_size(clip_nn_model)
            if calculated_size > 0:
                self._MODEL_SIZES[self.model_name] = calculated_size
                model_size = calculated_size
                logger.debug(
                    f"CLIPモデル '{self.model_name}' のサイズを計算・キャッシュしました: {model_size:.2f} MB"
                )
            else:
                logger.warning(f"CLIPモデル '{self.model_name}' のサイズ計算に失敗しました。")

        self._MODEL_STATES[self.model_name] = f"on_{self.device}"
        self._MEMORY_USAGE[self.model_name] = model_size if model_size else 0.0
        self._MODEL_LAST_USED[self.model_name] = time.time()
        logger.info(
            f"CLIPモデル '{self.model_name}' のロード成功 (デバイス: {self.device}, サイズ: {(model_size or 0.0) / 1024:.3f}GB)"
        )
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
    ) -> dict[str, Any] | None:
        """CLIPモデルをロード (複雑度削減版)"""
        if self.model_name in self._MODEL_STATES:
            logger.debug(
                f"モデル '{self.model_name}' は既にロード済み、状態: {self._MODEL_STATES[self.model_name]}"
            )
            return None  # Indicate no load occurred

        if not self._check_memory_before_load():
            return None  # Memory check failed

        # Prepare for cache clearing before attempting load
        model_size_mb = self.get_model_size()
        if model_size_mb > 0:
            ModelLoad._clear_cache_if_needed(self.model_name, model_size_mb)
        else:
            logger.warning(
                f"CLIPモデル '{self.model_name}' サイズ不明のため、キャッシュクリアはベストエフォートになります。"
            )

        try:
            # 1. Create CLIP model components using the external function
            model_dict = self._create_clip_model_internal(
                base_model, model_path, activation_type, final_activation_type
            )
            if model_dict is None:
                # Error already logged in helper
                return None  # Creation failed

            # 2. Handle success (state, size, logs)
            components = self._handle_load_success(model_dict)
            return components

        except (torch.cuda.OutOfMemoryError, MemoryError, OSError) as e:
            if isinstance(e, OSError) and "allocate memory" not in str(e).lower():
                self._handle_generic_error(e)
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
    _MODEL_SIZES: ClassVar[dict[str, float]] = {}

    @staticmethod
    def get_model_size(model_name: str) -> float:
        """モデルの推定メモリ使用量を取得(MB単位)"""
        base_loader = BaseModelLoader(model_name, "cpu")
        return base_loader.get_model_size()

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
        if model_size <= 0:
            logger.warning(
                f"モデル '{model_name}' のサイズが不明なため、キャッシュサイズチェックをスキップしてキャッシュ試行します。"
            )
            # Return 0 or None to indicate unknown size but continue?
            # Returning 0.0 for now, but None might be clearer.
            return 0.0  # Indicate unknown size but continue
        else:
            model_size_gb = model_size / 1024
            logger.info(f"モデル '{model_name}' の推定サイズ: {model_size_gb:.3f}GB")
            ModelLoad._clear_cache_if_needed(model_name, model_size)
            return model_size  # Return the known size

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

        max_cache = ModelLoad.get_max_cache_size()
        current_usage = sum(ModelLoad._MEMORY_USAGE.values())
        size_log = f"{known_size / 1024:.3f}GB" if known_size > 0 else "不明"
        logger.info(
            f"モデル '{model_name}' をキャッシュしました "
            f"(サイズ: {size_log}, "
            f"現在のキャッシュ使用量: {current_usage / 1024:.3f}GB/{max_cache / 1024:.3f}GB)"
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
    # These methods now correctly return Optional[dict[str, Any]]

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> dict[str, Any] | None:
        """Transformersモデルをロード"""
        loader = TransformersLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> dict[str, Any] | None:
        """TransformersPipelineモデルをロード"""
        loader = TransformersPipelineLoader(model_name, device)
        return loader.load_components(task, model_path, batch_size)

    @staticmethod
    def load_onnx_components(model_name: str, model_path: str, device: str) -> dict[str, Any] | None:
        """ONNXモデルをロード"""
        loader = ONNXLoader(model_name, device)
        return loader.load_components(model_path)

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,  # Device might not be directly used by TF loader but kept for consistency
        model_format: str,
    ) -> dict[str, Any] | None:
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
    ) -> dict[str, Any] | None:
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
