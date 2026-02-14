"""ONNX モデルローダー。

onnxruntime の InferenceSession を使用した ONNX モデルのロードを提供する。

Dependencies:
    - onnxruntime: ONNX Runtime (遅延import)
    - torch: CUDA キャッシュクリア用 (遅延import, 省略可)
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, override

from .. import utils
from ..types import ONNXComponents
from ..utils import logger
from .loader_base import LoaderBase


class ONNXLoader(LoaderBase):
    """ONNX モデルのローダー。"""

    def _resolve_model_path_internal(self, model_path: str) -> tuple[Path | None, Path | None]:
        """ONNX モデルパスと関連 CSV パスを解決する。"""
        try:
            csv_path, model_repo_or_path_obj = utils.download_onnx_tagger_model(model_path)
            if model_repo_or_path_obj is None:
                logger.error(f"ONNX モデルパス/リポジトリ解決失敗: {model_path}")
                return None, None
            logger.debug(f"ONNXモデルパス解決: {model_repo_or_path_obj}")
            return csv_path, model_repo_or_path_obj
        except Exception as e:
            logger.error(f"ONNXモデルパス解決中にエラー ({model_path}): {e}", exc_info=True)
            return None, None

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """解決された ONNX ファイルサイズに基づいてサイズを計算する (乗数付き)。"""
        _, resolved_path = self._resolve_model_path_internal(model_path)
        if resolved_path and resolved_path.is_file():
            return LoaderBase._calculate_file_size_mb(resolved_path) * 1.5
        elif resolved_path and resolved_path.is_dir():
            logger.warning(f"ONNX パス {resolved_path} はディレクトリです。サイズ計算はベストエフォート。")
            onnx_files = list(resolved_path.glob("*.onnx"))
            if onnx_files:
                largest_onnx = max(onnx_files, key=lambda p: p.stat().st_size)
                return LoaderBase._calculate_file_size_mb(largest_onnx) * 1.5
            return LoaderBase._calculate_dir_size_mb(resolved_path) * 1.5
        logger.warning(f"ONNXモデル有効パス見つからず ({resolved_path})。サイズ計算スキップ。")
        return 0.0

    @override
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> ONNXComponents:
        """ONNX InferenceSession をロードし CSV パスを解決する。"""
        import onnxruntime as ort

        csv_path, resolved_model_path = self._resolve_model_path_internal(model_path)
        if resolved_model_path is None or csv_path is None:
            raise FileNotFoundError(f"ONNXモデルパス解決失敗: {model_path}")

        logger.debug("ONNXキャッシュクリア試行...")
        gc.collect()
        if self.device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                logger.debug("torch not available for CUDA cache clearing")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        logger.debug(f"ONNX実行プロバイダー: {providers}")

        logger.info(f"ONNXモデル '{self.model_name}' ロード中: '{resolved_model_path}' on {providers}... ")
        session = ort.InferenceSession(str(resolved_model_path), providers=providers)
        logger.info(f"ONNXモデル '{self.model_name}' ロード成功。")

        return {"session": session, "csv_path": csv_path}
