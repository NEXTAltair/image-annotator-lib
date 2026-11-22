"""ONNX モデルローダー。

onnxruntime の InferenceSession を使用した ONNX モデルのロードを提供する。

Dependencies:
    - onnxruntime: ONNX Runtime (遅延import)
    - torch: CUDA キャッシュクリア用 (遅延import, 省略可)
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, cast, override

from .. import onnx_runtime, utils
from ..types import LoaderComponents, ONNXComponents
from ..utils import logger
from .loader_base import LoaderBase


class ONNXLoader(LoaderBase):
    """ONNX モデルのローダー。"""

    def __init__(self, model_name: str, device: str) -> None:
        """ONNX ローダーを初期化する。

        Issue #175: 実行先は PyTorch の CUDA 可用性だけでなく、onnxruntime が CUDA EP を
        提供しているかでも判定する (CUDA EP が無ければ CUDA admission を行わない)。

        Args:
            model_name: モデルインスタンスの名前。
            device: 要求されたデバイス (``cuda`` / ``cpu`` 等)。
        """
        self.model_name = model_name
        self.device = onnx_runtime.determine_onnx_device(device, model_name)

    def _resolve_model_path_internal(
        self,
        model_path: str,
        model_filename: str = utils.WD_MODEL_FILENAME,
        metadata_filename: str | None = utils.WD_LABEL_FILENAME,
        metadata_extension: str | None = ".csv",
    ) -> tuple[Path | None, Path | None]:
        """ONNX モデルパスと関連 metadata パスを解決する。"""
        try:
            metadata_path, model_repo_or_path_obj = utils.download_onnx_model_files(
                model_path,
                model_filename=model_filename,
                metadata_filename=metadata_filename,
                metadata_extension=metadata_extension,
            )
            if model_repo_or_path_obj is None:
                logger.error(f"ONNX モデルパス/リポジトリ解決失敗: {model_path}")
                return None, None
            logger.debug(f"ONNXモデルパス解決: {model_repo_or_path_obj}")
            return metadata_path, model_repo_or_path_obj
        except Exception as e:
            logger.error(f"ONNXモデルパス解決中にエラー ({model_path}): {e}", exc_info=True)
            return None, None

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """解決された ONNX ファイルサイズに基づいてサイズを計算する (乗数付き)。"""
        _, resolved_path = self._resolve_model_path_internal(model_path, **kwargs)
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
        """ONNX InferenceSession をロードし metadata パスを解決する。"""
        import onnxruntime as ort

        metadata_path, resolved_model_path = self._resolve_model_path_internal(model_path, **kwargs)
        if resolved_model_path is None or metadata_path is None:
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

        providers = onnx_runtime.select_providers(self.device)
        logger.debug(f"ONNX要求プロバイダー: {providers}")

        logger.info(f"ONNXモデル '{self.model_name}' ロード中: '{resolved_model_path}' on {providers}... ")
        session = ort.InferenceSession(str(resolved_model_path), providers=providers)
        # onnxruntime は CUDA EP を作れないと例外なしで CPU のみの session を返す。
        # ログは要求ではなく session が実際に使う provider を出す。
        logger.info(
            f"ONNXモデル '{self.model_name}' ロード成功 (実際のプロバイダー: {session.get_providers()})。"
        )

        return {"session": session, "csv_path": metadata_path, "metadata_path": metadata_path}

    @override
    def _resolve_execution_device(self, components: LoaderComponents) -> str:
        """session.get_providers() から実際の実行先を決め、CPU フォールバックを警告する。"""
        session = cast(ONNXComponents, components)["session"]
        actual = onnx_runtime.session_execution_device(session)
        if self.device.startswith("cuda") and actual != "cuda":
            logger.warning(
                f"ONNXモデル '{self.model_name}': {self.device} を要求しましたが、session は "
                f"{session.get_providers()} で作成されたため CPU で実行されます。"
            )
        return actual
