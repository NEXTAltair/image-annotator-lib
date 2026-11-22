"""ONNX Runtime の実行先を、実際の session provider から決める (Issue #175)。

onnxruntime は CUDA Execution Provider (EP) が使えないとき、例外を出さずに
CPUExecutionProvider だけで session を作る。ロード前の判定 (``determine_onnx_device``) は
「CUDA EP が使えると分かっているか」だけを見る。ロード後の実行先は、要求した provider ではなく
``session.get_providers()`` から決める (``session_execution_device``)。

onnxruntime はモジュール先頭で import しない (関数内で遅延 import する)。
"""

from __future__ import annotations

from typing import Protocol

from . import utils
from .utils import logger

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class _SessionWithProviders(Protocol):
    """``onnxruntime.InferenceSession`` のうち本モジュールが使うメソッド。"""

    def get_providers(self) -> list[str]: ...


def available_onnx_providers() -> list[str]:
    """インストール済みの onnxruntime が提供する provider の一覧を返す。"""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def determine_onnx_device(requested_device: str, model_name: str | None = None) -> str:
    """ロード前に、ONNX モデルを置く実行先を決める。

    PyTorch の CUDA 可用性に加えて、onnxruntime が CUDA EP を提供しているかを見る。
    CPU 版 onnxruntime や CPU/GPU 版の混在で CUDA EP が無い環境では、VRAM の確保
    (CUDA admission) を行わずに ``cpu`` で扱う。``session`` は device_id を渡さないため、
    ``cuda`` 以外 (``cuda:N``) も CUDA 実行にはならず ``cpu`` で扱う。

    Args:
        requested_device: 設定で要求されたデバイス (``cuda`` / ``cuda:N`` / ``cpu``)。
        model_name: ログ用のモデル名。

    Returns:
        ``cuda`` または ``cpu`` (CUDA を要求しなかった場合は要求値のまま)。
    """
    device = utils.determine_effective_device(requested_device, model_name)
    if not device.startswith("cuda"):
        return device

    label = model_name or "N/A"
    if device != "cuda":
        logger.warning(
            f"モデル '{label}': ONNX Runtime の session は GPU の番号指定 ('{device}') を使えないため、"
            " CPU で扱います。"
        )
        return "cpu"
    if CUDA_PROVIDER not in available_onnx_providers():
        logger.warning(
            f"モデル '{label}': onnxruntime に {CUDA_PROVIDER} が無いため、要求デバイス '{device}' "
            "ではなく CPU で扱います (onnxruntime と onnxruntime-gpu の混在を確認してください)。"
        )
        return "cpu"
    return device


def select_providers(device: str) -> list[str]:
    """``InferenceSession`` に渡す provider の優先順を返す。"""
    if device == "cuda":
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


def session_execution_device(session: _SessionWithProviders) -> str:
    """作成済み session が実際に使う実行先を返す。

    Args:
        session: 作成済みの ``InferenceSession``。

    Returns:
        session の provider に CUDA EP が含まれれば ``cuda``、それ以外は ``cpu``。
    """
    return "cuda" if CUDA_PROVIDER in session.get_providers() else "cpu"
