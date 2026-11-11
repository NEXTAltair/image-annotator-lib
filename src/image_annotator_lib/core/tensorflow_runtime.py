"""TensorFlow 推論の実行先・スレッド数・推論枚数の決定 (ADR 0013)。

TensorFlow はモジュール先頭で import しない。関数内で遅延 import する。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import psutil

from .utils import logger

DEFAULT_INFERENCE_BATCH_SIZE = 8
"""TensorFlow 系の 1 回の推論枚数の既定値 (モデル設定 ``inference_batch_size`` で上書き)。"""

DEFAULT_INFERENCE_MEMORY_MULTIPLIER = 320.0
"""CPU 推論時、1 枚の入力テンソルのバイト数に掛けて作業領域込みの必要量を見積もる倍率。

Issue #172 の事例 (299x299 入力 100 枚、約 0.107GB の入力で空き約 33GB を使い切った) から
得た観測下限 (約 308 倍) を丸めた保守値。モデル設定 ``inference_memory_multiplier`` で上書きする。
"""

DEFAULT_MEMORY_SAFETY_MARGIN_GB = 2.0
"""推論前の見積もりで空き RAM から差し引く安全余裕 (GB)。``memory_safety_margin_gb`` で上書きする。"""

_BYTES_PER_GB = 1024**3


@dataclass
class _ThreadLimitRequest:
    """``set_tensorflow_cpu_threads`` で要求されたスレッド数。None は TF 既定のまま。"""

    intra_op_threads: int | None = None
    inter_op_threads: int | None = None


_THREAD_REQUEST = _ThreadLimitRequest()


def _tensorflow_module() -> Any:
    """TensorFlow を遅延 import して返す (テストでは sys.modules に fake を差し込む)。"""
    import tensorflow as tf

    return tf


def determine_tensorflow_device(requested_device: str, model_name: str | None = None) -> str:
    """TensorFlow から見える (visible な) GPU から実際の実行先を決める。

    PyTorch の CUDA 可用性は使わない (Issue #172: torch は ``cuda`` と判定したが
    TensorFlow は GPU を検出しておらず、表示と実行先が食い違った)。

    Args:
        requested_device: 設定で要求されたデバイス (``"cuda"`` / ``"cuda:N"`` / ``"cpu"``)。
        model_name: ログ用のモデル名。

    Returns:
        ``"cpu"`` / ``"cuda"`` / ``"cuda:N"``。CUDA 要求でも TF が GPU を検出しなければ ``"cpu"``。
    """
    if not requested_device.startswith("cuda"):
        return "cpu"

    tf = _tensorflow_module()
    # 物理 GPU ではなく、実行時に TensorFlow から見える GPU で判定する。
    # set_visible_devices([], "GPU") で隠された GPU は使われず soft placement で CPU 実行になるため。
    # list_logical_devices はランタイムを初期化し、以降のスレッド数・memory growth 設定を
    # できなくするので使わない (get_visible_devices は初期化しない)。
    gpus = list(tf.config.get_visible_devices("GPU"))
    label = model_name or "N/A"
    if not gpus:
        logger.warning(
            f"モデル '{label}': TensorFlow が GPU を検出しないため、要求デバイス "
            f"'{requested_device}' ではなく CPU で推論します。"
        )
        return "cpu"

    if ":" not in requested_device:
        return "cuda"
    index_text = requested_device.split(":", 1)[1]
    if index_text.isdecimal() and int(index_text) < len(gpus):
        return requested_device
    logger.warning(
        f"モデル '{label}': TensorFlow の GPU 数は {len(gpus)} で、'{requested_device}' は無効です。"
        " cuda:0 を使います。"
    )
    return "cuda"


def tensorflow_device_name(execution_device: str) -> str:
    """``determine_tensorflow_device`` の結果を ``tf.device`` 用の名前に変換する。"""
    if not execution_device.startswith("cuda"):
        return "/CPU:0"
    index = execution_device.split(":", 1)[1] if ":" in execution_device else "0"
    return f"/GPU:{index}"


def set_tensorflow_cpu_threads(
    intra_op_threads: int | None = None, inter_op_threads: int | None = None
) -> None:
    """TensorFlow の CPU スレッド数の上限を設定する。

    TensorFlow の制約で、スレッド数は TF ランタイムの初期化前にしか変更できない。
    TF 未 import なら要求を保存し、最初の TensorFlow モデル生成時に適用する。
    import 済みなら即座に適用を試み、初期化済みで反映できない場合は WARNING を出す。

    Args:
        intra_op_threads: 1 演算内の並列スレッド数の上限。None は変更しない。
        inter_op_threads: 演算間の並列スレッド数の上限。None は変更しない。

    Raises:
        ValueError: 1 未満の値が指定された場合。
    """
    for name, value in (("intra_op_threads", intra_op_threads), ("inter_op_threads", inter_op_threads)):
        if value is not None and (isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} は 1 以上の整数で指定してください: {value!r}")
    if intra_op_threads is not None:
        _THREAD_REQUEST.intra_op_threads = intra_op_threads
    if inter_op_threads is not None:
        _THREAD_REQUEST.inter_op_threads = inter_op_threads
    if "tensorflow" in sys.modules:
        apply_tensorflow_thread_limits()


def apply_tensorflow_thread_limits() -> None:
    """保存済みのスレッド数要求を TensorFlow に適用する (ライブラリ内部用)。"""
    request = _THREAD_REQUEST
    if request.intra_op_threads is None and request.inter_op_threads is None:
        return
    threading_api = _tensorflow_module().config.threading
    pairs = (
        (
            "intra_op",
            request.intra_op_threads,
            threading_api.get_intra_op_parallelism_threads,
            threading_api.set_intra_op_parallelism_threads,
        ),
        (
            "inter_op",
            request.inter_op_threads,
            threading_api.get_inter_op_parallelism_threads,
            threading_api.set_inter_op_parallelism_threads,
        ),
    )
    for label, wanted, getter, setter in pairs:
        if wanted is None or getter() == wanted:
            continue
        try:
            setter(wanted)
            logger.info(f"TensorFlow {label} スレッド数を {wanted} に設定しました。")
        except RuntimeError as error:
            # TF ランタイム初期化後は変更できない (TensorFlow の仕様)
            logger.warning(f"TensorFlow {label} スレッド数 {wanted} を反映できません (初期化済み): {error}")


def current_tensorflow_threads() -> tuple[int | None, int | None]:
    """現在の TensorFlow の (intra_op, inter_op) スレッド数。TF 未 import なら (None, None)。"""
    if "tensorflow" not in sys.modules:
        return None, None
    threading_api = _tensorflow_module().config.threading
    intra = threading_api.get_intra_op_parallelism_threads()
    inter = threading_api.get_inter_op_parallelism_threads()
    return (intra if isinstance(intra, int) else None, inter if isinstance(inter, int) else None)


def reset_tensorflow_thread_request() -> None:
    """保存済みのスレッド数要求を消す (テスト用)。"""
    _THREAD_REQUEST.intra_op_threads = None
    _THREAD_REQUEST.inter_op_threads = None


def available_memory_bytes() -> int:
    """現在の空き RAM (bytes)。テストではこの関数を差し替える。"""
    return int(psutil.virtual_memory().available)


def plan_inference_chunk_size(
    requested: int,
    remaining: int,
    per_image_bytes: int,
    available_bytes: int,
    safety_margin_bytes: int,
) -> int:
    """空き RAM から 1 回の推論枚数を決める。

    Args:
        requested: 設定上の推論枚数上限。
        remaining: 未処理の画像数。
        per_image_bytes: 1 枚あたりの見積もり必要量 (作業領域込み)。0 以下なら見積もりなし。
        available_bytes: 現在の空き RAM。
        safety_margin_bytes: 空き RAM から差し引く安全余裕。

    Returns:
        今回推論する枚数。0 は 1 枚分の空きも無いことを表す。
    """
    upper = max(0, min(requested, remaining))
    if per_image_bytes <= 0:
        return upper
    usable = available_bytes - safety_margin_bytes
    if usable <= 0:
        return 0
    return min(upper, usable // per_image_bytes)


def bytes_to_gb(value: int) -> float:
    """bytes を GB に変換する (ログ用)。"""
    return value / _BYTES_PER_GB


def gb_to_bytes(value: float) -> int:
    """GB を bytes に変換する。"""
    return int(value * _BYTES_PER_GB)
