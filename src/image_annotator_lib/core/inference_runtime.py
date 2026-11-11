"""ローカル推論の実行状況を呼び出し側へ公開する (ADR 0013)。

アプリ (LoRAIro #1468) が「実際の実行先」「OOM だったか」「小分けの枚数」を
モデル単位で読めるようにする。値は各モデルの直近の ``predict`` 1 回分のスナップショット。
torch / tensorflow を import しない (公開 import 経路を軽く保つため)。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

_RUNTIME_INFO: dict[str, InferenceRuntimeInfo] = {}
_RUNTIME_INFO_LOCK = Lock()


@dataclass(frozen=True)
class InferenceRuntimeInfo:
    """1 モデルの直近の推論実行の記録。

    Attributes:
        model_name: モデル名 (registry の canonical 名)。
        framework: 推論バックエンド名 (現在は ``"tensorflow"`` のみ記録する)。
        requested_device: 設定で要求されたデバイス (例: ``"cuda"``)。
        execution_device: バックエンド自身の可用性判定で決めた実際の実行先
            (``"cpu"`` / ``"cuda"`` / ``"cuda:N"``)。
        configured_batch_size: 設定上の 1 回の推論枚数上限 (``inference_batch_size``)。
        effective_batch_size: 実際に使った 1 回の推論枚数の最小値。
            推論を 1 回も実行しなかった場合は None。
        total_images: ``predict`` に渡された画像数。
        processed_images: 推論が成功した画像数。
        out_of_memory: 推論中にバックエンドがメモリ不足 (OOM) を報告した場合 True。
            このときモデルは解放済みで、未処理の画像は ``OUT_OF_MEMORY`` のエラー結果になる。
        insufficient_memory: 推論前の見積もりで 1 枚分の空きも無いと判定され、
            推論せずに残りの画像を ``INSUFFICIENT_MEMORY`` のエラー結果にした場合 True。
        intra_op_threads: 実行時点の TensorFlow intra-op スレッド数 (0 は TF 既定)。
            取得できない場合は None。
        inter_op_threads: 実行時点の TensorFlow inter-op スレッド数 (0 は TF 既定)。
            取得できない場合は None。
    """

    model_name: str
    framework: str
    requested_device: str
    execution_device: str
    configured_batch_size: int
    effective_batch_size: int | None
    total_images: int
    processed_images: int
    out_of_memory: bool
    insufficient_memory: bool
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None


def get_inference_runtime_info(model_name: str) -> InferenceRuntimeInfo | None:
    """モデルの直近の推論実行の記録を返す。

    Args:
        model_name: ``annotate`` に渡したモデル名 (canonical 名)。

    Returns:
        直近の記録。該当モデルがまだ記録対象の推論を実行していない場合は None。
    """
    with _RUNTIME_INFO_LOCK:
        return _RUNTIME_INFO.get(model_name)


def record_inference_runtime_info(info: InferenceRuntimeInfo) -> None:
    """推論実行の記録を保存する (ライブラリ内部用)。"""
    with _RUNTIME_INFO_LOCK:
        _RUNTIME_INFO[info.model_name] = info


def clear_inference_runtime_info() -> None:
    """保存済みの記録をすべて消す (テスト・再初期化用)。"""
    with _RUNTIME_INFO_LOCK:
        _RUNTIME_INFO.clear()
