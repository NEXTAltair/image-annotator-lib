---
type: ADR
title: "TensorFlow Inference Chunking, Memory Estimate and Runtime Report"
status: Accepted
timestamp: 2026-11-11
tags: [models, memory, tensorflow]
depends_on: [tensorflow, psutil]
---
# ADR 0013: TensorFlow Inference Chunking, Memory Estimate and Runtime Report

## Context

[Issue #172](https://github.com/NEXTAltair/image-annotator-lib/issues/172): 100 画像を DeepDanbooru
(TensorFlow CPU 推論) に一括投入して TensorFlow がメモリ不足を報告し、後続モデルへ進んだ後に PC
全体の操作が困難になった。確認できた問題は次の通り。

- TensorFlow 系は全画像を一度に前処理し、1 回のモデル呼び出しに渡していた。
- ロード前ガード (ADR 0012) はモデル重みだけを見ており、推論中の入力・中間データを見積もらない。
- 実行先の判定に PyTorch の CUDA 可用性を流用しており、TensorFlow が GPU を検出していないのに
  `cuda` と表示されていた。
- OOM は各画像の「メモリ不足エラー」文字列になり、他のエラーと機械的に区別できなかった。
- TensorFlow の CPU スレッド数に上限を設定する手段がなかった。

アプリ側の実行制御 ([LoRAIro #1468](https://github.com/NEXTAltair/LoRAIro/issues/1468)) は、
ライブラリが「実際の実行先」「OOM だったか」「小分けの枚数」を返す契約を必要とする。

## Decision

### 1. 推論の小分け

TensorFlow 系 (`TensorflowBaseAnnotator`) は、`predict` に渡された画像を設定枚数ずつ前処理・推論・整形する。
既定は 8 枚で、モデル設定 `inference_batch_size` で上書きできる。結果は入力画像の順序で連結し、
画像と結果の対応を保つ。他のバックエンドの一括方式は変えない (#164 と方向が逆のため一律にしない)。

### 2. 推論前の見積もり

各回の推論の前に、1 枚あたりの必要量を見積もる。

```text
1 枚の必要量 = H × W × C × 4 bytes (float32 入力) × 倍率
倍率 = inference_memory_multiplier (CPU 推論、既定 320) / 1 (GPU 推論: 作業領域は VRAM 側)
使える量   = 空き RAM (psutil.virtual_memory().available) − memory_safety_margin_gb (既定 2.0GB)
今回の枚数 = min(inference_batch_size, 残り枚数, 使える量 // 1 枚の必要量)
```

今回の枚数が設定より小さければ枚数を減らして続け (WARNING を 1 回)、0 なら推論せず残りの画像を
`AnnotationErrorCode.INSUFFICIENT_MEMORY` のエラー結果にする。既定倍率 320 は #172 の事例
(299×299 入力 100 枚 ≒ 0.107GB で空き約 33GB を使い切った、約 308 倍) を丸めた保守値で、
モデルごとに上書きする。入力形状が不明なモデル (`_input_shape()` が None) は見積もりを省略する。
VRAM の見積もりは行わない (TensorFlow は memory growth で確保し、GPU OOM は 4. で扱う)。

### 3. 実デバイスの判定

TensorFlow 系の実行先は `tf.config.list_physical_devices("GPU")` で決める
(`determine_tensorflow_device`)。CUDA 要求でも TensorFlow が GPU を検出しなければ `cpu`。推論は
`tf.device("/CPU:0" | "/GPU:N")` の中で実行し、`TensorFlowLoader` も PyTorch で再判定しない。
実行先は結果 (`UnifiedAnnotationResult.execution_device`)、推論開始/完了の INFO ログ、
`InferenceRuntimeInfo` に出す。

### 4. OOM の通知と解放

推論中の OOM (`tf.errors.ResourceExhaustedError` ほか `_is_memory_failure` が認識するもの) では、
既存の `LocalModelAnnotator` がそのモデルの世代を無効化して解放し (ADR 0012)、加えて
`tf.keras.backend.clear_session()` と `gc.collect()` を行う。完了済みの回の結果は保持し、
残りの画像を `AnnotationErrorCode.OUT_OF_MEMORY` のエラー結果にする。同じモデルで再試行はしない
(無制限の再試行を避ける)。他バックエンドの OOM 結果と、`annotation_runner` がモデル単位で記録する
メモリ不足にも同じ `OUT_OF_MEMORY` を付ける。

後続モデルへは進む。進んだ先で推論するかどうかは、そのモデルの推論前の見積もり (2.) と既存の
ロード前ガードで決める。全体を止めるか待つかはアプリ (#1468) が `InferenceRuntimeInfo` と
`error_code` を見て判断する。

### 5. CPU スレッド数

`set_tensorflow_cpu_threads(intra_op_threads, inter_op_threads)` で上限を設定する。TensorFlow は
ランタイム初期化後にスレッド数を変更できないため、要求を保存し、最初の TensorFlow モデル生成時
(GPU 検出の前) に適用する。初期化後で反映できない場合は WARNING を出して続ける。既定は設定なし
(TensorFlow 既定)。プロセス全体の設定のため、モデル設定ではなく API で受ける。

### 6. アプリ向けの公開 API

```python
from image_annotator_lib import (
    AnnotationErrorCode,          # OUT_OF_MEMORY / INSUFFICIENT_MEMORY を追加
    InferenceRuntimeInfo,
    get_inference_runtime_info,
    set_tensorflow_cpu_threads,
)

def get_inference_runtime_info(model_name: str) -> InferenceRuntimeInfo | None: ...
def set_tensorflow_cpu_threads(
    intra_op_threads: int | None = None, inter_op_threads: int | None = None
) -> None: ...

@dataclass(frozen=True)
class InferenceRuntimeInfo:
    model_name: str
    framework: str                 # "tensorflow"
    requested_device: str          # 設定の device
    execution_device: str          # "cpu" / "cuda" / "cuda:N"
    configured_batch_size: int     # inference_batch_size
    effective_batch_size: int | None   # 実際に使った最小の枚数 (推論なしは None)
    total_images: int
    processed_images: int          # 推論が成功した枚数
    out_of_memory: bool
    insufficient_memory: bool
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None
```

`get_inference_runtime_info` は各モデルの直近の `predict` 1 回分を返す (未実行や TensorFlow 以外は
None)。結果側の追加フィールド `UnifiedAnnotationResult.execution_device: str | None` と
`error_code` は既定値付きの追加で、既存の引数・戻り値・`annotate` のシグネチャは変えない。

設定キー (`LocalMLModelConfig`、いずれも省略可): `inference_batch_size`、
`inference_memory_multiplier`、`memory_safety_margin_gb`。

## Rationale

TensorFlow 2.10 系への固定は採用しない。2.10.1 の Windows 向け wheel は Python 3.7〜3.10 用で、
本ライブラリと LoRAIro は Python 3.12 以上を要求するため pin 変更だけでは成立せず、`tf-keras>=2.19`
など関連依存とも整合しない。Python 3.10 の別プロセスで TensorFlow を動かす構成も採用しない。
別の Python 環境・依存の二重管理、プロセス間の画像受け渡し、異常終了時の回収が増え、運用負担に
見合わない。今回の障害は「一括投入」と「見積もりの欠如」で説明でき、同一プロセス内の小分けと
見積もりで抑えられる。WSL2 / 別バックエンドへの移行は、実機確認 (#183) の結果を見て改めて判断する。

小分けを TensorFlow 系に限るのは、他のバックエンドは既に 1 枚ずつ、または別の上限で動いており、
一律に変えると #164 / #165 の方針と衝突するため。

## Consequences

- 100 枚を渡しても TensorFlow への 1 回の入力は既定 8 枚に制限される。
- 見積もりは推定であり、ピークを保証しない。倍率の妥当性は Windows 実機で確認する (#183)。
- OOM・見積もり不足は `error_code` で区別でき、完了済みの結果は失われない。
- CLEF 用 `llama-server` の孤児プロセス回収は本 ADR の範囲外とし、#182 で扱う。

## Related

- [Issue #172](https://github.com/NEXTAltair/image-annotator-lib/issues/172)
- Follow-up: [#182](https://github.com/NEXTAltair/image-annotator-lib/issues/182) (CLEF 子プロセス回収)、
  [#183](https://github.com/NEXTAltair/image-annotator-lib/issues/183) (Windows 実機確認)
- [ADR 0006](0006-annotation-outcome-contract.md)、[ADR 0012](0012-local-model-resource-ownership.md)
- [LoRAIro #1468](https://github.com/NEXTAltair/LoRAIro/issues/1468)
