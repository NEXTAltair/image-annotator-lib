---
type: ADR
title: "Object Detector API (Person / Face / Head ONNX Detectors)"
status: Accepted
timestamp: 2026-11-12
tags: [detection, onnx, models]
depends_on: [onnxruntime, huggingface-hub, numpy, pillow]
---
# ADR 0014: Object Detector API (Person / Face / Head ONNX Detectors)

## Context

呼び出し側 (LoRAIro) は学習用のクロップを自動生成するため、画像中の人物・顔・頭部の
矩形を必要とする。既存の `annotate()` は pHash をキーにタグ・キャプション・スコア・
レーティングを返す契約で、矩形の列を表現する場所がない。矩形を tags や scores に
押し込むと結果契約 (ADR 0002 / 0003 / 0006) の意味が崩れる。

候補は deepghs が公開している YOLOv8 系のアニメ検出器 (ONNX)。モデルカードの
front matter でライセンスが確認できるもののみを採用対象とする。

## Decision

### 採用モデル

| 検出器名 | クラス | repo | ONNX | 既定 score 閾値 | ライセンス |
|---|---|---|---|---|---|
| `person_detect_v1.3_s` | person | `deepghs/anime_person_detection` | `person_detect_v1.3_s/model.onnx` | 0.324 | MIT |
| `face_detect_v1.4_s` | face | `deepghs/anime_face_detection` | `face_detect_v1.4_s/model.onnx` | 0.307 | MIT |
| `head_detect_v2.0_s` | head | `deepghs/anime_head_detection` | `head_detect_v2.0_s/model.onnx` | 0.413 | MIT |

- 既定閾値はモデルカードの推奨値。各リポジトリにはより小さい `_n` 変種もあるが、
  精度を優先して `_s` を既定とする。
- ライセンス: 3 モデルともモデルカードの front matter に `license: mit` と明記されている。
  リポジトリに LICENSE ファイルは存在しないため、front matter の記載を根拠とする。
- **halfbody 検出器は採用しない。** OpenRAIL 系の表記で、利用制限条項の本文を確認
  できなかったため。条項を確認できた時点で別途検討する。
- 学習は ultralytics (AGPL-3.0) で行われている (head の ONNX メタデータにも
  `license: AGPL-3.0`, `author: Ultralytics` が埋め込まれている)。本ライブラリは
  onnxruntime で ONNX を推論するだけで ultralytics のコードには依存しない。
  重みの利用条件はモデル配布者の表記 (MIT) に従う、という判断であり、
  この点は利用者向けに `DetectorInfo.license` とこの ADR で明示する。

### 公開 API

`annotate()` とは別の型付き API `image_annotator_lib.detection` を追加する
(ADR 0011 の decisions API と同じ方針)。トップレベルからも遅延 import ラッパーで公開する。

```python
detect_objects(
    images: Sequence[PIL.Image.Image],
    detector: str,
    classes: Iterable[str] | None = None,   # None = 検出器の全クラス
    min_score: float | None = None,         # None = DetectorInfo.default_min_score
    nms_iou: float = 0.5,
) -> list[ImageDetections]
list_detectors() -> list[DetectorInfo]
release_detectors(names: Iterable[str] | None = None) -> int
```

- 戻り値は frozen dataclass。`ImageDetections(detector, image_size=(w, h), detections)`、
  `Detection(class_name, score, bbox=(x1, y1, x2, y2))`。bbox は入力画像のピクセル座標
  (float、画像範囲に clip 済み)、detections は score 降順で、0 件なら空タプル。
- `DetectorInfo(name, classes, default_min_score, license, repo_id, filename, input_size)`。
- 未知の検出器名は `ModelNotFoundError`、画像 0 枚・非 PIL 入力・未知クラス・範囲外の閾値は
  `InvalidInputError`。いずれもモデル取得前に検証する。取得・セッション生成失敗は
  `ModelLoadError`、推論失敗は `ModelExecutionError` (メモリ確保失敗は `OutOfMemoryError`)、
  出力形状の不整合は `InvalidOutputError`。

### registry / config を使わない

検出器はアノテーションモデルの registry と `annotator_config.toml` に登録しない。
registry に載ると `annotate()` / `list_annotator_info()` に検出器が混ざり、矩形を
返せない結果契約で呼ばれてしまう。検出器の定義は `detection/catalog.py` の固定表を
唯一の定義元 (SSoT) とし、config の template と runtime の同期 (drift) 対象を増やさない。

### 出力形式と NMS

ONNX は NMS を含まない生の YOLOv8 出力 (`output0`、nc=1 で 4+1 チャネル × 8400 anchor)
を返す。入力名は `images`、形状は `[batch, 3, height, width]` (動的)。出力の軸順は
モデルにより `[1, 5, 8400]` と宣言されたものと `[1, anchors, 5]` と宣言されたものが
混在するため、チャネル数 `4 + nc` に一致する軸で向きを判定する。

NMS は numpy で lib 側に実装する (クラス別、貪欲法、IoU が `nms_iou` を超える低 score 側を抑制、
同点は入力順を保つ安定ソート)。onnxruntime 以外の推論依存 (torch / torchvision /
ultralytics) を増やさず、決定論的に単体テストできるため。

### 前処理: letterbox を採用

RGB 変換 → 縦横比を保って 640×640 に収めるリサイズ (bilinear) → 余白を (114,114,114) で
埋める letterbox → float32 NCHW (0-1)。座標は軸ごとの実倍率 (リサイズ後辺 / 元の辺) と
余白で元画像へ逆変換する。

実画像 (lib の `tests/resources/img/1_img` の 9 枚) で 3 方式を比較した。

| 方式 | 結果 |
|---|---|
| 640×640 への stretch (imgutils の既定とされる方式) | 人物のいない料理画像 (file08) で person を score 0.375 で誤検出。縦長画像 (file02) の head 矩形が左へ 40px ずれた |
| letterbox 640×640 | 誤検出なし。他方式と矩形がほぼ一致 |
| 縦横比維持 + 32 の倍数に整列した可変サイズ | letterbox と同等 (誤検出なし) |

正方形画像では 3 方式とも同一の結果。letterbox は stretch で見られた誤検出と矩形のずれが
なく、入力形状が常に 640×640 で一定、逆変換も一様な倍率と余白だけで済むため採用した。
ultralytics の学習・検証時の既定前処理とも一致する。

### EXIF は呼び出し側の責務

lib は EXIF の向き補正を行わない。呼び出し側が表示向きに補正済み (EXIF 適用済み) の
PIL 画像を渡し、返る座標はその画像の座標系とする。lib 側で補正すると、呼び出し側が
保持する画像と座標系が食い違う経路が生まれるため。

### CPU 実行とセッションキャッシュ

セッションは `CPUExecutionProvider` 固定で生成する。3 モデルの ONNX は各約 45MB で、CPU でも
1 枚 0.1 秒前後で推論できる。CUDA で実行すると ADR 0012 の VRAM 台帳に載らない VRAM を
消費し、ローカルモデルの admission 判断を狂わせるため、GPU プロバイダーは使わない。
CPU/GPU 版 onnxruntime の共存問題にも影響されない。

セッションはプロセス内で検出器名ごとに 1 個生成してキャッシュし (生成はロックで直列化)、
`release_detectors()` で明示的に破棄する。モデルファイルは `huggingface_hub.hf_hub_download`
で取得し、HF キャッシュを再利用する。onnxruntime / huggingface_hub は関数内で遅延 import する。

### メモリ管理 (ADR 0012) との関係

検出器セッションは `ModelLoad` / `LoaderBase` の RAM・VRAM 台帳、所有権リース、LRU 退避に
参加しない。台帳と退避の対象を変えないため、ADR 0012 の契約 (既存ローカルモデルの
admission / release の挙動) は変わらない。検出器のホストメモリ (3 種合計で数百 MB 未満) は
台帳外であり、不要になったら呼び出し側が `release_detectors()` で解放する。

## Rationale

- registry 登録案: `annotate()` から一律に呼べる利点はあるが、結果契約が矩形を表現できず、
  `list_annotator_info()` の利用者に検出器が混入する。独立 API の方が契約が明確。
- NMS 付き ONNX への再変換案: 配布元のモデルをそのまま使えなくなり、更新追従の負担が増える。
- GPU 優先案: 速度向上は小さく、ADR 0012 の台帳外 VRAM 消費の方が害が大きい。

## Consequences

- 呼び出し側は `list_detectors()` で検出器名・クラス・既定閾値・ライセンスを取得して
  UI / CLI の既定値に使える。
- 検出器セッションの解放は呼び出し側の責務 (`release_detectors()`)。`ModelLoad` の
  解放 API では解放されない。
- 実モデルの DL と推論の検証は `downloads_and_runs_model` marker のテスト
  (`tests/runtime_validation/test_real_detector_runtime.py`) で行い、CI の必須ゲートには含めない。
- 検出器を追加するときは `detection/catalog.py` に 1 行追加し、モデルカードのライセンスを
  確認して ADR を更新する。

## Related

- [ADR 0001](0001-runtime-validation-test-lanes.md) — 実モデル検証レーン
- [ADR 0011](0011-typed-clef-decision-api.md) — `annotate()` と別の型付き API の前例
- [ADR 0012](0012-local-model-resource-ownership.md) — ローカルモデルの資源管理
- [NEXTAltair/LoRAIro#1488](https://github.com/NEXTAltair/LoRAIro/issues/1488)
- モデルカード: [anime_person_detection](https://huggingface.co/deepghs/anime_person_detection),
  [anime_face_detection](https://huggingface.co/deepghs/anime_face_detection),
  [anime_head_detection](https://huggingface.co/deepghs/anime_head_detection)
