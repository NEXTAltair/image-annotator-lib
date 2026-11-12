"""YOLOv8 検出器の前処理・出力デコード・NMS・座標逆変換 (numpy のみ)。

onnxruntime に依存しない純粋関数だけを置き、fake セッションで決定論的に検証できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

from ..exceptions.errors import InvalidOutputError
from .types import BoundingBox, Detection

FloatArray = npt.NDArray[np.float32]

LETTERBOX_FILL = (114, 114, 114)
"""letterbox の余白色。ultralytics の学習・検証時の既定値に合わせる。"""


@dataclass(frozen=True)
class LetterboxTransform:
    """letterbox で行った変換。モデル座標から元画像座標へ戻すのに使う。

    Attributes:
        scale_x: 元画像 x からモデル入力 x への倍率 (リサイズ後幅 / 元幅)。
        scale_y: 元画像 y からモデル入力 y への倍率 (リサイズ後高さ / 元高さ)。
        pad_x: モデル入力の左余白 (ピクセル)。
        pad_y: モデル入力の上余白 (ピクセル)。
        image_width: 元画像の幅。
        image_height: 元画像の高さ。
    """

    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    image_width: int
    image_height: int


def letterbox(image: Image.Image, input_size: int) -> tuple[FloatArray, LetterboxTransform]:
    """画像を縦横比を保って正方形に収め、float32 NCHW (0-1) テンソルに変換する。

    Args:
        image: 入力画像。RGB 以外のモードは RGB へ変換する。EXIF の向きは適用しない。
        input_size: モデル入力の一辺。

    Returns:
        形状 ``(1, 3, input_size, input_size)`` のテンソルと、座標逆変換用の情報。
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    ratio = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, round(width * ratio)))
    resized_height = max(1, min(input_size, round(height * ratio)))
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = Image.new("RGB", (input_size, input_size), LETTERBOX_FILL)
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
    transform = LetterboxTransform(
        scale_x=resized_width / width,
        scale_y=resized_height / height,
        pad_x=pad_x,
        pad_y=pad_y,
        image_width=width,
        image_height=height,
    )
    return tensor, transform


def normalize_yolo_output(raw_output: npt.NDArray[np.generic], num_classes: int) -> FloatArray:
    """YOLOv8 の生出力を ``(anchors, 4 + num_classes)`` に揃える。

    ultralytics の書き出し時期により ``[1, 4+nc, anchors]`` と ``[1, anchors, 4+nc]`` の
    両方があるため、チャネル数 ``4 + num_classes`` に一致する軸で向きを判定する。

    Args:
        raw_output: セッションの 1 枚分の出力 (バッチ次元 1 を含む 3 次元、または含まない 2 次元)。
        num_classes: 検出器のクラス数。

    Returns:
        行が anchor、列が ``cx, cy, w, h, class_0, ...`` の float32 配列。

    Raises:
        InvalidOutputError: 形状が YOLOv8 出力として解釈できない場合。
    """
    array = np.asarray(raw_output, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise InvalidOutputError(f"検出器出力のバッチ次元が 1 ではありません (shape={array.shape})")
        array = array[0]
    if array.ndim != 2:
        raise InvalidOutputError(f"検出器出力が 2/3 次元ではありません (shape={array.shape})")

    channels = 4 + num_classes
    if array.shape[1] == channels:
        return array
    if array.shape[0] == channels:
        return np.ascontiguousarray(array.T)
    raise InvalidOutputError(
        f"検出器出力のチャネル数が 4+{num_classes} と一致しません (shape={array.shape})"
    )


def box_iou(box: FloatArray, others: FloatArray) -> FloatArray:
    """1 個の矩形と複数矩形の IoU を返す (xyxy 形式)。

    Args:
        box: 形状 ``(4,)`` の矩形。
        others: 形状 ``(n, 4)`` の矩形群。

    Returns:
        形状 ``(n,)`` の IoU。和集合の面積が 0 の組は 0.0。
    """
    inter_x1 = np.maximum(box[0], others[:, 0])
    inter_y1 = np.maximum(box[1], others[:, 1])
    inter_x2 = np.minimum(box[2], others[:, 2])
    inter_y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(inter_x2 - inter_x1, 0.0, None) * np.clip(inter_y2 - inter_y1, 0.0, None)
    area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    other_areas = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(
        others[:, 3] - others[:, 1], 0.0, None
    )
    union = area + other_areas - inter
    iou = np.zeros_like(inter, dtype=np.float32)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou


def non_max_suppression(boxes: FloatArray, scores: FloatArray, iou_threshold: float) -> list[int]:
    """貪欲法の NMS。IoU が閾値を超える低 score 側の矩形を除く。

    同点 score は入力順を保つ (安定ソート) ため結果は決定論的。

    Args:
        boxes: 形状 ``(n, 4)`` の xyxy 矩形。
        scores: 形状 ``(n,)`` の score。
        iou_threshold: これを **超える** IoU の矩形を抑制する。

    Returns:
        残した矩形の index (score 降順)。
    """
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def decode_detections(
    raw_output: npt.NDArray[np.generic],
    class_names: tuple[str, ...],
    transform: LetterboxTransform,
    *,
    min_score: float,
    nms_iou: float,
    allowed_classes: frozenset[str],
) -> list[Detection]:
    """YOLOv8 生出力を元画像ピクセル座標の検出結果へ変換する。

    処理順: 向きの正規化 → クラス別最大 score → 閾値とクラスで絞り込み →
    クラス別 NMS → letterbox 逆変換 → 画像範囲へ clip → 面積 0 の矩形を除外 → score 降順。

    Args:
        raw_output: セッションの 1 枚分の出力。
        class_names: クラス index 順のクラス名。
        transform: 前処理で得た letterbox 変換情報。
        min_score: これ以上の score を残す。
        nms_iou: NMS の IoU 閾値。
        allowed_classes: 返すクラス名の集合。

    Returns:
        score 降順の検出結果。該当なしなら空リスト。

    Raises:
        InvalidOutputError: 出力形状が不正な場合。
    """
    predictions = normalize_yolo_output(raw_output, len(class_names))
    if predictions.shape[0] == 0:
        return []

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    allowed_ids = np.array(
        [index for index, name in enumerate(class_names) if name in allowed_classes], dtype=np.int64
    )
    mask = (scores >= min_score) & np.isin(class_ids, allowed_ids)
    if not np.any(mask):
        return []

    centers = predictions[mask, :4]
    scores = scores[mask]
    class_ids = class_ids[mask]
    boxes = np.stack(
        [
            centers[:, 0] - centers[:, 2] / 2,
            centers[:, 1] - centers[:, 3] / 2,
            centers[:, 0] + centers[:, 2] / 2,
            centers[:, 1] + centers[:, 3] / 2,
        ],
        axis=1,
    ).astype(np.float32)

    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        kept.extend(int(indices[i]) for i in non_max_suppression(boxes[indices], scores[indices], nms_iou))

    detections: list[Detection] = []
    for index in kept:
        bbox = _to_image_coordinates(boxes[index], transform)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        detections.append(
            Detection(
                class_name=class_names[int(class_ids[index])],
                score=float(scores[index]),
                bbox=bbox,
            )
        )
    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections


def _to_image_coordinates(box: FloatArray, transform: LetterboxTransform) -> BoundingBox:
    """モデル入力座標の矩形を元画像ピクセル座標へ戻し、画像範囲へ clip する。"""
    width = float(transform.image_width)
    height = float(transform.image_height)
    x1 = (float(box[0]) - transform.pad_x) / transform.scale_x
    y1 = (float(box[1]) - transform.pad_y) / transform.scale_y
    x2 = (float(box[2]) - transform.pad_x) / transform.scale_x
    y2 = (float(box[3]) - transform.pad_y) / transform.scale_y
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )
