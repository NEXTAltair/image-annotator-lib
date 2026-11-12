"""物体検出 API の公開データ型。

numpy / onnxruntime に依存しない純粋な dataclass のみを置く。
`import image_annotator_lib` 時に読み込まれても重い依存を引き込まない。
"""

from __future__ import annotations

from dataclasses import dataclass

BoundingBox = tuple[float, float, float, float]
"""ピクセル座標の矩形 ``(x1, y1, x2, y2)``。左上原点、x2/y2 は排他的な右下端。"""


@dataclass(frozen=True)
class DetectorInfo:
    """利用可能な検出器のメタデータ。

    Attributes:
        name: 検出器名。`detect_objects(detector=...)` に渡す値。
        classes: 検出器が出力するクラス名 (モデルのクラス index 順)。
        default_min_score: モデル提供元の推奨 score 閾値。`min_score=None` のとき使われる。
        license: モデル重みのライセンス表記。
        repo_id: Hugging Face Hub のリポジトリ ID。
        filename: リポジトリ内の ONNX ファイルパス。
        input_size: モデル入力の一辺 (正方形、ピクセル)。
    """

    name: str
    classes: tuple[str, ...]
    default_min_score: float
    license: str
    repo_id: str
    filename: str
    input_size: int


@dataclass(frozen=True)
class Detection:
    """1 件の検出結果。

    Attributes:
        class_name: 検出クラス名。
        score: 信頼度 (0.0 から 1.0)。
        bbox: 入力画像のピクセル座標 ``(x1, y1, x2, y2)``。画像範囲内に clip 済みの float。
    """

    class_name: str
    score: float
    bbox: BoundingBox


@dataclass(frozen=True)
class ImageDetections:
    """1 枚の画像に対する検出結果。

    Attributes:
        detector: 使用した検出器名。
        image_size: 入力画像のサイズ ``(width, height)``。bbox はこの座標系。
        detections: score 降順の検出結果。検出 0 件なら空タプル。
    """

    detector: str
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]
