"""組み込み検出器のカタログ (SSoT)。

検出器はアノテーションモデルの registry / `annotator_config.toml` には登録しない。
出力形式 (矩形の列) が `annotate()` の結果契約と異なるため、本モジュールの固定表を
唯一の定義元とする。採用理由とライセンスの確認内容は ADR 0014 を参照。
"""

from __future__ import annotations

from .types import DetectorInfo

_YOLO_INPUT_SIZE = 640

# NOTE: いずれも deepghs 提供の YOLOv8 (nc=1) を ONNX 化したもの。
# ライセンスは各モデルカードの front matter に MIT と明記されている (LICENSE ファイルは無い)。
# 既定閾値はモデルカード記載の推奨 score 閾値。
BUILTIN_DETECTORS: dict[str, DetectorInfo] = {
    info.name: info
    for info in (
        DetectorInfo(
            name="person_detect_v1.3_s",
            classes=("person",),
            default_min_score=0.324,
            license="MIT",
            repo_id="deepghs/anime_person_detection",
            filename="person_detect_v1.3_s/model.onnx",
            input_size=_YOLO_INPUT_SIZE,
        ),
        DetectorInfo(
            name="face_detect_v1.4_s",
            classes=("face",),
            default_min_score=0.307,
            license="MIT",
            repo_id="deepghs/anime_face_detection",
            filename="face_detect_v1.4_s/model.onnx",
            input_size=_YOLO_INPUT_SIZE,
        ),
        DetectorInfo(
            name="head_detect_v2.0_s",
            classes=("head",),
            default_min_score=0.413,
            license="MIT",
            repo_id="deepghs/anime_head_detection",
            filename="head_detect_v2.0_s/model.onnx",
            input_size=_YOLO_INPUT_SIZE,
        ),
    )
}
