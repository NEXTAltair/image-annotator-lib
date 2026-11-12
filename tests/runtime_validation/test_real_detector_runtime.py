"""実検出器モデルの DL + 推論 smoke (ADR 0014)。

`downloads_and_runs_model` marker により CI からは除外される (ADR 0001 amended)。
ローカルで `uv run pytest -m downloads_and_runs_model tests/runtime_validation` で実行する。
"""

from pathlib import Path

import pytest
from PIL import Image

from image_annotator_lib import detect_objects, list_detectors, release_detectors

_IMAGE_DIR = Path(__file__).parent.parent / "resources" / "img" / "1_img"
# file04: 人物 1 人が全身で写る縦長画像 / file08: 人物の写っていない料理画像
_CHARACTER_IMG = _IMAGE_DIR / "file04.webp"
_NO_PERSON_IMG = _IMAGE_DIR / "file08.webp"


@pytest.mark.downloads_and_runs_model
@pytest.mark.parametrize("detector", [info.name for info in list_detectors()])
def test_real_detector_runtime(detector: str) -> None:
    """各検出器が人物画像で 1 件以上、人物なし画像で 0 件を返すこと。"""
    if not _CHARACTER_IMG.exists() or not _NO_PERSON_IMG.exists():
        pytest.skip(f"resource images not found under {_IMAGE_DIR}")

    character = Image.open(_CHARACTER_IMG).convert("RGB")
    no_person = Image.open(_NO_PERSON_IMG).convert("RGB")
    try:
        character_result, no_person_result = detect_objects([character, no_person], detector=detector)
    finally:
        release_detectors([detector])

    assert character_result.image_size == character.size
    assert len(character_result.detections) >= 1
    for detection in character_result.detections:
        x1, y1, x2, y2 = detection.bbox
        assert 0.0 <= x1 < x2 <= character.width
        assert 0.0 <= y1 < y2 <= character.height
        assert 0.0 <= detection.score <= 1.0
    assert no_person_result.detections == ()
