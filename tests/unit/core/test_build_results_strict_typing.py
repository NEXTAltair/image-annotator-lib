"""`_build_results` が UnifiedAnnotationResult 以外を受け取った場合に TypeError を出すことを検証する。

Issue #2 で後方互換パス (raw_output={"formatted_output": ...} へのラップ) を削除したため、
全アノテーターは `_format_predictions` で UnifiedAnnotationResult を返すことが必須となった。
旧形式 (dict / list[str] 等) を返した場合は TypeError で早期失敗する。
"""

import pytest
from PIL import Image

from image_annotator_lib.core.base.annotator import BaseAnnotator
from image_annotator_lib.core.model_config import LocalMLModelConfig
from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult


class _StubAnnotator(BaseAnnotator):
    """テスト用スタブアノテーター。_format_predictions の戻り値を差し替えるため。"""

    def __init__(self, model_name: str, formatted_output: object):
        config = LocalMLModelConfig(
            model_name=model_name,
            model_path="dummy",
            class_name="StubAnnotator",
            device="cpu",
            estimated_size_gb=0.01,
        )
        super().__init__(model_name, config=config)
        self._formatted_output = formatted_output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def _preprocess_images(self, images):
        return images

    def _run_inference(self, processed):
        return processed

    def _format_predictions(self, raw_outputs):
        return [self._formatted_output for _ in raw_outputs]


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color="red")


@pytest.mark.unit
def test_build_results_accepts_unified_annotation_result(dummy_image: Image.Image) -> None:
    """正常系: _format_predictions が UnifiedAnnotationResult を返せばそのまま返却される。"""
    expected = UnifiedAnnotationResult(
        model_name="stub",
        capabilities={TaskCapability.TAGS},
        tags=["dog"],
    )
    annotator = _StubAnnotator(model_name="stub", formatted_output=expected)

    results = annotator.predict([dummy_image])

    assert len(results) == 1
    assert results[0] is expected


@pytest.mark.unit
def test_build_results_rejects_dict(dummy_image: Image.Image) -> None:
    """旧形式 (dict) を返すと TypeError が発生する (predict は内部で例外をエラー結果に変換)。"""
    annotator = _StubAnnotator(model_name="stub", formatted_output={"score": 0.9})

    results = annotator.predict([dummy_image])

    assert len(results) == 1
    assert results[0].error is not None
    assert "UnifiedAnnotationResult" in (results[0].error or "")


@pytest.mark.unit
def test_build_results_rejects_str(dummy_image: Image.Image) -> None:
    """旧形式 (str) を返すと TypeError が発生する。"""
    annotator = _StubAnnotator(model_name="stub", formatted_output="caption text")

    results = annotator.predict([dummy_image])

    assert len(results) == 1
    assert results[0].error is not None
    assert "UnifiedAnnotationResult" in (results[0].error or "")
