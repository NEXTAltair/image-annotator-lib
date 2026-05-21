"""ローカル ML モデル実 DL + 実推論の on-demand validation (ADR 0001 amended)。

各 base class (ONNX / Transformers / Tensorflow / Pipeline / CLIP) を 1 model ずつ
network 越しに実 DL し、`image_annotator_lib.annotate()` public API 経由で実推論を
通すことで、image-annotator-lib の public API contract と heavy native dep 経路を
ローカル only で検証する。

Runtime cost (初回 cache 構築時):
    DL ~5-10 GB (5 model 合算)、wall time ~10-20 分。cache 後は ~3-5 分。

Marker:
    `@pytest.mark.downloads_and_runs_model` (CI 不経由、ローカル only)。

Related:
    LoRAIro umbrella #276 / Tier 1 #277 / iam-lib #71 / ADR 0001 amended 2026-05-18
"""

from pathlib import Path

import pytest
from PIL import Image

from image_annotator_lib import annotate

_RESOURCE_IMG = Path(__file__).parent.parent / "resources" / "img" / "1_img" / "file07.webp"


@pytest.mark.downloads_and_runs_model
@pytest.mark.parametrize(
    "model_name",
    [
        "wd-vit-tagger-v3",  # ONNXBaseAnnotator
        "BLIPLargeCaptioning",  # TransformersBaseAnnotator
        "deepdanbooru-v4-20200814-sgd-e30",  # TensorflowBaseAnnotator
        pytest.param(
            "cafe_aesthetic",  # PipelineBaseAnnotator
            marks=pytest.mark.xfail(
                reason=(
                    "LoRAIro #273: CafePredictor._format_predictions が "
                    "score_labels=['aesthetic'] を返すが capabilities に "
                    "SCORE_LABELS を含めないため UnifiedAnnotationResult "
                    "validation error。fix 後 strict=True で xpassed 検知。"
                ),
                strict=True,
            ),
        ),
        "WaifuAesthetic",  # ClipBaseAnnotator
    ],
)
def test_real_model_runtime(model_name: str) -> None:
    """5 base class の 1 model ずつで実 DL + 実推論を通し、output が non-empty かを確認する。

    出力 field (`tags` / `captions` / `scores` / `score_labels`) は model の capability で
    異なるため、本 test は「いずれかの output field が non-empty」を smoke 条件とする。
    具体的な capability 別検証は別 test で扱う方針。
    """
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    img = Image.open(_RESOURCE_IMG).convert("RGB")
    result = annotate(images_list=[img], model_name_list=[model_name])

    assert len(result) == 1, f"expected 1 phash entry, got {len(result)}"
    for _phash, models in result.items():
        assert model_name in models, f"{model_name} missing in result keys: {list(models.keys())}"
        ann = models[model_name]
        assert ann.error is None, f"{model_name} returned error: {ann.error}"
        output_present = (
            bool(ann.tags)
            or bool(ann.captions)
            or bool(ann.scores)
            or bool(ann.score_labels)
            or bool(ann.ratings)
        )
        assert output_present, (
            f"{model_name}: all output fields empty "
            f"(tags={ann.tags!r}, captions={ann.captions!r}, "
            f"scores={ann.scores!r}, score_labels={ann.score_labels!r}, ratings={ann.ratings!r})"
        )


@pytest.mark.downloads_and_runs_model
def test_real_anime_rating_runtime() -> None:
    """deepghs/anime_rating smoke test for model-native rating output."""
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    img = Image.open(_RESOURCE_IMG).convert("RGB")
    model_name = "anime_rating_mobilenetv3_sce_dist"
    result = annotate(images_list=[img], model_name_list=[model_name])

    assert len(result) == 1, f"expected 1 phash entry, got {len(result)}"
    for _phash, models in result.items():
        ann = models[model_name]
        assert ann.error is None, f"{model_name} returned error: {ann.error}"
        assert ann.tags is None
        assert ann.ratings is not None and len(ann.ratings) == 1
        rating = ann.ratings[0]
        assert rating.raw_label in {"safe", "r15", "r18"}
        assert rating.source_scheme == "sankaku3"
        assert rating.confidence_score is not None
        assert 0.0 <= rating.confidence_score <= 1.0


@pytest.mark.downloads_and_runs_model
def test_real_camie_tagger_runtime() -> None:
    """CamieTagger heavy smoke test for tags + rating output."""
    if not _RESOURCE_IMG.exists():
        pytest.skip(f"resource image not found: {_RESOURCE_IMG}")

    img = Image.open(_RESOURCE_IMG).convert("RGB")
    model_name = "camie_tagger_initial"
    result = annotate(images_list=[img], model_name_list=[model_name])

    assert len(result) == 1, f"expected 1 phash entry, got {len(result)}"
    for _phash, models in result.items():
        ann = models[model_name]
        assert ann.error is None, f"{model_name} returned error: {ann.error}"
        assert ann.tags is not None
        assert ann.ratings is not None and len(ann.ratings) == 1
        rating = ann.ratings[0]
        assert rating.raw_label in {"general", "sensitive", "questionable", "explicit"}
        assert rating.source_scheme == "danbooru4"
        assert rating.confidence_score is not None
        assert 0.0 <= rating.confidence_score <= 1.0
