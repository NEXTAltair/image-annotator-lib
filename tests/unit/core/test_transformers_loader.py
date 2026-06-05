"""Issue #139 (A-2) regression: transformers 5.x の legacy image_processor 解決。

`_resolve_pipeline_image_processor` は古い repo (preprocessor_config.json が
legacy な `image_processor_type` を持ち AutoImageProcessor で解決できない) でも
model_type 経由で ImageProcessor クラスを引いて構築できることを検証する。
ネットワーク / 実モデルには触れず、transformers の auto クラスをモックする。

注: ヘルパー内の `from transformers import ...` は関数ローカルかつ transformers は
_LazyModule なので、モジュール属性の patch (`patch("transformers.AutoImageProcessor")`)
は効かない。共有されるクラスオブジェクトの from_pretrained を patch.object で差し替える。
IMAGE_PROCESSOR_MAPPING は通常のサブモジュール属性なので dict 差し替えで patch する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig, AutoImageProcessor

from image_annotator_lib.core.loaders.transformers_loader import (
    _resolve_pipeline_image_processor,
)

_MAPPING = "transformers.models.auto.image_processing_auto.IMAGE_PROCESSOR_MAPPING"


@pytest.mark.unit
def test_resolve_image_processor_auto_success():
    """AutoImageProcessor が成功する正常系はそのまま結果を返す (cafe / v2 相当)。"""
    expected = MagicMock(name="image_processor")
    with patch.object(AutoImageProcessor, "from_pretrained", return_value=expected) as mock_auto:
        result = _resolve_pipeline_image_processor("some/repo")
    assert result is expected
    mock_auto.assert_called_once_with("some/repo")


@pytest.mark.unit
def test_resolve_image_processor_legacy_fallback_via_model_type():
    """AutoImageProcessor 失敗時 (legacy image_processor_type) は model_type 経由で構築する。

    Issue #139 の aesthetic_shadow_v1 (ViTFeatureExtractor) を再現。
    """
    built = MagicMock(name="vit_image_processor_instance")
    vit_cls = MagicMock(name="ViTImageProcessor")
    vit_cls.from_pretrained.return_value = built

    fake_config = MagicMock(name="ViTConfig")
    # transformers 5.x の mapping 値は backend 別 dict
    mapping_entry = {"pil": None, "torchvision": vit_cls}

    with (
        patch.object(
            AutoImageProcessor,
            "from_pretrained",
            side_effect=ValueError("Unrecognized image processor"),
        ),
        patch.object(AutoConfig, "from_pretrained", return_value=fake_config),
        patch(_MAPPING, {type(fake_config): mapping_entry}),
    ):
        result = _resolve_pipeline_image_processor("shadowlilac/aesthetic-shadow")

    assert result is built
    vit_cls.from_pretrained.assert_called_once_with("shadowlilac/aesthetic-shadow")


@pytest.mark.unit
def test_resolve_image_processor_unresolvable_returns_none():
    """AutoImageProcessor も model_type 経由も失敗するなら None (pipeline auto に委ねる)。"""
    with (
        patch.object(
            AutoImageProcessor,
            "from_pretrained",
            side_effect=ValueError("Unrecognized image processor"),
        ),
        patch.object(AutoConfig, "from_pretrained", side_effect=OSError("config not found")),
    ):
        result = _resolve_pipeline_image_processor("broken/repo")

    assert result is None
