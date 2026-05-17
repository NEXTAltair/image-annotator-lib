"""CLIP ベースの Scorer モデルの実装。

ADR 0002 contract: 純 regression scorer (`["scores"]` のみ、`score_labels=None`)。
配布元 (`christophschuhmann/improved-aesthetic-predictor` / `waifu-diffusion/aesthetic`) は
categorical label を提供しないため、lib では label 化を行わない。
"""

from typing import Any

from ..core.base import ClipBaseAnnotator
from ..core.utils import logger

# --- Improved Aesthetic / Waifu Aesthetic ---


class ImprovedAesthetic(ClipBaseAnnotator):
    """Improved Aesthetic Predictor v2 モデル (1-10 系 regression)。"""

    def __init__(self, model_name: str, **kwargs: Any):
        """ImprovedAestheticPredictor を初期化します。"""
        super().__init__(model_name=model_name, **kwargs)
        logger.debug(f"ImprovedAestheticPredictor '{model_name}' initialized.")


class WaifuAesthetic(ClipBaseAnnotator):
    """Waifu Diffusion Aesthetic Predictor v2 モデル (0-1 系 regression)。"""

    def __init__(self, model_name: str, **kwargs: Any):
        """WaifuAestheticPredictor を初期化します。"""
        super().__init__(model_name=model_name, **kwargs)
        logger.debug(f"WaifuAestheticPredictor '{model_name}' initialized.")
