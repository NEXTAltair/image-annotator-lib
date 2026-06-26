"""Pipeline ベースの Aesthetic Score モデルの実装。"""

from typing import Any, ClassVar

from image_annotator_lib.core import utils

# PipelineBaseAnnotator をインポート
from image_annotator_lib.core.base import PipelineBaseAnnotator

from ..core.types import ScoreScale, UnifiedAnnotationResult
from ..core.utils import logger


class AestheticShadow(PipelineBaseAnnotator):
    """Aesthetic Shadow モデル (v1, v2) の共通処理を含む基底クラス (内部用)。

    Hugging Face Pipeline を使用して美的スコアを計算します。
    """

    # ADR 0009: hq/lq は softmax 確率 (各 0-1)。lq は値が小さいほど良い。
    SCORE_SCALE: ClassVar[dict[str, ScoreScale]] = {
        "hq": ScoreScale((0.0, 1.0), higher_is_better=True),
        "lq": ScoreScale((0.0, 1.0), higher_is_better=False),
    }

    def __init__(self, model_name: str):  # kwargs は不要
        """AestheticShadow を初期化します。"""
        super().__init__(model_name=model_name)
        logger.debug(f"AestheticShadow '{model_name}' initialized.")
        self.SCORE_THRESHOLDS = {
            "very aesthetic": 0.71,
            "aesthetic": 0.45,
            "displeasing": 0.27,
        }

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> list[UnifiedAnnotationResult]:
        """Pipeline の生出力リストから、各画像のhqとlqスコアを UnifiedAnnotationResult で返します。

        ADR 0002 contract: scorer は scores と score_labels (canonical label) を返し、
        tags は None。tags field は content tag (WDTagger 等) 専用。

        Args:
            raw_outputs: モデルからの生の出力リスト
                例: [[{'label': 'hq', 'score': 0.9}, {'label': 'lq', 'score': 0.1}], ...]

        Returns:
            list[UnifiedAnnotationResult]: 各画像の統一アノテーション結果
        """
        capabilities = utils.get_model_capabilities(self.model_name)
        results = []

        for single_output in raw_outputs:
            final_scores: dict[str, float] = {}

            for item in single_output:
                label = item["label"]
                if label in ["hq", "lq"]:
                    final_scores[label] = float(item["score"])

            # ADR 0002: 4-tier 閾値マッチで canonical score_labels を生成
            score_labels = self._generate_score_labels(final_scores)

            result = UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                scores=final_scores,
                score_scales=self.SCORE_SCALE,
                tags=None,
                score_labels=score_labels,
                framework="pipeline",
                raw_output={"scores": final_scores},
            )
            results.append(result)

        return results

    def _generate_score_labels(self, formatted_score: dict[str, float]) -> list[str]:
        """hq スコアに 4-tier 閾値マッチを適用して canonical score_labels を返す (ADR 0002)。

        Args:
            formatted_score: hqとlqのスコアを含む辞書

        Returns:
            list[str]: 該当 tier 1 つを単一要素として返す
                ("very aesthetic" / "aesthetic" / "displeasing" / "very displeasing")
        """
        if "hq" in formatted_score:
            hq_score = formatted_score["hq"]
            for label, threshold in self.SCORE_THRESHOLDS.items():
                if hq_score >= threshold:
                    return [label]
        return ["very displeasing"]


# --- Cafe Aesthetic ---
class CafePredictor(PipelineBaseAnnotator):
    """Cafe Aesthetic モデル ("model_name/cafe_aesthetic")。

    Hugging Face Pipeline を使用して美的スコアを計算します。
    """

    # ADR 0009: aesthetic/not_aesthetic は softmax 確率 (各 0-1、sum=1)。
    # not_aesthetic は値が小さいほど良い。
    SCORE_SCALE: ClassVar[dict[str, ScoreScale]] = {
        "aesthetic": ScoreScale((0.0, 1.0), higher_is_better=True),
        "not_aesthetic": ScoreScale((0.0, 1.0), higher_is_better=False),
    }

    def __init__(self, model_name: str):
        """CafePredictor を初期化します。"""
        super().__init__(model_name=model_name)
        logger.debug(f"CafePredictor '{model_name}' initialized.")

    # _format_predictions を実装 (PipelineBaseAnnotator の抽象メソッド)
    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> list[UnifiedAnnotationResult]:
        """Pipeline の生出力から UnifiedAnnotationResult を返す (ADR 0002 contract)。

        scores には aesthetic / not_aesthetic 両 label の probability を保持し、
        score_labels は argmax label の単一要素を返す。tags は None。
        """
        capabilities = utils.get_model_capabilities(self.model_name)
        results = []

        for single_output in raw_outputs:
            # raw 例: [{'label': 'aesthetic', 'score': 0.67}, {'label': 'not_aesthetic', 'score': 0.33}]
            if not isinstance(single_output, list):
                logger.warning(
                    f"予期しない single_output の型: {type(single_output)}。スコア 0.0 を使用します。"
                )
                aesthetic_score = 0.0
            else:
                aesthetic_score = 0.0
                score_found = False
                for entry in single_output:
                    if isinstance(entry, dict) and entry.get("label") == "aesthetic" and "score" in entry:
                        try:
                            aesthetic_score = float(entry["score"])
                            score_found = True
                            break
                        except (TypeError, ValueError):
                            logger.error(f"モデルからの戻り値 'aesthetic' のスコアが不正です: {entry}")
                            aesthetic_score = 0.0
                            score_found = True
                            break
                if not score_found:
                    logger.warning(f"出力に 'aesthetic' ラベルが見つかりませんでした: {single_output}")
                    aesthetic_score = 0.0

            # ADR 0002: binary classification の両 label probability を保持 (sum=1 を前提)
            not_aesthetic_score = 1.0 - aesthetic_score
            scores = {"aesthetic": aesthetic_score, "not_aesthetic": not_aesthetic_score}

            # ADR 0002: argmax label を score_labels に
            label = "aesthetic" if aesthetic_score > 0.5 else "not_aesthetic"

            result = UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                scores=scores,
                score_scales=self.SCORE_SCALE,
                tags=None,
                score_labels=[label],
                framework="pipeline",
                raw_output={"aesthetic_score": aesthetic_score, "not_aesthetic_score": not_aesthetic_score},
            )
            results.append(result)

        return results


class AestheticShadowV2(AestheticShadow):
    """Aesthetic Shadow V2 モデル ("NEXTAltair/cache_aestheic-shadow-v2")。"""

    # model_name は config ファイルで "aesthetic-shadow-v2" のように定義されることを想定
    pass  # 実装は AestheticShadow に依存


# 他の Aesthetic 系モデルも同様に PipelineBaseAnnotator を継承して実装可能
