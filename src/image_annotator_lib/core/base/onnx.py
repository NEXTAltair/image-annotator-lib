"""ONNX Runtime を使用するモデル用の基底クラス。"""

from abc import abstractmethod
from typing import Any, Self, cast

import numpy as np
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import OutOfMemoryError
from ..model_factory import ModelLoad
from ..types import ONNXComponents, TaskCapability, UnifiedAnnotationResult
from ..utils import logger
from .annotator import BaseAnnotator


class ONNXBaseAnnotator(BaseAnnotator):
    """ONNX Runtime を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.all_tags: list[str] = []
        self.target_size: tuple[int, int] | None = None
        self.is_nchw_expected: bool = False
        # components の型ヒントを具体的に指定
        self.components: ONNXComponents | None = None

    def __enter__(self) -> Self:
        """
        ModelLoad を使用して ONNX モデルコンポーネントをロードします。
        """
        try:
            if self.model_path is None:
                raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")
            logger.info(f"Loading/Restoring ONNX components: model='{self.model_path}'")
            self.components = ModelLoad.load_onnx_components(self.model_name, self.model_path, self.device)
            self._load_tags()
            self._analyze_model_input_format()

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"ONNXモデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """ONNX モデルのリソースを解放します。"""
        logger.debug(f"Exiting context for ONNX model '{self.model_name}' (exception: {exc_type})")
        if self.components:
            released_components = ModelLoad.release_model_components(
                self.model_name, cast(dict[str, Any], self.components)
            )
            self.components = cast(ONNXComponents, released_components)
        if exc_type:
            logger.error(f"ONNX モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

    @abstractmethod
    def _load_tags(self) -> None:
        """タグ情報 (語彙) をロードし、必要に応じてカテゴリインデックスを設定します (サブクラスで実装)。"""
        raise NotImplementedError("ONNX サブクラスは _load_tags を実装する必要があります。")

    def _extract_category_tags(
        self, attr_name: str, tags_with_probs: list[tuple[str, float]]
    ) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (ONNX タガー用)。"""
        category_tags: dict[str, float] = {}
        indexes = getattr(self, attr_name, [])
        all_tags_list = getattr(self, "all_tags", [])
        for i in indexes:
            if 0 <= i < len(tags_with_probs):
                tag_name, prob = tags_with_probs[i]
                category_tags[tag_name] = prob
            else:
                logger.warning(f"インデックス {i} が範囲外です (タグ総数: {len(all_tags_list)})。")
        return category_tags

    def _format_predictions_single(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> UnifiedAnnotationResult:
        """ONNX生出力を統一UnifiedAnnotationResultにフォーマットする。

        Args:
            raw_output: ONNX推論の生出力配列。

        Returns:
            フォーマット済みのUnifiedAnnotationResult。
        """
        from ..utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        all_tags_list = getattr(self, "all_tags", [])
        threshold = getattr(self, "tag_threshold", 0.35)

        # バリデーション
        predictions, error = self._validate_onnx_output(raw_output, all_tags_list, capabilities)
        if error:
            return error

        # カテゴリ別にタグを分類
        tags_with_probs = list(zip(all_tags_list, predictions, strict=True))
        category_scores = self._classify_tags_by_category(tags_with_probs)

        # 閾値フィルタリング
        final_tags = self._filter_tags_by_threshold(category_scores, threshold)

        return UnifiedAnnotationResult(
            model_name=self.model_name,
            capabilities=capabilities,
            tags=final_tags if TaskCapability.TAGS in capabilities else None,
            captions=None,
            scores=None,
            framework="onnx",
            raw_output={
                "predictions": predictions.tolist(),
                "category_scores": category_scores,
                "threshold": threshold,
                "total_tags_count": len(all_tags_list),
            },
        )

    def _validate_onnx_output(
        self,
        raw_output: np.ndarray[Any, np.dtype[Any]],
        all_tags_list: list[str],
        capabilities: set,
    ) -> tuple[np.ndarray | None, UnifiedAnnotationResult | None]:
        """ONNX出力のバリデーションを行う。

        Args:
            raw_output: ONNX推論の生出力。
            all_tags_list: タグ候補リスト。
            capabilities: モデルのケイパビリティ。

        Returns:
            (predictions, error) のタプル。バリデーション成功時はerrorがNone。
        """
        if not all_tags_list:
            logger.warning("タグ候補リスト (all_tags) がロードされていません。")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error="タグ候補リストが未ロード",
                framework="onnx",
            )

        if not isinstance(raw_output, np.ndarray):
            logger.error(f"予期しない生出力型: {type(raw_output)}")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"予期しない生出力型: {type(raw_output)}",
                framework="onnx",
            )

        if raw_output.ndim == 2 and raw_output.shape[0] == 1:
            predictions = raw_output[0].astype(float)
        elif raw_output.ndim == 1:
            predictions = raw_output.astype(float)
        else:
            logger.error(f"予期しない生出力形状: {raw_output.shape}")
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"予期しない生出力形状: {raw_output.shape}",
                framework="onnx",
            )

        if len(all_tags_list) != len(predictions):
            logger.error(
                f"タグ候補リスト数 ({len(all_tags_list)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return None, UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"タグ数と予測数の不一致: {len(all_tags_list)} != {len(predictions)}",
                framework="onnx",
            )

        return predictions, None

    def _classify_tags_by_category(
        self, tags_with_probs: list[tuple[str, float]]
    ) -> dict[str, dict[str, float]]:
        """タグをカテゴリ別に分類する。

        Args:
            tags_with_probs: (タグ名, 確率) のリスト。

        Returns:
            カテゴリ名→{タグ名: 確率} の辞書。
        """
        category_scores: dict[str, dict[str, float]] = {}

        category_map = getattr(self, "_category_attr_map", None)
        if category_map is None:
            logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            category_scores["general"] = {tag: float(prob) for tag, prob in tags_with_probs}
        else:
            for category_key, attr_name in category_map.items():
                category_tags = self._extract_category_tags(attr_name, tags_with_probs)
                if category_tags:
                    category_scores[category_key] = category_tags

            if "rating" in category_scores and "ratings" not in category_scores:
                category_scores["ratings"] = category_scores.pop("rating")

        return category_scores

    @staticmethod
    def _filter_tags_by_threshold(
        category_scores: dict[str, dict[str, float]], threshold: float
    ) -> list[str]:
        """閾値を超えるタグを信頼度順でフィルタリングする。

        Args:
            category_scores: カテゴリ別スコア辞書。
            threshold: フィルタ閾値。

        Returns:
            信頼度降順のタグ名リスト。
        """
        filtered_tags = []
        for _category, tag_dict in category_scores.items():
            for tag, confidence in tag_dict.items():
                if confidence >= threshold:
                    filtered_tags.append((tag, confidence))

        filtered_tags.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in filtered_tags]

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        # tag_threshold はサブクラスで設定される想定
        threshold = getattr(self, "tag_threshold", 0.35)

        for category, tag_dict in formatted_output.items():
            if category == "error":
                continue
            for tag, confidence in tag_dict.items():
                conf_value = (
                    confidence["confidence"]
                    if isinstance(confidence, dict) and "confidence" in confidence
                    else confidence
                )

                if isinstance(conf_value, (float)) and conf_value >= threshold:
                    tags.append((tag, float(conf_value)))

        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]

    def _analyze_model_input_format(self) -> None:
        """モデル入力形式を分析し、ターゲットサイズと次元形式を判定･保存する"""
        if not self.components or "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_shape = session.get_inputs()[0].shape

        target_size: tuple[int, int] | None = None
        is_nchw = False
        if len(input_shape) == 4:
            if (
                isinstance(input_shape[1], int)
                and input_shape[1] == 3
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                target_size = (input_shape[2], input_shape[3])
                is_nchw = True
            elif (
                isinstance(input_shape[3], int)
                and input_shape[3] == 3
                and isinstance(input_shape[1], int)
                and isinstance(input_shape[2], int)
            ):
                target_size = (input_shape[1], input_shape[2])
                is_nchw = False
            else:
                if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
                    logger.warning(
                        f"モデル {self.model_name} の不明な入力形状フォーマット: {input_shape}。ターゲットサイズとしてNHWC (インデックス 1, 2) を想定します。"
                    )
                    target_size = (input_shape[1], input_shape[2])
                    is_nchw = False

        if target_size is None:
            raise ValueError(f"入力形状 {input_shape} から有効なターゲットサイズ (H, W) を決定できません。")

        self.target_size = target_size
        self.is_nchw_expected = is_nchw

        logger.debug(
            f"モデル {self.model_name} の入力形状: {input_shape}, ターゲットサイズ: {self.target_size}, NCHW形式: {self.is_nchw_expected}"
        )

    def _preprocess_images(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        if self.target_size is None:
            raise ValueError(f"モデル {self.model_name} の target_size が設定されていません。")

        results = []
        for image in images:
            if image.mode == "RGBA":
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                canvas.paste(image, mask=image.split()[3])
                img_rgb = canvas
            elif image.mode != "RGB":
                img_rgb = image.convert("RGB")
            else:
                img_rgb = image

            width, height = img_rgb.size
            max_dim = max(width, height)
            pad_width = (max_dim - width) // 2
            pad_height = (max_dim - height) // 2
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(img_rgb, (pad_width, pad_height))

            resized = padded.resize(self.target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized, dtype=np.float32)[:, :, ::-1]

            if self.is_nchw_expected:
                input_data = np.transpose(img_array, (2, 0, 1))
            else:
                input_data = img_array

            input_data = np.expand_dims(input_data, axis=0)
            results.append(input_data.astype(np.float32))
        return results

    def _run_inference(
        self, processed: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """バッチの各画像に対してONNX推論を実行します。"""
        if not self.components or "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        results = []
        for input_data in processed:
            try:
                raw_output = session.run([output_name], {input_name: input_data})
                results.append(raw_output[0])
            except Exception as e:
                if "Failed to allocate memory" in str(e) or "out of memory" in str(e).lower():
                    error_message = f"ONNX Runtime メモリ不足: モデル {self.model_name} の推論中"
                    logger.error(error_message)
                    logger.error(f"元のONNX Runtimeエラー: {e}")
                    raise OutOfMemoryError(error_message) from e
                else:
                    logger.exception(f"ONNX Runtime エラー: モデル {self.model_name} の推論中: {e}")
                    raise
        return results

    def _format_predictions(
        self, raw_outputs: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[UnifiedAnnotationResult]:
        """バッチ出力結果を統一バリデーションスキーマでフォーマット"""
        result_list = []
        for raw_output in raw_outputs:
            formatted = self._format_predictions_single(raw_output)
            result_list.append(formatted)
        return result_list

    def _generate_tags(self, formatted_output: UnifiedAnnotationResult) -> list[str]:
        """統一バリデーションスキーマからタグリストを生成"""
        if isinstance(formatted_output, UnifiedAnnotationResult):
            return formatted_output.tags or []
        return []
