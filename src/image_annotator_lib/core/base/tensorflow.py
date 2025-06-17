"""TensorFlow モデルを使用するモデル用の基底クラス。"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
import tensorflow as tf
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import ModelLoadError, OutOfMemoryError
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import TensorFlowComponents
from ..utils import logger
from .annotator import BaseAnnotator


class TensorflowBaseAnnotator(BaseAnnotator):
    """TensorFlow モデルを使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        if tf is None:
            raise ImportError("TensorFlow がインストールされていません。")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.debug("TensorFlow GPU メモリ成長を有効化しました。")
            else:
                logger.debug("TensorFlow: 利用可能な GPU が見つかりません。")
        except Exception as gpu_e:
            logger.warning(f"TensorFlow GPU 設定中にエラー: {gpu_e}")

        # model_format の取得と検証 (config_registry を使用)
        model_format_input = config_registry.get(self.model_name, "model_format", "h5")
        allowed_formats = ("h5", "saved_model", "pb")
        if model_format_input not in allowed_formats:
            raise ValueError(
                f"設定 '{self.model_name}' の 'model_format' が不正です: '{model_format_input}'. "
                f"許可される形式: {allowed_formats}"
            )
        self.model_format = model_format_input
        # components の型ヒントを具体的に指定
        self.components: TensorFlowComponents | None = None

    def __enter__(self) -> "TensorflowBaseAnnotator":
        """TensorFlow モデルコンポーネントをロードします。状態管理は ModelLoad に委譲します。"""
        logger.debug(f"Entering context for TensorFlow model '{self.model_name}'")
        try:
            logger.info(
                f"Loading/Restoring TensorFlow components: model='{self.model_path}', format='{self.model_format}'"
            )
            if self.model_path is None:
                raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")
            loaded_components = ModelLoad.load_tensorflow_components(
                self.model_name,
                self.model_path,
                self.device,
                self.model_format,
            )
            if loaded_components is None:
                raise ModelLoadError(f"モデル '{self.model_name}' のロード/復元に失敗しました。")
            self.components = loaded_components
            self._load_tags()  # TFモデル固有のタグロード処理
            logger.info(f"モデル '{self.model_name}' を正常にロードしました")
        except (ModelLoadError, OutOfMemoryError, FileNotFoundError, ValueError) as e:
            logger.error(f"TensorFlow モデル '{self.model_name}' のロード/準備中にエラー: {e}")
            self.components = None
            raise
        except Exception as e:
            logger.exception(f"TensorFlow モデル '{self.model_name}' のロード/準備中に予期せぬエラー: {e}")
            self.components = None
            raise ModelLoadError(f"予期せぬロードエラー: {e}") from e
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """TensorFlow モデルのリソースを解放します。"""
        logger.debug(f"Exiting context for TensorFlow model '{self.model_name}' (exception: {exc_type})")
        if self.components:
            try:
                components_to_release = self.components
                released_components = ModelLoad.release_model_components(self.model_name, cast(dict[str, Any], components_to_release))
                self.components = cast(TensorFlowComponents, released_components)
                logger.debug("TensorFlow Keras セッションクリアを試行 (必要な場合)。")
                if tf:
                    tf.keras.backend.clear_session()
            except Exception as e:
                logger.exception(f"TensorFlow モデル '{self.model_name}' の解放中にエラー: {e}")
            finally:
                self.components = None
        if exc_type:
            logger.error(f"TensorFlow モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

    @abstractmethod
    def _load_tags(self) -> None:
        """モデル固有のタグ情報 (例: tags.txt) をロードします。"""
        raise NotImplementedError("サブクラスは _load_tags を実装する必要があります。")

    def _load_tag_file(self, tags_path: Path) -> list[str]:
        """タグファイルを読み込み、タグのリストを返します。"""
        if not tags_path.is_file():
            logger.error(f"タグファイルが見つかりません: {tags_path}")
            return []
        try:
            with open(tags_path, encoding="utf-8") as f:
                tags = [line.strip() for line in f if line.strip()]
                logger.debug(f"{tags_path.name} から {len(tags)} 個のタグをロードしました。")
                return tags
        except Exception as e:
            logger.exception(f"タグファイル '{tags_path}' の読み込みエラー: {e}")
            return []

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """画像リストを前処理し、単一の NumPy 配列バッチを返します。"""
        raise NotImplementedError("TensorFlow サブクラスは _preprocess_images を実装する必要があります。")

    def _run_inference(self, processed: np.ndarray[Any, np.dtype[Any]]) -> tf.Tensor:
        """前処理済みバッチで推論を実行します (TensorFlow用)。"""
        return self._run_inference_tf(processed)

    @abstractmethod
    def _format_predictions(self, raw_output: tf.Tensor) -> list[Any]:
        """モデルの生出力バッチをフォーマットします。"""
        raise NotImplementedError("TensorFlow サブクラスは _format_predictions を実装する必要があります。")

    def _run_inference_tf(self, processed: np.ndarray[Any, np.dtype[Any]]) -> tf.Tensor:
        """TensorFlow モデルでバッチ推論を実行します。"""
        if not self.components or "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("TensorFlow モデルがロードされていません。")
        tf_model = self.components["model"]
        try:
            logger.debug(f"TF 推論実行: 入力形状={processed.shape}")
            raw_output = tf_model(processed, training=False)
            logger.debug(f"TF 推論完了: 出力形状={raw_output.shape}")
            return raw_output
        except tf.errors.ResourceExhaustedError as e:
            error_message = f"TensorFlow リソース枯渇 (OOM?) : モデル '{self.model_name}' の推論実行中"
            logger.error(error_message)
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            logger.exception(f"TensorFlow モデル '{self.model_name}' の推論実行中にエラーが発生: {e}")
            raise RuntimeError(f"TensorFlow 推論エラー: {e}") from e

    def _generate_tags(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX/TF タガー用)。"""
        return self._generate_tags_single(formatted_output)

    def _extract_category_tags(
        self, attr_name: str, tags_with_probs: list[tuple[str, float]]
    ) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (TF タガー用)。"""
        category_tags: dict[str, float] = {}
        # サブクラスで定義される属性 (e.g., self.general_indexes) を取得
        indexes = getattr(self, attr_name, [])
        all_tags_list = getattr(self, "all_tags", [])  # all_tags もサブクラスで設定される
        for i in indexes:
            if 0 <= i < len(tags_with_probs):
                tag_name, prob = tags_with_probs[i]
                category_tags[tag_name] = prob
            else:
                logger.warning(f"インデックス {i} が範囲外です (タグ総数: {len(all_tags_list)})。")
        return category_tags

    def _format_predictions_single(
        self,
        raw_output: np.ndarray[Any, np.dtype[Any]] | tf.Tensor,  # TFテンソルも受け入れる
    ) -> dict[str, dict[str, float]]:
        """単一の生出力をカテゴリ別にフォーマットします (TF タガー用)。"""
        result: dict[str, dict[str, float]] = {}
        all_tags_list = getattr(self, "all_tags", [])  # サブクラスで設定される all_tags を取得
        if not all_tags_list:
            logger.warning("タグ候補リスト (all_tags) がロードされていません。フォーマットできません。")
            return {"error": {}}  # エラーを示す辞書を返す

        # 生出力が NumPy 配列であることを確認し、適切な次元から予測値を取得
        if isinstance(raw_output, tf.Tensor):  # TFテンソルの場合 NumPy に変換
            try:
                # raw_output が None でないことを確認 (tf_model呼び出しがNoneを返さない想定)
                # tf_modelの呼び出し結果がNoneになるケースは現状考えにくいが、より安全にするならNoneチェック
                if raw_output is None:
                    logger.error("TensorFlow推論結果がNoneです。")
                    return {"error": {}}
                predictions = raw_output.numpy().astype(float)
            except Exception as e:
                logger.exception(f"TF テンソルの NumPy 変換中にエラー: {e}")
                return {"error": {}}
        elif isinstance(raw_output, np.ndarray):
            predictions = raw_output.astype(float)

        # 予測値の次元をチェック
        if predictions.ndim == 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        elif predictions.ndim != 1:
            logger.error(f"予期しない予測値形状: {predictions.shape}")
            return {"error": {}}

        # タグ数と予測数が一致するか確認
        if len(all_tags_list) != len(predictions):
            logger.error(
                f"タグ候補リスト数 ({len(all_tags_list)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return {"error": {}}

        tags_with_probs = list(zip(all_tags_list, predictions, strict=True))

        # _category_attr_map はサブクラス (e.g., DeepDanbooruTagger) で定義される
        category_map = getattr(self, "_category_attr_map", None)
        if category_map is None:
            logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            result["general"] = {tag: float(prob) for tag, prob in tags_with_probs}
            return result

        # カテゴリごとにタグを抽出
        for category_key, attr_name in category_map.items():
            category_tags = self._extract_category_tags(attr_name, tags_with_probs)
            if category_tags:
                result[category_key] = category_tags

        # rating キーを ratings にリネーム (後方互換性のため)
        if "rating" in result and "ratings" not in result:
            result["ratings"] = result.pop("rating")

        return result

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (TF タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        # tag_threshold はサブクラスで設定される想定
        threshold = getattr(self, "tag_threshold", 0.35)

        for category, tag_dict in formatted_output.items():
            if category == "error":  # エラーカテゴリはスキップ
                continue
            for tag, confidence in tag_dict.items():
                # 確信度の値が辞書型の場合 (古い形式への対応?)、confidence キーの値を取得
                conf_value = (
                    confidence["confidence"]
                    if isinstance(confidence, dict) and "confidence" in confidence
                    else confidence
                )

                # confidence が数値型であることを確認してから比較
                if isinstance(conf_value, (float)) and conf_value >= threshold:
                    tags.append((tag, float(conf_value)))  # 確信度も float に統一

        # タグ名で重複を除去し、最も高い確信度を採用
        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        # 確信度で降順ソートしてタグ名のみを返す
        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]
