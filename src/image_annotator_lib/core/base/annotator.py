"""基底アノテータークラス - すべてのアノテーターの共通基底クラス"""

import time
from abc import ABC, abstractmethod
from typing import Any, Self

import imagehash
from PIL import Image

# --- ローカルインポート ---
from ...exceptions.errors import OutOfMemoryError
from ..config import config_registry
from ..types import AnnotationResult, LoaderComponents
from ..utils import logger


class BaseAnnotator(ABC):
    """すべてのアノテーターの基底クラス。

    このクラスは、画像アノテーションの共通インターフェースを定義し、
    各フレームワーク固有の実装に必要な抽象メソッドを提供します。
    """

    def __init__(self, model_name: str):
        """BaseAnnotator を初期化します。

        Args:
            model_name (str): モデルの名前。設定ファイルでの識別子として使用されます。
        """
        self.model_name = model_name
        self.model_path = config_registry.get(model_name, "model_path")
        self.device = config_registry.get(model_name, "device", "cpu")
        self.components: LoaderComponents | None = None

    @abstractmethod
    def __enter__(self) -> Self:
        """コンテキストマネージャーの開始処理。モデルのロードを行います。"""
        raise NotImplementedError("サブクラスは __enter__ を実装する必要があります。")

    @abstractmethod
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """コンテキストマネージャーの終了処理。リソースの解放を行います。"""
        raise NotImplementedError("サブクラスは __exit__ を実装する必要があります。")

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        """画像リストを前処理します。

        Args:
            images: 前処理する PIL Image のリスト。

        Returns:
            前処理済みのデータ。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _preprocess_images を実装する必要があります。")

    @abstractmethod
    def _run_inference(self, processed: Any) -> Any:
        """前処理済みデータで推論を実行します。

        Args:
            processed: _preprocess_images の出力。

        Returns:
            推論結果の生データ。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _run_inference を実装する必要があります。")

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> Any:
        """推論結果を整形します。

        Args:
            raw_outputs: _run_inference の出力。

        Returns:
            整形済みの予測結果。形式はサブクラスによって異なります。
        """
        raise NotImplementedError("サブクラスは _format_predictions を実装する必要があります。")

    @abstractmethod
    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """整形済み出力からタグリストを生成します。

        Args:
            formatted_output: _format_predictions の出力。

        Returns:
            タグの文字列リスト。
        """
        raise NotImplementedError("サブクラスは _generate_tags を実装する必要があります。")

    def predict(
        self, images: list[Image.Image], phash_list: list[str] | None = None
    ) -> list[AnnotationResult]:
        """画像リストに対してアノテーションを実行します。

        Args:
            images: アノテーションする PIL Image のリスト。
            phash_list: 事前計算された知覚ハッシュのリスト（オプション）。

        Returns:
            各画像のアノテーション結果のリスト。
        """
        results: list[AnnotationResult] = []
        if not images:
            logger.warning("空の画像リストが渡されました。アノテーションをスキップします。")
            return results
        try:
            # 前処理
            start_time = time.time()
            processed = self._preprocess_images(images)
            preprocess_time = time.time() - start_time
            logger.debug(f"前処理時間: {preprocess_time:.3f}秒")

            # 推論実行
            start_time = time.time()
            raw_outputs = self._run_inference(processed)
            inference_time = time.time() - start_time
            logger.debug(f"推論時間: {inference_time:.3f}秒")

            # 結果の整形
            start_time = time.time()
            formatted_outputs = self._format_predictions(raw_outputs)
            format_time = time.time() - start_time
            logger.debug(f"整形時間: {format_time:.3f}秒")

            # 各画像の結果を処理
            if not isinstance(formatted_outputs, list):
                formatted_outputs = [formatted_outputs] * len(images)

            for i, (image, formatted_output) in enumerate(zip(images, formatted_outputs, strict=True)):
                try:
                    # 知覚ハッシュの計算
                    phash = (
                        phash_list[i]
                        if phash_list and i < len(phash_list)
                        else self._calculate_phash(image)
                    )

                    # タグ生成
                    tags = self._generate_tags(formatted_output)

                    result: AnnotationResult = {
                        "phash": phash,
                        "tags": tags,
                        "formatted_output": formatted_output,
                        "error": None,
                    }
                    results.append(result)

                except Exception as e:
                    logger.exception(f"画像 {i} の処理中にエラー: {e}")
                    err_result: AnnotationResult = {
                        "phash": phash_list[i] if phash_list and i < len(phash_list) else None,
                        "tags": [],
                        "formatted_output": None,
                        "error": f"タグ生成エラー: {e}",
                    }
                    results.append(err_result)

        except OutOfMemoryError as mem_e:
            logger.error(f"メモリ不足エラー: {mem_e}")
            # メモリ不足の場合、全画像に対してエラー結果を返す
            for i in range(len(images)):
                err_result: AnnotationResult = {
                    "phash": phash_list[i] if phash_list and i < len(phash_list) else None,
                    "tags": [],
                    "formatted_output": None,
                    "error": "メモリ不足エラー",
                }
                results.append(err_result)
        except Exception as e:
            logger.exception(f"予期せぬエラー: {e}")
            # その他のエラーの場合も、全画像に対してエラー結果を返す
            for i in range(len(images)):
                err_result: AnnotationResult = {
                    "phash": phash_list[i] if phash_list and i < len(phash_list) else None,
                    "tags": [],
                    "formatted_output": None,
                    "error": f"予期せぬエラー: {e}",
                }
                results.append(err_result)

        return results

    def _calculate_phash(self, image: Image.Image) -> str | None:
        """画像の知覚ハッシュを計算します。

        Args:
            image: ハッシュを計算する PIL Image。

        Returns:
            知覚ハッシュの文字列表現。計算に失敗した場合は None。
        """
        try:
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            logger.warning(f"知覚ハッシュの計算に失敗: {e}")
            return None
