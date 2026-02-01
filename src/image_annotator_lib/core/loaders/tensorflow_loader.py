"""TensorFlow モデルローダー。

SavedModel および H5 形式の TensorFlow モデルのロードを提供する。

Dependencies:
    - tensorflow: TensorFlow (遅延import)
    - keras: Keras (遅延import)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, override

from .. import utils
from ..types import TensorFlowComponents
from ..utils import logger
from .loader_base import LoaderBase

if __name__ != "__main__":
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import tensorflow as tf
        from tensorflow import keras


class TensorFlowLoader(LoaderBase):
    """TensorFlow モデル (SavedModel, H5) のローダー。"""

    def _resolve_model_dir_internal(self, model_path: str) -> Path | None:
        """TensorFlow モデルディレクトリを解決する。"""
        try:
            model_dir_obj = utils.load_file(model_path)
            if model_dir_obj is None or not model_dir_obj.is_dir():
                logger.error(f"有効な TensorFlow モデルディレクトリが見つかりません: {model_path}")
                return None
            return model_dir_obj
        except Exception as e:
            logger.error(f"TFモデルディレクトリ解決中にエラー ({model_path}): {e}", exc_info=True)
            return None

    def _get_tf_calc_params(self, model_dir: Path, model_format: str) -> tuple[Path, float]:
        """TF モデルフォーマットに基づいてターゲットパスとサイズ乗数を取得する。"""
        if model_format == "h5":
            h5_files = list(model_dir.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"H5ファイルが見つかりません: {model_dir}")
            return h5_files[0], 1.2
        elif model_format == "saved_model":
            if (
                not (model_dir / "saved_model.pb").exists()
                and not (model_dir / "saved_model.pbtxt").exists()
            ):
                raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {model_dir}")
            return model_dir, 1.3
        elif model_format == "pb":
            raise NotImplementedError(".pb 単体フォーマットのサイズ計算は未サポート。")
        else:
            raise ValueError(f"未対応のTFフォーマット: {model_format}")

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """モデルフォーマットに基づいてサイズを計算する。

        kwargs に 'model_format' が必要。
        """
        model_format = cast(str, kwargs.get("model_format"))
        if not model_format:
            return 0.0

        model_dir = self._resolve_model_dir_internal(model_path)
        if not model_dir:
            return 0.0

        try:
            target_path, multiplier = self._get_tf_calc_params(model_dir, model_format)
            if target_path.is_file():
                return LoaderBase._calculate_file_size_mb(target_path) * multiplier
            elif target_path.is_dir():
                return LoaderBase._calculate_dir_size_mb(target_path) * multiplier
            return 0.0
        except (FileNotFoundError, NotImplementedError, ValueError) as e:
            logger.error(f"TFモデル '{self.model_name}' サイズ計算エラー: {e}", exc_info=False)
            return 0.0
        except Exception as e:
            logger.warning(f"TFモデル '{self.model_name}' サイズ計算中にエラー: {e}", exc_info=False)
            return 0.0

    @override
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> TensorFlowComponents:
        """TensorFlow モデル (SavedModel または H5) をロードする。

        kwargs に 'model_format' が必要。
        """
        import tensorflow as tf
        from tensorflow import keras

        model_format = cast(str, kwargs.get("model_format"))
        if not model_format:
            raise ValueError("TensorFlow loader requires 'model_format' kwarg.")

        model_dir = self._resolve_model_dir_internal(model_path)
        if model_dir is None:
            raise FileNotFoundError(f"TensorFlow モデルディレクトリ解決失敗: {model_path}")

        model_instance: tf.Module | keras.Model | None = None
        if model_format == "h5":
            h5_files = list(model_dir.glob("*.h5"))
            if not h5_files:
                raise FileNotFoundError(f"H5ファイルが見つかりません: {model_dir}")
            target_path = h5_files[0]
            logger.info(f"H5モデルロード中: {target_path}")
            model_instance = keras.models.load_model(target_path, compile=False)
        elif model_format == "saved_model":
            target_path = model_dir
            logger.info(f"SavedModelロード中: {target_path}")
            if (
                not (target_path / "saved_model.pb").exists()
                and not (target_path / "saved_model.pbtxt").exists()
            ):
                raise FileNotFoundError(f"有効な SavedModel ディレクトリではありません: {target_path}")
            model_instance = tf.saved_model.load(str(target_path))
        elif model_format == "pb":
            raise NotImplementedError("Direct loading from .pb is not supported.")
        else:
            raise ValueError(f"未対応のTensorFlowモデルフォーマット: {model_format}")

        if model_instance is None:
            from ...exceptions.errors import ModelLoadError

            raise ModelLoadError("TensorFlowモデルインスタンスのロード失敗。")

        return {"model_dir": model_dir, "model": model_instance}
