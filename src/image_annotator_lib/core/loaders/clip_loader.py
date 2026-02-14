"""CLIP ベースモデルローダー。

CLIP プロセッサ、ベースモデル、分類器ヘッドのロードを提供する。

Dependencies:
    - transformers: CLIP モデル (遅延import)
    - torch: PyTorch (遅延import)
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, cast, override

from ..types import CLIPComponents
from ..utils import logger
from .loader_base import LoaderBase

if TYPE_CHECKING:
    import torch.nn as nn
    from transformers.models.clip import CLIPModel, CLIPProcessor

from ...exceptions.errors import ModelLoadError
from ..classifier import Classifier


class CLIPLoader(LoaderBase):
    """CLIP ベースモデル (Scorer など) のローダー。

    ベースの CLIP モデル/プロセッサと分類器ヘッドをロードする。
    """

    def _infer_classifier_structure(self, state_dict: dict[str, Any]) -> list[int]:
        """state_dict から分類器ヘッドの隠れ層サイズを推測する。

        不連続なレイヤー番号付けに対応する。
        推測に失敗した場合はデフォルト構造を使用する。
        """
        hidden_features: list[int] = []
        current_layer = 0
        while True:
            weight_key = f"layers.{current_layer}.weight"
            bias_key = f"layers.{current_layer}.bias"
            if weight_key not in state_dict or bias_key not in state_dict:
                found_next = False
                for lookahead in range(1, 5):
                    next_weight_key = f"layers.{current_layer + lookahead}.weight"
                    if next_weight_key in state_dict:
                        current_layer += lookahead
                        weight_key = next_weight_key
                        found_next = True
                        break
                if not found_next:
                    break

            if weight_key in state_dict:
                hidden_features.append(state_dict[weight_key].shape[0])
                current_layer += 1
            else:
                break

        hidden_sizes = hidden_features[:-1] if hidden_features else []
        if not hidden_sizes:
            logger.warning(
                f"CLIP分類器 '{self.model_name}' 構造推測失敗。デフォルト [1024, 128, 64, 16] 使用。"
            )
            hidden_sizes = [1024, 128, 64, 16]
        logger.info(f"推測された隠れ層サイズ: {hidden_sizes}")
        return hidden_sizes

    def _load_base_clip_components(self, base_model: str) -> tuple[CLIPProcessor, CLIPModel, int]:
        """CLIP プロセッサとベースモデルをロードし、特徴量次元を返す。"""
        from transformers.models.clip import CLIPModel, CLIPProcessor

        logger.debug(f"CLIPプロセッサロード中: {base_model}")
        clip_processor = CLIPProcessor.from_pretrained(base_model)
        logger.debug(f"CLIPモデルロード中: {base_model} on {self.device}")
        clip_model = cast(
            CLIPModel,
            CLIPModel.from_pretrained(base_model).to(self.device).eval(),  # type: ignore[no-untyped-call]
        )
        input_size = clip_model.config.projection_dim
        logger.debug(f"CLIPモデル {base_model} 特徴量次元: {input_size}")
        return clip_processor, clip_model, input_size  # type: ignore[return-value]

    def _create_and_load_classifier_head(
        self,
        input_size: int,
        hidden_sizes: list[int],
        state_dict: dict[str, Any],
        activation_type: str | None,
        final_activation_type: str | None,
        model_path_for_log: str,
    ) -> nn.Module:
        """分類器ヘッドモジュールを作成し、重みをロードしてデバイスに移動する。"""
        import torch.nn as nn

        activation_map: dict[str, type[nn.Module]] = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
        }
        use_activation = activation_type is not None
        activation_func = (
            activation_map.get(activation_type, nn.ReLU)  # type: ignore[arg-type]
            if use_activation and activation_type in activation_map
            else nn.ReLU
        )
        use_final_activation = final_activation_type is not None
        final_activation_func = (
            activation_map.get(final_activation_type, nn.Sigmoid)  # type: ignore[arg-type]
            if use_final_activation and final_activation_type in activation_map
            else nn.Sigmoid
        )

        logger.info("分類器モデル初期化...")
        classifier_head = Classifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            use_activation=use_activation,
            activation=activation_func,
            use_final_activation=use_final_activation,
            final_activation=final_activation_func,
        )
        classifier_head.load_state_dict(state_dict, strict=False)
        classifier_head = classifier_head.to(self.device).eval()
        logger.debug(f"CLIP分類器ヘッド '{model_path_for_log}' ロード完了 (デバイス: {self.device})")
        return classifier_head

    def _create_clip_model_internal(
        self,
        base_model: str,
        model_path: str,
        activation_type: str | None,
        final_activation_type: str | None,
    ) -> CLIPComponents | None:
        """CLIP コンポーネント一式を作成する。

        プロセッサ、ベースモデル、分類器ヘッドを含む。
        """
        import torch
        from transformers.models.clip import CLIPModel, CLIPProcessor

        from .. import utils

        try:
            # 1. ベース CLIP モデルとプロセッサをロード
            clip_processor, clip_model, input_size = self._load_base_clip_components(base_model)

            # 2. 分類器ヘッドの重みをロード
            logger.debug(f"分類器ヘッド重みロード中: {model_path}")
            local_path = utils.load_file(model_path)
            if local_path is None:
                logger.error(f"分類器ヘッドパス '{model_path}' 解決失敗。")
                return None
            state_dict = torch.load(local_path, map_location=self.device)
            logger.debug("重みロード完了、構造推測開始...")

            # 3. 分類器構造を推測
            hidden_sizes_for_classifier = self._infer_classifier_structure(state_dict)

            # 4. 分類器ヘッドを作成・ロード
            classifier_head = self._create_and_load_classifier_head(
                input_size=input_size,
                hidden_sizes=hidden_sizes_for_classifier,
                state_dict=state_dict,
                activation_type=activation_type,
                final_activation_type=final_activation_type,
                model_path_for_log=model_path,
            )

            # 5. 型安全チェック
            if not isinstance(classifier_head, torch.nn.Module):
                raise TypeError("Classifier head not a Module")
            if not isinstance(clip_processor, CLIPProcessor):
                raise TypeError("Processor not a CLIPProcessor")
            if not isinstance(clip_model, CLIPModel):
                raise TypeError("Base model not a CLIPModel")

            result: CLIPComponents = {
                "model": classifier_head,
                "processor": clip_processor,
                "clip_model": clip_model,
            }
            return result

        except FileNotFoundError as e:
            logger.error(f"CLIPモデル作成エラー: ファイル未検出: {e}")
            return None
        except KeyError as e:
            logger.error(f"CLIPモデル作成エラー: state_dict キーエラー: {e}")
            return None
        except Exception as e:
            logger.error(
                f"CLIPモデル作成中に予期せぬエラー ({base_model}/{model_path}): {e}", exc_info=True
            )
            return None

    def _calculate_specific_size(self, model_path: str, **kwargs: Any) -> float:
        """CPU 上で CLIP コンポーネントを一時作成してサイズを計算する。

        kwargs に 'base_model' が必要。
        """
        import torch

        base_model = cast(str, kwargs.get("base_model"))
        activation_type = cast(str | None, kwargs.get("activation_type"))
        final_activation_type = cast(str | None, kwargs.get("final_activation_type"))
        if not base_model:
            return 0.0

        logger.debug(f"一時ロードによる CLIP サイズ計算開始: base={base_model}, head={model_path}")
        calculated_size_mb = 0.0
        temp_components: CLIPComponents | None = None
        temp_helper_instance: CLIPLoader | None = None
        try:
            temp_helper_instance = CLIPLoader(self.model_name, "cpu")
            temp_components = temp_helper_instance._create_clip_model_internal(
                base_model, model_path, activation_type, final_activation_type
            )

            if temp_components and isinstance(temp_components.get("model"), torch.nn.Module):
                classifier_model = temp_components["model"]
                calculated_size_mb = LoaderBase._calculate_transformer_size_mb(classifier_model)
            else:
                logger.warning(f"一時ロード CLIP '{self.model_name}' から有効な分類器取得失敗。")
        except Exception as e:
            logger.warning(f"一時ロード CLIP 計算中にエラー: {e}", exc_info=False)
            if "calculated_size_mb" not in locals():
                calculated_size_mb = 0.0
        finally:
            temp_components = None
            temp_helper_instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.debug(f"一時ロード CLIP サイズ計算完了: {calculated_size_mb:.2f} MB")
        return calculated_size_mb

    @override
    def _load_components_internal(self, model_path: str, **kwargs: Any) -> CLIPComponents:
        """内部ヘルパーを使用して CLIP コンポーネント一式をロードする。

        kwargs に 'base_model' が必要。
        """
        base_model = cast(str, kwargs.get("base_model"))
        activation_type = cast(str | None, kwargs.get("activation_type"))
        final_activation_type = cast(str | None, kwargs.get("final_activation_type"))
        if not base_model:
            raise ValueError("CLIP loader requires 'base_model' kwarg.")

        components = self._create_clip_model_internal(
            base_model, model_path, activation_type, final_activation_type
        )
        if components is None:
            raise ModelLoadError(f"CLIPモデル '{self.model_name}' のコンポーネント作成失敗。")
        return components
