"""Transformers ライブラリを使用するモデル用の基底クラス。"""

from typing import Any, cast

import torch
from PIL import Image
from transformers.models.clip import CLIPProcessor

# --- ローカルインポート ---
from ...exceptions.errors import ConfigurationError, ModelLoadError, OutOfMemoryError
from ..config import config_registry
from ..model_factory import ModelLoad
from ..types import TransformersComponents, UnifiedAnnotationResult
from ..utils import logger
from .annotator import BaseAnnotator


class TransformersBaseAnnotator(BaseAnnotator):
    """Transformers ライブラリを使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        """TransformerModel を初期化します。
        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name)
        # 設定ファイルから追加パラメータを取得
        self.max_length = config_registry.get(self.model_name, "max_length", 75)
        self.processor_path = config_registry.get(self.model_name, "processor_path")
        # components の型ヒントを具体的に指定
        self.components: TransformersComponents | None = None
        # device, model_pathの型を保証
        self.device = str(self.device) if isinstance(self.device, str) else "cpu"
        self.model_path = str(self.model_path) if isinstance(self.model_path, str) else ""

    def __enter__(self) -> "TransformersBaseAnnotator":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        メモリ不足エラーをハンドリングし、VRAM使用量をログに出力
        """
        try:
            if self.model_path is None:
                raise ConfigurationError(f"モデル '{self.model_name}' の model_path が設定されていません。")

            # --- モデルロード処理 ---
            logger.info(f"モデルコンポーネントのロード試行: {self.model_name} をデバイス {self.device} へ")
            loaded_components = ModelLoad.load_transformers_components(
                self.model_name,
                str(self.model_path),
                str(self.device),
            )

            # Load Failure Detection (致命的エラー)
            if not loaded_components:
                error_msg = f"Failed to load components for model '{self.model_name}'."
                logger.error(error_msg)
                raise ModelLoadError(error_msg, model_path=self.model_path)

            self.components = loaded_components  # ← 必ず有効な components が代入される
            logger.info(f"モデルコンポーネントのロード成功: {self.model_name}")

            # --- CUDAへの復元処理 ---
            logger.debug(f"モデル {self.model_name} を {self.device} へ復元試行")
            # Restoration Attempt (失敗しても継続可能)
            restored_components = ModelLoad.restore_model_to_cuda(
                self.model_name,
                dict(self.components),
                str(self.device),
            )

            # Restoration Failure Handling (警告のみ、CPU で継続)
            if restored_components is not None:
                if not isinstance(restored_components, dict):
                    raise TypeError("restored_componentsはdict型である必要があります。")
                self.components = cast(TransformersComponents, restored_components)
                logger.debug(f"モデル {self.model_name} の {self.device} への復元成功")
            else:
                # restore_model_to_cuda() は既に CPU フォールバック済み（None 返却）
                # self.components は CPU 版を維持したまま継続
                logger.warning(
                    f"Model '{self.model_name}' will run on CPU. "
                    f"CUDA restoration failed but CPU fallback is already complete."
                )
        except (OutOfMemoryError, MemoryError, OSError) as mem_e:
            # メモリ関連エラーはそのまま上位に伝播させる
            raise mem_e
        except Exception as e:
            # メモリ関連以外の予期せぬエラー
            logger.exception(f"モデル {self.model_name} のロード/復元中に予期せぬエラーが発生: {e}")
            raise

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if self.components and isinstance(self.components, dict):
            cached_components = ModelLoad.cache_to_main_memory(self.model_name, dict(self.components))
            self.components = cast(TransformersComponents, cached_components)
        else:
            self.components = None

    def _preprocess_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        results = []
        if not self.components or "processor" not in self.components:
            raise ConfigurationError("Transformersプロセッサがロードされていません。")
        processor = self.components["processor"]
        if not callable(processor):  # callable かどうかでチェック
            raise ConfigurationError(f"プロセッサが呼び出し可能ではありません: {type(processor)}")

        for image in images:
            # プロセッサの出力を取得してデバイスに移動
            processed_output = processor(images=image, return_tensors="pt").to(self.device)
            logger.debug(f"辞書のキー: {processed_output.keys()}")
            results.append(processed_output)
        return results

    def _run_inference(self, processed: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """前処理済みバッチで推論を実行します (Transformers用)。"""
        if not self.components or "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("Transformer モデルがロードされていません。")
        model: Any = self.components["model"]
        outputs = []
        # generateメソッドの一般的な引数やモデルのforwardメソッドの引数を想定
        KNOWN_ARGS = {
            "input_ids",
            "pixel_values",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "labels",
        }

        with torch.no_grad():
            for processed_image in processed:
                # モデルに渡す引数をフィルタリング
                model_kwargs = {k: v for k, v in processed_image.items() if k in KNOWN_ARGS}

                if hasattr(model, "generate"):
                    # generateメソッドにmax_lengthを追加
                    if self.max_length is not None:  # Noneチェック
                        # model_kwargs の型は Dict[str, Tensor] だが、generateは他の型の引数も取る
                        model_kwargs_any: dict[str, Any] = model_kwargs  # Anyにキャスト
                        model_kwargs_any["max_length"] = self.max_length
                        model_out = model.generate(**model_kwargs_any)
                    else:
                        model_out = model(**model_kwargs)
                    if hasattr(model_out, "last_hidden_state"):
                        model_out = model_out.last_hidden_state
                    elif hasattr(model_out, "logits"):
                        model_out = model_out.logits
                outputs.append(model_out)
        return outputs

    def _format_predictions(self, token_ids_list: list[torch.Tensor]) -> list[UnifiedAnnotationResult]:
        """生出力バッチをフォーマットします (Transformers用、テキストデコード)。

        capabilities に応じて captions / tags のどちらか一方にデコード文字列を格納する。
        両方を設定すると capability バリデーションで弾かれるため、CAPTIONS が含まれれば
        captions を優先し、そうでなければ tags にフォールバックする。

        Returns:
            list[UnifiedAnnotationResult]: 統一フォーマットのアノテーション結果リスト
        """
        # 関数レベルでインポート（循環インポート回避）
        from ..types import TaskCapability, UnifiedAnnotationResult
        from ..utils import get_model_capabilities

        if (
            not self.components
            or "processor" not in self.components
            or self.components["processor"] is None
        ):
            raise RuntimeError("Transformer プロセッサがロードされていません。")

        processor_obj = self.components["processor"]
        all_formatted: list[UnifiedAnnotationResult] = []

        # get_model_capabilities を一度だけ呼び出す
        try:
            capabilities = get_model_capabilities(self.model_name)
        except Exception as e:
            logger.error(f"モデル '{self.model_name}' のcapabilities取得に失敗: {e}")
            capabilities = set()

        use_captions = TaskCapability.CAPTIONS in capabilities

        try:
            for token_ids in token_ids_list:
                # batch_decode属性の有無を安全に判定
                batch_decode = getattr(processor_obj, "batch_decode", None)
                if callable(batch_decode):
                    decoded_texts = batch_decode(token_ids, skip_special_tokens=True)
                    if isinstance(decoded_texts, str):
                        decoded_text = decoded_texts
                    elif isinstance(decoded_texts, list) and decoded_texts:
                        decoded_text = decoded_texts[0]
                    else:
                        decoded_text = ""
                elif isinstance(processor_obj, CLIPProcessor):
                    logger.warning(
                        "CLIPProcessorにはbatch_decodeがありません。デコード処理をスキップします。"
                    )
                    decoded_text = ""
                else:
                    raise TypeError(f"Unsupported processor type: {type(processor_obj)}")

                payload = [decoded_text] if decoded_text else None
                result = UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    captions=payload if use_captions else None,
                    tags=payload if not use_captions else None,
                    framework="transformers",
                )
                all_formatted.append(result)

            return all_formatted
        except Exception as e:
            logger.exception(f"予測結果のフォーマット中にエラー発生: {e}")
            # エラー時は1つのエラー結果を返す
            error_result = UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities=capabilities,
                error=f"予測結果のフォーマット失敗: {e}",
                framework="transformers",
            )
            return [error_result]
