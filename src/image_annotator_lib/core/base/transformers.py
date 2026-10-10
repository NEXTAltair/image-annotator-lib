"""Transformers ライブラリを使用するモデル用の基底クラス。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from PIL import Image

# Issue #59: module-level `import torch` / `from transformers.models.clip import CLIPProcessor` は
# transformers が torch を eager に load するため CUDA driver 不在 + triton 在り環境で SIGSEGV を
# 引き起こす。型ヒントは TYPE_CHECKING 内、runtime 利用箇所は関数内 import に分離する。
if TYPE_CHECKING:
    import torch

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
        # device 判定はローカル ML 系 base class の責務 (Issue #35 で BaseAnnotator から移譲)
        from ..utils import determine_effective_device

        self.device = determine_effective_device(self._config.device, self.model_name)
        self._requested_device = self._config.device
        # 設定ファイルから追加パラメータを取得
        self.max_length = config_registry.get(self.model_name, "max_length", 75)
        self.processor_path = config_registry.get(self.model_name, "processor_path")
        # components の型ヒントを具体的に指定
        self.components: TransformersComponents | None = None
        self._prepared = False
        # model_path の型を保証 (LocalMLModelConfig では str を期待)
        self.model_path = str(self.model_path) if isinstance(self.model_path, str) else ""

    def __enter__(self) -> TransformersBaseAnnotator:
        """準備済みモデルを保持し、必要な場合だけロード/復元する (Issue #165)。"""
        try:
            if not self.model_path:
                raise ConfigurationError(f"モデル '{self.model_name}' の model_path が設定されていません。")

            state = ModelLoad._get_model_state(self.model_name)
            if self._prepared and self.components and state == f"on_{self.device}":
                ModelLoad._update_model_state(self.model_name)
                logger.debug(f"Transformers components already prepared for '{self.model_name}'; reusing")
                return self

            if state and not self._prepared:
                # 別インスタンスが所有する実体も解放してからロードする。
                ModelLoad.release_model(self.model_name)
            elif not state:
                self._release_retained_components()

            if not self._prepared:
                from ..utils import determine_effective_device

                self.device = determine_effective_device(self._requested_device, self.model_name)
            ModelLoad._make_cuda_room(self.model_name, self.device)
            loaded = ModelLoad.load_transformers_components(
                self.model_name, str(self.model_path), str(self.device)
            )
            if loaded is not None:
                self.components = loaded
            if (
                not self.components
                or self.components.get("model") is None
                or self.components.get("processor") is None
            ):
                raise ModelLoadError(
                    f"Failed to load components for model '{self.model_name}'.", model_path=self.model_path
                )

            self._restore_components()

            self._prepared = True
            ModelLoad.register_component_releaser(self.model_name, self._release_retained_components)
        except Exception:
            self._discard_components()
            raise
        return self

    def _restore_components(self) -> None:
        """保持コンポーネントを復元し、CPU fallback 時は入力のデバイスも揃える。"""
        had_loaded_state = ModelLoad._get_model_state(self.model_name) is not None
        if self.components is None:
            raise ModelLoadError(f"Missing components for model {self.model_name!r}.")
        restored = ModelLoad.restore_model_to_cuda(self.model_name, dict(self.components), str(self.device))
        if restored is not None:
            self.components = cast(TransformersComponents, restored)
        else:
            state = ModelLoad._get_model_state(self.model_name)
            if had_loaded_state and state is None:
                raise ModelLoadError(
                    f"Failed to restore components for model '{self.model_name}'.",
                    model_path=self.model_path,
                )
            if state == "on_cpu":
                self.device = "cpu"
            logger.warning(f"Model '{self.model_name}' will run on CPU after restoration fallback.")

    def _release_retained_components(self) -> None:
        """LRU / 明示解放から呼ばれ、モデルとプロセッサの参照を手放す。"""
        components = self.components
        self.components = None
        self._prepared = False
        if components:
            cast(dict[str, Any], components).clear()

    def _discard_components(self) -> None:
        """準備失敗や OOM 後に、次回クリーンなロードを要求する。"""
        self._release_retained_components()
        ModelLoad.unregister_component_releaser(self.model_name)
        try:
            ModelLoad.release_model(self.model_name)
        except Exception as e:
            logger.error(f"コンポーネント解放失敗 ({self.model_name}): {e}")

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """正常終了では保持を続け、解放は ModelLoad の LRU / release_model に委ねる。"""
        if exc_type is not None:
            self._discard_components()

    def _execute_pipeline(self, images: list[Image.Image]) -> list:
        """predict が OOM をエラー結果に変換する前に、保持中の実体を無効化する。"""
        import torch

        try:
            return super()._execute_pipeline(images)
        except (OutOfMemoryError, MemoryError, torch.cuda.OutOfMemoryError) as e:
            self._discard_components()
            raise OutOfMemoryError(
                f"モデル '{self.model_name}' の処理中にメモリ不足が発生しました。"
            ) from e

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
        import torch

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

        capabilities に応じて captions / tags の両方または片方にデコード文字列を格納する。
        TAGS と CAPTIONS の両方が capabilities に含まれるマルチタスクモデルでは
        両フィールドに同じ payload を入れて情報損失を防ぐ
        (UnifiedAnnotationResult は両フィールド同時設定を許可している)。

        Returns:
            list[UnifiedAnnotationResult]: 統一フォーマットのアノテーション結果リスト
        """
        # 関数レベルでインポート (循環インポート回避 + Issue #59 torch eager load 回避)
        from transformers.models.clip import CLIPProcessor

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
                captions_value = payload if (TaskCapability.CAPTIONS in capabilities and payload) else None
                tags_value = payload if (TaskCapability.TAGS in capabilities and payload) else None
                result = UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    captions=captions_value,
                    tags=tags_value,
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
