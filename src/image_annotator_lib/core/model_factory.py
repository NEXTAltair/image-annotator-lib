"""モデルロード・キャッシュ管理のファサードクラス。

ModelLoad は各フレームワーク固有ローダーへの公開インターフェースを提供する。
実際のロード・状態管理ロジックは core/loaders/ パッケージに委譲される。

後方互換性のため、既存の公開 API シグネチャは完全に維持される。

Dependencies:
    - core.loaders: フレームワーク固有ローダーと LoaderBase
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from .loaders import (
    CLIPLoader,
    LoaderBase,
    ONNXLoader,
    TensorFlowLoader,
    TransformersLoader,
    TransformersPipelineLoader,
)
from .types import (
    CLIPComponents,
    ONNXComponents,
    TensorFlowComponents,
    TransformersComponents,
    TransformersPipelineComponents,
)
from .utils import logger


class ModelLoad:
    """モデルのロード、キャッシュ、メモリ管理のファサード。

    各フレームワーク固有のロードは loaders パッケージに委譲し、
    このクラスは公開 API と ClassVar への後方互換アクセスを提供する。

    公開インターフェース:
        - load_transformers_components / load_transformers_pipeline_components
        - load_onnx_components / load_tensorflow_components / load_clip_components
        - cache_to_main_memory / restore_model_to_cuda
        - release_model / release_model_components
        - get_model_size / get_max_cache_size
    """

    # --- ClassVar (LoaderBase への委譲プロパティ) ---
    # 後方互換性のためにクラス属性として公開。
    # 実体は LoaderBase の ClassVar。
    _MODEL_STATES: ClassVar[dict[str, str]] = LoaderBase._MODEL_STATES
    _MEMORY_USAGE: ClassVar[dict[str, float]] = LoaderBase._MEMORY_USAGE
    _MODEL_LAST_USED: ClassVar[dict[str, float]] = LoaderBase._MODEL_LAST_USED
    _CACHE_RATIO: ClassVar[float] = LoaderBase._CACHE_RATIO
    _MODEL_SIZES: ClassVar[dict[str, float]] = LoaderBase._MODEL_SIZES

    # --- 内部ヘルパーの後方互換委譲 ---
    # classmethod/staticmethod ディスクリプタを直接コピーし、
    # cls が ModelLoad に正しく解決されるようにする。

    _get_model_size_from_config = LoaderBase.__dict__["_get_model_size_from_config"]
    _calculate_file_size_mb = LoaderBase.__dict__["_calculate_file_size_mb"]
    _calculate_dir_size_mb = LoaderBase.__dict__["_calculate_dir_size_mb"]
    _calculate_transformer_size_mb = LoaderBase.__dict__["_calculate_transformer_size_mb"]
    _save_size_to_config = LoaderBase.__dict__["_save_size_to_config"]
    _get_or_calculate_size = LoaderBase.__dict__["_get_or_calculate_size"]
    _get_current_cache_usage = LoaderBase.__dict__["_get_current_cache_usage"]
    _get_models_sorted_by_last_used = LoaderBase.__dict__["_get_models_sorted_by_last_used"]
    _get_model_memory_usage = LoaderBase.__dict__["_get_model_memory_usage"]
    _check_memory_before_load = LoaderBase.__dict__["_check_memory_before_load"]
    _get_max_cache_size = LoaderBase.__dict__["_get_max_cache_size"]
    _clear_cache_internal = LoaderBase.__dict__["_clear_cache_internal"]
    _update_model_state = LoaderBase.__dict__["_update_model_state"]
    _get_model_state = LoaderBase.__dict__["_get_model_state"]
    _move_components_to_device = LoaderBase.__dict__["_move_components_to_device"]
    _release_model_state = LoaderBase.__dict__["_release_model_state"]
    _release_model_internal = LoaderBase.__dict__["_release_model_internal"]
    _handle_load_error = LoaderBase.__dict__["_handle_load_error"]

    # --- 後方互換: 内部ローダークラスへのエイリアス ---
    _BaseLoaderInternal = LoaderBase
    _TransformersLoader = TransformersLoader
    _TransformersPipelineLoader = TransformersPipelineLoader
    _ONNXLoader = ONNXLoader
    _TensorFlowLoader = TensorFlowLoader
    _CLIPLoader = CLIPLoader

    # --- 公開 API ---

    @staticmethod
    def get_model_size(model_name: str) -> float | None:
        """Config からモデルの推定メモリ使用量 (MB) を取得する。"""
        return ModelLoad._get_model_size_from_config(model_name)

    @staticmethod
    def get_max_cache_size() -> float:
        """計算された最大キャッシュサイズ (MB) を取得する。"""
        return ModelLoad._get_max_cache_size()

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントをメインメモリ (CPU) に移動してキャッシュする。

        Args:
            model_name: モデル名。
            components: モデルコンポーネント辞書 (GPU 上の可能性あり)。

        Returns:
            CPU 上のコンポーネント辞書。失敗時は空の dict。
        """
        state = ModelLoad._get_model_state(model_name)
        if state == "on_cpu":
            logger.debug(f"モデル '{model_name}' は既にCPUキャッシュにあります。")
            ModelLoad._update_model_state(model_name)
            return components

        model_size = ModelLoad._MODEL_SIZES.get(model_name, 0.0)
        can_cache = True
        if model_size <= 0:
            logger.warning(f"モデル '{model_name}' サイズ不明/0、CPUキャッシュ前の容量確認スキップ。")
        else:
            max_cache = ModelLoad._get_max_cache_size()
            current_usage = ModelLoad._get_current_cache_usage()
            usage_without_model = current_usage - ModelLoad._get_model_memory_usage(model_name)
            if usage_without_model + model_size > max_cache:
                logger.warning(
                    f"CPUキャッシュ不可: モデル '{model_name}'"
                    f" ({model_size / 1024:.3f}GB) を追加するとキャッシュ容量超過。解放試行..."
                )
                if not ModelLoad._clear_cache_internal(model_name, model_size):
                    can_cache = False

        if not can_cache:
            logger.error(f"モデル '{model_name}' CPUキャッシュ失敗: 十分な空き容量なし。")
            ModelLoad._release_model_state(model_name)
            return {}

        try:
            ModelLoad._move_components_to_device(components, "cpu")
            ModelLoad._update_model_state(model_name, "cpu", "cached_cpu", model_size)
            logger.info(f"モデル '{model_name}' をCPUにキャッシュしました。")
            return components
        except Exception as e:
            logger.error(f"モデル '{model_name}' CPUキャッシュ中にエラー: {e}", exc_info=True)
            ModelLoad._release_model_state(model_name)
            return {}

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> TransformersComponents | None:
        """Transformers モデルコンポーネントをロードする。"""
        loader = TransformersLoader(model_name, device)
        result = loader.load_components(model_path)
        return cast(TransformersComponents | None, result)

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> TransformersPipelineComponents | None:
        """Transformers Pipeline コンポーネントをロードする。"""
        loader = TransformersPipelineLoader(model_name, device)
        result = loader.load_components(model_path, task=task, batch_size=batch_size)
        return cast(TransformersPipelineComponents | None, result)

    @staticmethod
    def load_onnx_components(model_name: str, model_path: str, device: str) -> ONNXComponents | None:
        """ONNX モデルコンポーネントをロードする。"""
        loader = ONNXLoader(model_name, device)
        result = loader.load_components(model_path)
        return cast(ONNXComponents | None, result)

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,
        model_format: str,
    ) -> TensorFlowComponents | None:
        """TensorFlow モデルコンポーネントをロードする。"""
        loader = TensorFlowLoader(model_name, device)
        result = loader.load_components(model_path, model_format=model_format)
        return cast(TensorFlowComponents | None, result)

    @staticmethod
    def load_clip_components(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> CLIPComponents | None:
        """CLIP モデルコンポーネントをロードする。"""
        loader = CLIPLoader(model_name, device)
        result = loader.load_components(
            model_path,
            base_model=base_model,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )
        return cast(CLIPComponents | None, result)

    @staticmethod
    def restore_model_to_cuda(
        model_name: str,
        components: dict[str, Any],
        device: str,
    ) -> dict[str, Any] | None:
        """モデルを指定 CUDA デバイスに復元する。

        Args:
            model_name: モデル名。
            components: モデルコンポーネント辞書 (CPU 上の可能性あり)。
            device: ターゲット CUDA デバイス (例: "cuda", "cuda:0")。

        Returns:
            ターゲットデバイスに移動したコンポーネント辞書。失敗時は None。
        """
        state = ModelLoad._get_model_state(model_name)
        if not state:
            logger.warning(f"モデル '{model_name}' 状態不明。CUDA復元スキップ。")
            return None
        if state == f"on_{device}":
            logger.debug(f"モデル '{model_name}' は既に {device} にあります。")
            ModelLoad._update_model_state(model_name)
            return components

        logger.info(f"モデル '{model_name}' を {device} に復元中...")
        model_size = ModelLoad._MODEL_SIZES.get(model_name, 0.0)

        if not ModelLoad._check_memory_before_load(model_size, model_name):
            logger.error(f"CUDA復元失敗: メモリ不足 ({model_name} -> {device})")
            return None

        if model_size > 0:
            if not ModelLoad._clear_cache_internal(model_name, model_size):
                logger.error(f"CUDA復元失敗: キャッシュ解放失敗 ({model_name} -> {device})")
                return None

        try:
            ModelLoad._move_components_to_device(components, device)
            ModelLoad._update_model_state(model_name, device, "loaded", model_size)
            logger.info(f"モデル '{model_name}' を {device} に復元完了。")
            return components
        except Exception as e:
            logger.error(f"モデル '{model_name}' の {device} への復元中にエラー: {e}", exc_info=True)
            try:
                logger.warning(f"CUDA復元エラー後、CPUへのフォールバック試行 ({model_name})...")
                ModelLoad._move_components_to_device(components, "cpu")
                ModelLoad._update_model_state(model_name, "cpu", "cached_cpu", model_size)
                logger.warning(f"CUDA復元エラーのため、モデル '{model_name}' をCPUに戻しました。")
            except Exception as fallback_e:
                logger.error(f"CPUフォールバック中にエラー ({model_name}): {fallback_e}", exc_info=True)
                ModelLoad._release_model_state(model_name)
            return None

    @staticmethod
    def release_model(model_name: str) -> None:
        """モデルをキャッシュから解放し、リソースをクリーンアップする。"""
        ModelLoad._release_model_internal(model_name, components=None)

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントと関連する状態を解放する。

        Args:
            model_name: モデル名。
            components: リソース解放が必要なコンポーネント辞書。

        Returns:
            空の辞書 (解放完了を示す)。
        """
        ModelLoad._release_model_internal(model_name, components)
        return {}
