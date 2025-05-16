# Stub file for image_annotator_lib.core.factory
import logging
from pathlib import Path
from typing import Any, TypedDict

import onnxruntime as ort
import tensorflow as tf
import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.clip import CLIPModel, CLIPProcessor
from transformers.pipelines.base import Pipeline

# --- TypedDict Definitions (Copied from implementation for clarity in stub) ---
class TransformersComponents(TypedDict):
    model: AutoModelForVision2Seq
    processor: AutoProcessor

class TransformersPipelineComponents(TypedDict):
    pipeline: Pipeline

class ONNXComponents(TypedDict):
    session: ort.InferenceSession
    csv_path: Path

class TensorFlowComponents(TypedDict):
    model_dir: Path
    model: tf.Module | tf.keras.Model

class CLIPComponents(TypedDict):
    model: nn.Module
    processor: CLIPProcessor
    clip_model: CLIPModel

# --- Classifier Stub ---
# Assuming Classifier is defined elsewhere or here temporarily
class Classifier(torch.nn.Module): ...

# --- ModelLoad Stub ---
class ModelLoad:
    """Handles loading, caching, and memory management for various model types. / 様々なモデルタイプのロード、キャッシュ、メモリ管理を扱います。"""

    _MODEL_STATES: dict[str, str]
    _MEMORY_USAGE: dict[str, float]
    _MODEL_LAST_USED: dict[str, float]
    _CACHE_RATIO: float
    _MODEL_SIZES: dict[str, float]
    logger: logging.Logger

    @staticmethod
    def get_model_size(model_name: str) -> float | None:
        """モデルの推定メモリ使用量を取得（MB単位）。
        Gets the estimated memory usage of a model in megabytes (MB).

        Args:
            model_name (str): The name of the model. / モデルの名前。

        Returns:
            Optional[float]: Estimated memory usage in MB, or None if unknown. / 推定メモリ使用量 (MB)、不明な場合は None。
        """
        ...

    @staticmethod
    def get_max_cache_size() -> float:
        """システムの最大メモリに基づいてキャッシュサイズを計算します (MB単位)。
        Calculates the cache size based on the system's maximum memory (in MB).

        Returns:
            float: The calculated maximum cache size in MB. / 計算された最大キャッシュサイズ (MB)。
        """
        ...

    @staticmethod
    def cache_to_main_memory(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネントをメインメモリ (CPU) にキャッシュします。
        Caches the model components to main memory (CPU).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            components (dict[str, Any]): The dictionary containing model components. / モデルコンポーネントを含む辞書。

        Returns:
            dict[str, Any]: The components dictionary (potentially modified in place, or empty on failure). / コンポーネント辞書 (インプレースで変更される可能性あり、失敗時は空)。
        """
        ...

    @staticmethod
    def load_transformers_components(
        model_name: str, model_path: str, device: str
    ) -> TransformersComponents | None:
        """Transformer モデルのコンポーネント (モデル、プロセッサ) をロードします。
        Loads Transformer model components (model, processor).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path or identifier for the model. / モデルのパスまたは識別子。
            device (str): The device to load the model onto ("cuda" or "cpu"). / モデルをロードするデバイス ("cuda" または "cpu")。

        Returns:
            Optional[TransformersComponents]: Dictionary with "model" and "processor", or None if load fails or already loaded. / "model" と "processor" を含む辞書、またはロード失敗･ロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_onnx_components(
        model_name: str,
        model_path: str,
        device: str,  # Implementation takes model_path, not model_repo
    ) -> ONNXComponents | None:
        """ONNX モデルのコンポーネント (セッション、CSVパス) をロードします。
        Loads ONNX model components (session, csv_path).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path or identifier for the ONNX model. / ONNX モデルのパスまたは識別子。
            device (str): The device to run inference on ("cuda" or "cpu"). / 推論を実行するデバイス ("cuda" または "cpu")。

        Returns:
            Optional[ONNXComponents]: Dictionary with "session" and "csv_path", or None if load fails or already loaded. / "session" と "csv_path" を含む辞書、またはロード失敗･ロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_tensorflow_components(
        model_name: str,
        model_path: str,
        device: str,
        model_format: str,  # No default in implementation
    ) -> TensorFlowComponents | None:
        """TensorFlow モデルのコンポーネント (モデル、モデルディレクトリ) をロードします。
        Loads TensorFlow model components (model, model_dir).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path to the model file or directory. / モデルファイルまたはディレクトリへのパス。
            device (str): Target device (currently unused for TF). / ターゲットデバイス (現在 TF では未使用)。
            model_format (str): The format of the model ("h5", "saved_model", "pb"). / モデルのフォーマット ("h5", "saved_model", "pb")。

        Returns:
            Optional[TensorFlowComponents]: Dictionary with "model" and "model_dir", or None if load fails or already loaded. / "model" と "model_dir" を含む辞書、またはロード失敗･ロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_transformers_pipeline_components(
        task: str, model_name: str, model_path: str, device: str, batch_size: int
    ) -> TransformersPipelineComponents | None:
        """Hugging Face Pipeline モデルのコンポーネントをロードします。
        Loads Hugging Face Pipeline model components.

        Args:
            task (str): The task for the pipeline. / パイプラインのタスク。
            model_name (str): The name of the model. / モデルの名前。
            model_path (str): Path or identifier for the pipeline model. / パイプラインモデルのパスまたは識別子。
            device (str): The device to load the pipeline onto ("cuda" or "cpu"). / パイプラインをロードするデバイス ("cuda" または "cpu")。
            batch_size (int): Batch size for the pipeline. / パイプラインのバッチサイズ。


        Returns:
            Optional[TransformersPipelineComponents]: Dictionary with "pipeline", or None if load fails or already loaded. / "pipeline" を含む辞書、またはロード失敗･ロード済みの場合は None。
        """
        ...

    @staticmethod
    def load_clip_components(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: str | None = None,
        final_activation_type: str | None = None,
    ) -> CLIPComponents | None:
        """CLIP ベースの Scorer モデルのコンポーネント (分類器、プロセッサ、CLIPモデル) をロードします。
        Loads CLIP-based Scorer model components (classifier, processor, clip_model).

        Args:
            model_name (str): The name of the scorer model. / スコアラーモデルの名前。
            base_model (str): Path or identifier for the base CLIP model. / ベースとなる CLIP モデルのパスまたは識別子。
            model_path (str): Path to the classifier head weights. / 分類器ヘッドの重みへのパス。
            device (str): The device to load models onto ("cuda" or "cpu"). / モデルをロードするデバイス ("cuda" または "cpu")。
            activation_type (Optional[str]): Type of activation for hidden layers. / 隠れ層の活性化関数のタイプ。
            final_activation_type (Optional[str]): Type of activation for the final layer. / 最終層の活性化関数のタイプ。

        Returns:
            Optional[CLIPComponents]: Dictionary with "model", "processor", "clip_model", or None if load fails or already loaded. / "model", "processor", "clip_model" を含む辞書、またはロード失敗･ロード済みの場合は None。
        """
        ...

    @staticmethod
    def restore_model_to_cuda(
        model_name: str,
        device: str,
        components: dict[str, Any],  # 'model' arg name changed to 'components'
    ) -> dict[str, Any] | None:  # Implementation returns None on failure
        """キャッシュされたモデルを指定された CUDA デバイスに復元します。
        Restores a cached model to the specified CUDA device.

        Args:
            model_name (str): The name of the model. / モデルの名前。
            device (str): The target CUDA device (e.g., "cuda:0"). / ターゲット CUDA デバイス (例: "cuda:0")。
            components (dict[str, Any]): The dictionary containing model components (potentially on CPU). / モデルコンポーネントを含む辞書 (CPU 上にある可能性あり)。

        Returns:
            Optional[dict[str, Any]]: The components dictionary with components moved to the target device, or None on failure. / コンポーネントがターゲットデバイスに移動されたコンポーネント辞書、失敗時は None。
        """
        ...

    @staticmethod
    def release_model(model_name: str) -> None:
        """指定されたモデルをキャッシュから解放し、関連リソース (GPUメモリなど) をクリアします。
        Releases the specified model from the cache and clears associated resources (like GPU memory).

        Args:
            model_name (str): The name of the model to release. / 解放するモデルの名前。
        """
        ...

    @staticmethod
    def release_model_components(model_name: str, components: dict[str, Any]) -> dict[str, Any]:
        """モデルコンポーネント (ONNX セッションなど) のリソースを明示的に解放します。
        Explicitly releases resources held by model components (like ONNX sessions).

        Args:
            model_name (str): The name of the model. / モデルの名前。
            components (dict[str, Any]): The dictionary containing model components. / モデルコンポーネントを含む辞書。

        Returns:
            dict[str, Any]: An empty dictionary, as components are released. / コンポーネントが解放されるため、空の辞書。
        """
        ...
