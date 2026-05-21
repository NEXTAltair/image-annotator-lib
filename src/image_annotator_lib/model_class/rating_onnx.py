"""Rating-only ONNX annotators."""

import json
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
from PIL import Image

from ..core.base import BaseAnnotator
from ..core.config import config_registry
from ..core.types import RatingPrediction, TaskCapability, UnifiedAnnotationResult
from ..core.utils import determine_effective_device, logger


class AnimeRatingAnnotator(BaseAnnotator):
    """deepghs/anime_rating ONNX annotator.

    Returns model-native Sankaku-style labels: safe, r15, r18.
    """

    rating_source_scheme: ClassVar[str] = "sankaku3"
    default_model_variant: ClassVar[str] = "mobilenetv3_sce_dist"
    default_labels: ClassVar[list[str]] = ["safe", "r15", "r18"]
    input_size: ClassVar[int] = 384
    image_mean: ClassVar[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_std: ClassVar[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.device = determine_effective_device(self._config.device, self.model_name)
        self.model_variant = config_registry.get(
            self.model_name, "model_variant", self.default_model_variant
        )
        self.components: dict[str, Any] | None = None
        self.labels = list(self.default_labels)

    def __enter__(self) -> Self:
        if self.model_path is None:
            raise ValueError(f"モデル '{self.model_name}' の model_path が設定されていません。")

        import onnxruntime as ort

        model_path, meta_path = self._resolve_model_files(self.model_path, self.model_variant)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        logger.info(f"Anime rating ONNX model '{self.model_name}' loading: {model_path}")
        session = ort.InferenceSession(str(model_path), providers=providers)
        self.labels = self._load_labels(meta_path)
        self.components = {
            "session": session,
            "model_path": model_path,
            "meta_path": meta_path,
        }
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        logger.debug(f"Exiting context for anime rating model '{self.model_name}'")
        self.components = None
        if exc_type:
            logger.error(f"Anime rating model '{self.model_name}' context error: {exc_val}")

    @staticmethod
    def _resolve_model_files(model_path: str, model_variant: str) -> tuple[Path, Path]:
        local_path = Path(model_path)
        if local_path.exists():
            if local_path.is_dir() and (local_path / "model.onnx").exists():
                base_path = local_path
            else:
                base_path = local_path / model_variant if local_path.is_dir() else local_path.parent
            return base_path / "model.onnx", base_path / "meta.json"

        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(model_path, f"{model_variant}/model.onnx")
        meta_file = hf_hub_download(model_path, f"{model_variant}/meta.json")
        return Path(model_file), Path(meta_file)

    def _load_labels(self, meta_path: Path) -> list[str]:
        try:
            labels = json.loads(meta_path.read_text(encoding="utf-8")).get("labels")
        except FileNotFoundError:
            logger.warning(f"Anime rating meta file not found: {meta_path}; fallback labels used")
            return list(self.default_labels)

        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            logger.warning(f"Invalid anime rating labels in {meta_path}; fallback labels used")
            return list(self.default_labels)
        return [label.strip().lower().replace(" ", "_") for label in labels]

    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray:
        processed = []
        for image in images:
            rgb = image.convert("RGB").resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            array = (array - self.image_mean) / self.image_std
            processed.append(np.transpose(array, (2, 0, 1)))
        return np.stack(processed, axis=0).astype(np.float32)

    def _run_inference(self, processed: np.ndarray) -> np.ndarray:
        if not self.components or "session" not in self.components:
            raise RuntimeError("Anime rating ONNX session is not loaded")
        session = self.components["session"]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return cast(np.ndarray, session.run([output_name], {input_name: processed})[0])

    def _format_predictions(self, raw_outputs: np.ndarray) -> list[UnifiedAnnotationResult]:
        from ..core.utils import get_model_capabilities

        capabilities = get_model_capabilities(self.model_name)
        probabilities = self._ensure_probabilities(raw_outputs.astype(np.float32))
        results = []
        for scores in probabilities:
            score_map = {label: float(score) for label, score in zip(self.labels, scores, strict=True)}
            raw_label, confidence = max(score_map.items(), key=lambda item: item[1])
            results.append(
                UnifiedAnnotationResult(
                    model_name=self.model_name,
                    capabilities=capabilities,
                    tags=None,
                    captions=None,
                    scores=None,
                    score_labels=None,
                    ratings=[
                        RatingPrediction(
                            raw_label=raw_label,
                            confidence_score=float(confidence),
                            source_scheme=self.rating_source_scheme,
                        )
                    ]
                    if TaskCapability.RATINGS in capabilities
                    else None,
                    framework="onnx",
                    raw_output={"ratings": score_map},
                )
            )
        return results

    @staticmethod
    def _ensure_probabilities(raw_outputs: np.ndarray) -> np.ndarray:
        if raw_outputs.ndim == 1:
            raw_outputs = raw_outputs.reshape(1, -1)
        row_sums = raw_outputs.sum(axis=1)
        if (
            np.all(raw_outputs >= 0.0)
            and np.all(raw_outputs <= 1.0)
            and np.allclose(row_sums, 1.0, atol=1e-3)
        ):
            return raw_outputs

        shifted = raw_outputs - np.max(raw_outputs, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
