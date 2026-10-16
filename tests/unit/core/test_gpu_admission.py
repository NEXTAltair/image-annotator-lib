"""GPU admission and requested-device recovery with real loader/lifecycle state."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from image_annotator_lib.core.base.clip import ClipBaseAnnotator
from image_annotator_lib.core.base.onnx import ONNXBaseAnnotator
from image_annotator_lib.core.base.pipeline import PipelineBaseAnnotator
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.loaders.clip_loader import CLIPLoader
from image_annotator_lib.core.loaders.loader_base import LoaderBase
from image_annotator_lib.core.loaders.onnx_loader import ONNXLoader
from image_annotator_lib.core.loaders.transformers_loader import (
    TransformersLoader,
    TransformersPipelineLoader,
)
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult
from image_annotator_lib.exceptions.errors import OutOfMemoryError
from image_annotator_lib.model_class.tagger_transformers import BLIPTagger

pytestmark = pytest.mark.unit


class DeviceInput:
    """Fake a native tensor while enforcing the model/input device contract."""

    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def norm(self, **kwargs):
        return 1

    def __truediv__(self, other):
        return self


class Processor:
    def __call__(self, **kwargs):
        class Batch(dict):
            def to(self, device):
                for value in self.values():
                    value.to(device)
                return self

        return Batch(pixel_values=DeviceInput())

    def batch_decode(self, outputs, **kwargs):
        return ["test caption"]


class NativeModel:
    def __init__(self, device, inference_devices):
        self.device = device
        self.inference_devices = inference_devices

    def generate(self, pixel_values, **kwargs):
        assert pixel_values.device == self.device
        self.inference_devices.append(self.device)
        return [1]

    def get_image_features(self, pixel_values):
        assert pixel_values.device == self.device
        return pixel_values

    def __call__(self, features):
        import torch

        assert features.device == self.device
        self.inference_devices.append(self.device)
        return torch.tensor([[0.75]])


class NativePipeline:
    def __init__(self, model):
        self.model = model

    def __call__(self, images):
        self.model.inference_devices.append(self.model.device)
        return ["test caption" for _ in images]


class CaptionPipeline(PipelineBaseAnnotator):
    def _format_predictions(self, outputs):
        return [
            UnifiedAnnotationResult(
                model_name=self.model_name,
                capabilities={TaskCapability.CAPTIONS},
                captions=[caption],
                framework="pytorch",
            )
            for caption in outputs
        ]


class TaggedONNX(ONNXBaseAnnotator):
    def _load_tags(self):
        self.all_tags = ["test-tag"]


def download_components(kind, device, inference_devices):
    model = NativeModel(device, inference_devices)
    if kind == "transformers":
        return {"model": model, "processor": Processor()}
    if kind == "pipeline":
        return {"pipeline": NativePipeline(model)}
    if kind == "clip":
        return {"model": model, "clip_model": model, "processor": Processor()}
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[1, 3, 448, 448])]
    session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return {"session": session}


def move_components(components, device):
    for key in ("model", "clip_model"):
        if key in components:
            components[key].device = device
    if "pipeline" in components:
        components["pipeline"].model.device = device


@pytest.fixture
def backend_factory(monkeypatch):
    """Only fake native allocations, tensor moves and hardware probes."""
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: d)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda device: 0)
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda device: 0)
    for owner in (LoaderBase, ModelLoad):
        monkeypatch.setattr(owner, "_check_memory_before_load", classmethod(lambda cls, *a: True))
        monkeypatch.setattr(owner, "_get_max_cache_size", classmethod(lambda cls: 8192.0))
    allocations = {}

    def native_load(loader, **kwargs):
        return allocations[loader.model_name](loader.device)

    for loader_class in (TransformersLoader, TransformersPipelineLoader, CLIPLoader, ONNXLoader):
        monkeypatch.setattr(loader_class, "_load_components_internal", native_load)

    monkeypatch.setattr(ModelLoad, "_move_components_to_device", move_components)

    def create(kind, name):
        config = {
            "class": "BLIPTagger" if kind == "transformers" else "AestheticShadow",
            "model_path": "fake/model",
            "device": "cuda",
            "estimated_size_gb": 1.0,
            "capabilities": ["scores"] if kind == "clip" else ["captions"],
        }
        if kind == "clip":
            config["base_model"] = "fake/clip"
        monkeypatch.setitem(config_registry._merged_config_data, name, config)
        inference_devices = []

        loads = Mock(side_effect=lambda device: download_components(kind, device, inference_devices))
        allocations[name] = loads
        annotator_class = {
            "transformers": BLIPTagger,
            "pipeline": CaptionPipeline,
            "clip": ClipBaseAnnotator,
            "onnx": TaggedONNX,
        }[kind]
        return annotator_class(name), loads, inference_devices

    return create


def assert_prediction(annotator):
    result = annotator.predict([Image.new("RGB", (4, 4))])[0]
    assert result.error is None
    if isinstance(annotator, ClipBaseAnnotator):
        assert result.scores == {"aesthetic": 0.75}
    else:
        assert result.captions == ["test caption"]


@pytest.mark.parametrize("kind", ["pipeline", "clip", "onnx"])
@pytest.mark.parametrize("active", [False, True], ids=["idle-evicted", "active-protected"])
def test_fresh_gpu_backend_admits_before_native_allocation(backend_factory, monkeypatch, kind, active):
    retained, retained_loads, _ = backend_factory("transformers", "retained-transformer")
    with retained:
        assert_prediction(retained)
    retained_components = retained.components
    assert ModelLoad._get_model_state(retained.model_name) == "on_cuda"
    fresh, allocations, _ = backend_factory(kind, f"fresh-{kind}")
    download = allocations.side_effect

    def allocate(device):
        if retained.components:
            raise OutOfMemoryError("native allocation before idle eviction")
        return download(device)

    allocations.side_effect = allocate
    with retained if active else nullcontext():
        monkeypatch.setattr(
            "torch.cuda.mem_get_info",
            lambda device: (512 * 1024**2 if retained.components else 2 * 1024**3, 2 * 1024**3),
        )
        if active:
            with pytest.raises(OutOfMemoryError):
                with fresh:
                    pass
            allocations.assert_not_called()
            assert retained.components is retained_components
            assert ModelLoad._get_model_state(retained.model_name) == "on_cuda"
            assert ModelLoad._MODEL_OWNERS[retained.model_name] is retained
            assert_prediction(retained)
        else:
            with fresh:
                assert retained.components is None
                assert retained_components == {}
                assert ModelLoad._get_model_state(retained.model_name) is None
                assert retained.model_name not in ModelLoad._MEMORY_USAGE
                assert retained.model_name not in ModelLoad._HOST_MEMORY_USAGE
                assert ModelLoad._get_model_state(fresh.model_name) == "on_cuda"
            allocations.assert_called_once_with("cuda")
        assert not ModelLoad.is_model_active(fresh.model_name)
    retained_loads.assert_called_once_with("cuda")
    assert not ModelLoad._ACTIVE_CONTEXTS
    assert not ModelLoad._ACTIVE_THREADS


@pytest.mark.parametrize("kind", ["pipeline", "clip"])
@pytest.mark.parametrize("release_before_retry", [False, True], ids=["restore-retained", "release-reload"])
def test_cpu_fallback_keeps_nested_placement_and_retries_configured_cuda(
    backend_factory, monkeypatch, kind, release_before_retry
):
    annotator, loads, inference_devices = backend_factory(kind, f"recover-{kind}")
    with annotator:
        assert_prediction(annotator)
    components = annotator.components
    assert ModelLoad._get_model_state(annotator.model_name) == "on_cpu"
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (512 * 1024**2, 16 * 1024**3))
    with annotator:
        assert annotator.components is components
        assert annotator.device == "cpu"
        assert_prediction(annotator)
        assert inference_devices[-1] == "cpu"
        assert ModelLoad._get_model_state(annotator.model_name) == "on_cpu"
        # GPU recovery must not migrate an outer context's live CPU generation.
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
        with annotator:
            assert annotator.device == "cpu"
            assert annotator.components is components
            assert ModelLoad._get_model_state(annotator.model_name) == "on_cpu"
            assert_prediction(annotator)
            assert inference_devices[-1] == "cpu"
        assert annotator.device == "cpu"
        assert annotator.components is components
    if release_before_retry:
        ModelLoad.release_model(annotator.model_name)
        assert annotator.components is None
        assert components == {}
    with annotator:
        assert annotator.device == "cuda"
        assert ModelLoad._get_model_state(annotator.model_name) == "on_cuda"
        assert_prediction(annotator)
        assert inference_devices[-1] == "cuda"
        if not release_before_retry:
            assert annotator.components is components
    assert loads.call_count == (2 if release_before_retry else 1)
    assert [call.args[0] for call in loads.call_args_list] == ["cuda"] * loads.call_count
    assert not ModelLoad._ACTIVE_CONTEXTS
    assert not ModelLoad._ACTIVE_THREADS
