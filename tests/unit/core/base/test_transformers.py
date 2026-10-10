"""Transformers retention with real loader state and fake model downloads (Issue #165)."""

import gc
import weakref
from unittest.mock import Mock

import pytest
from PIL import Image

from image_annotator_lib.api import annotate
from image_annotator_lib.core import annotation_runner, registry
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.loaders.loader_base import LoaderBase
from image_annotator_lib.core.loaders.transformers_loader import TransformersLoader
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.exceptions.errors import ModelLoadError, OutOfMemoryError
from image_annotator_lib.model_class.tagger_transformers import BLIPTagger

pytestmark = pytest.mark.unit
MODEL_NAME = "test-transformers-retention"


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, **kwargs):
        return FakeBatch(pixel_values=object())

    def batch_decode(self, tokens, **kwargs):
        return ["test caption"]


class FakeModel:
    def __init__(self, device):
        self.device = device
        self.generate = Mock(return_value=[1])


@pytest.fixture
def backend(monkeypatch):
    """Keep loader/LRU/release real; replace downloads, memory probes, and device moves."""
    for attr in (
        "_MODEL_STATES",
        "_MODEL_SIZES",
        "_MEMORY_USAGE",
        "_MODEL_LAST_USED",
        "_COMPONENT_RELEASERS",
    ):
        data = {}
        monkeypatch.setattr(LoaderBase, attr, data)
        monkeypatch.setattr(ModelLoad, attr, data)
    monkeypatch.setattr(annotation_runner, "_MODEL_INSTANCE_REGISTRY", {})
    monkeypatch.setattr(registry, "_MODEL_CLASS_OBJ_REGISTRY", {MODEL_NAME: BLIPTagger})
    monkeypatch.setitem(
        config_registry._merged_config_data,
        MODEL_NAME,
        {
            "class": "BLIPTagger",
            "model_path": "fake/model",
            "device": "cuda",
            "estimated_size_gb": 1.0,
            "capabilities": ["captions"],
        },
    )
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: d)
    monkeypatch.setattr(LoaderBase, "_check_memory_before_load", classmethod(lambda cls, *a: True))
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 2048.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 2048.0))
    monkeypatch.setattr(ModelLoad, "_check_memory_before_load", classmethod(lambda cls, *a: True))
    loads = Mock(side_effect=lambda device: {"model": FakeModel(device), "processor": FakeProcessor()})
    monkeypatch.setattr(
        TransformersLoader, "_load_components_internal", lambda self, **kwargs: loads(self.device)
    )
    moves = Mock(side_effect=lambda components, device: setattr(components["model"], "device", device))
    monkeypatch.setattr(ModelLoad, "_move_components_to_device", moves)
    yield loads, moves


def call_annotate():
    return annotate(
        [Image.new("RGB", (4, 4))], [MODEL_NAME], phash_list=["image"], api_keys={"openai": "test-key"}
    )["image"][MODEL_NAME]


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_repeated_public_annotate_keeps_model_on_device(backend, device, monkeypatch):
    loads, moves = backend
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: device)
    assert call_annotate().captions == ["test caption"]
    first = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    components = first.components
    assert call_annotate().captions == ["test caption"]
    assert annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME] is first
    assert first.components is components
    assert ModelLoad._get_model_state(MODEL_NAME) == f"on_{device}"
    loads.assert_called_once_with(device)
    moves.assert_not_called()


def test_explicit_cpu_offload_is_restored_once(backend):
    loads, moves = backend
    call_annotate()
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    components = annotator.components
    ModelLoad.cache_to_main_memory(MODEL_NAME, dict(components))
    assert ModelLoad._get_model_state(MODEL_NAME) == "on_cpu"
    assert call_annotate().error is None
    assert call_annotate().error is None
    assert annotator.components["model"] is components["model"]
    assert [c.args[1] for c in moves.call_args_list] == ["cpu", "cuda"]
    loads.assert_called_once()


def test_failed_cuda_restore_uses_cpu_for_inputs(backend):
    loads, moves = backend
    call_annotate()
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    ModelLoad.cache_to_main_memory(MODEL_NAME, dict(annotator.components))

    def move(components, device):
        if device == "cuda":
            raise RuntimeError("CUDA unavailable")
        components["model"].device = device

    moves.side_effect = move
    assert call_annotate().error is None
    assert annotator.device == "cpu"
    moves.reset_mock()
    assert call_annotate().error is None
    moves.assert_not_called()
    loads.assert_called_once()


def test_failed_cpu_fallback_does_not_retain_invalid_components(backend):
    _, moves = backend
    call_annotate()
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    ModelLoad.cache_to_main_memory(MODEL_NAME, dict(annotator.components))
    moves.side_effect = RuntimeError("both devices unavailable")
    assert call_annotate().error is not None
    assert annotator.components is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None


@pytest.mark.parametrize("release", ["explicit", "lru"])
def test_release_frees_component_objects_and_next_call_reloads(backend, release):
    loads, moves = backend
    call_annotate()
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    model_ref = weakref.ref(annotator.components["model"])
    processor_ref = weakref.ref(annotator.components["processor"])
    if release == "explicit":
        ModelLoad.release_model(MODEL_NAME)
    else:
        assert ModelLoad._clear_cache_internal("next-model", 2048.0)
    gc.collect()
    assert annotator.components is None
    assert model_ref() is None
    assert processor_ref() is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None
    assert MODEL_NAME not in ModelLoad._MEMORY_USAGE
    assert call_annotate().error is None
    assert loads.call_count == 2
    moves.assert_not_called()


def test_retained_use_refreshes_lru_timestamp(backend, monkeypatch):
    call_annotate()
    monkeypatch.setitem(LoaderBase._MODEL_LAST_USED, MODEL_NAME, 0.0)
    call_annotate()
    assert LoaderBase._MODEL_LAST_USED[MODEL_NAME] > 0.0


def test_new_instance_releases_old_owner_before_reloading(backend):
    loads, _ = backend
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    replacement = BLIPTagger(MODEL_NAME)
    with replacement:
        assert old.components is None
        assert replacement.components is not None
    assert loads.call_count == 2


def test_restoration_exception_discards_components_and_allows_retry(backend, monkeypatch):
    loads, _ = backend
    annotator = BLIPTagger(MODEL_NAME)
    restore = Mock(side_effect=RuntimeError("preparation failed"))
    with monkeypatch.context() as m:
        m.setattr(ModelLoad, "restore_model_to_cuda", restore)
        with pytest.raises(RuntimeError, match="preparation failed"):
            annotator.__enter__()
    assert annotator.components is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None
    with annotator:
        assert annotator.components is not None
    assert loads.call_count == 2


@pytest.mark.parametrize("components", [None, {}, {"model": FakeModel("cuda")}])
def test_incomplete_load_is_not_retained(backend, components):
    loads, _ = backend
    loads.side_effect = None
    loads.return_value = components
    annotator = BLIPTagger(MODEL_NAME)
    with pytest.raises(ModelLoadError):
        annotator.__enter__()
    assert annotator.components is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None


@pytest.mark.parametrize("error_type", [OutOfMemoryError, MemoryError, "cuda"])
@pytest.mark.parametrize("stage", ["preprocess", "inference"])
def test_oom_error_result_discards_components_and_next_call_reloads(backend, error_type, stage):
    loads, _ = backend
    call_annotate()
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    if error_type == "cuda":
        import torch

        error_type = torch.cuda.OutOfMemoryError
    with pytest.MonkeyPatch.context() as m:
        if stage == "preprocess":
            m.setattr(annotator, "_preprocess_images", Mock(side_effect=error_type("oom")))
        else:
            annotator.components["model"].generate.side_effect = error_type("oom")
        assert "メモリ不足" in call_annotate().error
    assert annotator.components is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None
    assert call_annotate().error is None
    assert loads.call_count == 2


def test_exception_leaving_context_discards_components(backend):
    annotator = BLIPTagger(MODEL_NAME)
    with pytest.raises(RuntimeError, match="interrupted"):
        with annotator:
            raise RuntimeError("interrupted")
    assert annotator.components is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None
