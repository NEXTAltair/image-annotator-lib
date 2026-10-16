"""Transformers retention with real loader state and fake model downloads (Issue #165)."""

import gc
import weakref
from queue import Queue
from threading import Thread
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

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def backend(monkeypatch):
    """Keep loader/LRU/release real; replace downloads, memory probes, and device moves."""
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
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
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


def configure_next_model(monkeypatch, name="test-next-transformers", **overrides):
    config = dict(config_registry._merged_config_data[MODEL_NAME])
    config.update(overrides)
    monkeypatch.setitem(config_registry._merged_config_data, name, config)
    return BLIPTagger(name)


def assert_predicts(annotator):
    result = annotator.predict([Image.new("RGB", (4, 4))])[0]
    assert result.error is None
    assert result.captions == ["test caption"]


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

    ModelLoad.release_model(MODEL_NAME)
    moves.side_effect = lambda components, device: setattr(components["model"], "device", device)
    assert call_annotate().error is None
    assert annotator.device == "cuda"
    assert loads.call_count == 2


@pytest.mark.parametrize("size_known", [True, False])
def test_switching_models_evicts_old_model_when_vram_is_insufficient(backend, monkeypatch, size_known):
    loads, moves = backend
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    next_name = "test-next-transformers"
    config = dict(config_registry._merged_config_data[MODEL_NAME])
    if not size_known:
        config.pop("estimated_size_gb")
    monkeypatch.setitem(config_registry._merged_config_data, next_name, config)
    # Releasing the idle model recovers its VRAM; a constant insufficient probe
    # would correctly reject the next known-size model even after eviction.
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda device: (512 * 1024**2 if old.components else 2 * 1024**3, 2 * 1024**3),
    )
    if not size_known:
        monkeypatch.setattr(TransformersLoader, "_calculate_specific_size", lambda *args, **kwargs: 0.0)
    next_annotator = BLIPTagger(next_name)
    with next_annotator:
        assert old.components is None
        assert ModelLoad._get_model_state(MODEL_NAME) is None
    with next_annotator:
        assert next_annotator.components is not None
    assert loads.call_count == 2
    moves.assert_not_called()


def test_switching_models_preserves_cached_model_when_vram_is_sufficient(backend):
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    next_name = "test-next-transformers"
    config_registry._merged_config_data[next_name] = dict(config_registry._merged_config_data[MODEL_NAME])
    with BLIPTagger(next_name):
        assert old.components is not None


def test_gpu_eviction_skips_other_devices_and_cpu_models(backend, monkeypatch):
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    ModelLoad._update_model_state(MODEL_NAME, "cuda:1", "loaded", 1024.0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (0, 2 * 1024**3))
    ModelLoad._make_cuda_room("next-model", "cuda:0")
    assert old.components is not None
    ModelLoad._update_model_state(MODEL_NAME, "cpu", "cached_cpu", 1024.0)
    ModelLoad._make_cuda_room("next-model", "cuda")
    assert old.components is not None


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError])
def test_gpu_memory_probe_failure_evicts_other_resident_models(backend, monkeypatch, error_type):
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    ModelLoad._MODEL_SIZES["next-model"] = 512.0
    monkeypatch.setattr("torch.cuda.current_device", Mock(side_effect=error_type("unavailable")))
    monkeypatch.setattr("torch.cuda.mem_get_info", Mock(side_effect=error_type("unavailable")))
    ModelLoad._make_cuda_room("next-model", "cuda")
    assert old.components is None


def test_gpu_eviction_stops_after_enough_vram_is_recovered(backend, monkeypatch):
    call_annotate()
    old = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    ModelLoad._update_model_state("newer-model", "cuda", "loaded", 256.0)
    monkeypatch.setitem(ModelLoad._MODEL_LAST_USED, MODEL_NAME, 0.0)
    ModelLoad._MODEL_SIZES["next-model"] = 512.0
    available = Mock(side_effect=[(0, 2 * 1024**3), (1024**3, 2 * 1024**3)])
    monkeypatch.setattr("torch.cuda.mem_get_info", available)
    ModelLoad._make_cuda_room("next-model", "cuda")
    assert old.components is None
    assert ModelLoad._get_model_state("newer-model") == "on_cuda"


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
        # Host LRU evicts host-resident weights, so explicitly offload first.
        ModelLoad.cache_to_main_memory(MODEL_NAME, dict(annotator.components))
        assert [call.args[1] for call in moves.call_args_list] == ["cpu"]
        # Mock call arguments would otherwise keep their component copy alive.
        moves.reset_mock()
        assert ModelLoad._clear_cache_internal("next-model", 2048.0)
    gc.collect()
    assert annotator.components is None
    assert model_ref() is None
    assert processor_ref() is None
    assert ModelLoad._get_model_state(MODEL_NAME) is None
    assert MODEL_NAME not in ModelLoad._MEMORY_USAGE
    assert MODEL_NAME not in ModelLoad._HOST_MEMORY_USAGE
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


@pytest.mark.parametrize("size_known", [True, False])
def test_nested_other_model_under_vram_pressure_preserves_active_prediction(
    backend, monkeypatch, size_known
):
    loads, _ = backend
    outer = BLIPTagger(MODEL_NAME)
    inner = configure_next_model(monkeypatch)
    if not size_known:
        config_registry._merged_config_data[inner.model_name].pop("estimated_size_gb")
        monkeypatch.setattr(TransformersLoader, "_calculate_specific_size", lambda *args, **kwargs: 0.0)

    with outer:
        components = outer.components
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (512 * 1024**2, 2 * 1024**3))
        if size_known:
            with pytest.raises(OutOfMemoryError):
                with inner:
                    pytest.fail("A known-size model cannot fit while the outer model is active")
            assert inner.components is None
            assert not ModelLoad.is_model_active(inner.model_name)
        else:
            with inner:
                assert inner.components is not None
                assert outer.components is components
        assert outer.components is components
        assert ModelLoad._get_model_state(MODEL_NAME) == "on_cuda"
        assert_predicts(outer)

    assert not ModelLoad.is_model_active(MODEL_NAME)
    assert loads.call_count == (1 if size_known else 2)


def test_host_lru_defers_eviction_of_active_cpu_transformer(backend, monkeypatch):
    monkeypatch.setitem(config_registry._merged_config_data[MODEL_NAME], "device", "cpu")
    annotator = BLIPTagger(MODEL_NAME)
    with annotator:
        components = annotator.components
        assert not ModelLoad._clear_cache_internal("incoming-cpu", 2048.0)
        assert annotator.components is components
        assert ModelLoad._get_current_cache_usage() == 1024.0
        assert_predicts(annotator)
    assert ModelLoad._clear_cache_internal("incoming-cpu", 2048.0)
    assert annotator.components is None
    assert ModelLoad._get_current_cache_usage() == 0.0


def test_same_instance_nested_exit_keeps_outer_lease_and_components(backend):
    loads, moves = backend
    annotator = BLIPTagger(MODEL_NAME)
    with annotator:
        components = annotator.components
        with annotator:
            assert annotator.components is components
            assert not ModelLoad.is_final_model_context(MODEL_NAME, annotator)
        assert ModelLoad.is_model_active(MODEL_NAME)
        assert ModelLoad.is_final_model_context(MODEL_NAME, annotator)
        assert annotator.components is components
        assert_predicts(annotator)
    assert not ModelLoad.is_model_active(MODEL_NAME)
    assert annotator.components is components
    loads.assert_called_once()
    moves.assert_not_called()


def test_failed_active_same_name_replacement_cannot_discard_current_owner(backend):
    loads, _ = backend
    original = BLIPTagger(MODEL_NAME)
    replacement = BLIPTagger(MODEL_NAME)
    with original:
        components = original.components
        with pytest.raises(ModelLoadError):
            replacement.__enter__()
        assert replacement.components is None
        assert original.components is components
        assert ModelLoad._MODEL_OWNERS[MODEL_NAME] is original
        assert ModelLoad.is_final_model_context(MODEL_NAME, original)
        assert_predicts(original)
    assert not ModelLoad.is_model_active(MODEL_NAME)
    loads.assert_called_once()


def test_old_owner_cleanup_cannot_unregister_or_release_replacement(backend):
    loads, _ = backend
    original = BLIPTagger(MODEL_NAME)
    with original:
        pass
    replacement = BLIPTagger(MODEL_NAME)
    with replacement:
        components = replacement.components
        original._discard_components()
        assert replacement.components is components
        assert ModelLoad._MODEL_OWNERS[MODEL_NAME] is replacement
        assert_predicts(replacement)
    ModelLoad.release_model(MODEL_NAME)
    assert replacement.components is None
    assert loads.call_count == 2


def test_explicit_release_waits_for_last_nested_exit(backend):
    annotator = BLIPTagger(MODEL_NAME)
    with annotator:
        components = annotator.components
        with annotator:
            ModelLoad.release_model(MODEL_NAME)
            assert annotator.components is components
            assert_predicts(annotator)
        assert ModelLoad.is_model_active(MODEL_NAME)
        assert annotator.components is components
        assert_predicts(annotator)
    assert annotator.components is None
    assert MODEL_NAME not in ModelLoad._MEMORY_USAGE
    assert MODEL_NAME not in ModelLoad._HOST_MEMORY_USAGE
    assert MODEL_NAME not in ModelLoad._ACTIVE_CONTEXTS
    assert MODEL_NAME not in ModelLoad._ACTIVE_THREADS
    assert MODEL_NAME not in ModelLoad._RELEASE_PENDING


def test_ordinary_nested_exception_does_not_invalidate_outer_prediction(backend):
    annotator = BLIPTagger(MODEL_NAME)
    with annotator:
        components = annotator.components
        with pytest.raises(ValueError, match="inner caller failed"):
            with annotator:
                raise ValueError("inner caller failed")
        assert annotator.components is components
        assert ModelLoad.is_final_model_context(MODEL_NAME, annotator)
        assert_predicts(annotator)
    assert not ModelLoad.is_model_active(MODEL_NAME)


def test_own_nested_oom_discards_only_its_model_and_balances_contexts(backend, monkeypatch):
    annotator = BLIPTagger(MODEL_NAME)
    other = configure_next_model(monkeypatch)
    with other:
        other_components = other.components
        with annotator:
            with annotator:
                annotator.components["model"].generate.side_effect = OutOfMemoryError("own inference OOM")
                result = annotator.predict([Image.new("RGB", (4, 4))])[0]
                assert "メモリ不足" in result.error
                assert annotator.components is None
                assert ModelLoad._get_model_state(MODEL_NAME) is None
                assert other.components is other_components
            assert ModelLoad.is_model_active(MODEL_NAME)
        assert not ModelLoad.is_model_active(MODEL_NAME)
        assert MODEL_NAME not in ModelLoad._ACTIVE_THREADS
        assert_predicts(other)
    assert not ModelLoad._ACTIVE_CONTEXTS


def test_cross_thread_reentry_rejects_without_corrupting_outer_context(backend):
    annotator = BLIPTagger(MODEL_NAME)
    outcomes = Queue()

    def enter_from_other_thread():
        try:
            with annotator:
                outcomes.put(None)
        except Exception as error:
            outcomes.put(error)

    with annotator:
        components = annotator.components
        thread = Thread(target=enter_from_other_thread, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
        assert not thread.is_alive(), "Lease conflict detection deadlocked"
        assert isinstance(outcomes.get_nowait(), ModelLoadError)
        assert ModelLoad.is_final_model_context(MODEL_NAME, annotator)
        assert annotator.components is components
        assert_predicts(annotator)
    assert not ModelLoad._ACTIVE_CONTEXTS
    assert not ModelLoad._ACTIVE_THREADS


def test_sufficient_gpu_memory_retains_two_models_with_one_model_host_budget(backend, monkeypatch):
    loads, moves = backend
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 1024.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 1024.0))
    first = BLIPTagger(MODEL_NAME)
    second = configure_next_model(monkeypatch)
    with first:
        pass
    first_components = first.components
    with second:
        assert first.components is first_components
        assert_predicts(second)
    assert first.components is first_components
    assert ModelLoad._MEMORY_USAGE == {MODEL_NAME: 1024.0, second.model_name: 1024.0}
    assert ModelLoad._get_current_cache_usage() == 0.0
    with first:
        assert_predicts(first)
    assert loads.call_count == 2
    moves.assert_not_called()


def test_cpu_loading_under_real_host_pressure_evicts_idle_cpu_model(backend, monkeypatch):
    loads, _ = backend
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 1024.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 1024.0))
    monkeypatch.setitem(config_registry._merged_config_data[MODEL_NAME], "device", "cpu")
    first = BLIPTagger(MODEL_NAME)
    second = configure_next_model(monkeypatch, device="cpu")
    with first:
        pass
    assert ModelLoad._get_current_cache_usage() == 1024.0
    with second:
        assert first.components is None
        assert ModelLoad._get_model_state(MODEL_NAME) is None
        assert_predicts(second)
    assert ModelLoad._get_current_cache_usage() == 1024.0
    assert loads.call_count == 2


def test_offload_and_restore_replace_host_usage_without_evicting_unrelated_cpu_model(backend, monkeypatch):
    _, moves = backend
    gpu = BLIPTagger(MODEL_NAME)
    cpu = configure_next_model(monkeypatch, device="cpu")
    with gpu:
        pass
    with cpu:
        pass
    cpu_components = cpu.components
    ModelLoad.cache_to_main_memory(MODEL_NAME, dict(gpu.components))
    assert ModelLoad._get_current_cache_usage() == 2048.0
    assert ModelLoad._MEMORY_USAGE[MODEL_NAME] == 1024.0
    with gpu:
        assert gpu.device == "cuda"
        assert cpu.components is cpu_components
        assert ModelLoad._HOST_MEMORY_USAGE[MODEL_NAME] == 0.0
        assert ModelLoad._get_current_cache_usage() == 1024.0
        assert_predicts(gpu)
    assert [call.args[1] for call in moves.call_args_list] == ["cpu", "cuda"]


def test_mid_context_offload_is_rejected_and_prediction_stays_on_gpu(backend):
    _, moves = backend
    annotator = BLIPTagger(MODEL_NAME)
    with annotator:
        components = annotator.components
        with pytest.raises(ModelLoadError):
            ModelLoad.cache_to_main_memory(MODEL_NAME, dict(components))
        assert annotator.components is components
        assert annotator.device == "cuda"
        assert ModelLoad._get_model_state(MODEL_NAME) == "on_cuda"
        assert_predicts(annotator)
    moves.assert_not_called()


def test_retained_cpu_model_survives_known_vram_deficit_without_redownload(backend, monkeypatch):
    loads, moves = backend
    assert call_annotate().error is None
    annotator = annotation_runner._MODEL_INSTANCE_REGISTRY[MODEL_NAME]
    model = annotator.components["model"]
    ModelLoad.cache_to_main_memory(MODEL_NAME, dict(annotator.components))
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (512 * 1024**2, 2 * 1024**3))
    assert call_annotate().captions == ["test caption"]
    assert call_annotate().captions == ["test caption"]
    assert annotator.components["model"] is model
    assert annotator.device == "cpu"
    assert model.device == "cpu"
    assert ModelLoad._get_model_state(MODEL_NAME) == "on_cpu"
    assert ModelLoad._HOST_MEMORY_USAGE[MODEL_NAME] == 1024.0
    assert not ModelLoad.is_model_active(MODEL_NAME)
    loads.assert_called_once()
    assert all(call.args[1] == "cpu" for call in moves.call_args_list)


def test_public_annotate_oom_cleanup_allows_immediate_next_chunk(backend, monkeypatch):
    """A failed chunk must return reserved VRAM before the next admission probe."""
    import torch

    loads, _ = backend
    available = {"bytes": 16 * 1024**3}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (available["bytes"], 16 * 1024**3))
    empty_cache = Mock(side_effect=lambda: available.update(bytes=16 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    def fail_allocation(**kwargs):
        available["bytes"] = 512 * 1024**2
        raise torch.cuda.OutOfMemoryError("allocator retains reserved memory until empty_cache")

    def download(device):
        model = FakeModel(device)
        if loads.call_count == 1:
            model.generate.side_effect = fail_allocation
        return {"model": model, "processor": FakeProcessor()}

    loads.side_effect = download
    assert "メモリ不足" in call_annotate().error
    assert available["bytes"] == 16 * 1024**3
    assert not ModelLoad.is_model_active(MODEL_NAME)
    assert not ModelLoad._CLEANUP_PENDING
    result = call_annotate()
    assert result.error is None
    assert result.captions == ["test caption"]
    assert loads.call_count == 2
    empty_cache.assert_called()
