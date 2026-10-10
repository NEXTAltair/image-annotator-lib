"""Lease ownership and host-memory regressions with real loader bookkeeping."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from image_annotator_lib.core.base.clip import ClipBaseAnnotator
from image_annotator_lib.core.base.onnx import ONNXBaseAnnotator
from image_annotator_lib.core.base.pipeline import PipelineBaseAnnotator
from image_annotator_lib.core.base.tensorflow import TensorflowBaseAnnotator
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.loaders.clip_loader import CLIPLoader
from image_annotator_lib.core.loaders.loader_base import LoaderBase
from image_annotator_lib.core.loaders.onnx_loader import ONNXLoader
from image_annotator_lib.core.loaders.tensorflow_loader import TensorFlowLoader
from image_annotator_lib.core.loaders.transformers_loader import TransformersPipelineLoader
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.exceptions.errors import ModelLoadError, OutOfMemoryError

pytestmark = pytest.mark.unit


class TestONNXAnnotator(ONNXBaseAnnotator):
    """Use the real ONNX context lifecycle without loading a tags file."""

    __test__ = False

    def _load_tags(self):
        self.all_tags = ["test-tag"]


class TestTensorflowAnnotator(TensorflowBaseAnnotator):
    __test__ = False

    def _load_tags(self):
        self.all_tags = ["test-tag"]

    def _preprocess_images(self, images):
        return images

    def _format_predictions(self, outputs):
        return outputs


@pytest.fixture(params=["pipeline", "clip", "tensorflow"])
def transient_backend(monkeypatch, request):
    """Exercise final-exit offloading/release of the other local backends."""
    kind = request.param
    model_name = f"test-nested-{kind}"
    config = {
        "class": "AestheticShadow",
        "model_path": "fake/model",
        "device": "cuda",
        "estimated_size_gb": 1.0,
    }
    if kind == "clip":
        config["base_model"] = "fake/clip"
    monkeypatch.setitem(config_registry._merged_config_data, model_name, config)
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: d)
    monkeypatch.setattr(LoaderBase, "_check_memory_before_load", classmethod(lambda cls, *a: True))
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 4096.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 4096.0))
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
    model = SimpleNamespace(device="cuda")
    clear_session = Mock()
    if kind == "pipeline":
        loader_class = TransformersPipelineLoader
        components = {"pipeline": SimpleNamespace(model=model)}
        annotator_class = PipelineBaseAnnotator
    elif kind == "clip":
        loader_class = CLIPLoader
        components = {"model": model, "clip_model": model, "processor": object()}
        annotator_class = ClipBaseAnnotator
    else:
        fake_tensorflow = Mock()
        fake_tensorflow.config.list_physical_devices.return_value = []
        fake_tensorflow.keras.backend.clear_session = clear_session
        monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
        loader_class = TensorFlowLoader
        components = {"model": model}
        annotator_class = TestTensorflowAnnotator
    loads = Mock(return_value=components)
    monkeypatch.setattr(loader_class, "_load_components_internal", lambda self, **kwargs: loads())
    moves = Mock(side_effect=lambda components, device: setattr(model, "device", device))
    monkeypatch.setattr(ModelLoad, "_move_components_to_device", moves)
    return kind, annotator_class(model_name), loads, moves, clear_session


@pytest.fixture
def onnx_backend(monkeypatch):
    """Fake ONNX downloads/session only; keep ownership, loading and LRU real."""
    model_name = "test-lease-onnx"
    monkeypatch.setitem(
        config_registry._merged_config_data,
        model_name,
        {
            "class": "WDTagger",
            "model_path": "fake/onnx",
            "device": "cuda",
            "estimated_size_gb": 1.0,
        },
    )
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: d)
    monkeypatch.setattr(LoaderBase, "_check_memory_before_load", classmethod(lambda cls, *a: True))
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 4096.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 4096.0))
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[1, 3, 448, 448])]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    loads = Mock(return_value={"session": session})
    monkeypatch.setattr(ONNXLoader, "_load_components_internal", lambda self, **kwargs: loads())
    return TestONNXAnnotator(model_name), loads


def enter_lease(model_name, owner):
    ModelLoad.begin_model_context(model_name, owner)
    ModelLoad.finish_model_entry(model_name, owner)


def test_owner_is_reserved_during_model_preparation():
    owner = object()
    contender = object()
    ModelLoad.begin_model_context("preparing", owner)
    try:
        assert ModelLoad.is_model_active("preparing")
        with pytest.raises(ModelLoadError):
            ModelLoad.begin_model_context("preparing", contender)
        assert ModelLoad.is_final_model_context("preparing", owner)
    finally:
        ModelLoad.end_model_context("preparing", owner, failed=True)
    assert not ModelLoad.is_model_active("preparing")
    assert "preparing" not in ModelLoad._ACTIVE_THREADS
    assert "preparing" not in ModelLoad._PREPARING_MODELS


def test_other_local_backend_nested_exit_only_offloads_or_releases_at_final_exit(transient_backend):
    kind, annotator, loads, moves, clear_session = transient_backend
    with annotator:
        components = annotator.components
        with annotator:
            assert annotator.components is components
        assert annotator.components is components
        assert ModelLoad._get_model_state(annotator.model_name) == "on_cuda"
        assert ModelLoad.is_final_model_context(annotator.model_name, annotator)
        moves.assert_not_called()
        clear_session.assert_not_called()
    assert not ModelLoad.is_model_active(annotator.model_name)
    loads.assert_called_once_with()
    if kind == "tensorflow":
        assert annotator.components is None
        assert components == {}
        assert ModelLoad._get_model_state(annotator.model_name) is None
        clear_session.assert_called_once_with()
    else:
        assert ModelLoad._get_model_state(annotator.model_name) == "on_cpu"
        assert [call.args[1] for call in moves.call_args_list] == ["cpu"]


def test_other_local_backend_caught_inner_exception_preserves_outer_context(transient_backend):
    _, annotator, loads, moves, clear_session = transient_backend
    with annotator:
        components = annotator.components
        with pytest.raises(ValueError, match="inner caller failed"):
            with annotator:
                raise ValueError("inner caller failed")
        assert annotator.components is components
        assert ModelLoad._get_model_state(annotator.model_name) == "on_cuda"
        assert ModelLoad.is_final_model_context(annotator.model_name, annotator)
        moves.assert_not_called()
        clear_session.assert_not_called()
    assert not ModelLoad.is_model_active(annotator.model_name)
    assert not ModelLoad._ACTIVE_THREADS
    loads.assert_called_once_with()


def test_nested_deferred_release_calls_owner_releaser_once():
    owner = object()
    release = Mock()
    enter_lease("nested", owner)
    ModelLoad._update_model_state("nested", "cpu", "loaded", 512.0)
    ModelLoad.register_component_releaser("nested", release, owner=owner)
    enter_lease("nested", owner)
    ModelLoad.release_model("nested")
    release.assert_not_called()
    ModelLoad.end_model_context("nested", owner)
    assert ModelLoad.is_model_active("nested")
    assert ModelLoad._get_current_cache_usage() == 512.0
    release.assert_not_called()
    ModelLoad.end_model_context("nested", owner)
    release.assert_called_once_with()
    assert ModelLoad._get_current_cache_usage() == 0.0
    assert not ModelLoad._ACTIVE_CONTEXTS
    assert not ModelLoad._ACTIVE_THREADS
    assert not ModelLoad._RELEASE_PENDING


def test_foreign_owner_cannot_invalidate_or_unregister_current_components():
    owner = object()
    stale_owner = object()
    release = Mock()
    enter_lease("owned", owner)
    ModelLoad._update_model_state("owned", "cpu", "loaded", 512.0)
    ModelLoad.register_component_releaser("owned", release, owner=owner)
    with pytest.raises(ModelLoadError):
        ModelLoad.unregister_component_releaser("owned", owner=stale_owner)
    with pytest.raises(ModelLoadError):
        ModelLoad.invalidate_model("owned", stale_owner)
    assert ModelLoad._get_model_state("owned") == "on_cpu"
    assert ModelLoad.is_final_model_context("owned", owner)
    release.assert_not_called()
    ModelLoad.release_model("owned")
    ModelLoad.end_model_context("owned", owner)
    release.assert_called_once_with()
    assert ModelLoad._get_model_state("owned") is None


def test_own_invalidation_clears_resources_but_waits_to_balance_existing_leases():
    owner = object()
    release = Mock()
    enter_lease("failed", owner)
    enter_lease("failed", owner)
    ModelLoad._update_model_state("failed", "cuda", "loaded", 512.0, host_size_mb=0.0)
    ModelLoad.register_component_releaser("failed", release, owner=owner)
    ModelLoad.invalidate_model("failed", owner)
    release.assert_called_once_with()
    assert ModelLoad._get_model_state("failed") is None
    assert "failed" not in ModelLoad._MEMORY_USAGE
    assert "failed" not in ModelLoad._HOST_MEMORY_USAGE
    assert ModelLoad.is_model_active("failed")
    ModelLoad.end_model_context("failed", owner)
    assert ModelLoad.is_model_active("failed")
    ModelLoad.end_model_context("failed", owner)
    assert not ModelLoad.is_model_active("failed")
    assert not ModelLoad._ACTIVE_THREADS


def test_known_vram_insufficiency_raises_after_idle_eviction_without_touching_active_model(monkeypatch):
    owner = object()
    active_release = Mock()
    idle_release = Mock()
    enter_lease("active", owner)
    ModelLoad._update_model_state("active", "cuda", "loaded", 1024.0, host_size_mb=0.0)
    ModelLoad.register_component_releaser("active", active_release, owner=owner)
    ModelLoad._update_model_state("idle", "cuda", "loaded", 256.0, host_size_mb=0.0)
    ModelLoad.register_component_releaser("idle", idle_release)
    ModelLoad._MODEL_SIZES["incoming"] = 2048.0
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (512 * 1024**2, 2 * 1024**3))
    try:
        with pytest.raises(OutOfMemoryError):
            ModelLoad._make_cuda_room("incoming", "cuda")
        idle_release.assert_called_once_with()
        active_release.assert_not_called()
        assert ModelLoad._get_model_state("active") == "on_cuda"
        assert ModelLoad.is_final_model_context("active", owner)
    finally:
        ModelLoad.end_model_context("active", owner)


@pytest.mark.parametrize("unknown", ["size", "probe"])
def test_unknown_vram_capacity_only_evicts_idle_models(monkeypatch, unknown):
    owner = object()
    active_release = Mock()
    idle_release = Mock()
    enter_lease("active", owner)
    ModelLoad._update_model_state("active", "cuda", "loaded", 1024.0, host_size_mb=0.0)
    ModelLoad.register_component_releaser("active", active_release, owner=owner)
    ModelLoad._update_model_state("idle", "cuda", "loaded", 256.0, host_size_mb=0.0)
    ModelLoad.register_component_releaser("idle", idle_release)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    if unknown == "size":
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (0, 2 * 1024**3))
    else:
        ModelLoad._MODEL_SIZES["incoming"] = 2048.0
        monkeypatch.setattr("torch.cuda.mem_get_info", Mock(side_effect=RuntimeError("probe unavailable")))
    try:
        ModelLoad._make_cuda_room("incoming", "cuda")
        idle_release.assert_called_once_with()
        active_release.assert_not_called()
        assert ModelLoad._get_model_state("active") == "on_cuda"
    finally:
        ModelLoad.end_model_context("active", owner)


def test_host_usage_uses_explicit_placement_and_legacy_entry_fallback():
    ModelLoad._MEMORY_USAGE.update({"gpu": 8192.0, "partial-host": 1024.0, "legacy": 256.0})
    ModelLoad._HOST_MEMORY_USAGE.update({"gpu": 0.0, "partial-host": 512.0})
    assert ModelLoad._get_current_cache_usage() == 768.0
    assert ModelLoad._get_model_memory_usage("gpu") == 8192.0


def test_unknown_backend_placement_remains_host_accounted():
    ModelLoad._update_model_state("unknown-placement", "cuda", "loaded", 1024.0)
    assert ModelLoad._HOST_MEMORY_USAGE["unknown-placement"] == 1024.0
    assert ModelLoad._get_current_cache_usage() == 1024.0


@pytest.mark.parametrize("requested_device", ["cuda", "cpu"])
def test_onnx_cpu_fallback_stays_in_host_budget(onnx_backend, requested_device):
    annotator, loads = onnx_backend
    annotator.device = requested_device
    with annotator:
        assert annotator.components["session"].get_providers() == ["CPUExecutionProvider"]
        assert ModelLoad._MEMORY_USAGE[annotator.model_name] == 1024.0
        assert ModelLoad._HOST_MEMORY_USAGE[annotator.model_name] == 1024.0
        assert ModelLoad._get_current_cache_usage() == 1024.0
    assert not ModelLoad.is_model_active(annotator.model_name)
    loads.assert_called_once_with()


def test_host_lru_preserves_active_onnx_and_zero_host_gpu_but_releases_idle_cpu(onnx_backend):
    annotator, _ = onnx_backend
    gpu_release = Mock()
    cpu_release = Mock()
    with annotator:
        session = annotator.components["session"]
        ModelLoad._update_model_state("oldest-gpu", "cuda", "loaded", 2048.0, host_size_mb=0.0)
        ModelLoad.register_component_releaser("oldest-gpu", gpu_release)
        ModelLoad._MODEL_LAST_USED["oldest-gpu"] = 0.0
        ModelLoad._MODEL_LAST_USED[annotator.model_name] = 1.0
        ModelLoad._update_model_state("idle-cpu", "cpu", "loaded", 2048.0)
        ModelLoad.register_component_releaser("idle-cpu", cpu_release)
        assert ModelLoad._clear_cache_internal("incoming-cpu", 2048.0)
        cpu_release.assert_called_once_with()
        gpu_release.assert_not_called()
        assert annotator.components["session"] is session
        assert ModelLoad._get_model_state("oldest-gpu") == "on_cuda"
        assert ModelLoad._get_current_cache_usage() == 1024.0
    assert annotator.components["session"] is session


def test_onnx_explicit_release_waits_until_nested_context_has_finished(onnx_backend):
    annotator, loads = onnx_backend
    with annotator:
        session = annotator.components["session"]
        with annotator:
            ModelLoad.release_model(annotator.model_name)
        assert annotator.components["session"] is session
        assert ModelLoad.is_model_active(annotator.model_name)
        assert ModelLoad._get_current_cache_usage() == 1024.0
    assert annotator.components is None
    assert ModelLoad._get_model_state(annotator.model_name) is None
    assert ModelLoad._get_current_cache_usage() == 0.0
    assert not ModelLoad._ACTIVE_CONTEXTS
    loads.assert_called_once_with()


def test_active_onnx_same_name_replacement_failure_preserves_original_session(onnx_backend):
    annotator, loads = onnx_backend
    replacement = TestONNXAnnotator(annotator.model_name)
    with annotator:
        session = annotator.components["session"]
        with pytest.raises(ModelLoadError):
            replacement.__enter__()
        assert replacement.components is None
        assert annotator.components["session"] is session
        assert ModelLoad._MODEL_OWNERS[annotator.model_name] is annotator
        assert ModelLoad.is_final_model_context(annotator.model_name, annotator)
    assert not ModelLoad._ACTIVE_CONTEXTS
    loads.assert_called_once_with()


def test_cpu_recache_counts_protected_model_as_replacement_once(monkeypatch):
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 2048.0))
    other_release = Mock()
    ModelLoad._update_model_state("transitioning", "cuda", "loaded", 1024.0, host_size_mb=512.0)
    ModelLoad._MODEL_SIZES["transitioning"] = 1024.0
    ModelLoad._update_model_state("other-cpu", "cpu", "loaded", 1024.0)
    ModelLoad.register_component_releaser("other-cpu", other_release)
    components = {"model": object()}
    move = Mock()
    monkeypatch.setattr(ModelLoad, "_move_components_to_device", move)
    assert ModelLoad.cache_to_main_memory("transitioning", components) is components
    other_release.assert_not_called()
    assert ModelLoad._get_current_cache_usage() == 2048.0
    assert ModelLoad._HOST_MEMORY_USAGE["transitioning"] == 1024.0
    move.assert_called_once_with(components, "cpu")


def test_replacement_host_demand_does_not_count_protected_model_twice(monkeypatch):
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 2048.0))
    release = Mock()
    ModelLoad._update_model_state("transitioning", "cpu", "loaded", 1024.0)
    ModelLoad._update_model_state("other-cpu", "cpu", "loaded", 1024.0)
    ModelLoad.register_component_releaser("other-cpu", release)
    assert ModelLoad._clear_cache_internal("transitioning", 1024.0)
    release.assert_not_called()
    assert ModelLoad._get_model_state("other-cpu") == "on_cpu"


def test_state_release_clears_both_memory_ledgers_and_owner_callback():
    owner = object()
    release = Mock()
    enter_lease("released", owner)
    ModelLoad._update_model_state("released", "cuda", "loaded", 1024.0, host_size_mb=128.0)
    ModelLoad.register_component_releaser("released", release, owner=owner)
    ModelLoad.end_model_context("released", owner)
    ModelLoad.release_model("released")
    release.assert_called_once_with()
    for mapping in (
        ModelLoad._MODEL_STATES,
        ModelLoad._MEMORY_USAGE,
        ModelLoad._HOST_MEMORY_USAGE,
        ModelLoad._MODEL_LAST_USED,
        ModelLoad._MODEL_OWNERS,
        ModelLoad._COMPONENT_RELEASERS,
    ):
        assert "released" not in mapping
