"""ONNX Runtime の実行先を、実際の session provider で判定する回帰テスト (Issue #175)。

onnxruntime は CUDA EP が使えない (または session 作成時に外れた) とき、例外を出さずに
CPUExecutionProvider だけで session を作る。要求した provider ではなく
``session.get_providers()`` を基準にしないと、CPU 実行を ``on_cuda`` と表示してしまう。
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from image_annotator_lib.core import onnx_runtime
from image_annotator_lib.core.base.onnx import ONNXBaseAnnotator
from image_annotator_lib.core.config import config_registry
from image_annotator_lib.core.loaders.loader_base import LoaderBase
from image_annotator_lib.core.loaders.onnx_loader import ONNXLoader
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.core.utils import logger
from image_annotator_lib.model_class.rating_onnx import AnimeRatingAnnotator

pytestmark = pytest.mark.unit

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class TaggedONNX(ONNXBaseAnnotator):
    """タグ読み込みだけを差し替え、ONNX のロード・推論経路は実物を使う。"""

    __test__ = False

    def _load_tags(self) -> None:
        self.all_tags = ["test-tag"]
        self._category_attr_map = {"general": "general_indexes"}
        self.general_indexes = [0]


def fake_session(providers: list[str]) -> Mock:
    session = Mock()
    session.get_providers.return_value = providers
    session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[1, 3, 8, 8])]
    session.get_outputs.return_value = [SimpleNamespace(name="output")]
    session.run.return_value = [np.array([[0.9]], dtype=np.float32)]
    return session


@pytest.fixture
def hardware(monkeypatch: pytest.MonkeyPatch):
    """torch の CUDA 判定と ORT の provider 一覧 (ハードウェア調査) だけを差し替える。"""
    state = SimpleNamespace(available=[CPU])
    monkeypatch.setattr("image_annotator_lib.core.utils.determine_effective_device", lambda d, n: d)
    monkeypatch.setattr(onnx_runtime, "available_onnx_providers", lambda: list(state.available))
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda device: 0)
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda device: 0)
    monkeypatch.setattr(LoaderBase, "_check_memory_before_load", classmethod(lambda cls, *a: True))
    monkeypatch.setattr(LoaderBase, "_get_max_cache_size", classmethod(lambda cls: 8192.0))
    monkeypatch.setattr(ModelLoad, "_get_max_cache_size", classmethod(lambda cls: 8192.0))
    return state


# --- 純粋な判定関数 ---------------------------------------------------------------


def test_cuda_request_falls_back_to_cpu_when_cuda_provider_is_unavailable(hardware) -> None:
    hardware.available = [CPU]
    with patch.object(logger, "warning") as warning:
        assert onnx_runtime.determine_onnx_device("cuda", "m") == "cpu"
    assert any("CUDAExecutionProvider" in call.args[0] for call in warning.call_args_list)


def test_cuda_request_is_kept_when_cuda_provider_is_available(hardware) -> None:
    hardware.available = [CUDA, CPU]
    assert onnx_runtime.determine_onnx_device("cuda", "m") == "cuda"
    assert onnx_runtime.determine_onnx_device("cpu", "m") == "cpu"


def test_indexed_cuda_request_is_cpu_because_session_has_no_device_id(hardware) -> None:
    """ONNX の session は device_id を渡さないため cuda:N は実在する GPU 選択にならない。"""
    hardware.available = [CUDA, CPU]
    assert onnx_runtime.determine_onnx_device("cuda:1", "m") == "cpu"


@pytest.mark.parametrize(
    ("providers", "expected"),
    [
        ([CPU], "cpu"),
        ([CUDA, CPU], "cuda"),
        (["AzureExecutionProvider", CPU], "cpu"),
        ([], "cpu"),
    ],
)
def test_execution_device_follows_session_providers(providers: list[str], expected: str) -> None:
    assert onnx_runtime.session_execution_device(fake_session(providers)) == expected


def test_provider_selection_requests_cuda_only_for_cuda_device() -> None:
    assert onnx_runtime.select_providers("cuda") == [CUDA, CPU]
    assert onnx_runtime.select_providers("cpu") == [CPU]


# --- ローダー: 状態・admission・ログ --------------------------------------------------


def load_with_session(session: Mock, device: str = "cuda"):
    """ONNXLoader.load_components を、InferenceSession の生成だけ差し替えて通す。"""
    loader = ONNXLoader("test-onnx-model", device)
    meta = Path("meta.csv")
    model = Path("model.onnx")
    with (
        patch.object(ONNXLoader, "_resolve_model_path_internal", return_value=(meta, model)),
        patch.object(ONNXLoader, "_calculate_specific_size", return_value=100.0),
        patch("onnxruntime.InferenceSession", return_value=session) as create,
    ):
        components = loader.load_components("fake/onnx")
    return loader, components, create


def test_cpu_fallback_after_session_creation_is_not_recorded_as_on_cuda(hardware) -> None:
    """CUDA EP が一覧にあっても session が CPU だけなら on_cpu。警告も出す。"""
    hardware.available = [CUDA, CPU]
    session = fake_session([CPU])
    with patch.object(logger, "warning") as warning:
        loader, components, create = load_with_session(session)

    assert components is not None
    assert create.call_args.kwargs["providers"] == [CUDA, CPU]
    assert loader.device == "cpu"
    assert ModelLoad._get_model_state("test-onnx-model") == "on_cpu"
    assert any("CPU" in call.args[0] and "session" in call.args[0] for call in warning.call_args_list)


def test_cuda_session_is_recorded_as_on_cuda(hardware) -> None:
    hardware.available = [CUDA, CPU]
    loader, components, _ = load_with_session(fake_session([CUDA, CPU]))

    assert components is not None
    assert loader.device == "cuda"
    assert ModelLoad._get_model_state("test-onnx-model") == "on_cuda"


def test_known_unavailable_cuda_provider_skips_cuda_admission(hardware) -> None:
    """CUDA EP が無いと分かっている load で、VRAM admission (on_cuda 扱い) を走らせない。"""
    hardware.available = [CPU]
    with patch.object(ModelLoad, "_make_cuda_room") as make_room:
        loader, components, create = load_with_session(fake_session([CPU]))

    make_room.assert_not_called()
    assert create.call_args.kwargs["providers"] == [CPU]
    assert components is not None
    assert loader.device == "cpu"
    assert ModelLoad._get_model_state("test-onnx-model") == "on_cpu"


def test_cuda_admission_still_runs_when_cuda_provider_is_available(hardware) -> None:
    hardware.available = [CUDA, CPU]
    with patch.object(ModelLoad, "_make_cuda_room") as make_room:
        load_with_session(fake_session([CUDA, CPU]))

    make_room.assert_called_once_with("test-onnx-model", "cuda")


def test_session_load_logs_actual_providers(hardware) -> None:
    hardware.available = [CUDA, CPU]
    with patch.object(logger, "info") as info:
        load_with_session(fake_session([CPU]))

    loaded = [c.args[0] for c in info.call_args_list if "ロード成功" in c.args[0]]
    provider_logs = [m for m in loaded if "実際のプロバイダー" in m]
    device_logs = [m for m in loaded if "デバイス:" in m]
    assert provider_logs
    assert all("CPUExecutionProvider" in m and "CUDAExecutionProvider" not in m for m in provider_logs)
    assert device_logs
    assert all("デバイス: cpu" in m for m in device_logs)


# --- annotator: 結果の execution_device --------------------------------------------


@pytest.fixture
def tagger_factory(hardware, monkeypatch: pytest.MonkeyPatch):
    def create(name: str, providers: list[str], available: list[str]) -> TaggedONNX:
        hardware.available = available
        monkeypatch.setitem(
            config_registry._merged_config_data,
            name,
            {
                "class": "WDTagger",
                "model_path": "fake/onnx",
                "device": "cuda",
                "estimated_size_gb": 0.1,
                "capabilities": ["tags"],
            },
        )
        session = fake_session(providers)
        monkeypatch.setattr(
            ONNXLoader,
            "_load_components_internal",
            lambda self, **kwargs: {"session": session, "csv_path": Path("m"), "metadata_path": Path("m")},
        )
        return TaggedONNX(name)

    return create


def test_annotator_reports_cpu_execution_device_after_fallback(tagger_factory) -> None:
    annotator = tagger_factory("test-onnx-fallback", [CPU], [CUDA, CPU])
    with annotator:
        results = annotator.predict([Image.new("RGB", (8, 8))])

    assert results[0].error is None
    assert results[0].execution_device == "cpu"
    assert ModelLoad._get_model_state("test-onnx-fallback") == "on_cpu"
    assert TaskCapability.TAGS in results[0].capabilities


def test_annotator_reports_cuda_execution_device_for_cuda_session(tagger_factory) -> None:
    annotator = tagger_factory("test-onnx-cuda", [CUDA, CPU], [CUDA, CPU])
    with annotator:
        results = annotator.predict([Image.new("RGB", (8, 8))])

    assert results[0].error is None
    assert results[0].execution_device == "cuda"
    assert ModelLoad._get_model_state("test-onnx-cuda") == "on_cuda"


def test_annotator_with_unavailable_cuda_provider_starts_on_cpu(tagger_factory) -> None:
    annotator = tagger_factory("test-onnx-nocuda", [CPU], [CPU])

    assert annotator.device == "cpu"


# --- rating ONNX --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("providers", "expected_state"),
    [([CPU], "on_cpu"), ([CUDA, CPU], "on_cuda")],
    ids=["cpu-fallback", "cuda"],
)
def test_rating_onnx_state_follows_session_providers(
    hardware, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, providers: list[str], expected_state: str
) -> None:
    hardware.available = [CUDA, CPU]
    name = f"test-rating-{expected_state}"
    monkeypatch.setitem(
        config_registry._merged_config_data,
        name,
        {
            "class": "AnimeRatingAnnotator",
            "model_path": str(tmp_path),
            "device": "cuda",
            "estimated_size_gb": 0.1,
            "capabilities": ["ratings"],
        },
    )
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "meta.json").write_text('{"labels": ["a", "b"]}', encoding="utf-8")
    session = fake_session(providers)
    session.run.return_value = [np.array([[0.2, 0.8]], dtype=np.float32)]

    with patch("onnxruntime.InferenceSession", return_value=session):
        annotator = AnimeRatingAnnotator(name)
        with annotator:
            assert ModelLoad._get_model_state(name) == expected_state
            result = annotator.predict([Image.new("RGB", (8, 8))])[0]

    assert result.execution_device == ("cuda" if expected_state == "on_cuda" else "cpu")


def test_rating_onnx_skips_cuda_admission_without_cuda_provider(
    hardware, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hardware.available = [CPU]
    name = "test-rating-nocuda"
    monkeypatch.setitem(
        config_registry._merged_config_data,
        name,
        {
            "class": "AnimeRatingAnnotator",
            "model_path": str(tmp_path),
            "device": "cuda",
            "estimated_size_gb": 0.1,
            "capabilities": ["ratings"],
        },
    )
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "meta.json").write_text('{"labels": ["a", "b"]}', encoding="utf-8")

    with (
        patch("onnxruntime.InferenceSession", return_value=fake_session([CPU])) as create,
        patch.object(ModelLoad, "_make_cuda_room") as make_room,
    ):
        with AnimeRatingAnnotator(name):
            pass

    make_room.assert_not_called()
    assert create.call_args.kwargs["providers"] == [CPU]
