"""Offline process ownership, bounded startup and shared Clef lifetime tests."""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from image_annotator_lib.decisions import (
    DecisionErrorCode,
    DecisionRequest,
    LocalDecisionClient,
    NoulQuestion,
)
from image_annotator_lib.decisions import runtime as runtime_module
from image_annotator_lib.decisions.runtime import LocalRuntimeError, RuntimeSettings, _LocalRuntime

pytestmark = [pytest.mark.unit, pytest.mark.standard]


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("local-server", timeout)
        return self.returncode


@pytest.fixture
def settings(tmp_path: Path) -> RuntimeSettings:
    paths = [tmp_path / name for name in ("server.exe", "model.gguf", "vision.gguf")]
    for path in paths:
        path.write_bytes(b"test")
    return LocalDecisionClient(*paths)._configuration()


@pytest.fixture(autouse=True)
def isolate_runtime() -> Any:
    runtime_module.shutdown_local_runtime()
    yield
    runtime_module.shutdown_local_runtime()


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[dict[str, Any]]:
    original = httpx.Client
    options: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> httpx.Client:
        options.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(runtime_module.httpx, "Client", client)
    return options


def test_runtime_launch_is_hidden_loopback_and_ready_before_use(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    process = FakeProcess()

    def popen(args: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200, json={"status": "ok"})

    options = _mock_http(monkeypatch, handler)
    runtime = _LocalRuntime(settings)
    runtime.start(2)
    assert len(calls) == 1 and len(requests) == 2
    args, kwargs = calls[0]
    assert args[0] == str(settings.server_path)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("-ngl") + 1] == "10"
    assert args[args.index("-ub") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--no-context-shift" in args and "--offline" in args
    assert kwargs["shell"] is False
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert options[0]["trust_env"] is options[0]["follow_redirects"] is False
    assert requests[0].url.host == "127.0.0.1" and requests[0].url.path == "/health"
    runtime.close()
    assert process.terminated and runtime.process is None
    runtime.close()


@pytest.mark.parametrize("failure", ["exit", "timeout", "spawn"])
def test_startup_failures_are_bounded_typed_and_cleaned_up(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    process = FakeProcess()

    def popen(*args: Any, **kwargs: Any) -> FakeProcess:
        if failure == "spawn":
            raise OSError("private system details")
        if failure == "exit":
            process.returncode = 1
        return process

    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    _mock_http(monkeypatch, lambda request: httpx.Response(503))
    runtime = _LocalRuntime(settings)
    with pytest.raises(LocalRuntimeError) as caught:
        runtime.start(0.01)
    assert caught.value.code == (
        DecisionErrorCode.TRANSPORT if failure == "timeout" else DecisionErrorCode.CONFIGURATION
    )
    assert "private system details" not in str(caught.value)
    assert runtime.process is None
    if failure == "timeout":
        assert process.terminated


def test_close_kills_and_reaps_a_process_that_ignores_termination(settings: RuntimeSettings) -> None:
    class StubbornProcess(FakeProcess):
        def terminate(self) -> None:
            self.terminated = True

    process = StubbornProcess()
    runtime = _LocalRuntime(settings)
    runtime.process = process  # type: ignore[assignment]
    runtime.close()
    assert process.terminated and process.killed and runtime.process is None


def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    instances: list[Any] = []

    class FakeRuntime:
        def __init__(self, settings: RuntimeSettings) -> None:
            self.settings = settings
            self.process: FakeProcess | None = FakeProcess()
            self.base_url = "http://127.0.0.1:11437"
            self.closed = False
            self.starts = 0
            instances.append(self)

        def start(self, timeout: float) -> None:
            self.starts += 1

        def close(self) -> None:
            self.closed = True
            self.process = None

    monkeypatch.setattr(runtime_module, "_LocalRuntime", FakeRuntime)
    return instances


def test_clients_reuse_one_runtime_then_release_old_model_on_settings_or_file_change(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = _fake_runtime(monkeypatch)
    for _ in range(2):
        with runtime_module.runtime_session(settings, 1) as endpoint:
            assert endpoint == "http://127.0.0.1:11437"
    assert len(instances) == 1 and instances[0].starts == 1
    changed = replace(settings, n_gpu_layers=0)
    with runtime_module.runtime_session(changed, 1):
        assert instances[0].closed
    assert len(instances) == 2
    settings.model_path.write_bytes(b"replaced model")
    replaced = LocalDecisionClient(
        settings.server_path, settings.model_path, settings.mmproj_path, n_gpu_layers=0
    )._configuration()
    with runtime_module.runtime_session(replaced, 1):
        assert instances[1].closed
    assert len(instances) == 3
    runtime_module.shutdown_local_runtime()
    assert instances[2].closed


def test_concurrent_evaluations_do_not_duplicate_load_or_overlap_inference(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = _fake_runtime(monkeypatch)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with runtime_module.runtime_session(settings, 2):
            first_entered.set()
            assert release_first.wait(2)

    def second() -> None:
        with runtime_module.runtime_session(settings, 2):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first)
        assert first_entered.wait(2)
        two = pool.submit(second)
        assert not second_entered.wait(0.02)
        release_first.set()
        one.result(timeout=2)
        two.result(timeout=2)
    assert second_entered.is_set() and len(instances) == 1


def test_busy_runtime_returns_error_within_caller_timeout(settings: RuntimeSettings) -> None:
    with runtime_module._runtime_lock, pytest.raises(LocalRuntimeError, match="still running"):
        with runtime_module.runtime_session(settings, 0.01):
            pytest.fail("a busy runtime must not be entered")


def test_timed_out_inference_stops_owned_runtime_before_next_call(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = _fake_runtime(monkeypatch)
    with pytest.raises(httpx.ReadTimeout):
        with runtime_module.runtime_session(settings, 1):
            raise httpx.ReadTimeout("request timed out")
    assert instances[0].closed and runtime_module._active_runtime is None
    with runtime_module.runtime_session(settings, 1):
        assert len(instances) == 2


def test_constructing_client_never_launches_model_and_startup_errors_are_results(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("missing executable dependency")

    monkeypatch.setattr(runtime_module.subprocess, "Popen", fail)
    client = LocalDecisionClient(settings.server_path, settings.model_path, settings.mmproj_path)
    result = client.evaluate(DecisionRequest("check", {}, {"tag": NoulQuestion("Supported?")}))
    assert result.error is not None and result.error.code == DecisionErrorCode.CONFIGURATION
    assert result.provider == "llamacpp" and result.model_name == "clef-flash"
    assert not result.answers
