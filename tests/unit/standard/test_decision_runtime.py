"""Offline process ownership, bounded startup and shared Clef lifetime tests."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
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
from image_annotator_lib.decisions.runtime import (
    LocalRuntimeError,
    ManagedEndpoint,
    RuntimeSettings,
    _LocalRuntime,
)

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
        options.append(dict(kwargs))
        if kwargs.get("transport") is None:
            kwargs["transport"] = httpx.MockTransport(handler)
        return original(**kwargs)

    monkeypatch.setattr(runtime_module.httpx, "Client", client)
    return options


@pytest.mark.parametrize("n_gpu_layers", [0, 10])
def test_runtime_launch_is_hidden_loopback_and_ready_before_use(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, n_gpu_layers: int
) -> None:
    settings = replace(settings, n_gpu_layers=n_gpu_layers)
    inherited_options = (
        "LLAMA_ARG_API_PREFIX",
        "LLAMA_API_KEY",
        "MTMD_BACKEND_DEVICE",
        "GGML_RPC_SERVERS",
        "AIP_MODE",
    )
    for name in inherited_options:
        monkeypatch.setenv(name, "unmanaged-value")
    monkeypatch.setenv("CLEF_TEST_KEEP", "preserved")
    parent_env = dict(os.environ)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    process = FakeProcess()

    def popen(args: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(503 if len(requests) == 1 else 200, json={"status": "ok"})
        if request.headers["Authorization"] != f"Bearer {runtime.api_key}":
            return httpx.Response(401)
        return httpx.Response(200, json={"model_path": str(settings.model_path)})

    options = _mock_http(monkeypatch, handler)
    runtime = _LocalRuntime(settings)
    runtime.start(2)
    assert len(calls) == 1 and len(requests) == 4
    args, kwargs = calls[0]
    assert args[0] == str(settings.server_path)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("-ngl") + 1] == str(n_gpu_layers)
    assert args[args.index("-ub") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--no-context-shift" in args and "--offline" in args
    key_file, key_directory = runtime._key_file, runtime._key_directory
    assert key_file is not None and key_directory is not None
    assert args[args.index("--api-key-file") + 1] == str(key_file)
    assert "--api-key" not in args and runtime.api_key not in repr(args)
    assert key_file.read_text(encoding="ascii").strip() == runtime.api_key
    if os.name == "posix":
        assert key_file.stat().st_mode & 0o777 == 0o600
        assert key_directory.stat().st_mode & 0o777 == 0o700
    assert len(runtime.api_key) >= 32
    assert "--no-cors-credentials" in args
    assert args[args.index("--cors-origins") + 1] == "http://127.0.0.1"
    assert not any(name in kwargs["env"] for name in inherited_options)
    assert kwargs["env"]["CLEF_TEST_KEEP"] == "preserved" and dict(os.environ) == parent_env
    if n_gpu_layers == 0:
        assert args[args.index("--device") + 1] == "none"
        assert "--no-mmproj-offload" in args
    else:
        assert "--device" not in args and "--no-mmproj-offload" not in args
    assert kwargs["shell"] is False
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert options[0]["trust_env"] is options[0]["follow_redirects"] is False
    assert requests[0].url.host == "127.0.0.1" and requests[0].url.path == "/health"
    runtime.close()
    assert process.terminated and runtime.process is None
    assert not key_file.exists() and not key_directory.exists()
    assert runtime._key_file is runtime._key_directory is None
    runtime.close()


@pytest.mark.parametrize("failure", ["exit", "timeout", "spawn"])
def test_startup_failures_are_bounded_typed_and_cleaned_up(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    process = FakeProcess()
    created: list[Path] = []

    def popen(*args: Any, **kwargs: Any) -> FakeProcess:
        assert runtime._key_directory is not None
        created.append(runtime._key_directory)
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
    assert runtime._key_file is runtime._key_directory is None
    assert created and not created[0].exists()
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


def test_close_handles_process_exit_between_poll_and_terminate(settings: RuntimeSettings) -> None:
    class ExitingProcess(FakeProcess):
        def terminate(self) -> None:
            self.returncode = 0
            raise ProcessLookupError("already exited")

    runtime = _LocalRuntime(settings)
    runtime.process = ExitingProcess()  # type: ignore[assignment]
    runtime.close()
    assert runtime.process is None


def test_failed_termination_remains_typed_and_keeps_process_ownership(settings: RuntimeSettings) -> None:
    class ProtectedProcess(FakeProcess):
        def terminate(self) -> None:
            raise PermissionError("private system detail")

    runtime = _LocalRuntime(settings)
    runtime.process = ProtectedProcess()  # type: ignore[assignment]
    with pytest.raises(LocalRuntimeError, match="could not be stopped"):
        runtime.close()
    assert runtime.process is not None


@pytest.mark.parametrize("health", [b"not JSON", b"[]", b'{"status": "loading"}'])
def test_http_success_without_ready_health_does_not_start_inference(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, health: bytes
) -> None:
    process = FakeProcess()
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *args, **kwargs: process)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, content=health))
    runtime = _LocalRuntime(settings)
    with pytest.raises(LocalRuntimeError, match="timed out"):
        runtime.start(0.01)
    assert process.terminated and runtime.process is None


@pytest.mark.parametrize("other_service", ["unprotected", "other-key", "other-model", "redirect"])
def test_port_collision_never_receives_images_or_questions(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch, other_service: str
) -> None:
    process = FakeProcess()
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *args, **kwargs: process)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if other_service == "unprotected":
            return httpx.Response(200, json={"model_path": str(settings.model_path)})
        if other_service == "redirect":
            return httpx.Response(302, headers={"Location": "https://example.invalid"})
        if other_service == "other-key" or len(requests) == 2:
            return httpx.Response(401)
        return httpx.Response(200, json={"model_path": "another-model.gguf"})

    _mock_http(monkeypatch, handler)
    client = LocalDecisionClient(settings.server_path, settings.model_path, settings.mmproj_path)
    result = client.evaluate(
        DecisionRequest("private-id", {"private": "state"}, {"tag": NoulQuestion("Fit?")})
    )
    assert result.error is not None and result.error.code == DecisionErrorCode.TRANSPORT
    assert not result.answers and process.terminated
    assert all(request.method == "GET" and request.url.host == "127.0.0.1" for request in requests)
    assert all(not request.content for request in requests)


def test_runtime_keys_are_unique_and_not_part_of_endpoint_repr(settings: RuntimeSettings) -> None:
    first, second = _LocalRuntime(settings), _LocalRuntime(settings)
    assert first.api_key != second.api_key
    assert first.api_key not in repr(ManagedEndpoint("http://127.0.0.1:11437", first.api_key))


def test_key_file_creation_failure_cleans_up_directory(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _LocalRuntime(settings)
    created: list[Path] = []
    original_open = runtime_module.os.open

    def fail_key_open(path: Any, *args: Any, **kwargs: Any) -> int:
        if Path(path).name == "api-key":
            created.append(Path(path).parent)
            raise PermissionError("cannot create credential")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(runtime_module.os, "open", fail_key_open)
    with pytest.raises(LocalRuntimeError) as caught:
        runtime.start(1)
    assert caught.value.code == DecisionErrorCode.CONFIGURATION
    assert created and not created[0].exists()
    assert runtime._key_file is runtime._key_directory is None


def test_older_windows_python_rejects_insecure_temporary_directory_support(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _LocalRuntime(settings)
    with monkeypatch.context() as scoped:
        scoped.setattr(runtime_module.sys, "platform", "win32")
        scoped.setattr(runtime_module.sys, "version_info", (3, 12, 3))
        with pytest.raises(LocalRuntimeError, match=r"Python 3\.12\.4") as caught:
            runtime.start(1)
    assert caught.value.code == DecisionErrorCode.CONFIGURATION
    assert runtime._key_file is runtime._key_directory is None


def test_failed_key_cleanup_keeps_path_for_a_later_shutdown(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _LocalRuntime(settings)
    runtime._create_key_file()
    key_file, key_directory = runtime._key_file, runtime._key_directory
    assert key_file is not None and key_directory is not None
    original_unlink = Path.unlink

    def fail_key_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == key_file:
            raise PermissionError("file is temporarily locked")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", fail_key_unlink)
        with pytest.raises(LocalRuntimeError, match="temporary credential"):
            runtime.close()
        assert runtime._key_file == key_file and key_file.exists()
    runtime.close()
    assert not key_file.exists() and not key_directory.exists()


def test_inherited_runtime_reference_never_deletes_parent_key_or_stops_parent(
    settings: RuntimeSettings,
) -> None:
    runtime = _LocalRuntime(settings)
    runtime._create_key_file()
    key_file = runtime._key_file
    assert key_file is not None
    process = FakeProcess()
    runtime.process = process  # type: ignore[assignment]
    inherited = copy.copy(runtime)
    inherited._owner_pid = os.getpid() + 1
    inherited.close()
    assert key_file.exists() and not process.terminated
    assert inherited.process is inherited._key_file is inherited._key_directory is None
    runtime.close()
    assert not key_file.exists() and process.terminated


def test_fork_hook_discards_state_without_closing_parent_resources(
    settings: RuntimeSettings,
) -> None:
    runtime = _LocalRuntime(settings)
    runtime._create_key_file()
    key_file = runtime._key_file
    assert key_file is not None
    runtime_module._active_runtime = runtime
    inherited_lock = runtime_module._runtime_lock
    inherited_lock.acquire()
    try:
        runtime_module._reset_after_fork()
        assert runtime_module._active_runtime is None
        assert runtime_module._runtime_lock is not inherited_lock
        assert runtime_module._runtime_lock.acquire(blocking=False)
        runtime_module._runtime_lock.release()
        runtime_module.shutdown_local_runtime()
        assert key_file.exists()
    finally:
        inherited_lock.release()
        runtime.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX fork lifecycle")
@pytest.mark.parametrize("scenario", ["other-thread", "session"])
def test_posix_fork_child_has_independent_ownership(scenario: str) -> None:
    script = """
import os, signal, sys, threading, time
from pathlib import Path
from image_annotator_lib.decisions import runtime as module

settings = module.RuntimeSettings(Path(sys.executable), Path(sys.executable), Path(sys.executable), "clef-flash", 0, 4096, ())
runtime = module._LocalRuntime(settings)
runtime._create_key_file()
key_file = runtime._key_file
directory = runtime._key_directory
class Process:
    def poll(self): return None
    def terminate(self): raise AssertionError("inherited process must not be terminated")
runtime.process = Process()
runtime.ready = True
module._active_runtime = runtime
release, held = threading.Event(), threading.Event()
def child_check():
    try:
        assert module._active_runtime is None
        assert module._runtime_lock.acquire(blocking=False)
        module._runtime_lock.release()
        module.shutdown_local_runtime()
        runtime.close()
        assert key_file.exists()
        return 0
    except BaseException:
        return 1
def lock_holder():
    with module._runtime_lock:
        held.set()
        release.wait(10)
thread = None
pid = -1
try:
    if sys.argv[1] == "other-thread":
        thread = threading.Thread(target=lock_holder)
        thread.start()
        assert held.wait(5)
        pid = os.fork()
        if pid == 0: os._exit(child_check())
    else:
        with module.runtime_session(settings, 1):
            pid = os.fork()
            if pid == 0: code = child_check()
        if pid == 0: os._exit(code)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            pid = -1
            assert os.waitstatus_to_exitcode(status) == 0
            break
        time.sleep(0.01)
    else:
        raise AssertionError("fork child hung")
    assert key_file.exists()
finally:
    if pid > 0:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    release.set()
    if thread is not None: thread.join(5)
    runtime.process = None
    module.shutdown_local_runtime()
assert not key_file.exists() and not directory.exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, scenario], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    instances: list[Any] = []

    class FakeRuntime:
        def __init__(self, settings: RuntimeSettings) -> None:
            self.settings = settings
            self.process: FakeProcess | None = FakeProcess()
            self.base_url = "http://127.0.0.1:11437"
            self.api_key = "test-runtime-key"
            self.closed = False
            self.starts = 0
            self.ready = False
            instances.append(self)

        def start(self, timeout: float) -> None:
            self.starts += 1
            self.ready = True

        def close(self) -> None:
            self.closed = True
            self.ready = False
            self.process = None

    monkeypatch.setattr(runtime_module, "_LocalRuntime", FakeRuntime)
    return instances


def test_clients_reuse_one_runtime_then_release_old_model_on_settings_or_file_change(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = _fake_runtime(monkeypatch)
    for _ in range(2):
        with runtime_module.runtime_session(settings, 1) as endpoint:
            assert endpoint == ManagedEndpoint("http://127.0.0.1:11437", "test-runtime-key")
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
