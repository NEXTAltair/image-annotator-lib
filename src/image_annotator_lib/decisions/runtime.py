"""Own one lazy llama.cpp Clef process, shared across decision clients."""

from __future__ import annotations

import atexit
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from .types import DecisionErrorCode


class LocalRuntimeError(RuntimeError):
    """A bounded local runtime failure safe to show to the caller."""

    def __init__(self, message: str, code: DecisionErrorCode = DecisionErrorCode.TRANSPORT) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeSettings:
    server_path: Path
    model_path: Path
    mmproj_path: Path
    model_name: str
    n_gpu_layers: int
    context_size: int
    file_versions: tuple[tuple[int, int, int], ...]


class _LocalRuntime:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url = ""
        self.ready = False

    def start(self, timeout: float) -> None:
        """Start a loopback server and wait for readiness within the time budget."""
        started = time.monotonic()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            self.base_url = f"http://127.0.0.1:{port}"
            settings = self.settings
            self.process = subprocess.Popen(
                [
                    str(settings.server_path),
                    "-m",
                    str(settings.model_path),
                    "--mmproj",
                    str(settings.mmproj_path),
                    "-a",
                    settings.model_name,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "-c",
                    str(settings.context_size),
                    "-b",
                    str(settings.context_size),
                    "-ub",
                    str(settings.context_size),
                    "-ngl",
                    str(settings.n_gpu_layers),
                    "--parallel",
                    "1",
                    "--no-webui",
                    "--no-context-shift",
                    "--offline",
                    *(["--device", "none", "--no-mmproj-offload"] if settings.n_gpu_layers == 0 else []),
                ],
                cwd=settings.server_path.parent,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._wait_ready(started + timeout)
            self.ready = True
        except OSError:
            self.close()
            raise LocalRuntimeError(
                "Local Clef server could not start. Check the executable and its runtime dependencies.",
                DecisionErrorCode.CONFIGURATION,
            ) from None
        except BaseException:
            self.close()
            raise

    def _wait_ready(self, deadline: float) -> None:
        assert self.process is not None
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise LocalRuntimeError(
                        "Local Clef server exited during startup. Check model/projector compatibility "
                        "and available memory.",
                        DecisionErrorCode.CONFIGURATION,
                    )
                try:
                    response = client.get(
                        f"{self.base_url}/health", timeout=max(0.001, min(1.0, deadline - time.monotonic()))
                    )
                    if response.status_code == 200:
                        health = response.json()
                        if (
                            isinstance(health, dict)
                            and health.get("status") == "ok"
                            and self.process.poll() is None
                        ):
                            return
                except (httpx.RequestError, ValueError):
                    pass  # Loading a model precedes the server's health endpoint.
                time.sleep(max(0, min(0.1, deadline - time.monotonic())))
        raise LocalRuntimeError("Local Clef model loading timed out. Check memory or increase the timeout.")

    def close(self) -> None:
        """Terminate only the subprocess owned by this runtime, then reap it."""
        self.ready = False
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    # It can exit between poll() and terminate(), especially on Windows.
                    if self.process.poll() is None:
                        raise
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            raise LocalRuntimeError(
                "The owned local Clef process could not be stopped. Close it before evaluating again."
            ) from None
        self.process = None


_runtime_lock = threading.Lock()
_active_runtime: _LocalRuntime | None = None


@contextmanager
def runtime_session(settings: RuntimeSettings, timeout: float) -> Iterator[str]:
    """Serialize loading and inference; replace obsolete settings before loading."""
    global _active_runtime
    if not _runtime_lock.acquire(timeout=timeout):
        raise LocalRuntimeError(
            "Another local Clef evaluation is still running. Try again when it finishes."
        )
    try:
        if _active_runtime is not None and (
            _active_runtime.settings != settings
            or not _active_runtime.ready
            or _active_runtime.process is None
            or _active_runtime.process.poll() is not None
        ):
            _active_runtime.close()
            _active_runtime = None
        if _active_runtime is None:
            _active_runtime = _LocalRuntime(settings)
            _active_runtime.start(timeout)
        try:
            yield _active_runtime.base_url
        except BaseException:
            # A timed-out request may still consume VRAM/compute in the server.
            _active_runtime.close()
            _active_runtime = None
            raise
    finally:
        _runtime_lock.release()


def shutdown_local_runtime() -> None:
    """Release the shared model; subsequent evaluations may start it again."""
    global _active_runtime
    with _runtime_lock:
        if _active_runtime is not None:
            _active_runtime.close()
            _active_runtime = None


atexit.register(shutdown_local_runtime)
