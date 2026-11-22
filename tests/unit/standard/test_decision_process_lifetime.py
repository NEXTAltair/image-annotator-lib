"""子プロセスを所有プロセスの寿命へ結び付ける境界のテスト。"""

from __future__ import annotations

import ctypes
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from image_annotator_lib.decisions import DecisionErrorCode, process_lifetime
from image_annotator_lib.decisions import runtime as runtime_module
from image_annotator_lib.decisions.runtime import LocalRuntimeError, RuntimeSettings, _LocalRuntime

pytestmark = [pytest.mark.unit, pytest.mark.standard]

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death launcher")

JOB, PROCESS_HANDLE, PID = 0x7000_0000_0010, 0x7000_0000_0020, 4242


class Child:
    pid = PID

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.wait_error: Exception | None = None

    def kill(self) -> None:
        self.events.append(("kill",))

    def wait(self, timeout: float) -> int:
        self.events.append(("wait", timeout))
        if self.wait_error is not None:
            raise self.wait_error
        return 1


class Kernel32:
    """Job Object への登録に使う Win32 API の fake。全ての呼び出しを記録する。"""

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.failing: str | None = None

    def CreateJobObjectW(self, security: None, name: None) -> int | None:
        self.events.append(("create-job", security, name))
        return None if self.failing == "create-job" else JOB

    def SetInformationJobObject(self, job: int, kind: int, info: Any, size: int) -> int:
        raw = ctypes.string_at(info, size)
        self.events.append(("set-limits", job, kind, size, int.from_bytes(raw[16:20], sys.byteorder)))
        assert not any(raw[:16]) and not any(raw[20:]), "only the kill-on-close limit may be set"
        return 0 if self.failing == "set-limits" else 1

    def OpenProcess(self, access: int, inherit: int, pid: int) -> int | None:
        self.events.append(("open", access, inherit, pid))
        return None if self.failing == "open" else PROCESS_HANDLE

    def AssignProcessToJobObject(self, job: int, process: int) -> int:
        self.events.append(("assign", job, process))
        return 0 if self.failing == "assign" else 1

    def CloseHandle(self, handle: int) -> int:
        self.events.append(("close-handle", handle))
        return 1


class Windows:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.kernel32 = Kernel32(self.events)
        self.child = Child(self.events)
        self.resume_error: BaseException | None = None
        windows = self

        class Handle:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def resume(self) -> None:
                windows.events.append(("resume", self.pid))
                if windows.resume_error is not None:
                    raise windows.resume_error

        monkeypatch.setattr(process_lifetime, "_kernel32", lambda: self.kernel32)
        monkeypatch.setattr(process_lifetime, "_job_handle", None)
        monkeypatch.setattr(process_lifetime.psutil, "Process", Handle)

    def launch(self, flags: int) -> Any:
        self.events.append(("launch", flags))
        return self.child

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> Windows:
    return Windows(monkeypatch)


def test_windows_child_joins_kill_on_close_job_before_it_runs(windows: Windows) -> None:
    assert process_lifetime._spawn_in_job(windows.launch) is windows.child
    assert windows.events == [
        ("create-job", None, None),  # security 属性なし = handle は継承不可。
        ("set-limits", JOB, 9, ctypes.sizeof(process_lifetime._ExtendedLimits), 0x2000),
        ("launch", 0x4),  # CREATE_SUSPENDED
        ("open", 0x101, 0, PID),
        ("assign", JOB, PROCESS_HANDLE),
        ("close-handle", PROCESS_HANDLE),
        ("resume", PID),
    ]


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="64-bit Win32 layout")
def test_job_limit_structures_match_the_win64_layout() -> None:
    assert ctypes.sizeof(process_lifetime._BasicLimits) == 64
    assert process_lifetime._BasicLimits.LimitFlags.offset == 16
    assert process_lifetime._BasicLimits.Affinity.offset == 48
    assert ctypes.sizeof(process_lifetime._ExtendedLimits) == 144
    assert process_lifetime._ExtendedLimits.ProcessMemoryLimit.offset == 112


def test_windows_job_handle_is_shared_and_held_until_the_owner_ends(
    windows: Windows, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(3):
        process_lifetime._spawn_in_job(windows.launch)
    assert windows.names().count("create-job") == windows.names().count("set-limits") == 1
    assert windows.names().count("assign") == windows.names().count("resume") == 3
    runtime_module.shutdown_local_runtime()
    assert ("close-handle", JOB) not in windows.events
    assert process_lifetime._job_handle == JOB


@pytest.mark.parametrize("stage", ["create-job", "set-limits"])
def test_windows_job_setup_failure_starts_no_process(windows: Windows, stage: str) -> None:
    windows.kernel32.failing = stage
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._spawn_in_job(windows.launch)
    assert "launch" not in windows.names() and process_lifetime._job_handle is None
    assert (("close-handle", JOB) in windows.events) is (stage == "set-limits")

    windows.kernel32.failing = None
    assert process_lifetime._spawn_in_job(windows.launch) is windows.child
    assert process_lifetime._job_handle == JOB


@pytest.mark.parametrize("stage", ["open", "assign", "resume", "interrupt"])
def test_windows_child_that_cannot_be_bound_is_killed_and_reaped(windows: Windows, stage: str) -> None:
    expected: type[BaseException] = process_lifetime.ProcessLifetimeError
    if stage == "resume":
        windows.resume_error = psutil.AccessDenied(PID)
    elif stage == "interrupt":
        windows.resume_error, expected = KeyboardInterrupt(), KeyboardInterrupt
    else:
        windows.kernel32.failing = stage
    with pytest.raises(expected):
        process_lifetime._spawn_in_job(windows.launch)
    assert windows.events[-2:] == [("kill",), ("wait", 5)]
    if stage in {"open", "assign"}:
        assert "resume" not in windows.names(), "an unbound child must never run"
    assert (("close-handle", PROCESS_HANDLE) in windows.events) is (stage != "open")
    assert ("close-handle", JOB) not in windows.events


def test_windows_reap_timeout_does_not_hide_the_binding_error(windows: Windows) -> None:
    windows.kernel32.failing = "assign"
    windows.child.wait_error = subprocess.TimeoutExpired("server", 5)
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._spawn_in_job(windows.launch)


def test_windows_launch_failure_keeps_the_job_for_the_next_attempt(windows: Windows) -> None:
    def missing(flags: int) -> Any:
        raise FileNotFoundError("server")

    with pytest.raises(FileNotFoundError):
        process_lifetime._spawn_in_job(missing)
    assert process_lifetime._job_handle == JOB and ("close-handle", JOB) not in windows.events


def test_windows_spawn_is_hidden_and_suspended(
    windows: Windows, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(args: list[str], **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return windows.child

    monkeypatch.setattr(process_lifetime.subprocess, "CREATE_NO_WINDOW", 0x0800_0000, raising=False)
    monkeypatch.setattr(process_lifetime.subprocess, "Popen", popen)
    with monkeypatch.context() as scoped:
        scoped.setattr(process_lifetime.sys, "platform", "win32")
        child = process_lifetime.spawn_owned(["server.exe", "-m", "model"], cwd=tmp_path, env={"A": "1"})
    assert child is windows.child
    args, options = calls[0]
    assert args == ["server.exe", "-m", "model"]
    assert options["creationflags"] == 0x0800_0004 and options["shell"] is False
    assert options.get("close_fds", True) is True
    assert "resume" in windows.names()


def test_win32_api_is_declared_with_pointer_sized_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "CreateJobObjectW",
        "SetInformationJobObject",
        "OpenProcess",
        "AssignProcessToJobObject",
        "CloseHandle",
    )
    library = SimpleNamespace(**{name: SimpleNamespace() for name in names})
    loaded: list[tuple[str, bool]] = []

    def load(name: str, use_last_error: bool) -> Any:
        loaded.append((name, use_last_error))
        return library

    monkeypatch.setattr(process_lifetime.ctypes, "WinDLL", load, raising=False)
    with monkeypatch.context() as scoped:
        scoped.setattr(process_lifetime.sys, "platform", "win32")
        assert process_lifetime._kernel32() is library
    assert loaded == [("kernel32", True)]
    handle, dword = ctypes.c_void_p, ctypes.c_uint32
    assert library.CreateJobObjectW.restype is library.OpenProcess.restype is handle
    assert library.CreateJobObjectW.argtypes == (ctypes.c_void_p, ctypes.c_wchar_p)
    assert library.SetInformationJobObject.argtypes == (handle, ctypes.c_int, ctypes.c_void_p, dword)
    assert library.OpenProcess.argtypes == (dword, ctypes.c_int, dword)
    assert library.AssignProcessToJobObject.argtypes == (handle, handle)
    assert library.CloseHandle.argtypes == (handle,)


def test_binding_failure_is_a_typed_startup_error_without_leftovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[Path] = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        assert runtime._key_directory is not None
        created.append(runtime._key_directory)
        raise process_lifetime.ProcessLifetimeError("private system detail")

    monkeypatch.setattr(runtime_module, "spawn_owned", refuse)
    server = tmp_path / "server"
    runtime = _LocalRuntime(RuntimeSettings(server, server, server, "clef-flash", 0, 4096, ()))
    with pytest.raises(LocalRuntimeError, match="could not be tied") as caught:
        runtime.start(1)
    assert caught.value.code == DecisionErrorCode.CONFIGURATION
    assert "private system detail" not in str(caught.value)
    assert runtime.process is None and created and not created[0].exists()


def test_every_child_is_created_by_one_thread_that_outlives_the_callers() -> None:
    creators: list[threading.Thread] = []
    child = Child([])

    def launch() -> Any:
        creators.append(threading.current_thread())
        return child

    for _ in range(3):
        caller = threading.Thread(target=process_lifetime._spawn_on_lasting_thread, args=(launch,))
        caller.start()
        caller.join(5)
        assert not caller.is_alive()
    assert len(creators) == 3 and len(set(creators)) == 1
    assert creators[0].daemon and creators[0].is_alive()


def test_failed_launch_does_not_end_the_owning_thread() -> None:
    creators: list[threading.Thread] = []

    def fail() -> Any:
        creators.append(threading.current_thread())
        raise FileNotFoundError("server")

    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            process_lifetime._spawn_on_lasting_thread(fail)
    assert creators[0] is creators[1] and creators[0].is_alive()


def test_child_of_an_interrupted_caller_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[Any, ...]] = []
    child = Child(events)
    launching, release = threading.Event(), threading.Event()
    reaped = threading.Event()

    class InterruptedFuture(process_lifetime.Future):  # 最初の待機が Ctrl-C で中断される。
        interrupted = False

        def result(self, timeout: float | None = None) -> Any:
            if not self.interrupted:
                self.interrupted = True
                assert launching.wait(5)
                raise KeyboardInterrupt
            return super().result(timeout)

    def launch() -> Any:
        launching.set()
        assert release.wait(5)
        return child

    monkeypatch.setattr(child, "wait", lambda timeout: reaped.set())
    monkeypatch.setattr(process_lifetime, "Future", InterruptedFuture)
    try:
        with pytest.raises(KeyboardInterrupt):
            process_lifetime._spawn_on_lasting_thread(launch)
    finally:
        release.set()
    assert reaped.wait(5) and events == [("kill",)]


def test_owning_thread_start_failure_is_typed_and_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(thread: threading.Thread) -> None:
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(process_lifetime, "_spawn_requests", None)
    monkeypatch.setattr(process_lifetime.threading.Thread, "start", refuse)
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._spawn_on_lasting_thread(lambda: pytest.fail("must not launch"))
    assert process_lifetime._spawn_requests is None


def test_missing_interpreter_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_lifetime.sys, "executable", "")
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._parent_bound_command(["/bin/true"], 3, 4)


def test_no_child_is_started_during_interpreter_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_lifetime.sys, "is_finalizing", lambda: True)
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._spawn_on_lasting_thread(lambda: pytest.fail("must not launch"))


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (b"", process_lifetime.ProcessLifetimeError),  # launcher が exec 前に異常終了した
        (b"P", process_lifetime.ProcessLifetimeError),
        (b"R", process_lifetime.ProcessLifetimeError),
        (b"+E2", FileNotFoundError),
        (b"+E13", PermissionError),
        (b"+Ex", process_lifetime.ProcessLifetimeError),
        (b"++", process_lifetime.ProcessLifetimeError),
    ],
)
def test_launch_report_distinguishes_binding_from_executable_failures(
    report: bytes, expected: type[BaseException]
) -> None:
    with pytest.raises(expected) as caught:
        process_lifetime._check_launch_report(report)
    assert type(caught.value) is expected or issubclass(type(caught.value), expected)
    if expected is process_lifetime.ProcessLifetimeError:
        assert type(caught.value) is process_lifetime.ProcessLifetimeError
    process_lifetime._check_launch_report(b"+")


def _run_bound(command: list[str], **options: Any) -> tuple[subprocess.Popen[bytes], bytes]:
    """本番と同じ手順で launcher を起動し、標準出力を返す。"""
    started: list[subprocess.Popen[bytes]] = []

    def launch(args: Any, fds: Any) -> subprocess.Popen[bytes]:
        started.append(subprocess.Popen(args, pass_fds=fds, stdout=subprocess.PIPE, **options))
        return started[-1]

    try:
        process = process_lifetime._launch_parent_bound(command, launch)
    except BaseException:
        for child in started:
            child.communicate(timeout=10)
        raise
    output, _ = process.communicate(timeout=10)
    return process, output


@linux_only
@pytest.mark.parametrize("env", [{}, {"LANG": "C", "EXTRA": "a=b"}, {"LC_ALL": "C.UTF-8"}])
def test_launcher_hands_over_the_same_environment_signals_and_fds_as_a_direct_start(
    env: dict[str, str],
) -> None:
    # 直接起動の比較対象も pass_fds 付きにする。CPython は pass_fds の有無で起動経路を変え、
    # 経路によって glibc 内部用 signal (32, 33) の無視が子に残る。launcher 由来の差だけを見る。
    spare_read, spare_write = os.pipe()
    commands = (
        (["/usr/bin/env", "-0"], (spare_read,)),
        (["/bin/grep", "-E", "^Sig(Ign|Blk)", "/proc/self/status"], (spare_read,)),
        (["/bin/ls", "/proc/self/fd"], ()),
    )
    for command, fds in commands:
        direct = subprocess.run(command, capture_output=True, env=env, check=True, timeout=10, pass_fds=fds)
        process, output = _run_bound(command, env=env)
        assert process.returncode == 0
        assert sorted(output.split(b"\0")) == sorted(direct.stdout.split(b"\0"))
    os.close(spare_read)
    os.close(spare_write)


@linux_only
def test_launcher_sets_sigkill_as_parent_death_signal_on_the_executed_program() -> None:
    query = (
        "import ctypes; value = ctypes.c_int(); "
        "assert ctypes.CDLL(None).prctl(2, ctypes.byref(value)) == 0; print(value.value)"
    )
    _, output = _run_bound([sys.executable, "-c", query])
    assert output.strip() == str(int(signal.SIGKILL)).encode()


@linux_only
@pytest.mark.parametrize("owner", ["exited", "exec"])
def test_launcher_does_not_run_the_program_when_the_owner_is_already_gone(
    tmp_path: Path, owner: str
) -> None:
    marker = tmp_path / "ran"
    go_read, go_write = os.pipe()
    report_read, report_write = os.pipe()
    command = process_lifetime._parent_bound_command(["/usr/bin/touch", str(marker)], go_read, report_write)
    if owner == "exited":
        command[5] = str(os.getppid())  # launcher の親ではない PID (付け替え後と同じ状態)。
    else:
        os.close(go_write)  # exec で close-on-exec の返信端が閉じ、返信が届かない状態。
    with subprocess.Popen(command, pass_fds=(go_read, report_write)) as launcher:
        for descriptor in (go_read, report_write):
            os.close(descriptor)
        report = process_lifetime._read_report(report_read, time.monotonic() + 10)
        assert launcher.wait(10) == process_lifetime.LAUNCH_FAILED
    os.close(report_read)
    if owner == "exited":
        os.close(go_write)
    assert report == (b"P" if owner == "exited" else b"R")
    assert not marker.exists()
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._check_launch_report(report)


@linux_only
def test_launcher_waits_for_the_owning_thread_before_executing(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    go_read, go_write = os.pipe()
    report_read, report_write = os.pipe()
    command = process_lifetime._parent_bound_command(["/usr/bin/touch", str(marker)], go_read, report_write)
    with subprocess.Popen(command, pass_fds=(go_read, report_write)) as launcher:
        for descriptor in (go_read, report_write):
            os.close(descriptor)
        assert process_lifetime._read_report(report_read, time.monotonic() + 10, until=b"R") == b"R"
        assert not select.select([report_read], [], [], 0.3)[0], "exec must wait for the reply"
        assert launcher.poll() is None and not marker.exists()
        os.write(go_write, b"G")
        os.close(go_write)
        assert process_lifetime._read_report(report_read, time.monotonic() + 10) == b"+"
        assert launcher.wait(10) == 0
    os.close(report_read)
    assert marker.exists()


@linux_only
@pytest.mark.parametrize("problem", ["missing", "not-executable"])
def test_launcher_reports_why_the_program_could_not_be_executed(tmp_path: Path, problem: str) -> None:
    program = tmp_path / "server"
    if problem == "not-executable":
        program.write_text("#!/bin/sh\n", encoding="utf-8")
        program.chmod(0o600)
    expected = FileNotFoundError if problem == "missing" else PermissionError
    with pytest.raises(expected):
        _run_bound([str(program)])


@linux_only
def test_launcher_that_dies_before_exec_is_a_binding_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_lifetime.sys, "executable", "/bin/false")
    with pytest.raises(process_lifetime.ProcessLifetimeError):
        _run_bound(["/bin/true"])


@linux_only
def test_silent_launcher_is_killed_after_the_report_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    silent = tmp_path / "python"
    silent.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    silent.chmod(0o700)
    monkeypatch.setattr(process_lifetime.sys, "executable", str(silent))
    monkeypatch.setattr(process_lifetime, "_LAUNCH_REPORT_TIMEOUT", 0.5)
    started: list[subprocess.Popen[bytes]] = []

    def launch(args: Any, fds: Any) -> subprocess.Popen[bytes]:
        started.append(subprocess.Popen(args, pass_fds=fds))
        return started[-1]

    with pytest.raises(process_lifetime.ProcessLifetimeError):
        process_lifetime._launch_parent_bound(["/bin/true"], launch)
    assert started[0].returncode == -signal.SIGKILL


@linux_only
def test_launcher_never_imports_code_from_the_server_directory(tmp_path: Path) -> None:
    for name in ("ctypes", "select", "signal"):
        (tmp_path / f"{name}.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    process, _ = _run_bound(["/bin/true"], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert process.returncode == 0


class _DescriptorLedger:
    """process_lifetime が作った pipe の記述子と、そのモジュールが行った close を記録する。"""

    def __init__(self) -> None:
        self.created: list[int] = []
        self.closed: list[int] = []

    def pipe(self) -> tuple[int, int]:
        pair = os.pipe()
        self.created.extend(pair)
        return pair

    def close(self, descriptor: int) -> None:
        self.closed.append(descriptor)
        os.close(descriptor)

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


def _prepare_launch_path(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[str], type[BaseException] | None]:
    """経路ごとに launcher の状況を作り、起動する command と期待する例外を返す。"""
    if path == "exec-failure":
        return [str(tmp_path / "missing")], FileNotFoundError
    if path == "launcher-dies":
        monkeypatch.setattr(process_lifetime.sys, "executable", "/bin/false")
        return ["/bin/true"], process_lifetime.ProcessLifetimeError
    if path == "timeout":
        silent = tmp_path / "python"
        silent.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
        silent.chmod(0o700)
        monkeypatch.setattr(process_lifetime.sys, "executable", str(silent))
        monkeypatch.setattr(process_lifetime, "_LAUNCH_REPORT_TIMEOUT", 0.5)
        return ["/bin/true"], process_lifetime.ProcessLifetimeError
    if path == "interrupt":
        read_report = process_lifetime._read_report

        def interrupted(descriptor: int, deadline: float, until: bytes | None = None) -> bytes:
            read_report(descriptor, deadline, until)
            raise KeyboardInterrupt

        monkeypatch.setattr(process_lifetime, "_read_report", interrupted)
        return ["/bin/true"], KeyboardInterrupt
    if path == "popen":
        return ["/bin/true"], FileNotFoundError
    return ["/bin/true"], None


@linux_only
@pytest.mark.parametrize(
    "path", ["success", "exec-failure", "launcher-dies", "timeout", "interrupt", "popen"]
)
def test_each_launch_pipe_end_is_closed_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    # 同じ番号を二度 close すると、間に別スレッドが開いた無関係な記述子を閉じてしまう。
    ledger = _DescriptorLedger()
    monkeypatch.setattr(process_lifetime, "os", ledger)
    command, expected = _prepare_launch_path(path, monkeypatch, tmp_path)
    started: list[subprocess.Popen[bytes]] = []

    def launch(args: Any, fds: Any) -> subprocess.Popen[bytes]:
        if path == "popen":
            raise FileNotFoundError("python")
        started.append(subprocess.Popen(args, pass_fds=fds))
        return started[-1]

    try:
        if expected is None:
            process_lifetime._launch_parent_bound(command, launch)
        else:
            with pytest.raises(expected):
                process_lifetime._launch_parent_bound(command, launch)
    finally:
        for child in started:  # このテストが起動した launcher だけを止める。
            if child.poll() is None:
                child.kill()
            child.wait(10)
    assert len(ledger.created) == 4
    assert sorted(ledger.closed) == sorted(ledger.created), ledger.closed


@linux_only
def test_spawn_succeeds_when_descriptor_numbers_exceed_select_limit(tmp_path: Path) -> None:
    # select.select() は FD_SETSIZE (1024) 以上の記述子を扱えず ValueError になる。
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 1100
    if soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            pytest.skip(f"RLIMIT_NOFILE の hard limit ({hard}) が {needed} 未満")
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    base = os.open(os.devnull, os.O_RDONLY)
    held = [base]
    try:
        while held[-1] < 1030:
            held.append(os.dup(base))
        child = process_lifetime.spawn_owned(["/bin/true"], cwd=tmp_path, env={})
        assert child.wait(10) == 0
    finally:
        for descriptor in held:
            os.close(descriptor)
        if soft < needed:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
