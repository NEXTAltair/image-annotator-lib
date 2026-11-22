"""子プロセスの寿命を、起動したこのプロセスの寿命に結び付ける。

``atexit`` は強制終了やクラッシュでは走らないため、回収は OS に任せる。

- Linux: 子に ``PR_SET_PDEATHSIG(SIGKILL)`` を設定する。この signal は子を作った「スレッド」の
  終了で届くので、子は必ずプロセスと同じ寿命の 1 スレッドから作る。設定は短い Python の launcher が
  行い、PID を保ったまま exec する (``preexec_fn`` はスレッドのあるプロセスで安全でない)。
- Windows: 子を停止状態で作り、kill-on-close の Job Object に入れてから再開する。
  Job の handle はこのプロセスだけが終了まで持つ。
- それ以外の OS には仕掛けが無く、所有プロセスの正常終了処理だけに依存する。

所有権の説明は docs/local-clef-process-lifetime.md を参照。
"""

from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path

import psutil

# launcher が exec できなかったときの終了 code。動的リンクの失敗時に loader が返す値と同じ。
LAUNCH_FAILED = 127

# `python -I -S -c` で動かし、server の置き場所や site の hook からコードを読み込ませない。
# Python の起動処理が変えて exec 後に残るものは戻す: 無視される SIGPIPE / SIGXFSZ と、
# locale coercion が足す LC_CTYPE (そのため受け取った時点の環境変数を渡す)。
_LINUX_LAUNCHER = f"""
import ctypes, os, signal, sys
try:
    owner, program = int(sys.argv[1]), sys.argv[2]
    prctl = ctypes.CDLL(None).prctl
    if prctl(1, ctypes.c_ulong(signal.SIGKILL)) != 0 or os.getppid() != owner:
        os._exit({LAUNCH_FAILED})
    with open("/proc/self/environ", "rb") as stream:
        entries = stream.read().split(b"\\0")
    env = dict(entry.split(b"=", 1) for entry in entries if b"=" in entry)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    os.execve(program, sys.argv[2:], env)
except OSError:
    os._exit({LAUNCH_FAILED})
"""

_CREATE_SUSPENDED = 0x00000004
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class ProcessLifetimeError(OSError):
    """子を所有プロセスに結び付けられなかった。その子は動かしたままにしない。"""


# ctypes.wintypes は使わない。LP64 では DWORD が 8 byte になり、構造体の配置を検証できない。
class _BasicLimits(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    )


class _ExtendedLimits(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


_SpawnRequest = tuple[Callable[[], subprocess.Popen[bytes]], Future[subprocess.Popen[bytes]]]
_lock = threading.Lock()
_spawn_requests: queue.SimpleQueue[_SpawnRequest] | None = None
_job_handle: int | None = None


def spawn_owned(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    """このプロセスが終わると OS に停止される子プロセスを、入出力なしで起動する。

    正常時の停止と回収は呼び出し元が行う。

    Args:
        command: 実行ファイルの絶対パスと引数。
        cwd: 子の作業ディレクトリ。
        env: 子へ渡す環境変数の全体。

    Returns:
        起動した子。PID は実行ファイル自身の PID と一致する。

    Raises:
        ProcessLifetimeError: このプロセスに結び付けられなかった。
        OSError: 子を作れなかった。
    """

    def launch(args: Sequence[str], creationflags: int = 0) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | creationflags,
        )

    if sys.platform == "win32":
        return _spawn_in_job(lambda flags: launch(command, flags))
    if sys.platform == "linux":
        bound_command = _parent_bound_command(command)
        return _spawn_on_lasting_thread(lambda: launch(bound_command))
    return launch(command)


def _parent_bound_command(command: Sequence[str]) -> list[str]:
    if not sys.executable:
        raise ProcessLifetimeError("子の起動に使う Python が見つかりません")
    return [sys.executable, "-I", "-S", "-c", _LINUX_LAUNCHER, str(os.getpid()), *command]


def _spawn_on_lasting_thread(launch: Callable[[], subprocess.Popen[bytes]]) -> subprocess.Popen[bytes]:
    """プロセスの終了まで終わらないスレッドで子を作る。

    ThreadPoolExecutor は使えない。atexit や他スレッドの終了より前に join され、
    使用中の server を停止させてしまう。
    """
    global _spawn_requests
    with _lock:
        if _spawn_requests is None:
            requests: queue.SimpleQueue[_SpawnRequest] = queue.SimpleQueue()
            try:
                threading.Thread(
                    target=_serve_spawns, args=(requests,), name="clef-process-owner", daemon=True
                ).start()
            except RuntimeError as exc:  # インタプリタ終了処理中など。
                raise ProcessLifetimeError("子を所有するスレッドを開始できません") from exc
            _spawn_requests = requests
        future: Future[subprocess.Popen[bytes]] = Future()
        _spawn_requests.put((launch, future))
    try:
        return future.result()
    except BaseException:
        # 待機を中断された呼び出し元は子を受け取れないので、ここで止める。
        future.add_done_callback(_discard_unclaimed)
        raise


def _serve_spawns(requests: queue.SimpleQueue[_SpawnRequest]) -> None:
    while True:
        launch, future = requests.get()
        try:
            future.set_result(launch())
        except BaseException as exc:
            # 呼び出し元へ渡す。このスレッドが終わると全ての子が停止するので、例外で抜けない。
            future.set_exception(exc)


def _discard_unclaimed(future: Future[subprocess.Popen[bytes]]) -> None:
    if future.exception() is None:
        _discard(future.result())


def _discard(process: subprocess.Popen[bytes]) -> None:
    process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def _spawn_in_job(launch: Callable[[int], subprocess.Popen[bytes]]) -> subprocess.Popen[bytes]:
    kernel32 = _kernel32()
    with _lock:
        job = _owner_job(kernel32)
    process = launch(_CREATE_SUSPENDED)
    try:
        # Popen が handle を保持しているので、この間に PID は再利用されない。
        _assign_to_job(kernel32, job, process.pid)
        try:
            psutil.Process(process.pid).resume()
        except psutil.Error as exc:
            raise ProcessLifetimeError("子プロセスを再開できません") from exc
    except BaseException:
        _discard(process)
        raise
    return process


def _owner_job(kernel32: ctypes.CDLL) -> int:
    """このプロセスの Job を返す。無ければ 1 度だけ作る。

    handle は意図して閉じない。閉じることが子の停止条件であり、所有プロセスの終了時に OS が閉じる。
    継承不可なので、子が handle を持ち続けることもない。
    """
    global _job_handle
    if _job_handle is not None:
        return _job_handle
    job: int | None = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise _api_error("CreateJobObjectW")
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
    ):
        error = _api_error("SetInformationJobObject")
        kernel32.CloseHandle(job)
        raise error
    _job_handle = job
    return job


def _assign_to_job(kernel32: ctypes.CDLL, job: int, pid: int) -> None:
    handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, 0, pid)
    if not handle:
        raise _api_error("OpenProcess")
    try:
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise _api_error("AssignProcessToJobObject")
    finally:
        kernel32.CloseHandle(handle)


def _api_error(operation: str) -> ProcessLifetimeError:
    code = 0
    if sys.platform == "win32":
        code = ctypes.get_last_error()
    return ProcessLifetimeError(f"{operation} に失敗しました (Windows error {code})")


def _kernel32() -> ctypes.CDLL:
    if sys.platform != "win32":
        raise ProcessLifetimeError("Job Object は Windows でのみ使えます")
    else:  # mypy が Windows のときだけ検査するよう分岐にする。
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, boolean, dword = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        kernel32.CreateJobObjectW.restype = handle
        kernel32.SetInformationJobObject.argtypes = (handle, ctypes.c_int, ctypes.c_void_p, dword)
        kernel32.SetInformationJobObject.restype = boolean
        kernel32.OpenProcess.argtypes = (dword, boolean, dword)
        kernel32.OpenProcess.restype = handle
        kernel32.AssignProcessToJobObject.argtypes = (handle, handle)
        kernel32.AssignProcessToJobObject.restype = boolean
        kernel32.CloseHandle.argtypes = (handle,)
        kernel32.CloseHandle.restype = boolean
        return kernel32


def _reset_after_fork() -> None:
    """fork した子は、親の起動用スレッドも所有権も引き継がない。"""
    global _lock, _spawn_requests
    _lock = threading.Lock()
    _spawn_requests = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
