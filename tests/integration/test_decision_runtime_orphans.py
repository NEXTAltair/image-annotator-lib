"""実プロセスのテスト: 管理下の Clef server は所有プロセスより長生きしない。

server は起動時の handshake だけに応答する代用プロセス。signal を送る相手はこのテストが作った
プロセスに限り、Popen か pidfd で指定するので、再利用された PID に当たらない。
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from image_annotator_lib.decisions import process_lifetime
from image_annotator_lib.decisions import runtime as runtime_module
from image_annotator_lib.decisions.runtime import LocalRuntimeError, RuntimeSettings
from image_annotator_lib.decisions.types import DecisionErrorCode

pytestmark = [
    pytest.mark.integration,
    pytest.mark.fast_integration,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal and pidfd"),
]

# package の __init__ は全モデルを登録して約 8 秒かかる。runtime には不要なので、
# package の場所だけを与えて decisions を実物のまま import する。
_IMPORT_RUNTIME = f"""
import types
package = types.ModuleType("image_annotator_lib")
package.__path__ = [{str(Path(runtime_module.__file__).parents[1])!r}]
sys.modules[package.__name__] = package
from image_annotator_lib.decisions import runtime as module
"""

_SERVER = """#!{python}
import http.server, json, os, signal, sys, time

signal.alarm(90)  # テストが失敗しても残り続けない。
args = sys.argv
key = open(args[args.index("--api-key-file") + 1]).read().strip()
record = os.path.join({records!r}, str(os.getpid()))
with open(record + ".tmp", "w") as stream:
    json.dump({{"pid": os.getpid(), "argv": args}}, stream)
os.replace(record + ".tmp", record + ".json")
if os.environ.get("CLEF_TEST_LOADING"):
    time.sleep(90)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            status, body = 200, {{"status": "ok"}}
        elif self.headers.get("Authorization") != "Bearer " + key:
            status, body = 401, {{}}
        else:
            status, body = 200, {{"model_path": args[args.index("-m") + 1]}}
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

port = int(args[args.index("--port") + 1])
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

_OWNER = """
import os, signal, sys, threading, time
from pathlib import Path
{imports}

server, model, scenario = sys.argv[1:]
settings = module.RuntimeSettings(Path(server), Path(model), Path(model), "clef-flash", 0, 4096, ())

def start():
    with module.runtime_session(settings, 20):
        pass

if scenario == "main-thread":
    start()
elif scenario == "loading":
    threading.Thread(target=start, daemon=True).start()
else:
    worker = threading.Thread(target=start)
    worker.start()
    worker.join()
print("started", flush=True)
command = sys.stdin.readline().strip()
if command == "os._exit":
    os._exit(0)
if command == "execv":
    os.execv(sys.executable, [sys.executable, "-c", "import time; time.sleep(30)"])
time.sleep(30)
"""


@pytest.fixture
def records(tmp_path: Path) -> Path:
    directory = tmp_path / "records"
    directory.mkdir()
    return directory


@pytest.fixture
def settings(tmp_path: Path, records: Path) -> RuntimeSettings:
    server = tmp_path / "llama-server"
    server.write_text(_SERVER.format(python=sys.executable, records=str(records)), encoding="utf-8")
    server.chmod(0o700)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test")
    return RuntimeSettings(server, model, model, "clef-flash", 0, 4096, ())


@pytest.fixture(autouse=True)
def isolate_runtime() -> Iterator[None]:
    runtime_module.shutdown_local_runtime()
    yield
    runtime_module.shutdown_local_runtime()


def _server_pid(records: Path) -> int:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        for path in records.glob("*.json"):
            return int(json.loads(path.read_text(encoding="utf-8"))["pid"])
        time.sleep(0.02)
    raise AssertionError("the stand-in server did not start")


def _exited(descriptor: int, timeout: float) -> bool:
    # pidfd は終了時に読めるようになる。未回収の zombie でも判定できる。
    return bool(select.select([descriptor], [], [], timeout)[0])


@contextmanager
def _watch(pid: int, *, kill: bool = True) -> Iterator[int]:
    descriptor = os.pidfd_open(pid)
    try:
        yield descriptor
    finally:
        if kill and not _exited(descriptor, 0):
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        os.close(descriptor)


@contextmanager
def _owner(settings: RuntimeSettings, scenario: str, tmp_path: Path) -> Iterator[subprocess.Popen[str]]:
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    if scenario == "loading":
        env["CLEF_TEST_LOADING"] = "1"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _OWNER.replace("{imports}", _IMPORT_RUNTIME),
            str(settings.server_path),
            str(settings.model_path),
            scenario,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert owner.stdout is not None
        assert select.select([owner.stdout], [], [], 30)[0], "the owner process did not start"
        assert owner.stdout.readline().strip() == "started"
        yield owner
    finally:
        owner.kill()
        owner.communicate(timeout=10)


@pytest.mark.parametrize("scenario", ["main-thread", "worker-thread", "loading"])
def test_sigkilled_owner_takes_its_server_down(
    settings: RuntimeSettings, records: Path, tmp_path: Path, scenario: str
) -> None:
    with _owner(settings, scenario, tmp_path) as owner, _watch(_server_pid(records)) as server:
        assert not _exited(server, 0.5), "the server must run while its owner is alive"
        os.kill(owner.pid, signal.SIGKILL)
        assert _exited(server, 10), "the server outlived its killed owner"


@pytest.mark.parametrize("ending", ["SIGTERM", "os._exit", "execv"])
def test_owner_ending_without_python_cleanup_takes_its_server_down(
    settings: RuntimeSettings, records: Path, tmp_path: Path, ending: str
) -> None:
    with _owner(settings, "worker-thread", tmp_path) as owner, _watch(_server_pid(records)) as server:
        assert not _exited(server, 0.5), "the server must run while its owner is alive"
        if ending == "SIGTERM":
            owner.terminate()
        else:
            assert owner.stdin is not None
            owner.stdin.write(f"{ending}\n")
            owner.stdin.flush()
        assert _exited(server, 10), f"the server outlived an owner ended by {ending}"


def test_server_survives_the_worker_thread_that_started_it(settings: RuntimeSettings) -> None:
    pids: list[int] = []

    def evaluate() -> None:
        with runtime_module.runtime_session(settings, 20):
            active = runtime_module._active_runtime
            assert active is not None and active.process is not None
            pids.append(active.process.pid)

    for _ in range(3):
        worker = threading.Thread(target=evaluate)
        worker.start()
        worker.join(30)
        assert not worker.is_alive() and pids
        with _watch(pids[-1], kill=False) as server:
            assert not _exited(server, 1), "the end of the starting thread stopped the server"
    assert len(pids) == 3 and len(set(pids)) == 1, "the loaded server must be reused"

    active = runtime_module._active_runtime
    assert active is not None and active.process is not None
    process = active.process
    runtime_module.shutdown_local_runtime()
    assert process.returncode == -signal.SIGTERM


def test_normal_exit_keeps_server_for_running_threads_then_cleans_up(
    settings: RuntimeSettings, records: Path, tmp_path: Path
) -> None:
    script = """
import sys, threading, time
from pathlib import Path
{imports}

settings = module.RuntimeSettings(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[2]), "clef-flash", 0, 4096, ())
entered = threading.Event()
def evaluate():
    with module.runtime_session(settings, 20):
        entered.set()
        time.sleep(1)  # この時点で main thread は終わっている。
        print("alive" if module._active_runtime.process.poll() is None else "stopped", flush=True)
threading.Thread(target=evaluate).start()
assert entered.wait(20)
"""
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script.replace("{imports}", _IMPORT_RUNTIME),
            str(settings.server_path),
            str(settings.model_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "TMPDIR": str(temporary)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "alive"
    assert not list(temporary.glob("lorairo-clef-*")), "atexit must still remove the credential"
    with pytest.raises(ProcessLookupError):
        os.kill(_server_pid(records), 0)


def test_forked_child_owns_only_the_server_it_started(settings: RuntimeSettings) -> None:
    script = """
import os, select, sys
from pathlib import Path
{imports}

settings = module.RuntimeSettings(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[2]), "clef-flash", 0, 4096, ())
with module.runtime_session(settings, 20):
    parent_server = module._active_runtime.process
idle = os.fork()
if idle == 0:
    os._exit(0)
assert os.waitpid(idle, 0)[1] == 0
assert parent_server.poll() is None, "an exiting fork child stopped the parent's server"

report_read, report_write = os.pipe()
leave_read, leave_write = os.pipe()
pid = os.fork()
if pid == 0:
    try:
        with module.runtime_session(settings, 20):
            os.write(report_write, str(module._active_runtime.process.pid).encode())
        os.read(leave_read, 1)
    finally:
        os._exit(0)
try:
    assert select.select([report_read], [], [], 20)[0], "the fork child could not start a server"
    child_server = int(os.read(report_read, 32))
    assert child_server != parent_server.pid
    descriptor = os.pidfd_open(child_server)
finally:
    os.write(leave_write, b"x")
    os.waitpid(pid, 0)
assert select.select([descriptor], [], [], 10)[0], "the child's server outlived the child"
assert parent_server.poll() is None, "the fork child stopped the parent's server"
module.shutdown_local_runtime()
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script.replace("{imports}", _IMPORT_RUNTIME),
            str(settings.server_path),
            str(settings.model_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr


def test_owner_that_execs_while_the_server_starts_leaves_no_server(
    settings: RuntimeSettings, records: Path, tmp_path: Path
) -> None:
    # 所有プロセスが launcher の起動中に exec する。PID が変わらないので親 PID の検査では見分けられない。
    script = """
import os, sys, threading, time
from pathlib import Path
{imports}
settings = module.RuntimeSettings(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[2]), "clef-flash", 0, 4096, ())
def start():
    try:
        with module.runtime_session(settings, 20):
            pass
    except module.LocalRuntimeError:
        pass
if sys.argv[3] == "after-start":
    start()  # 起動が返った直後に exec する。
else:
    threading.Thread(target=start, daemon=True).start()
    time.sleep(float(sys.argv[3]))
os.execv(sys.executable, [sys.executable, "-c", "import time; time.sleep(60)"])
"""
    owners = []
    try:
        # 起動開始から exec までの間隔を launcher の起動時間 (数十 ms) の前後に散らし、同時に走らせる。
        for index in range(40):
            owners.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script.replace("{imports}", _IMPORT_RUNTIME),
                        str(settings.server_path),
                        str(settings.model_path),
                        "after-start" if index % 10 == 9 else str((index % 10) * 0.01),
                    ],
                    env={**os.environ, "TMPDIR": str(tmp_path)},
                )
            )
        time.sleep(4)  # 遅れて exec される server も記録されるまで待つ。
        started = sorted(records.glob("*.json"))
        assert started, "no server was started, so the race was not exercised"
        watched: list[int] = []
        try:
            for path in started:
                with suppress(ProcessLookupError):  # 既に回収された server は監視不要。
                    watched.append(os.pidfd_open(int(json.loads(path.read_text(encoding="utf-8"))["pid"])))
            deadline = time.monotonic() + 10
            pending = list(watched)
            while pending and (remaining := deadline - time.monotonic()) > 0:
                ready = select.select(pending, [], [], remaining)[0]
                pending = [descriptor for descriptor in pending if descriptor not in ready]
            survivors = len(pending)
        finally:
            for descriptor in watched:
                if not _exited(descriptor, 0):
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                os.close(descriptor)
        assert not survivors, f"{survivors} of {len(started)} servers outlived an owner that exec'd"
    finally:
        for owner in owners:
            owner.kill()
            owner.wait(10)


def test_start_during_interpreter_finalization_fails_without_hanging(settings: RuntimeSettings) -> None:
    # 終了処理の後半では import できないため、runtime ではなく起動部分を直接呼ぶ。
    script = """
import os, sys
from pathlib import Path
{imports}
lifetime = sys.modules["image_annotator_lib.decisions.process_lifetime"]
settings = module.RuntimeSettings(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[2]), "clef-flash", 0, 4096, ())
with module.runtime_session(settings, 20):
    pass  # 起動用スレッドを作っておく。

class StartsAtFinalization:
    def __del__(self):
        try:
            lifetime.spawn_owned(["/bin/true"], cwd=Path("/"), env={})
            os.write(1, b"started")
        except lifetime.ProcessLifetimeError:
            os.write(1, b"refused")

late = StartsAtFinalization()  # module の後始末 (sys.is_finalizing() が真) で __del__ が走る。
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script.replace("{imports}", _IMPORT_RUNTIME),
            str(settings.server_path),
            str(settings.model_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout == "refused", (completed.stdout, completed.stderr)


def test_launcher_failure_is_not_reported_as_a_model_problem(
    settings: RuntimeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_lifetime.sys, "executable", "/bin/false")
    with pytest.raises(LocalRuntimeError) as caught:
        with runtime_module.runtime_session(settings, 20):
            pytest.fail("a server without a working launcher must not be used")
    assert "model/projector" not in str(caught.value)
    assert "could not be tied" in str(caught.value)
    assert caught.value.code == DecisionErrorCode.CONFIGURATION


@pytest.mark.parametrize("problem", ["not-executable", "invalid-format"])
def test_unstartable_executable_is_a_configuration_error(settings: RuntimeSettings, problem: str) -> None:
    settings.server_path.write_bytes(b"\x00not an executable")
    settings.server_path.chmod(0o600 if problem == "not-executable" else 0o700)
    with pytest.raises(LocalRuntimeError, match="could not start") as caught:
        with runtime_module.runtime_session(settings, 20):
            pytest.fail("an unstartable server must not be used")
    assert caught.value.code == DecisionErrorCode.CONFIGURATION
