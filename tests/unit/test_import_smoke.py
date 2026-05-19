"""`import image_annotator_lib` の eager heavy import 防止 smoke test (ADR 0001 amended)。

ADR 0001 amendment 2026-05-18 で「通常 CI に `import image_annotator_lib` smoke を含める」
ことが明文化された。本 test は subprocess clean import で実行し、in-process の sys.modules
汚染 (他 test が torch を import 済み等) で false negative になることを避ける。

Note:
    現状 `tensorflow` は `initialize_registry()` 経由で DeepDanbooru 系 TF tagger class
    の登録時に eager load される。本 smoke test ではその 1 件を許容しつつ、torch /
    torchvision / transformers / onnxruntime の 4 module の lazy 化を regression 防止する。
    tensorflow の lazy 化は別 Issue で追跡 (NEXTAltair/image-annotator-lib 側で起票予定)。
"""

import json
import subprocess
import sys

import pytest

_HEAVY_MODULES: tuple[str, ...] = ("torch", "torchvision", "transformers", "onnxruntime")
"""eager load を禁止する heavy native dep の module 名集合 (4 件)。

tensorflow は現状 registry 初期化で eager load されるため本 set から除外し、別 Issue
で追跡する (本 PR の scope 外)。
"""

_SENTINEL = "IAMLIB_SMOKE_JSON:"


@pytest.mark.unit
@pytest.mark.fast
def test_import_image_annotator_lib_does_not_eager_load_heavy_deps() -> None:
    """`import image_annotator_lib` 後の sys.modules に heavy native dep が居ないこと。

    subprocess で clean Python プロセスを起動し、bare な `import image_annotator_lib`
    のみを実行。完了後の `sys.modules` を JSON で sentinel marker 付きで stdout に出力し、
    本 test 側で parse する。loguru の INFO log と混在しても sentinel で分離可能。
    """
    code = (
        "import sys, json\n"
        "import image_annotator_lib\n"
        f"heavy = set({list(_HEAVY_MODULES)!r})\n"
        "loaded = sorted(heavy & set(sys.modules))\n"
        f"print({_SENTINEL!r} + json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (returncode={result.returncode}): "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    payload: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith(_SENTINEL):
            payload = line[len(_SENTINEL):]
            break
    assert payload is not None, (
        f"sentinel marker {_SENTINEL!r} not found in stdout: {result.stdout!r}"
    )

    loaded = json.loads(payload)
    assert loaded == [], (
        f"eager-loaded heavy deps after bare `import image_annotator_lib`: {loaded}"
    )
