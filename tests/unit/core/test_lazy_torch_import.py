"""Issue #59 regression: `import image_annotator_lib` must not eager-load torch."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.unit
def test_import_does_not_eager_load_torch() -> None:
    """`import image_annotator_lib` で torch / torchvision が load されないことを検証する。

    Linux + triton package 在り + CUDA driver 不在環境では `torch._dynamo` →
    `triton.knobs` の C extension create_module が SIGSEGV を起こす。
    本 test は fresh interpreter 内で `import image_annotator_lib` 実行直後の
    `sys.modules` を検査し、torch / torchvision package が含まれないことを確認する。

    subprocess を使う理由: 同一 pytest session 内で既に他テストが torch を
    import 済みだと `sys.modules` に残留して検出不能になるため、独立した
    Python interpreter で fresh state を保証する。
    """
    script = textwrap.dedent(
        """
        import sys
        import image_annotator_lib  # noqa: F401

        loaded = sorted(
            m for m in sys.modules
            if m == "torch" or m.startswith("torch.")
            or m == "torchvision" or m.startswith("torchvision.")
        )
        if loaded:
            for m in loaded:
                print(f"LEAK: {m}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"torch/torchvision eager-loaded at `import image_annotator_lib`:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
