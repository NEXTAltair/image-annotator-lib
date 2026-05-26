"""`import image_annotator_lib` の eager heavy import 防止 smoke test (ADR 0001 amended)。

ADR 0001 amendment 2026-05-18 で「通常 CI に `import image_annotator_lib` smoke を含める」
ことが明文化された。本 test は subprocess clean import で実行し、in-process の sys.modules
汚染 (他 test が torch を import 済み等) で false negative になることを避ける。

"""

import json
import subprocess
import sys

import pytest

from image_annotator_lib.webapi.annotator import WebApiAnnotator
from image_annotator_lib.webapi.api_model_discovery import discover_available_vision_models
from image_annotator_lib.webapi.http_retry import build_retry_http_client
from image_annotator_lib.webapi.image_preprocess import preprocess_images_to_binary
from image_annotator_lib.webapi.model_id import resolve_model_ref
from image_annotator_lib.webapi.output_normalization import normalize_annotation_output
from image_annotator_lib.webapi.provider_manager import ProviderManager
from image_annotator_lib.webapi.result_adapter import to_annotation_result

_HEAVY_MODULES: tuple[str, ...] = ("torch", "torchvision", "transformers", "onnxruntime", "tensorflow")
"""eager load を禁止する heavy native dep の module 名集合。"""

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


@pytest.mark.unit
@pytest.mark.fast
def test_webapi_package_import_surface() -> None:
    """WebAPI subsystem の最終 import surface が `image_annotator_lib.webapi` にあること。"""
    assert WebApiAnnotator.__name__ == "WebApiAnnotator"
    assert ProviderManager.__name__ == "ProviderManager"
    assert callable(resolve_model_ref)
    assert callable(discover_available_vision_models)
    assert callable(build_retry_http_client)
    assert callable(preprocess_images_to_binary)
    assert callable(normalize_annotation_output)
    assert callable(to_annotation_result)
