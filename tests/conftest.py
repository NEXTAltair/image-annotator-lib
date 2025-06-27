"""テスト全体で共有されるfixtures。

このモジュールでは、複数のテストファイルで使用される共通のfixtureを定義します。
"""

# Import shared fixtures from fixtures module
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

# Add the tests directory to sys.path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Temporarily disable shared fixtures for testing
# from unit.fixtures.mock_libraries import *
# from unit.fixtures.mock_configs import *
# from unit.fixtures.mock_components import *
# from unit.fixtures.shared_fixtures import *

resources_dir = Path("tests") / "resources"


@pytest.fixture(scope="session")
def load_image_files() -> Callable[[int], list[Image.Image]]:
    """指定された枚数の画像ファイルをリストとして読み込む関数を返す"""

    def _load_images(count: int = 1) -> list[Image.Image]:
        image_path = resources_dir / "img" / "1_img"
        files = list(image_path.rglob("*.webp"))

        # 指定された枚数だけファイルを取得(ディレクトリ内のファイル数を超えないように)
        count = min(count, len(files))
        files = files[:count]

        # すべての画像をリストに格納して返す
        return [Image.open(file) for file in files if file.exists()]

    return _load_images
