# tests/test_bdd_runner.py
"""pytest-bdd テストランナーファイル

このファイルは pytest によってテストとして検出され、
`scenarios()` 関数を呼び出すことで、関連する .feature ファイルと
ステップ定義を結びつけて BDD テストを実行します。
"""

from pytest_bdd import scenarios

from tests.features.step_definitions.annotation_steps import *
from tests.features.step_definitions.api_model_discovery_steps import *
from tests.features.step_definitions.base_steps import *
from tests.features.step_definitions.model_errors_steps import *
from tests.features.step_definitions.model_factory_steps import *
from tests.features.step_definitions.registry_steps import *
from tests.features.step_definitions.utils_steps import *
from tests.features.step_definitions.webapi_annotate_steps import *

# features ディレクトリ内のすべての .feature ファイルを対象にテストを実行
# pytest-bdd が conftest.py と step_definitions ディレクトリ内のステップを
# 自動的に発見することを期待する
scenarios('features')
