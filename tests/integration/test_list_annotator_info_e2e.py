"""list_annotator_info() の end-to-end 統合テスト (Issue #19)。

実 `initialize_registry()` でロードされるモデル群に対して、新規 API が
矛盾なく AnnotatorInfo を返せることを検証する。
"""

import pytest

from image_annotator_lib import AnnotatorInfo, list_annotator_info
from image_annotator_lib.core.registry import _MODEL_CLASS_OBJ_REGISTRY, initialize_registry


@pytest.mark.integration
def test_list_annotator_info_real_registry():
    """実 registry 初期化後に list_annotator_info() を呼び、登録モデルすべてが
    AnnotatorInfo として返されることを確認する。"""
    initialize_registry()

    infos = list_annotator_info()

    # レジストリに何かしら登録されている前提
    assert len(infos) >= len(_MODEL_CLASS_OBJ_REGISTRY)

    # 全エントリが AnnotatorInfo であり、不変条件を満たす
    for info in infos:
        assert isinstance(info, AnnotatorInfo)
        assert info.is_local != info.is_api, f"{info.name}: XOR 違反"
        if info.is_api:
            assert info.device is None, f"{info.name}: API モデルなのに device あり"
        assert info.model_type in ("tagger", "scorer", "captioner", "vision")

    # 名前ソートされている
    names = [info.name for info in infos]
    assert names == sorted(names), "list_annotator_info() の戻り値は name 昇順ソート済み"

    # レジストリ登録済みモデル名がすべて含まれる
    registry_names = set(_MODEL_CLASS_OBJ_REGISTRY.keys())
    info_names = {info.name for info in infos}
    missing = registry_names - info_names
    assert not missing, f"レジストリ登録済みなのに AnnotatorInfo に含まれないモデル: {missing}"
