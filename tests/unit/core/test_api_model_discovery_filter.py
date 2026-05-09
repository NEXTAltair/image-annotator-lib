"""ADR 0023 Phase 1 (Issue #45) — WebAPI モデル絞り込み条件の回帰防止テスト。

`_is_litellm_model_annotation_compatible()` が `supports_response_schema` ではなく
`supports_function_calling` を主条件として動作することを保証する。

受け入れ条件:
- vision=True, function_calling=True, response_schema=False のモデルが登録対象
- vision=True, function_calling=False, response_schema=True のモデルが除外対象
"""

from __future__ import annotations

import pytest

from image_annotator_lib.core.api_model_discovery import (
    _format_litellm_metadata,
    _is_litellm_model_annotation_compatible,
)


@pytest.mark.unit
class TestIsLitellmModelAnnotationCompatible:
    """`_is_litellm_model_annotation_compatible()` の判定ロジック検証。"""

    def test_vision_and_function_calling_true_is_compatible(self):
        """vision + function_calling 両方 True なら compatible (response_schema 値は不問)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_response_schema": False,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is True

    def test_function_calling_false_is_excluded(self):
        """vision=True でも function_calling=False なら除外 (response_schema=True でも)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": False,
            "supports_response_schema": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_function_calling_missing_is_excluded(self):
        """function_calling キー欠落でも除外 (`is True` 比較なので False/None/欠落いずれも弾く)。"""
        info = {
            "supports_vision": True,
            "supports_response_schema": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_vision_false_is_excluded(self):
        """vision=False なら function_calling=True でも除外。"""
        info = {
            "supports_vision": False,
            "supports_function_calling": True,
            "mode": "chat",
        }
        assert _is_litellm_model_annotation_compatible(info) is False

    def test_responses_mode_is_compatible(self):
        """mode=responses も chat と同様に compatible (Phase 1 仕様)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "responses",
        }
        assert _is_litellm_model_annotation_compatible(info) is True

    def test_completion_mode_is_excluded(self):
        """mode=completion 等は除外 (Phase 1 は chat / responses のみ)。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "completion",
        }
        assert _is_litellm_model_annotation_compatible(info) is False


@pytest.mark.unit
class TestFormatLitellmMetadata:
    """`_format_litellm_metadata()` が ADR 0023 Phase 1 の metadata 形式を返すこと。"""

    def test_response_schema_key_is_absent(self):
        """Issue #45: `supports_response_schema` キーは metadata から削除されている。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_response_schema": True,
            "mode": "chat",
        }
        metadata = _format_litellm_metadata("openai/gpt-4o", info)
        assert metadata is not None
        assert "supports_response_schema" not in metadata

    def test_function_calling_key_is_present(self):
        """`supports_function_calling` キーは metadata に保持される。"""
        info = {
            "supports_vision": True,
            "supports_function_calling": True,
            "mode": "chat",
        }
        metadata = _format_litellm_metadata("openai/gpt-4o", info)
        assert metadata is not None
        assert metadata["supports_function_calling"] is True

    def test_invalid_model_id_returns_none(self):
        """`provider/model` 形式でない ID は None を返す。"""
        assert _format_litellm_metadata("invalid_id_no_slash", {}) is None
