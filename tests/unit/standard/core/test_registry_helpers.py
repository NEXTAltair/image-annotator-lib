"""`_safe_float` / `_safe_int` / `_parse_discontinued_at` のユニットテスト (Issue #23)。

`available_api_models.toml` 由来のメタデータ値を `_WEBAPI_MODEL_METADATA` に格納する際の
型変換ヘルパー。malformed 値で `_register_webapi_models_from_discovery` 全体が abort
しないよう、warning + None フォールバックする方針。
"""

import datetime

import pytest

from image_annotator_lib.core.registry import (
    _parse_discontinued_at,
    _safe_float,
    _safe_int,
)


@pytest.mark.unit
@pytest.mark.fast
class TestSafeFloat:
    """`_safe_float` の境界条件。"""

    def test_returns_none_for_none(self):
        """None 入力は None を返す (optional フィールドの欠落)。"""
        assert _safe_float(None, "model-x", "max_output_tokens") is None

    def test_returns_float_for_int(self):
        """int 入力は float に変換される。"""
        assert _safe_float(1800, "model-x", "max_output_tokens") == 1800.0

    def test_returns_float_for_str_numeric(self):
        """数値文字列は float に変換される (TOML が文字列で渡してきた場合の保険)。"""
        assert _safe_float("3.14", "model-x", "estimated_size_gb") == 3.14

    def test_returns_none_for_malformed_str(self):
        """非数値文字列は warning + None フォールバック (モデル登録は失敗させない)。"""
        assert _safe_float("not-a-number", "model-x", "estimated_size_gb") is None

    def test_returns_none_for_unsupported_type(self):
        """list 等のサポート外型は None フォールバック。"""
        assert _safe_float([1, 2, 3], "model-x", "estimated_size_gb") is None


@pytest.mark.unit
@pytest.mark.fast
class TestSafeInt:
    """`_safe_int` の境界条件。"""

    def test_returns_none_for_none(self):
        assert _safe_int(None, "model-x", "max_output_tokens") is None

    def test_returns_int_for_int(self):
        assert _safe_int(1800, "model-x", "max_output_tokens") == 1800

    def test_returns_int_for_str_numeric(self):
        assert _safe_int("1800", "model-x", "max_output_tokens") == 1800

    def test_returns_none_for_malformed_str(self):
        assert _safe_int("not-an-int", "model-x", "max_output_tokens") is None

    def test_returns_none_for_unsupported_type(self):
        assert _safe_int({"a": 1}, "model-x", "max_output_tokens") is None


@pytest.mark.unit
@pytest.mark.fast
class TestParseDiscontinuedAt:
    """`_parse_discontinued_at` の境界条件。

    重要: `datetime.datetime` は `datetime.date` のサブクラスなので、
    isinstance チェックの順序が逆だと datetime 値が二重変換される。
    """

    def test_returns_none_for_none(self):
        assert _parse_discontinued_at(None, "model-x") is None

    def test_returns_datetime_unchanged(self):
        """datetime.datetime はそのまま返す (二重変換しない)。"""
        dt = datetime.datetime(2025, 12, 31, 23, 59, 59, tzinfo=datetime.UTC)
        assert _parse_discontinued_at(dt, "model-x") is dt

    def test_normalizes_date_to_utc_datetime(self):
        """datetime.date (TOML local-date) は UTC 00:00:00 に正規化される。"""
        d = datetime.date(2025, 12, 31)
        result = _parse_discontinued_at(d, "model-x")
        assert isinstance(result, datetime.datetime)
        assert result == datetime.datetime(2025, 12, 31, 0, 0, 0, tzinfo=datetime.UTC)

    def test_parses_iso_string_with_z_suffix(self):
        """ISO 8601 文字列 (Z suffix) を datetime に parse する。"""
        result = _parse_discontinued_at("2025-12-31T23:59:59Z", "model-x")
        assert result == datetime.datetime(2025, 12, 31, 23, 59, 59, tzinfo=datetime.UTC)

    def test_parses_iso_string_with_offset(self):
        """ISO 8601 文字列 (+HH:MM offset) も parse できる。"""
        result = _parse_discontinued_at("2025-12-31T23:59:59+09:00", "model-x")
        assert result is not None
        assert result.tzinfo is not None

    def test_returns_none_for_malformed_str(self):
        """非 ISO 文字列は warning + None フォールバック。"""
        assert _parse_discontinued_at("not-a-date", "model-x") is None

    def test_returns_none_for_unsupported_type(self):
        """int / list 等のサポート外型は warning + None フォールバック。"""
        assert _parse_discontinued_at(20251231, "model-x") is None
        assert _parse_discontinued_at([2025, 12, 31], "model-x") is None
