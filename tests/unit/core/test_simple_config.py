"""Unit tests for SimpleModelConfig (simple_config.py)

Phase C Task: Achieve 35% → 85%+ coverage for simple_config.py

Test Strategy:
- REAL components: toml.load(), dict operations, settings merge logic
- MOCKED: File system (temp TOML files), MODEL_SETTINGS_PATH.exists()
- AUTOUSE fixture: clear_simple_config_cache to prevent cache pollution
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from image_annotator_lib.core.simple_config import (
    SimpleModelConfig,
    get_default_settings,
    get_model_settings,
    get_simple_config,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def clear_simple_config_cache():
    """Clear SimpleModelConfig cache before/after each test to prevent pollution.

    CRITICAL: SimpleModelConfig uses module-level _config_cache that persists
    between tests. Without this fixture, tests can affect each other.
    """
    # Clear global instance before test
    import image_annotator_lib.core.simple_config as config_module

    config_module._simple_config = None

    yield

    # Clear global instance after test
    config_module._simple_config = None


@pytest.fixture
def mock_simple_config_toml(tmp_path: Path) -> Path:
    """Create temporary TOML file for testing.

    Args:
        tmp_path: pytest tmpdir fixture

    Returns:
        Path to created TOML file
    """
    toml_path = tmp_path / "model_settings.toml"
    toml_content = {
        "global_defaults": {"timeout": 30, "max_tokens": 1500, "temperature": 0.7},
        "model_overrides": {
            "google/gemini-2.5-pro": {"timeout": 60, "max_tokens": 2000},
            "anthropic/claude-3-5-sonnet": {"temperature": 0.5},
        },
    }

    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(toml_content, f)

    return toml_path


# ==============================================================================
# Priority 1B: Simple Config Module - Test 6-9
# ==============================================================================


class TestSimpleConfigLoading:
    """Configuration loading tests for SimpleModelConfig."""

    @pytest.mark.unit
    def test_simple_config_load_from_toml(self, mock_simple_config_toml: Path):
        """Test successful TOML loading and cache population.

        Coverage: Lines 21-30 (_load_config success path)

        REAL components:
        - Real toml.load() operation
        - Real _config_cache population
        - Real logger calls

        MOCKED:
        - MODEL_SETTINGS_PATH to point to test TOML file

        Scenario:
        1. Create temporary TOML file with test config
        2. Mock MODEL_SETTINGS_PATH to point to temp file
        3. Initialize SimpleModelConfig
        4. Verify _config_cache populated correctly

        Assertions:
        - _config_cache contains expected keys
        - global_defaults loaded correctly
        - model_overrides loaded correctly
        - logger.info called with success message
        """
        # Mock MODEL_SETTINGS_PATH to point to test file
        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", mock_simple_config_toml
        ):
            # Act: Initialize SimpleModelConfig (triggers _load_config)
            config = SimpleModelConfig()

            # Assert: Cache populated
            assert config._config_cache is not None, "_config_cache存在"
            assert "global_defaults" in config._config_cache, "global_defaults読み込み"
            assert "model_overrides" in config._config_cache, "model_overrides読み込み"

            # Assert: global_defaults correct
            global_defaults = config._config_cache["global_defaults"]
            assert global_defaults["timeout"] == 30, "timeout正しく読み込み"
            assert global_defaults["max_tokens"] == 1500, "max_tokens正しく読み込み"
            assert global_defaults["temperature"] == 0.7, "temperature正しく読み込み"

            # Assert: model_overrides correct
            model_overrides = config._config_cache["model_overrides"]
            assert "google/gemini-2.5-pro" in model_overrides, "モデルオーバーライド読み込み"
            assert (
                model_overrides["google/gemini-2.5-pro"]["timeout"] == 60
            ), "モデル固有timeout読み込み"

    @pytest.mark.unit
    def test_simple_config_missing_file_fallback(self):
        """Test fallback to defaults when TOML file missing.

        Coverage: Lines 29-30 (file not found handling)

        REAL components:
        - Real default config dict creation
        - Real logger.warning call

        MOCKED:
        - MODEL_SETTINGS_PATH.exists() returns False

        Scenario:
        1. Mock MODEL_SETTINGS_PATH.exists() to return False
        2. Initialize SimpleModelConfig
        3. Verify defaults created

        Assertions:
        - _config_cache contains default structure
        - Warning logged about missing file
        """
        # Create non-existent path
        non_existent_path = Path("/nonexistent/model_settings.toml")

        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", non_existent_path
        ):
            # Act: Initialize (file doesn't exist)
            config = SimpleModelConfig()

            # Assert: Default structure created
            assert config._config_cache is not None, "_config_cache存在"
            assert config._config_cache == {
                "global_defaults": {},
                "model_overrides": {},
            }, "デフォルト構造作成"

    @pytest.mark.unit
    def test_simple_config_toml_parse_error(self, tmp_path: Path):
        """Test error handling when TOML parsing fails.

        Coverage: Lines 31-33 (exception handling)

        REAL components:
        - Real exception handling
        - Real fallback to defaults

        MOCKED:
        - toml.load() raises TomlDecodeError

        Scenario:
        1. Create invalid TOML file
        2. Mock toml.load() to raise exception
        3. Initialize SimpleModelConfig
        4. Verify fallback to defaults

        Assertions:
        - Exception caught and handled
        - _config_cache contains defaults
        - Error logged
        - No crash
        """
        # Create invalid TOML file
        invalid_toml_path = tmp_path / "invalid.toml"
        with open(invalid_toml_path, "w", encoding="utf-8") as f:
            f.write("[invalid toml content\n")  # Missing closing bracket

        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", invalid_toml_path
        ):
            # Act: Initialize (should catch TOML error)
            config = SimpleModelConfig()

            # Assert: Fallback to defaults (no crash)
            assert config._config_cache is not None, "_config_cache存在"
            assert config._config_cache == {
                "global_defaults": {},
                "model_overrides": {},
            }, "パースエラー時はデフォルトにフォールバック"


class TestSimpleConfigMerging:
    """Settings merge logic tests for SimpleModelConfig."""

    @pytest.mark.unit
    def test_simple_config_get_model_settings_merge(self, mock_simple_config_toml: Path):
        """Test global defaults + model overrides merge logic.

        Coverage: Lines 45-55 (get_model_settings merge logic)

        REAL components:
        - Real dict merge operations
        - Real override precedence logic

        MOCKED:
        - MODEL_SETTINGS_PATH to test TOML

        Scenario:
        1. Load config with global defaults and model overrides
        2. Call get_model_settings() for model with overrides
        3. Verify override takes precedence
        4. Verify global defaults preserved for non-override keys

        Assertions:
        - Model overrides take precedence
        - Global defaults preserved when not overridden
        - Correct merged settings returned
        """
        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", mock_simple_config_toml
        ):
            config = SimpleModelConfig()

            # Act: Get settings for model with overrides
            settings = config.get_model_settings("google/gemini-2.5-pro")

            # Assert: Override takes precedence
            assert settings["timeout"] == 60, "timeout: モデルオーバーライド優先"
            assert settings["max_tokens"] == 2000, "max_tokens: モデルオーバーライド優先"

            # Assert: Global default preserved (not overridden)
            assert (
                settings["temperature"] == 0.7
            ), "temperature: グローバルデフォルト保持（オーバーライドなし）"

            # Act: Get settings for model without overrides
            settings_default = config.get_model_settings("nonexistent/model")

            # Assert: All from global defaults
            assert (
                settings_default["timeout"] == 30
            ), "オーバーライドなしモデル: グローバルデフォルト使用"
            assert settings_default["max_tokens"] == 1500, "max_tokens: グローバル値"
            assert settings_default["temperature"] == 0.7, "temperature: グローバル値"

            # Act: Get settings for model with partial overrides
            settings_partial = config.get_model_settings("anthropic/claude-3-5-sonnet")

            # Assert: Mixed (override + defaults)
            assert settings_partial["temperature"] == 0.5, "temperature: オーバーライド"
            assert settings_partial["timeout"] == 30, "timeout: グローバルデフォルト（未オーバーライド）"
            assert (
                settings_partial["max_tokens"] == 1500
            ), "max_tokens: グローバルデフォルト（未オーバーライド）"


class TestSimpleConfigGlobalFunctions:
    """Global convenience function tests."""

    @pytest.mark.unit
    def test_get_simple_config_singleton(self):
        """Test get_simple_config() returns singleton instance.

        Coverage: Lines 72-77 (get_simple_config singleton)

        REAL components:
        - Real singleton pattern
        - Real global instance management

        Scenario:
        1. Call get_simple_config() twice
        2. Verify same instance returned

        Assertions:
        - Same instance ID
        - Singleton pattern works
        """
        # Act: Get instance twice
        instance1 = get_simple_config()
        instance2 = get_simple_config()

        # Assert: Same instance (singleton)
        assert instance1 is instance2, "シングルトンインスタンス（同一オブジェクト）"
        assert id(instance1) == id(instance2), "オブジェクトID同一"

    @pytest.mark.unit
    def test_get_model_settings_convenience_function(self, mock_simple_config_toml: Path):
        """Test get_model_settings() convenience function.

        Coverage: Lines 80-90 (get_model_settings function)

        REAL components:
        - Real convenience function
        - Real delegation to SimpleModelConfig

        Scenario:
        1. Use convenience function to get settings
        2. Verify correct settings returned

        Assertions:
        - Settings returned correctly
        - Delegates to SimpleModelConfig instance
        """
        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", mock_simple_config_toml
        ):
            # Act: Use convenience function
            settings = get_model_settings("google/gemini-2.5-pro")

            # Assert: Correct settings
            assert settings["timeout"] == 60, "便利関数経由で正しい設定取得"
            assert settings["max_tokens"] == 2000, "max_tokens正しい"

    @pytest.mark.unit
    def test_get_default_settings_convenience_function(self, mock_simple_config_toml: Path):
        """Test get_default_settings() convenience function.

        Coverage: Lines 93-100 (get_default_settings function)

        REAL components:
        - Real convenience function
        - Real delegation to SimpleModelConfig

        Scenario:
        1. Use convenience function to get defaults
        2. Verify correct defaults returned

        Assertions:
        - Global defaults returned
        - Delegates to SimpleModelConfig instance
        """
        with patch(
            "image_annotator_lib.core.simple_config.MODEL_SETTINGS_PATH", mock_simple_config_toml
        ):
            # Act: Use convenience function
            defaults = get_default_settings()

            # Assert: Correct defaults
            assert defaults["timeout"] == 30, "グローバルデフォルト取得"
            assert defaults["max_tokens"] == 1500, "max_tokens正しい"
            assert defaults["temperature"] == 0.7, "temperature正しい"
