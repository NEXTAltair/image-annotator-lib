"""
Test capability management utilities
"""

from unittest.mock import patch

from image_annotator_lib.core.types import TaskCapability
from image_annotator_lib.core.utils import get_model_capabilities


class TestGetModelCapabilities:
    """Test get_model_capabilities utility function"""

    @patch("image_annotator_lib.core.config.config_registry")
    def test_get_model_capabilities_success(self, mock_config_registry):
        """Test successful capability retrieval"""
        # Mock config registry to return capabilities
        mock_config_registry.get.return_value = ["tags", "captions", "scores"]

        capabilities = get_model_capabilities("gpt-4o")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}
        mock_config_registry.get.assert_called_once_with("gpt-4o", "capabilities", [])

    @patch("image_annotator_lib.core.config.config_registry")
    def test_get_model_capabilities_single_capability(self, mock_config_registry):
        """Test single capability retrieval"""
        mock_config_registry.get.return_value = ["tags"]

        capabilities = get_model_capabilities("wd-tagger")

        assert capabilities == {TaskCapability.TAGS}

    @patch("image_annotator_lib.core.config.config_registry")
    @patch("image_annotator_lib.core.utils.logger")
    def test_get_model_capabilities_missing_config(self, mock_logger, mock_config_registry):
        """Test handling of missing capabilities configuration"""
        mock_config_registry.get.return_value = []

        capabilities = get_model_capabilities("unknown-model")

        assert capabilities == set()
        mock_logger.warning.assert_called_once_with(
            "モデル 'unknown-model' のcapabilitiesが設定されていません"
        )

    @patch("image_annotator_lib.core.config.config_registry")
    @patch("image_annotator_lib.core.utils.logger")
    def test_get_model_capabilities_invalid_capability(self, mock_logger, mock_config_registry):
        """Test handling of invalid capability values"""
        mock_config_registry.get.return_value = ["tags", "invalid_capability", "captions"]

        capabilities = get_model_capabilities("test-model")

        # Should include valid capabilities, skip invalid ones
        assert capabilities == {TaskCapability.TAGS, TaskCapability.CAPTIONS}
        mock_logger.error.assert_called_once_with(
            "無効なcapability 'invalid_capability' (model: test-model)"
        )

    @patch("image_annotator_lib.core.config.config_registry")
    def test_get_model_capabilities_mixed_valid_invalid(self, mock_config_registry):
        """Test mixed valid and invalid capabilities"""
        mock_config_registry.get.return_value = ["tags", "invalid1", "scores", "invalid2"]

        capabilities = get_model_capabilities("test-model")

        # Should only include valid capabilities
        assert capabilities == {TaskCapability.TAGS, TaskCapability.SCORES}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_get_model_capabilities_empty_result(self, mock_config_registry):
        """Test when all capabilities are invalid"""
        mock_config_registry.get.return_value = ["invalid1", "invalid2"]

        capabilities = get_model_capabilities("test-model")

        # Should return empty set
        assert capabilities == set()

    @patch("image_annotator_lib.core.config.config_registry")
    def test_get_model_capabilities_none_config(self, mock_config_registry):
        """Test when config registry returns None"""
        mock_config_registry.get.return_value = None

        capabilities = get_model_capabilities("test-model")

        # Should handle None as empty list and return empty set
        assert capabilities == set()
