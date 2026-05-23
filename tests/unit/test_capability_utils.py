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
        mock_config_registry.get.side_effect = [[], ""]  # 1回目: capabilities=[], 2回目: type=""

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

    @patch("image_annotator_lib.core.registry.get_webapi_metadata")
    def test_get_model_capabilities_webapi_returns_all_three(self, mock_get_webapi_metadata):
        """WebAPI モデルは Vision LLM として tags/captions/scores 全 capability を返す。

        WARNING ('capabilitiesが設定されていません') が ~80 件の WebAPI モデル
        全件で出ていた問題の修正。SSoT (`_WEBAPI_MODEL_METADATA`) に登録済みなら
        config_registry を見ずに即時全 capability を返す。
        """
        mock_get_webapi_metadata.return_value = {
            "api_model_id": "openai/gpt-4o",
            "provider": "openai",
            "type": "webapi",
        }

        capabilities = get_model_capabilities("gpt-4o")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES}

    @patch("image_annotator_lib.core.registry.get_webapi_metadata")
    def test_get_model_capabilities_webapi_respects_explicit_ratings(self, mock_get_webapi_metadata):
        """Issue #82: WebAPI metadata can explicitly request rating capability."""
        mock_get_webapi_metadata.return_value = {
            "api_model_id": "openai/gpt-4o",
            "provider": "openai",
            "type": "webapi",
            "capabilities": ["tags", "ratings"],
        }

        capabilities = get_model_capabilities("gpt-4o-rating")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.RATINGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_type_fallback_rating_capable_wd_tagger_includes_ratings(self, mock_config_registry):
        """LoRAIro #365: type 推論フォールバックで WDTagger は RATINGS も含める。

        ユーザー側 `config/annotator_config.toml` が古く `capabilities` 未指定でも、
        `class = "WDTagger"` であれば rating output を保持する。
        """
        # 1回目: capabilities=[] (未設定), 2回目: type="tagger", 3回目: class="WDTagger"
        mock_config_registry.get.side_effect = [[], "tagger", "WDTagger"]

        capabilities = get_model_capabilities("wd-vit-tagger-v3")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.RATINGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_type_fallback_rating_capable_z3d_tagger_includes_ratings(self, mock_config_registry):
        """LoRAIro #365: Z3D_E621Tagger も rating 対応として扱う。"""
        mock_config_registry.get.side_effect = [[], "tagger", "Z3D_E621Tagger"]

        capabilities = get_model_capabilities("Z3D-E621-Convnext")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.RATINGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_type_fallback_rating_capable_camie_tagger_includes_ratings(self, mock_config_registry):
        """LoRAIro #365: CamieTagger も rating 対応として扱う。"""
        mock_config_registry.get.side_effect = [[], "tagger", "CamieTagger"]

        capabilities = get_model_capabilities("camie_tagger_initial")

        assert capabilities == {TaskCapability.TAGS, TaskCapability.RATINGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_type_fallback_tags_only_tagger_excludes_ratings(self, mock_config_registry):
        """LoRAIro #365: rating 非対応 tagger (DeepDanbooruTagger) は TAGS のみ。

        DeepDanbooru モデルは rating index/source_scheme を持たないので、
        type 推論フォールバックでも RATINGS は付かない。
        """
        mock_config_registry.get.side_effect = [[], "tagger", "DeepDanbooruTagger"]

        capabilities = get_model_capabilities("deepdanbooru-v3-20211112-sgd-e28")

        assert capabilities == {TaskCapability.TAGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_type_fallback_tagger_without_class_field_excludes_ratings(self, mock_config_registry):
        """class フィールド未設定の tagger は TAGS のみ (安全側にフォールバック)。"""
        mock_config_registry.get.side_effect = [[], "tagger", ""]

        capabilities = get_model_capabilities("unknown-tagger")

        assert capabilities == {TaskCapability.TAGS}

    @patch("image_annotator_lib.core.config.config_registry")
    def test_explicit_capabilities_override_class_based_fallback(self, mock_config_registry):
        """明示 capabilities が指定されていれば class ベースフォールバックは適用しない。

        例: WDTagger ベースだが ratings を意図的に無効化したい場合。
        """
        mock_config_registry.get.return_value = ["tags"]

        capabilities = get_model_capabilities("wd-vit-tagger-v3-tags-only")

        assert capabilities == {TaskCapability.TAGS}
