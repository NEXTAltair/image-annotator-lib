"""Unit tests for OpenRouterApiAnnotator (Plan 1 architecture).

Plan 1: OpenRouterApiAnnotator is a thin wrapper that delegates to
ProviderManager.run_inference_with_model() via _run_inference().

Mock Strategy:
- Mock: ProviderManager.run_inference_with_model (external API calls)
- Mock: config_registry.get (configuration)
- Real: OpenRouterApiAnnotator._run_inference() and result conversion
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.model_config import WebAPIModelConfig
from image_annotator_lib.core.types import UnifiedAnnotationResult
from image_annotator_lib.model_class.annotator_webapi.openai_api_chat import (
    OpenRouterApiAnnotator,
)


def _make_config(model_name: str = "test_model", api_model_id: str = "test-model") -> WebAPIModelConfig:
    """Create a WebAPIModelConfig for testing."""
    return WebAPIModelConfig(
        model_name=model_name,
        **{"class": "OpenRouterApiAnnotator"},
        model_name_on_provider=api_model_id,
        device="cpu",
    )


class TestOpenRouterContextManager:
    """Context manager tests for OpenRouterApiAnnotator."""

    @pytest.mark.unit
    def test_context_manager_returns_self(self):
        """Test __enter__ returns the annotator instance."""
        config = _make_config()
        annotator = OpenRouterApiAnnotator(model_name="test_model", config=config)
        result = annotator.__enter__()
        assert result is annotator

    @pytest.mark.unit
    def test_context_manager_exit_no_error(self):
        """Test __exit__ completes without error."""
        config = _make_config()
        annotator = OpenRouterApiAnnotator(model_name="test_model", config=config)
        annotator.__enter__()
        annotator.__exit__(None, None, None)


class TestOpenRouterInference:
    """Inference tests for OpenRouterApiAnnotator via ProviderManager."""

    @pytest.mark.unit
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.ProviderManager")
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.config_registry")
    def test_run_inference_success(self, mock_config, mock_provider_mgr):
        """Test successful inference delegates to ProviderManager with openrouter: prefix."""
        mock_config.get.return_value = "meta-llama/llama-3.1-8b"

        mock_result = MagicMock(spec=UnifiedAnnotationResult)
        mock_result.error = None
        mock_result.tags = ["tag1", "tag2"]
        mock_provider_mgr.run_inference_with_model.return_value = {"phash_1": mock_result}

        with patch("image_annotator_lib.core.utils.calculate_phash", return_value="phash_1"):
            config = _make_config(model_name="test_or", api_model_id="meta-llama/llama-3.1-8b")
            annotator = OpenRouterApiAnnotator(model_name="test_or", config=config)
            test_images = [Image.new("RGB", (64, 64), color="red")]

            results = annotator._run_inference(test_images)

            assert len(results) == 1
            assert results[0] is mock_result

            # openrouter: プレフィックスが付与されることを確認
            call_kwargs = mock_provider_mgr.run_inference_with_model.call_args[1]
            assert call_kwargs["api_model_id"] == "openrouter:meta-llama/llama-3.1-8b"

    @pytest.mark.unit
    @patch("image_annotator_lib.core.utils.get_model_capabilities", return_value={"tags"})
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.ProviderManager")
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.config_registry")
    def test_run_inference_no_api_model_id(self, mock_config, mock_provider_mgr, _mock_caps):
        """Test _run_inference returns error when api_model_id not configured."""
        mock_config.get.return_value = ""

        config = _make_config(model_name="no_api_model", api_model_id="placeholder")
        annotator = OpenRouterApiAnnotator(model_name="no_api_model", config=config)
        test_images = [Image.new("RGB", (64, 64), color="blue")]

        results = annotator._run_inference(test_images)

        assert len(results) == 1
        assert results[0].error is not None
        assert "no api_model_id" in results[0].error.lower()
        mock_provider_mgr.run_inference_with_model.assert_not_called()

    @pytest.mark.unit
    @patch("image_annotator_lib.core.utils.get_model_capabilities", return_value={"tags"})
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.ProviderManager")
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.config_registry")
    def test_run_inference_image_not_in_results(self, mock_config, mock_provider_mgr, _mock_caps):
        """Test fallback when phash not found in results dict."""
        mock_config.get.return_value = "test-model"
        mock_provider_mgr.run_inference_with_model.return_value = {}

        with patch("image_annotator_lib.core.utils.calculate_phash", return_value="missing_phash"):
            config = _make_config()
            annotator = OpenRouterApiAnnotator(model_name="test_model", config=config)
            test_images = [Image.new("RGB", (64, 64), color="green")]

            results = annotator._run_inference(test_images)

            assert len(results) == 1
            assert results[0].error is not None
            assert "No result found" in results[0].error

    @pytest.mark.unit
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.ProviderManager")
    @patch("image_annotator_lib.model_class.annotator_webapi.openai_api_chat.config_registry")
    def test_batch_processing_multiple_images(self, mock_config, mock_provider_mgr):
        """Test batch processing with multiple images."""
        mock_config.get.return_value = "test-model"

        mock_results = {}
        for i, phash in enumerate(["p1", "p2", "p3"]):
            mock_result = MagicMock(spec=UnifiedAnnotationResult)
            mock_result.error = None
            mock_result.tags = [f"tag_{i}"]
            mock_results[phash] = mock_result
        mock_provider_mgr.run_inference_with_model.return_value = mock_results

        with patch("image_annotator_lib.core.utils.calculate_phash", side_effect=["p1", "p2", "p3"]):
            config = _make_config(model_name="test_batch")
            annotator = OpenRouterApiAnnotator(model_name="test_batch", config=config)
            test_images = [Image.new("RGB", (64, 64), color=c) for c in ["red", "green", "blue"]]

            results = annotator._run_inference(test_images)

            assert len(results) == 3
            for result in results:
                assert result.error is None
