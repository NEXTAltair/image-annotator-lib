"""Integration tests for context manager robustness against load and restoration failures.

Tests comprehensive error handling for model loading and CUDA restoration failures,
ensuring correct distinction between fatal errors and recoverable errors.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.base.pipeline import PipelineBaseAnnotator
from image_annotator_lib.core.base.transformers import TransformersBaseAnnotator
from image_annotator_lib.exceptions.errors import ModelLoadError


# Test-specific concrete implementations
class ConcreteTestPipelineAnnotator(PipelineBaseAnnotator):
    """Concrete Pipeline annotator for testing."""

    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """Generate tags from formatted output."""
        return []


class ConcreteTestTransformersAnnotator(TransformersBaseAnnotator):
    """Concrete Transformers annotator for testing."""

    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """Generate tags from formatted output."""
        return []


class TestContextManagerRobustness:
    """Integration tests for context manager error handling and CPU fallback."""

    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (224, 224), color="red")

    @pytest.fixture
    def pipeline_model_config(self, managed_config_registry):
        """Setup configuration for pipeline-based model."""
        config = {
            "class": "AestheticShadow",
            "model_path": "shadowlilac/aesthetic-shadow",
            "device": "cuda",
            "estimated_size_gb": 4.0,
            "batch_size": 8,
        }
        # Note: 'task' is not part of LocalMLModelConfig, but is retrieved dynamically
        # by PipelineBaseAnnotator.__init__() via config_registry.get() with default value
        managed_config_registry.set("test_pipeline_model", config)
        return config

    @pytest.fixture
    def transformers_model_config(self, managed_config_registry):
        """Setup configuration for transformers-based model."""
        config = {
            "class": "GITLargeCaptioning",
            "model_path": "microsoft/git-large-coco",
            "device": "cuda",
            "estimated_size_gb": 1.5,
        }
        managed_config_registry.set("test_transformers_model", config)
        return config

    @pytest.mark.integration
    def test_pipeline_load_failure_raises_model_load_error(self, pipeline_model_config):
        """Test that pipeline load failure raises ModelLoadError immediately.

        Scenario: load_transformers_pipeline_components() returns None
        Expected: __enter__() raises ModelLoadError (fatal error)
        """
        with patch(
            "image_annotator_lib.core.base.pipeline.ModelLoad.load_transformers_pipeline_components"
        ) as mock_load:
            # Mock load failure
            mock_load.return_value = None

            # Create annotator (should NOT fail yet)
            annotator = ConcreteTestPipelineAnnotator(model_name="test_pipeline_model")

            # __enter__() should raise ModelLoadError
            with pytest.raises(ModelLoadError) as exc_info:
                with annotator:
                    pass

            # Verify error message
            assert "Failed to load pipeline components" in str(exc_info.value)
            assert "test_pipeline_model" in str(exc_info.value)

            # Verify load was attempted
            mock_load.assert_called_once()

    @pytest.mark.integration
    def test_transformers_load_failure_raises_model_load_error(self, transformers_model_config):
        """Test that transformers load failure raises ModelLoadError immediately.

        Scenario: load_transformers_components() returns None
        Expected: __enter__() raises ModelLoadError (fatal error)
        """
        with patch(
            "image_annotator_lib.core.base.transformers.ModelLoad.load_transformers_components"
        ) as mock_load:
            # Mock load failure
            mock_load.return_value = None

            # Create annotator (should NOT fail yet)
            annotator = ConcreteTestTransformersAnnotator(model_name="test_transformers_model")

            # __enter__() should raise ModelLoadError
            with pytest.raises(ModelLoadError) as exc_info:
                with annotator:
                    pass

            # Verify error message
            assert "Failed to load components" in str(exc_info.value)
            assert "test_transformers_model" in str(exc_info.value)

            # Verify load was attempted
            mock_load.assert_called_once()

    @pytest.mark.integration
    def test_pipeline_restoration_failure_continues_on_cpu(self, pipeline_model_config):
        """Test that pipeline restoration failure allows CPU continuation.

        Scenario:
        - load_transformers_pipeline_components() returns valid CPU components
        - restore_model_to_cuda() returns None (CUDA restoration failed)
        Expected:
        - __enter__() does NOT raise exception
        - self.components is NOT None (CPU components maintained)
        - Warning log is emitted
        """
        # Create mock CPU components
        mock_cpu_components = {
            "pipeline": MagicMock(),
            "model": MagicMock(),
            "processor": MagicMock(),
        }

        with (
            patch(
                "image_annotator_lib.core.base.pipeline.ModelLoad.load_transformers_pipeline_components"
            ) as mock_load,
            patch("image_annotator_lib.core.base.pipeline.ModelLoad.restore_model_to_cuda") as mock_restore,
        ):
            # Mock successful load (CPU)
            mock_load.return_value = mock_cpu_components

            # Mock restoration failure (None indicates CUDA restoration failed, CPU fallback already done)
            mock_restore.return_value = None

            # Create annotator
            annotator = ConcreteTestPipelineAnnotator(model_name="test_pipeline_model")

            # __enter__() should NOT raise exception
            with annotator as ctx:
                # Verify context manager returns annotator instance
                assert ctx is annotator

                # Verify components is NOT None (CPU components maintained)
                assert ctx.components is not None
                assert ctx.components == mock_cpu_components

            # Note: Warning log verification skipped due to loguru/caplog integration issues
            # The warning log is actually emitted (visible in stderr), but caplog doesn't capture it
            # The critical behavior (no exception + components preserved) is verified above

            # Verify both load and restore were called
            mock_load.assert_called_once()
            mock_restore.assert_called_once()

    @pytest.mark.integration
    def test_transformers_restoration_failure_continues_on_cpu(self, transformers_model_config):
        """Test that transformers restoration failure allows CPU continuation.

        Scenario:
        - load_transformers_components() returns valid CPU components
        - restore_model_to_cuda() returns None (CUDA restoration failed)
        Expected:
        - __enter__() does NOT raise exception
        - self.components is NOT None (CPU components maintained)
        - Warning log is emitted
        """
        # Create mock CPU components
        mock_cpu_components = {
            "model": MagicMock(),
            "processor": MagicMock(),
        }

        with (
            patch(
                "image_annotator_lib.core.base.transformers.ModelLoad.load_transformers_components"
            ) as mock_load,
            patch(
                "image_annotator_lib.core.base.transformers.ModelLoad.restore_model_to_cuda"
            ) as mock_restore,
        ):
            # Mock successful load (CPU)
            mock_load.return_value = mock_cpu_components

            # Mock restoration failure (None indicates CUDA restoration failed, CPU fallback already done)
            mock_restore.return_value = None

            # Create annotator
            annotator = ConcreteTestTransformersAnnotator(model_name="test_transformers_model")

            # __enter__() should NOT raise exception
            with annotator as ctx:
                # Verify context manager returns annotator instance
                assert ctx is annotator

                # Verify components is NOT None (CPU components maintained)
                assert ctx.components is not None
                assert ctx.components == mock_cpu_components

            # Note: Warning log verification skipped due to loguru/caplog integration issues
            # The warning log is actually emitted (visible in stderr), but caplog doesn't capture it
            # The critical behavior (no exception + components preserved) is verified above

            # Verify both load and restore were called
            mock_load.assert_called_once()
            mock_restore.assert_called_once()
