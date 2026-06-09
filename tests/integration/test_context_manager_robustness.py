"""Integration tests for context manager robustness against load and restoration failures.

Tests comprehensive error handling for model loading and CUDA restoration failures,
ensuring correct distinction between fatal errors and recoverable errors.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_annotator_lib.core.base.clip import ClipBaseAnnotator
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


class ConcreteTestClipAnnotator(ClipBaseAnnotator):
    """Concrete CLIP annotator for testing."""

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


class TestClipContextManagerRobustness:
    """ClipBaseAnnotator の __enter__() エラーハンドリングと2回目ロードのリグレッションテスト (Issue #149)。"""

    @pytest.fixture
    def clip_model_config(self, managed_config_registry):
        """CLIP モデルの最小 config を登録する。"""
        config = {
            "class": "ImprovedAesthetic",
            "model_path": "/dummy/improved_aesthetic.pth",
            "device": "cpu",
            "estimated_size_gb": 0.5,
            "base_model": "openai/clip-vit-large-patch14",
        }
        managed_config_registry.set("test_clip_model", config)
        return config

    @pytest.mark.integration
    def test_clip_load_failure_raises_model_load_error(self, clip_model_config):
        """初回ロード失敗 (had_cached_state=False) は即 ModelLoadError を上げる。

        Regression: Issue #149 修正前は None 返却をサイレントに通過し、
        後続の _preprocess_images() で RuntimeError になっていた。
        """
        # アノテータは patch 外で生成して __init__ に正しい config を渡す
        annotator = ConcreteTestClipAnnotator(model_name="test_clip_model")

        with (
            patch("image_annotator_lib.core.base.clip.ModelLoad.load_clip_components") as mock_load,
            patch(
                "image_annotator_lib.core.base.clip.ModelLoad._get_model_state", return_value=None
            ),
        ):
            mock_load.return_value = None

            with pytest.raises(ModelLoadError) as exc_info:
                with annotator:
                    pass

            assert "Failed to load CLIP components" in str(exc_info.value)
            assert "test_clip_model" in str(exc_info.value)
            mock_load.assert_called_once()

    @pytest.mark.integration
    def test_clip_second_call_new_instance_with_cached_state_reloads(self, clip_model_config):
        """キャッシュ済み状態で新インスタンスが load_clip_components() から None を受けたとき、
        状態をリセットして強制再ロードし、正常に components を設定する。

        Regression: Issue #149 — ImprovedAesthetic / WaifuAesthetic の2バッチ目で
        `RuntimeError: CLIP プロセッサがロードされていません。` になるバグ。
        """
        mock_components = {
            "clip_model": MagicMock(),
            "model": MagicMock(),
            "processor": MagicMock(),
        }

        call_count: dict[str, int] = {"n": 0}

        def _load_side_effect(**kwargs: Any) -> dict | None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None  # 1回目: キャッシュ済み扱いで None を返す
            return mock_components  # 2回目: 強制再ロードで実コンポーネントを返す

        annotator = ConcreteTestClipAnnotator(model_name="test_clip_model")

        with (
            patch("image_annotator_lib.core.base.clip.ModelLoad.load_clip_components") as mock_load,
            patch(
                "image_annotator_lib.core.base.clip.ModelLoad._get_model_state",
                return_value="on_cpu",  # キャッシュ済み状態をシミュレート
            ),
            patch("image_annotator_lib.core.base.clip.ModelLoad._release_model_state") as mock_release,
        ):
            mock_load.side_effect = _load_side_effect

            with annotator as ctx:
                assert ctx.components is mock_components

            mock_release.assert_called_once_with("test_clip_model")
            assert mock_load.call_count == 2

    @pytest.mark.integration
    def test_clip_second_call_retry_also_fails_raises_model_load_error(self, clip_model_config):
        """キャッシュ済み状態でリセット後の再ロードも失敗した場合は ModelLoadError を上げる。"""
        annotator = ConcreteTestClipAnnotator(model_name="test_clip_model")

        with (
            patch("image_annotator_lib.core.base.clip.ModelLoad.load_clip_components") as mock_load,
            patch(
                "image_annotator_lib.core.base.clip.ModelLoad._get_model_state",
                return_value="on_cpu",
            ),
            patch("image_annotator_lib.core.base.clip.ModelLoad._release_model_state"),
        ):
            mock_load.return_value = None  # 両回ともNone

            with pytest.raises(ModelLoadError):
                with annotator:
                    pass

            assert mock_load.call_count == 2
