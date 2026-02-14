"""
Test capability-based validation for UnifiedAnnotationResult
"""

import pytest
from pydantic import ValidationError

from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult


class TestTaskCapability:
    """Test TaskCapability enum"""

    def test_task_capability_values(self):
        """Test that TaskCapability has expected values"""
        assert TaskCapability.TAGS == "tags"
        assert TaskCapability.CAPTIONS == "captions"
        assert TaskCapability.SCORES == "scores"

    def test_task_capability_creation(self):
        """Test TaskCapability creation from strings"""
        assert TaskCapability("tags") == TaskCapability.TAGS
        assert TaskCapability("captions") == TaskCapability.CAPTIONS
        assert TaskCapability("scores") == TaskCapability.SCORES

    def test_invalid_capability(self):
        """Test that invalid capability values raise errors"""
        with pytest.raises(ValueError):
            TaskCapability("invalid_capability")


class TestUnifiedAnnotationResult:
    """Test UnifiedAnnotationResult capability-based validation"""

    def test_valid_tags_capability(self):
        """Test valid tags with TAGS capability"""
        result = UnifiedAnnotationResult(
            model_name="test-tagger", capabilities={TaskCapability.TAGS}, tags=["tag1", "tag2"]
        )
        assert result.tags == ["tag1", "tag2"]
        assert result.captions is None
        assert result.scores is None

    def test_valid_captions_capability(self):
        """Test valid captions with CAPTIONS capability"""
        result = UnifiedAnnotationResult(
            model_name="test-captioner",
            capabilities={TaskCapability.CAPTIONS},
            captions=["A beautiful scene"],
        )
        assert result.captions == ["A beautiful scene"]
        assert result.tags is None
        assert result.scores is None

    def test_valid_scores_capability(self):
        """Test valid scores with SCORES capability"""
        result = UnifiedAnnotationResult(
            model_name="test-scorer", capabilities={TaskCapability.SCORES}, scores={"aesthetic": 0.85}
        )
        assert result.scores == {"aesthetic": 0.85}
        assert result.tags is None
        assert result.captions is None

    def test_multimodal_capabilities(self):
        """Test multimodal model with multiple capabilities"""
        result = UnifiedAnnotationResult(
            model_name="gpt-4o",
            capabilities={TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES},
            tags=["car", "red"],
            captions=["A red car on the street"],
            scores={"aesthetic": 0.85, "quality": 0.92},
        )
        assert len(result.capabilities) == 3
        assert result.tags == ["car", "red"]
        assert result.captions == ["A red car on the street"]
        assert result.scores == {"aesthetic": 0.85, "quality": 0.92}

    def test_invalid_tags_without_capability(self):
        """Test that tags without TAGS capability raises ValidationError"""
        with pytest.raises(ValidationError, match="tags provided but TAGS not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-scorer", capabilities={TaskCapability.SCORES}, tags=["invalid"]
            )

    def test_invalid_captions_without_capability(self):
        """Test that captions without CAPTIONS capability raises ValidationError"""
        with pytest.raises(ValidationError, match="captions provided but CAPTIONS not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-scorer", capabilities={TaskCapability.SCORES}, captions=["invalid caption"]
            )

    def test_invalid_scores_without_capability(self):
        """Test that scores without SCORES capability raises ValidationError"""
        with pytest.raises(ValidationError, match="scores provided but SCORES not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-tagger", capabilities={TaskCapability.TAGS}, scores={"invalid": 0.5}
            )

    def test_empty_capabilities_error(self):
        """Test that empty capabilities raises ValidationError"""
        with pytest.raises(ValidationError, match="capabilities cannot be empty"):
            UnifiedAnnotationResult(model_name="test-model", capabilities=set())

    def test_capability_based_field_validation(self):
        """Test capability-based field validation combinations"""
        # Valid single capability combinations
        tag_result = UnifiedAnnotationResult(
            model_name="wd-tagger", capabilities={TaskCapability.TAGS}, tags=["anime", "1girl"]
        )

        caption_result = UnifiedAnnotationResult(
            model_name="blip-captioner",
            capabilities={TaskCapability.CAPTIONS},
            captions=["A girl standing in a garden"],
        )

        score_result = UnifiedAnnotationResult(
            model_name="aesthetic-scorer", capabilities={TaskCapability.SCORES}, scores={"aesthetic": 0.75}
        )

        # All should be valid
        assert tag_result.tags == ["anime", "1girl"]
        assert caption_result.captions == ["A girl standing in a garden"]
        assert score_result.scores == {"aesthetic": 0.75}

    def test_raw_output_preservation(self):
        """Test raw output preservation"""
        result_with_raw = UnifiedAnnotationResult(
            model_name="test-model",
            capabilities={TaskCapability.TAGS},
            tags=["test"],
            raw_output={
                "original_tensor": [0.1, 0.9, 0.3],
                "processing_params": {"threshold": 0.5},
                "api_response": {"usage": {"tokens": 150}},
            },
        )

        assert result_with_raw.raw_output is not None
        assert "original_tensor" in result_with_raw.raw_output
        assert "processing_params" in result_with_raw.raw_output

    def test_error_handling(self):
        """Test error field handling"""
        error_result = UnifiedAnnotationResult(
            model_name="failed-model", capabilities={TaskCapability.TAGS}, error="Model processing failed"
        )

        assert error_result.error == "Model processing failed"
        assert error_result.tags is None

    def test_metadata_fields(self):
        """Test metadata fields"""
        result = UnifiedAnnotationResult(
            model_name="test-model",
            capabilities={TaskCapability.TAGS},
            tags=["test"],
            provider_name="openai",
            framework="pytorch",
        )

        assert result.provider_name == "openai"
        assert result.framework == "pytorch"
