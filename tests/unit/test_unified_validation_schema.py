"""
Test unified validation schema integration
"""

import pytest
from pydantic import ValidationError

from image_annotator_lib.core.types import TaskCapability, UnifiedAnnotationResult


class TestUnifiedValidationSchemaIntegration:
    """Test unified validation schema integration patterns"""

    def test_unified_annotation_result_capability_validation(self):
        """統一AnnotationResultのcapabilityバリデーションテスト"""
        # Tags capability
        result_with_tags = UnifiedAnnotationResult(
            model_name="test-tagger", capabilities={TaskCapability.TAGS}, tags=["tag1", "tag2"]
        )
        assert result_with_tags.tags == ["tag1", "tag2"]

        # Invalid combination - tags without TAGS capability
        with pytest.raises(ValidationError, match="tags provided but TAGS not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-scorer", capabilities={TaskCapability.SCORES}, tags=["invalid"]
            )

    def test_multimodal_annotation_result(self):
        """マルチモーダルLLM対応テスト"""
        # Multiple capabilities
        multimodal_result = UnifiedAnnotationResult(
            model_name="gpt-4o",
            capabilities={TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES},
            tags=["car", "red"],
            captions=["A red car on the street"],
            scores={"aesthetic": 0.85, "quality": 0.92},
        )
        assert len(multimodal_result.capabilities) == 3
        assert multimodal_result.tags == ["car", "red"]
        assert multimodal_result.captions == ["A red car on the street"]
        assert multimodal_result.scores == {"aesthetic": 0.85, "quality": 0.92}

    def test_capability_based_field_validation(self):
        """Capability-basedフィールドバリデーションテスト"""
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
        """生データ保持テスト"""
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

    def test_invalid_capability_combinations(self):
        """無効なcapability組み合わせテスト"""
        # Invalid captions without CAPTIONS capability
        with pytest.raises(ValidationError, match="captions provided but CAPTIONS not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-tagger", capabilities={TaskCapability.TAGS}, captions=["invalid caption"]
            )

        # Invalid scores without SCORES capability
        with pytest.raises(ValidationError, match="scores provided but SCORES not in capabilities"):
            UnifiedAnnotationResult(
                model_name="test-captioner", capabilities={TaskCapability.CAPTIONS}, scores={"invalid": 0.5}
            )

    def test_error_handling_with_capabilities(self):
        """エラーハンドリングとcapability組み合わせテスト"""
        error_result = UnifiedAnnotationResult(
            model_name="failed-model",
            capabilities={TaskCapability.TAGS, TaskCapability.CAPTIONS},
            error="Processing failed",
        )

        assert error_result.error == "Processing failed"
        assert error_result.tags is None
        assert error_result.captions is None

    def test_api_integration_pattern(self):
        """API統合パターンテスト"""
        # Simulate API layer returning unified results
        api_results = {
            "image_hash_1": {
                "wd-tagger": UnifiedAnnotationResult(
                    model_name="wd-tagger",
                    capabilities={TaskCapability.TAGS},
                    tags=["anime", "1girl", "blue_hair"],
                ),
                "gpt-4o": UnifiedAnnotationResult(
                    model_name="gpt-4o",
                    capabilities={TaskCapability.TAGS, TaskCapability.CAPTIONS, TaskCapability.SCORES},
                    tags=["anime", "character"],
                    captions=["An anime girl with blue hair"],
                    scores={"aesthetic": 0.92},
                ),
            }
        }

        # Verify structure
        assert "image_hash_1" in api_results
        assert "wd-tagger" in api_results["image_hash_1"]
        assert "gpt-4o" in api_results["image_hash_1"]

        # Verify capability-based access
        wd_result = api_results["image_hash_1"]["wd-tagger"]
        assert TaskCapability.TAGS in wd_result.capabilities
        assert wd_result.tags is not None
        assert wd_result.captions is None

        gpt_result = api_results["image_hash_1"]["gpt-4o"]
        assert len(gpt_result.capabilities) == 3
        assert gpt_result.tags is not None
        assert gpt_result.captions is not None
        assert gpt_result.scores is not None

    def test_framework_metadata_integration(self):
        """フレームワークメタデータ統合テスト"""
        results = [
            UnifiedAnnotationResult(
                model_name="onnx-tagger",
                capabilities={TaskCapability.TAGS},
                tags=["tag1"],
                framework="onnx",
            ),
            UnifiedAnnotationResult(
                model_name="clip-scorer",
                capabilities={TaskCapability.SCORES},
                scores={"aesthetic": 0.8},
                framework="pytorch",
            ),
            UnifiedAnnotationResult(
                model_name="gpt-4o",
                capabilities={TaskCapability.TAGS, TaskCapability.CAPTIONS},
                tags=["ai_generated"],
                captions=["AI generated image"],
                provider_name="openai",
                framework="api",
            ),
        ]

        # Verify framework information is preserved
        assert results[0].framework == "onnx"
        assert results[1].framework == "pytorch"
        assert results[2].framework == "api"
        assert results[2].provider_name == "openai"
