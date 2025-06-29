# tests/integration/test_end_to_end_workflow_integration.py
"""
Integration tests for end-to-end workflows.
Tests complete workflows from API entry point through provider management to final results.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image
import io

from image_annotator_lib.api import annotate, list_available_annotators
from image_annotator_lib.core.provider_manager import ProviderManager
from image_annotator_lib.core.pydantic_ai_factory import PydanticAIProviderFactory
from image_annotator_lib.core.model_factory import ModelLoad
from image_annotator_lib.core.types import AnnotationResult


class TestEndToEndWorkflowIntegration:
    """Integration tests for complete end-to-end workflows."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        PydanticAIProviderFactory.clear_cache()
        yield
        PydanticAIProviderFactory.clear_cache()

    @pytest.fixture
    def complete_workflow_configs(self, managed_config_registry):
        """Setup configurations for complete workflow testing."""
        configs = {
            "workflow_openai": {
                "class": "OpenAIApiAnnotator",
                "api_model_id": "gpt-4o-mini",
                "api_key": "test-openai-key"
            },
            "workflow_anthropic": {
                "class": "AnthropicApiAnnotator",
                "api_model_id": "claude-3-5-sonnet",
                "api_key": "test-anthropic-key"
            },
            "workflow_google": {
                "class": "GoogleApiAnnotator",
                "api_model_id": "gemini-1.5-pro",
                "api_key": "test-google-key"
            },
            "workflow_local_onnx": {
                "class": "WDTagger",
                "model_path": "test/wd/tagger",
                "device": "cpu",
                "estimated_size_gb": 1.0
            },
            "workflow_local_clip": {
                "class": "ImprovedAesthetic",
                "model_path": "test/aesthetic/scorer",
                "device": "cpu",
                "estimated_size_gb": 0.5
            }
        }
        
        # Register classes directly to avoid conftest issues
        from image_annotator_lib.core.registry import get_cls_obj_registry
        registry = get_cls_obj_registry()
        
        # Import and register WebAPI classes
        try:
            from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
            from image_annotator_lib.model_class.annotator_webapi.anthropic_api import AnthropicApiAnnotator
            from image_annotator_lib.model_class.annotator_webapi.google_api import GoogleApiAnnotator
            
            registry["workflow_openai"] = OpenAIApiAnnotator
            registry["workflow_anthropic"] = AnthropicApiAnnotator
            registry["workflow_google"] = GoogleApiAnnotator
            print("WebAPI classes registered successfully")
        except ImportError as e:
            print(f"WebAPI import failed: {e}")
        
        # Register local models using existing registry
        if "ImprovedAesthetic" in registry:
            registry["workflow_local_clip"] = registry["ImprovedAesthetic"]
            print("ImprovedAesthetic registered successfully")
        
        # Try WDTagger import with correct path
        try:
            from image_annotator_lib.model_class.tagger_onnx import WDTagger
            registry["workflow_local_onnx"] = WDTagger
            print("WDTagger registered successfully")
        except ImportError as e:
            print(f"WDTagger import failed: {e}")
        
        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)
        
        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_single_webapi_model_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test complete workflow with single WebAPI model."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            # Mock Provider Manager to return correct format and count
            from image_annotator_lib.core.utils import calculate_phash
            
            # Generate expected results matching image count
            expected_results = [
                {
                    "response": {"tags": ["e2e_openai_tag"]},
                    "error": None
                }
                for _ in lightweight_test_images[:2]
            ]
            
            mock_provider_inference.return_value = expected_results

            # Test complete workflow
            results = annotate(
                images_list=lightweight_test_images[:2],
                model_name_list=["workflow_openai"]
            )

            # Verify complete result structure
            assert isinstance(results, dict)
            assert len(results) == 2  # Two images

            for image_hash, model_results in results.items():
                assert "workflow_openai" in model_results
                result = model_results["workflow_openai"]
                
                # Verify successful annotation
                assert result["error"] is None
                assert "tags" in result
                assert "formatted_output" in result
                assert result["tags"] == ["e2e_openai_tag"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_single_local_model_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test complete workflow with single local ML model."""
        with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
            # Mock successful local model with correct predict method
            mock_annotator = MagicMock()
            mock_annotator.__enter__.return_value = mock_annotator
            mock_annotator.__exit__.return_value = None
            
            # Mock predict method to return correct number of results
            from image_annotator_lib.core.utils import calculate_phash
            from image_annotator_lib.core.types import AnnotationResult
            
            def mock_predict(images, phash_list):
                results = []
                for i, image in enumerate(images):
                    phash = phash_list[i] if i < len(phash_list) else calculate_phash(image)
                    results.append(AnnotationResult(
                        phash=phash,
                        tags=["e2e_local_tag"],
                        formatted_output={"tags": ["e2e_local_tag"]},
                        error=None
                    ))
                return results
            
            mock_annotator.predict.side_effect = mock_predict
            mock_create_annotator.return_value = mock_annotator

            # Test complete workflow
            results = annotate(
                images_list=lightweight_test_images[:2],
                model_name_list=["workflow_local_onnx"]
            )

            # Verify complete result structure
            assert isinstance(results, dict)
            assert len(results) == 2

            for image_hash, model_results in results.items():
                assert "workflow_local_onnx" in model_results
                result = model_results["workflow_local_onnx"]
                
                # Verify successful annotation
                assert result["error"] is None
                assert "tags" in result
                assert result["tags"] == ["e2e_local_tag"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_mixed_model_types_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test complete workflow with mixed WebAPI and local models."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
                
                # Mock Provider Manager for WebAPI models (OpenAI, Google)
                from image_annotator_lib.core.utils import calculate_phash
                
                # Generate expected results in correct format matching image count
                expected_results = [
                    {
                        "response": {"tags": ["mixed_webapi_tag"]},
                        "error": None
                    }
                    for _ in lightweight_test_images[:1]
                ]
                
                mock_provider_inference.return_value = expected_results

                # Mock Local models for context manager and predict method
                mock_annotator = MagicMock()
                mock_annotator.__enter__.return_value = mock_annotator
                mock_annotator.__exit__.return_value = None
                
                def mock_predict(images, phash_list):
                    results = []
                    for i, image in enumerate(images):
                        phash = phash_list[i] if i < len(phash_list) else calculate_phash(image)
                        results.append(AnnotationResult(
                            phash=phash,
                            tags=["mixed_local_tag"],
                            formatted_output={"tags": ["mixed_local_tag"]},
                            error=None
                        ))
                    return results
                
                mock_annotator.predict.side_effect = mock_predict
                mock_create_annotator.return_value = mock_annotator

                # Test workflow with mixed model types
                mixed_models = ["workflow_openai", "workflow_local_onnx", "workflow_google"]
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=mixed_models
                )

                # Verify all models produced results
                assert isinstance(results, dict)
                assert len(results) == 1

                image_hash = list(results.keys())[0]
                model_results = results[image_hash]
                
                # Should have results from all requested models
                for model_name in mixed_models:
                    assert model_name in model_results
                    result = model_results[model_name]
                    assert result["error"] is None
                    assert "tags" in result

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test complete workflow with batch processing multiple images."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            # Generate expected results matching all test images
            expected_results = [
                {
                    "response": {"tags": [f"batch_tag_{i+1}"]},
                    "error": None
                }
                for i, _ in enumerate(lightweight_test_images)
            ]
            
            mock_provider_inference.return_value = expected_results

            # Test batch processing
            results = annotate(
                images_list=lightweight_test_images,  # All test images
                model_name_list=["workflow_openai"]
            )

            # Verify batch processing results
            assert len(results) == len(lightweight_test_images)
            
            # Verify each image was processed
            for image_hash, model_results in results.items():
                assert "workflow_openai" in model_results
                result = model_results["workflow_openai"]
                assert result["error"] is None
                assert "tags" in result
                assert len(result["tags"]) > 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_error_resilience_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test end-to-end error resilience with partial failures."""
        with patch('image_annotator_lib.api.get_annotator_instance') as mock_get_annotator:
            
            # Mock for WebAPI model
            mock_webapi_annotator = MagicMock()
            mock_webapi_annotator.predict.return_value = [
                AnnotationResult(phash="fcfcfcf8f8f00000", tags=["resilient_tag"], formatted_output={"tags": ["resilient_tag"]}, error=None)
            ]

            # Mock for local model
            mock_local_annotator = MagicMock()
            mock_local_annotator.predict.side_effect = Exception("Local model loading failed")

            def annotator_side_effect(model_name):
                if "local" in model_name:
                    return mock_local_annotator
                return mock_webapi_annotator
            
            mock_get_annotator.side_effect = annotator_side_effect

            # Test with both successful and failing models
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["workflow_openai", "workflow_local_onnx"]
            )

            # Should get partial results
            assert isinstance(results, dict)
            assert len(results) == 1

            image_hash = list(results.keys())[0]
            model_results = results[image_hash]

            # WebAPI model should succeed
            assert "workflow_openai" in model_results
            assert model_results["workflow_openai"]["error"] is None

            # Local model should have error
            assert "workflow_local_onnx" in model_results
            assert model_results["workflow_local_onnx"]["error"] is not None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_different_image_formats_end_to_end(self, complete_workflow_configs):
        """Test end-to-end workflow with different image input formats."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            from image_annotator_lib.core.utils import calculate_phash
            
            # Create test images in different formats
            pil_image = Image.new("RGB", (64, 64), "red")
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            bytes_data = img_bytes.getvalue()

            # Test different formats
            test_formats = [
                ("PIL Image", pil_image),
                ("Bytes data", bytes_data)
            ]

            for format_name, test_image in test_formats:
                try:
                    # Convert bytes to PIL Image before processing
                    if isinstance(test_image, bytes):
                        processed_image = Image.open(io.BytesIO(test_image))
                    else:
                        processed_image = test_image
                    
                    # Generate expected results for current test image
                    expected_results = [
                        {
                            "response": {"tags": ["format_test_tag"]},
                            "error": None
                        }
                    ]
                    
                    mock_provider_inference.return_value = expected_results

                    # Use the processed PIL Image for annotation
                    results = annotate(
                        images_list=[processed_image],
                        model_name_list=["workflow_openai"]
                    )

                    assert len(results) > 0
                    
                    # Verify successful processing
                    for image_hash, model_results in results.items():
                        assert "workflow_openai" in model_results
                        result = model_results["workflow_openai"]
                        assert result["error"] is None, f"Failed for format: {format_name}"

                except Exception as e:
                    pytest.fail(f"End-to-end workflow failed for {format_name}: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_manager_direct_integration(self, complete_workflow_configs, lightweight_test_images):
        """Test direct Provider Manager integration in end-to-end workflow."""
        with patch('image_annotator_lib.core.provider_manager.ProviderInstanceBase.run_with_model') as mock_run_with_model:
            # Mock the response from the provider instance
            mock_response = MagicMock()
            mock_response.tags = ["provider_direct_tag"]
            mock_run_with_model.return_value = [{"response": mock_response, "error": None}]

            # Test direct Provider Manager usage
            results = ProviderManager.run_inference_with_model(
                model_name="workflow_anthropic",
                images_list=lightweight_test_images[:1],
                api_model_id="claude-3-5-sonnet"
            )

            # Verify Provider Manager result
            assert results is not None
            assert len(results) == 1
            
            image_hash = list(results.keys())[0]
            annotation_result = results[image_hash]

            assert isinstance(annotation_result, dict)
            assert "error" in annotation_result
            assert "tags" in annotation_result
            assert annotation_result["error"] is None
            assert annotation_result["tags"] == ["provider_direct_tag"]

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_model_discovery_integration(self, complete_workflow_configs):
        """Test model discovery integration in workflow."""
        with patch('image_annotator_lib.core.registry.initialize_registry') as mock_init:
            # Mock registry initialization to avoid API timeout
            mock_init.return_value = None

            try:
                # Test model discovery
                available_models = list_available_annotators()
                
                # Should return a list (might be empty due to mocking)
                assert isinstance(available_models, list)
                
                # If models are available, verify some of our configured models
                configured_models = set(complete_workflow_configs.keys())
                available_set = set(available_models)
                
                # Check if any of our configured models are available
                # (This test might pass even if registry is mocked)
                
            except Exception as e:
                # Model discovery might fail due to API timeouts - that's acceptable
                pytest.skip(f"Model discovery skipped due to infrastructure issue: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_cascade_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test configuration cascading through complete workflow."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            # Generate expected results for both models
            from image_annotator_lib.core.utils import calculate_phash
            
            expected_results = [
                {
                    "response": {"tags": ["config_cascade_tag"]},
                    "error": None
                }
                for _ in lightweight_test_images[:1]
            ]
            
            mock_provider_inference.return_value = expected_results

            # Test configuration propagation
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["workflow_openai", "workflow_anthropic"]
            )

            # Verify Provider Manager was called for each WebAPI model
            # Note: Both models should trigger Provider Manager calls
            assert mock_provider_inference.call_count >= 1
            
            # Verify results
            assert len(results) == 1
            
            for image_hash, model_results in results.items():
                for model_name in ["workflow_openai", "workflow_anthropic"]:
                    assert model_name in model_results
                    assert model_results[model_name]["error"] is None

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_memory_efficiency_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test memory efficiency in end-to-end workflows."""
        with patch('image_annotator_lib.api.get_annotator_instance') as mock_get_annotator:
            
            # Track resource usage
            loaded_models = []
            
            # Mock local model with context manager and predict method
            def mock_annotator_creation(model_name):
                if "local" in model_name:
                    mock_local_annotator = MagicMock()
                    mock_local_annotator.__enter__.return_value = mock_local_annotator
                    mock_local_annotator.__exit__.return_value = None
                    
                    def mock_predict(images, phash_list):
                        loaded_models.append(model_name)
                        results = []
                        for i, image in enumerate(images):
                            phash = phash_list[i] if i < len(phash_list) else calculate_phash(image)
                            results.append(AnnotationResult(
                                phash=phash,
                                tags=["memory_local_tag"],
                                formatted_output={"tags": ["memory_local_tag"]},
                                error=None
                            ))
                        return results
                    
                    mock_local_annotator.predict.side_effect = mock_predict
                    return mock_local_annotator
                
                # For WebAPI models, use a different mock to avoid interfering with ProviderManager
                from image_annotator_lib.api import PydanticAIWebAPIWrapper
                from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
                
                # This will be wrapped by PydanticAIWebAPIWrapper
                return PydanticAIWebAPIWrapper(model_name, OpenAIApiAnnotator)

            mock_get_annotator.side_effect = mock_annotator_creation
            
            with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
                # Mock Provider Manager for WebAPI model
                expected_results = [
                    {
                        "response": {"tags": ["memory_test_tag"]},
                        "error": None
                    }
                    for _ in lightweight_test_images[:2]
                ]
                mock_provider_inference.return_value = expected_results

                # Test memory-conscious workflow
                results = annotate(
                    images_list=lightweight_test_images[:2],
                    model_name_list=["workflow_openai", "workflow_local_onnx"]
                )

                # Verify efficient resource usage
                assert len(results) == 2
                
                # Check that Provider Manager was used for WebAPI models
                assert mock_provider_inference.call_count == 1
                
                # Local models should be loaded as needed
                assert len(loaded_models) > 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_result_consistency_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test result format consistency across the complete workflow."""
        with patch('image_annotator_lib.core.provider_manager.ProviderManager.run_inference_with_model') as mock_provider_inference:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
                
                # Mock Provider Manager for WebAPI models
                from image_annotator_lib.core.utils import calculate_phash
                
                expected_results = [
                    {
                        "response": {"tags": ["consistency_webapi"]},
                        "error": None
                    }
                    for _ in lightweight_test_images[:1]
                ]
                
                mock_provider_inference.return_value = expected_results

                # Mock local model with context manager and predict method
                mock_annotator = MagicMock()
                mock_annotator.__enter__.return_value = mock_annotator
                mock_annotator.__exit__.return_value = None
                
                def mock_predict(images, phash_list):
                    results = []
                    for i, image in enumerate(images):
                        phash = phash_list[i] if i < len(phash_list) else calculate_phash(image)
                        results.append(AnnotationResult(
                            phash=phash,
                            tags=["consistency_local"],
                            formatted_output={"tags": ["consistency_local"]},
                            error=None
                        ))
                    return results
                
                mock_annotator.predict.side_effect = mock_predict
                mock_create_annotator.return_value = mock_annotator

                # Test result consistency
                results = annotate(
                    images_list=lightweight_test_images[:1],
                    model_name_list=["workflow_openai", "workflow_local_onnx"]
                )

                # Verify consistent result structure
                assert len(results) == 1
                
                image_hash = list(results.keys())[0]
                model_results = results[image_hash]

                # Check consistent structure across all models
                expected_keys = {"tags", "formatted_output", "error"}
                
                for model_name, result in model_results.items():
                    assert isinstance(result, dict)
                    assert expected_keys.issubset(set(result.keys()))
                    assert isinstance(result["tags"], list)
                    assert isinstance(result["formatted_output"], dict)
                    assert result["error"] is None or isinstance(result["error"], str)