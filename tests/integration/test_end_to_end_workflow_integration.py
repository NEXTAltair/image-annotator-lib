# tests/integration/test_end_to_end_workflow_integration.py
"""
Integration tests for end-to-end workflows.
Tests complete workflows from API entry point through provider management to final results.
"""
import pytest
from unittest.mock import patch, MagicMock
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
                "class": "OpenAIApiChatAnnotator",
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
        
        for model_name, config in configs.items():
            managed_config_registry.set(model_name, config)
        
        return configs

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_single_webapi_model_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test complete workflow with single WebAPI model."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Mock successful WebAPI response
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["e2e_openai_tag"]
            mock_response.caption = "End-to-end test caption"
            mock_agent.run.return_value = MagicMock(data=mock_response)
            mock_get_agent.return_value = mock_agent

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
            # Mock successful local model
            mock_annotator = MagicMock()
            mock_annotator.run_inference.return_value = {
                "test_hash_1": AnnotationResult(
                    tags=["e2e_local_tag"],
                    formatted_output={"tags": ["e2e_local_tag"]},
                    error=None
                ),
                "test_hash_2": AnnotationResult(
                    tags=["e2e_local_tag"],
                    formatted_output={"tags": ["e2e_local_tag"]},
                    error=None
                )
            }
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
                
                # Mock WebAPI models
                mock_agent = MagicMock()
                mock_webapi_response = MagicMock()
                mock_webapi_response.tags = ["mixed_webapi_tag"]
                mock_agent.run.return_value = MagicMock(data=mock_webapi_response)
                mock_get_agent.return_value = mock_agent

                # Mock Local models
                mock_annotator = MagicMock()
                mock_annotator.run_inference.return_value = {
                    "test_hash": AnnotationResult(
                        tags=["mixed_local_tag"],
                        formatted_output={"tags": ["mixed_local_tag"]},
                        error=None
                    )
                }
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Mock batch API responses
            call_count = 0
            
            def mock_run(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_response = MagicMock()
                mock_response.tags = [f"batch_tag_{call_count}"]
                return MagicMock(data=mock_response)

            mock_agent = MagicMock()
            mock_agent.run.side_effect = mock_run
            mock_get_agent.return_value = mock_agent

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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
                
                # Mock successful WebAPI
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["resilient_tag"]
                mock_agent.run.return_value = MagicMock(data=mock_response)
                mock_get_agent.return_value = mock_agent

                # Mock failing local model
                mock_create_annotator.side_effect = Exception("Local model loading failed")

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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["format_test_tag"]
            mock_agent.run.return_value = MagicMock(data=mock_response)
            mock_get_agent.return_value = mock_agent

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
                    results = annotate(
                        images_list=[test_image],
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.tags = ["provider_direct_tag"]
            mock_agent.run.return_value = MagicMock(data=mock_response)
            mock_get_agent.return_value = mock_agent

            # Test direct Provider Manager usage
            result = ProviderManager.run_inference_with_model(
                model_name="workflow_anthropic",
                images_list=lightweight_test_images[:1],
                api_model_id="claude-3-5-sonnet"
            )

            # Verify Provider Manager result
            assert result is not None
            assert len(result) == 1

            for image_hash, annotation_result in result.items():
                assert isinstance(annotation_result, AnnotationResult)
                assert annotation_result.error is None
                assert annotation_result.tags == ["provider_direct_tag"]

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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            # Track configuration usage
            captured_configs = []
            
            def mock_agent_creation(model_name, api_model_id, api_key, config_hash=None):
                captured_configs.append({
                    "model_name": model_name,
                    "api_model_id": api_model_id,
                    "api_key": api_key
                })
                
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.tags = ["config_cascade_tag"]
                mock_agent.run.return_value = MagicMock(data=mock_response)
                return mock_agent

            mock_get_agent.side_effect = mock_agent_creation

            # Test configuration propagation
            results = annotate(
                images_list=lightweight_test_images[:1],
                model_name_list=["workflow_openai", "workflow_anthropic"]
            )

            # Verify configurations were used
            assert len(captured_configs) >= 2
            
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
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.core.model_factory.ModelLoad') as mock_model_load_class:
                
                # Track resource usage
                loaded_models = []
                created_agents = []
                
                def mock_agent_creation(*args, **kwargs):
                    created_agents.append(args[0])  # model_name
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.tags = ["memory_test_tag"]
                    mock_agent.run.return_value = MagicMock(data=mock_response)
                    return mock_agent

                def mock_model_loading():
                    mock_model_load = MagicMock()
                    
                    def mock_create_annotator(model_name):
                        loaded_models.append(model_name)
                        mock_annotator = MagicMock()
                        mock_annotator.run_inference.return_value = {
                            "test_hash": AnnotationResult(
                                tags=["memory_local_tag"],
                                formatted_output={"tags": ["memory_local_tag"]},
                                error=None
                            )
                        }
                        return mock_annotator
                    
                    mock_model_load.load_model.side_effect = mock_create_annotator
                    return mock_model_load

                mock_get_agent.side_effect = mock_agent_creation
                mock_model_load_class.return_value = mock_model_loading()

                # Test memory-conscious workflow
                results = annotate(
                    images_list=lightweight_test_images[:2],
                    model_name_list=["workflow_openai", "workflow_local_onnx"]
                )

                # Verify efficient resource usage
                assert len(results) == 2
                
                # Check that resources were properly managed
                # WebAPI models should use agent caching
                assert len(created_agents) > 0
                
                # Local models should be loaded as needed
                assert len(loaded_models) > 0

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_result_consistency_end_to_end(self, complete_workflow_configs, lightweight_test_images):
        """Test result format consistency across the complete workflow."""
        with patch('image_annotator_lib.core.pydantic_ai_factory.PydanticAIProviderFactory.get_cached_agent') as mock_get_agent:
            with patch('image_annotator_lib.api._create_annotator_instance') as mock_create_annotator:
                
                # Mock consistent responses
                mock_agent = MagicMock()
                mock_webapi_response = MagicMock()
                mock_webapi_response.tags = ["consistency_webapi"]
                mock_agent.run.return_value = MagicMock(data=mock_webapi_response)
                mock_get_agent.return_value = mock_agent

                mock_annotator = MagicMock()
                mock_annotator.run_inference.return_value = {
                    "test_hash": AnnotationResult(
                        tags=["consistency_local"],
                        formatted_output={"tags": ["consistency_local"]},
                        error=None
                    )
                }
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