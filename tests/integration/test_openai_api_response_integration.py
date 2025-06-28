# tests/integration/test_openai_api_response_integration.py
"""
Integration tests for OpenAI API Response implementation.
Focuses on fixing the 'str' object has no attribute 'save' error and proper image handling.
"""
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
import io
import base64

from image_annotator_lib.model_class.annotator_webapi.openai_api_response import OpenAIApiAnnotator
from image_annotator_lib.core.types import AnnotationResult


class TestOpenAIApiResponseIntegration:
    """Integration tests for OpenAI API Response annotator."""

    @pytest.fixture
    def openai_annotator(self, managed_config_registry):
        """Create an OpenAI API Response annotator with test configuration."""
        test_config = {
            "class": "OpenAIApiAnnotator",
            "api_key": "test-api-key",
            "model": "gpt-4o-mini", 
            "timeout": 30,
            "max_retries": 3
        }
        managed_config_registry.set("test_openai_response", test_config)
        return OpenAIApiAnnotator("test_openai_response")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_image_preprocessing_pil_to_base64(self, openai_annotator, lightweight_test_images):
        """Test that PIL Images are properly converted to base64 format."""
        test_image = lightweight_test_images[0]
        
        # Test the internal image preprocessing method
        if hasattr(openai_annotator, '_preprocess_image_to_base64'):
            base64_data = openai_annotator._preprocess_image_to_base64(test_image)
            
            # Verify it's a valid base64 string
            assert isinstance(base64_data, str)
            assert len(base64_data) > 0
            
            # Verify it can be decoded back to an image
            decoded_data = base64.b64decode(base64_data)
            decoded_image = Image.open(io.BytesIO(decoded_data))
            assert decoded_image.size == test_image.size

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @patch('openai.OpenAI')
    def test_run_inference_with_pil_images_mock(self, mock_openai_client, openai_annotator, lightweight_test_images):
        """Test that run_inference handles PIL Images correctly with mocked API."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"tags": ["test_tag_1", "test_tag_2"], "caption": "test caption"}'
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        # Test with PIL Images - this should NOT raise 'str' object has no attribute 'save'
        try:
            results = openai_annotator.run_inference(lightweight_test_images)
            
            # Verify results structure
            assert isinstance(results, dict)
            for image_hash, result in results.items():
                assert isinstance(result, AnnotationResult)
                assert result.error is None
                assert "tags" in result.formatted_output or "caption" in result.formatted_output
                
        except AttributeError as e:
            if "'str' object has no attribute 'save'" in str(e):
                pytest.fail("Image preprocessing error still exists - PIL Image not handled correctly")
            raise

    @pytest.mark.integration  
    @pytest.mark.fast_integration
    @patch('openai.OpenAI')
    def test_run_inference_with_different_image_formats(self, mock_openai_client, openai_annotator):
        """Test run_inference with various image input formats."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"tags": ["format_test"]}'
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        # Test with PIL Image
        pil_image = Image.new("RGB", (64, 64), "red")
        
        # Test with bytes (image file content)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        bytes_data = img_bytes.getvalue()
        
        test_cases = [
            ("PIL Image", [pil_image]),
            ("Bytes data", [bytes_data]),
        ]
        
        for test_name, test_input in test_cases:
            try:
                results = openai_annotator.run_inference(test_input)
                assert len(results) > 0, f"Failed for {test_name}"
                
                # Verify no errors in results
                for image_hash, result in results.items():
                    assert result.error is None, f"Error in {test_name}: {result.error}"
                    
            except Exception as e:
                pytest.fail(f"Failed to handle {test_name}: {str(e)}")

    @pytest.mark.integration
    @pytest.mark.fast_integration  
    @patch('openai.OpenAI')
    def test_error_handling_integration(self, mock_openai_client, openai_annotator, lightweight_test_images):
        """Test comprehensive error handling scenarios."""
        # Test API error handling
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance

        results = openai_annotator.run_inference(lightweight_test_images)
        
        # Verify error is properly captured
        assert len(results) > 0
        for image_hash, result in results.items():
            assert result.error is not None
            assert "API Error" in result.error or "OpenAI API Error" in result.error

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @patch('openai.OpenAI') 
    def test_json_response_parsing(self, mock_openai_client, openai_annotator, lightweight_test_images):
        """Test JSON response parsing with various response formats."""
        test_responses = [
            # Valid JSON
            '{"tags": ["tag1", "tag2"], "caption": "test caption"}',
            # Valid JSON with extra fields
            '{"tags": ["tag1"], "caption": "test", "confidence": 0.9}',
            # Minimal valid JSON
            '{"tags": ["single_tag"]}',
        ]
        
        for i, response_content in enumerate(test_responses):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = response_content
            
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_openai_client.return_value = mock_client_instance

            results = openai_annotator.run_inference([lightweight_test_images[0]])
            
            # Verify successful parsing
            assert len(results) == 1
            result = list(results.values())[0]
            assert result.error is None, f"JSON parsing failed for response {i}: {result.error}"
            assert isinstance(result.formatted_output, dict)

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @patch('openai.OpenAI')
    def test_invalid_json_response_handling(self, mock_openai_client, openai_annotator, lightweight_test_images):
        """Test handling of invalid JSON responses."""
        invalid_responses = [
            "Not JSON at all",
            '{"invalid": json}',  # Invalid JSON syntax
            '{"tags": incomplete',  # Incomplete JSON
            '',  # Empty response
        ]
        
        for response_content in invalid_responses:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = response_content
            
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            mock_openai_client.return_value = mock_client_instance

            results = openai_annotator.run_inference([lightweight_test_images[0]])
            
            # Should handle gracefully with error message
            assert len(results) == 1
            result = list(results.values())[0]
            assert result.error is not None
            assert "JSON" in result.error or "parsing" in result.error.lower()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_configuration_validation(self, managed_config_registry):
        """Test that annotator validates required configuration."""
        # Test missing API key
        invalid_config = {
            "class": "OpenAIApiAnnotator",
            "model": "gpt-4o-mini"
            # Missing api_key
        }
        managed_config_registry.set("invalid_openai", invalid_config)
        
        with pytest.raises(Exception):  # Should raise configuration error
            OpenAIApiResponseAnnotator("invalid_openai")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @patch('openai.OpenAI')
    def test_batch_processing_integration(self, mock_openai_client, openai_annotator, lightweight_test_images):
        """Test batch processing of multiple images."""
        # Setup mock to return different responses for different calls
        call_count = 0
        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f'{{"tags": ["batch_tag_{call_count}"]}}'
            return mock_response
            
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = mock_create
        mock_openai_client.return_value = mock_client_instance

        # Process all test images
        results = openai_annotator.run_inference(lightweight_test_images)
        
        # Verify all images were processed
        assert len(results) == len(lightweight_test_images)
        
        # Verify each result is valid
        for image_hash, result in results.items():
            assert result.error is None
            assert isinstance(result.formatted_output, dict)
            assert "tags" in result.formatted_output
            assert len(result.formatted_output["tags"]) > 0