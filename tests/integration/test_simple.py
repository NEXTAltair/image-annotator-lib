# tests/integration/test_simple.py
from unittest.mock import patch
from image_annotator_lib.api import annotate
from tests.integration.conftest import lightweight_test_images
from pydantic_ai.messages import ModelResponse, TextPart
import json

def test_simple_annotation(lightweight_test_images):
    with patch("pydantic_ai.Agent.run") as mock_run:
        annotation_data = {"tags": ["mocked_tag1", "mocked_tag2"], "formatted_output": "mocked_output"}
        response_json = json.dumps(annotation_data)
        mock_run.return_value = ModelResponse(parts=[TextPart(content=response_json)])
        
        annotate(images_list=lightweight_test_images[:1], model_name_list=["memory_openai_1"])
