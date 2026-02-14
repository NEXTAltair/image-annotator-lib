"""
モデルコンポーネントの統一モック
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def mock_model_components():
    """モデルコンポーネントの統一モック"""
    components = {
        "processor": MagicMock(),
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }

    # processor のモック設定
    components["processor"].return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
        "pixel_values": MagicMock(),
    }
    components["processor"].batch_decode.return_value = ["test_caption"]
    components["processor"].to.return_value = components["processor"]

    # model のモック設定
    generate_output = MagicMock()
    generate_output.last_hidden_state = MagicMock()
    components["model"].generate.return_value = generate_output
    components["model"].to.return_value = components["model"]
    components["model"].eval.return_value = components["model"]

    # tokenizer のモック設定
    components["tokenizer"].decode.return_value = "test_text"
    components["tokenizer"].encode.return_value = [1, 2, 3]

    return components


@pytest.fixture
def mock_pil_image():
    """PIL Image の軽量モック"""
    image_mock = MagicMock(spec=Image.Image)
    image_mock.size = (224, 224)
    image_mock.mode = "RGB"
    image_mock.convert.return_value = image_mock
    image_mock.resize.return_value = image_mock
    return image_mock


@pytest.fixture
def mock_imagehash():
    """imagehash ライブラリのモック"""
    with patch("image_annotator_lib.core.base.annotator.imagehash") as mock:
        hash_mock = MagicMock()
        hash_mock.__str__ = MagicMock(return_value="abc123def456")
        mock.phash.return_value = hash_mock
        yield mock


@pytest.fixture
def mock_psutil():
    """psutil ライブラリのモック"""
    with patch("psutil.virtual_memory") as mock:
        memory_mock = MagicMock()
        memory_mock.available = 8 * 1024 * 1024 * 1024  # 8GB available
        memory_mock.total = 16 * 1024 * 1024 * 1024  # 16GB total
        mock.return_value = memory_mock
        yield mock


@pytest.fixture
def mock_model_load():
    """ModelLoad クラスのモック"""
    with patch("image_annotator_lib.core.model_factory.ModelLoad") as mock:
        model_load_mock = MagicMock()
        model_load_mock.get_model.return_value = MagicMock()
        model_load_mock.is_memory_sufficient.return_value = True
        model_load_mock.get_estimated_size.return_value = 1.0
        mock.return_value = model_load_mock
        yield model_load_mock


@pytest.fixture
def mock_provider_manager():
    """ProviderManager のモック"""
    with patch("image_annotator_lib.core.provider_manager.ProviderManager") as mock:
        mock.get_provider_instance.return_value = MagicMock()
        mock.run_inference_with_model.return_value = {
            "test_hash": {
                "tags": ["test_tag1", "test_tag2"],
                "formatted_output": {"caption": "test caption"},
                "error": None,
            }
        }
        mock._determine_provider.return_value = "openai"
        yield mock


@pytest.fixture
def mock_pydantic_ai_factory():
    """PydanticAI Factory のモック"""
    with patch("image_annotator_lib.core.pydantic_ai_factory.PydanticAIAgentFactory") as mock:
        agent_mock = MagicMock()
        agent_mock.run_sync.return_value = MagicMock(data={"tags": ["test"], "caption": "test"})

        mock.get_provider.return_value = MagicMock()
        mock.create_agent.return_value = agent_mock
        mock.get_cached_agent.return_value = agent_mock
        yield mock
