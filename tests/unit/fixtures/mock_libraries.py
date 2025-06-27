"""
ML系ライブラリの統一モック
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_heavy_libraries():
    """重いMLライブラリを一括モック"""
    # torch のモック
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.OutOfMemoryError = Exception
    torch_mock.cuda.max_memory_allocated = MagicMock(return_value=1024 * 1024 * 1024)  # 1GB
    torch_mock.tensor = MagicMock()
    torch_mock.Tensor = MagicMock

    # transformers のモック
    transformers_mock = MagicMock()
    transformers_mock.CLIPProcessor = MagicMock()
    transformers_mock.BlipProcessor = MagicMock()
    transformers_mock.BlipForConditionalGeneration = MagicMock()

    # tensorflow のモック
    tensorflow_mock = MagicMock()
    tensorflow_mock.keras = MagicMock()
    tensorflow_mock.config = MagicMock()

    # onnxruntime のモック
    onnxruntime_mock = MagicMock()
    onnxruntime_mock.InferenceSession = MagicMock()

    # sys.modules への設定
    mocked_modules = {
        "torch": torch_mock,
        "transformers": transformers_mock,
        "tensorflow": tensorflow_mock,
        "onnxruntime": onnxruntime_mock,
        "clip": MagicMock(),
        "PIL": MagicMock(),
        "numpy": MagicMock(),
        "cv2": MagicMock(),
    }

    with patch.dict(sys.modules, mocked_modules):
        yield


@pytest.fixture
def mock_torch():
    """torch専用モック"""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.device.return_value = "cuda:0"
    torch_mock.tensor.return_value = MagicMock()
    return torch_mock


@pytest.fixture
def mock_transformers():
    """transformers専用モック"""
    transformers_mock = MagicMock()

    # プロセッサのモック
    processor_mock = MagicMock()
    processor_mock.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    processor_mock.batch_decode.return_value = ["mocked_text"]

    # モデルのモック
    model_mock = MagicMock()
    model_mock.generate.return_value = MagicMock(last_hidden_state=MagicMock())

    transformers_mock.CLIPProcessor = processor_mock
    transformers_mock.BlipProcessor = processor_mock
    transformers_mock.BlipForConditionalGeneration = model_mock

    return transformers_mock


@pytest.fixture
def mock_tensorflow():
    """tensorflow専用モック"""
    tf_mock = MagicMock()
    tf_mock.keras.models.load_model.return_value = MagicMock()
    tf_mock.config.experimental.set_memory_growth = MagicMock()
    return tf_mock


@pytest.fixture
def mock_onnxruntime():
    """onnxruntime専用モック"""
    onnx_mock = MagicMock()
    session_mock = MagicMock()
    session_mock.run.return_value = [MagicMock()]
    onnx_mock.InferenceSession.return_value = session_mock
    return onnx_mock
