from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from image_annotator_lib.core.base.transformers import TransformersBaseAnnotator


# --- テスト用ダミーサブクラス ---
class DummyTransformersAnnotator(TransformersBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # 必要なモックをセット（Any型で扱う）
        self.components: dict[str, Any] = {
            "processor": MagicMock(),
            "model": MagicMock(),
        }
        self.device = "cpu"
        self.max_length = 10

# --- _preprocess_images ---
def test_preprocess_images_calls_processor():
    annotator = DummyTransformersAnnotator("dummy-model")
    mock_processor = annotator.components["processor"]
    # processorの戻り値もto()を持つモック
    mock_tensor = MagicMock()
    mock_tensor.keys.return_value = ["input_ids"]
    # processor(images=..., return_tensors=...) -> mock_tensor
    def processor_side_effect(*args, **kwargs):
        return MagicMock(to=MagicMock(return_value=mock_tensor), keys=mock_tensor.keys)
    mock_processor.side_effect = processor_side_effect
    image = Image.new("RGB", (32, 32))
    result = annotator._preprocess_images([image])
    assert isinstance(result, list)
    assert mock_processor.call_count == 1
    assert result[0].keys() == ["input_ids"]

# --- _run_inference ---
def test_run_inference_generate_and_logits():
    annotator = DummyTransformersAnnotator("dummy-model")
    mock_model = annotator.components["model"]
    # generate()を持つ場合
    mock_out = MagicMock()
    mock_out.last_hidden_state = torch.tensor([[1, 2]])
    mock_model.generate.return_value = mock_out
    processed = [{"input_ids": torch.tensor([[1, 2]])}]
    out = annotator._run_inference(processed)
    assert isinstance(out, list)
    assert torch.equal(out[0], mock_out.last_hidden_state)

# --- _format_predictions ---
def test_format_predictions_batch_decode():
    annotator = DummyTransformersAnnotator("dummy-model")
    mock_processor = annotator.components["processor"]
    # batch_decodeが呼ばれる
    mock_processor.batch_decode = MagicMock(return_value=["text1"])
    token_ids = [torch.tensor([1, 2])]
    out = annotator._format_predictions(token_ids)
    assert out == ["text1"]
    mock_processor.batch_decode.assert_called_once()

def test_format_predictions_no_batch_decode():
    annotator = DummyTransformersAnnotator("dummy-model")
    # batch_decodeなし、CLIPProcessor型のisinstanceチェックを避けるためMagicMockで上書き
    class DummyCLIPProcessor:
        pass
    processor_mock = DummyCLIPProcessor()
    annotator.components["processor"] = processor_mock
    out = annotator._format_predictions([torch.tensor([1, 2])])
    assert out == [""]

# --- _generate_tags ---
def test_generate_tags_str():
    annotator = DummyTransformersAnnotator("dummy-model")
    out = annotator._generate_tags("caption")
    assert out == ["caption"]

def test_generate_tags_not_str():
    annotator = DummyTransformersAnnotator("dummy-model")
    out = annotator._generate_tags(["a", "b"])
    assert out == []

# --- 例外系 ---
def test_preprocess_images_no_processor():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components.pop("processor", None)
    with pytest.raises(Exception):
        annotator._preprocess_images([Image.new("RGB", (32, 32))])

def test_run_inference_no_model():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components["model"] = None
    with pytest.raises(Exception):
        annotator._run_inference([{"input_ids": torch.tensor([[1, 2]])}])

def test_format_predictions_no_processor():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components["processor"] = None
    with pytest.raises(Exception):
        annotator._format_predictions([torch.tensor([1, 2])])
