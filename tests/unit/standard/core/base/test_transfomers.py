from typing import Any
from unittest.mock import MagicMock, patch

import pytest
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
@pytest.mark.standard
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
@pytest.mark.standard
@patch("image_annotator_lib.core.base.transformers.torch")
def test_run_inference_generate_and_logits(mock_torch):
    # torch.tensorのモック
    mock_tensor = MagicMock()
    mock_torch.tensor.return_value = mock_tensor
    mock_torch.equal.return_value = True

    annotator = DummyTransformersAnnotator("dummy-model")
    mock_model = annotator.components["model"]
    # generate()を持つ場合
    mock_out = MagicMock()
    mock_out.last_hidden_state = mock_tensor
    mock_model.generate.return_value = mock_out
    processed = [{"input_ids": mock_tensor}]
    out = annotator._run_inference(processed)
    assert isinstance(out, list)
    assert out[0] == mock_out.last_hidden_state


# --- _format_predictions ---
@pytest.mark.standard
def test_format_predictions_batch_decode():
    annotator = DummyTransformersAnnotator("dummy-model")
    mock_processor = annotator.components["processor"]
    # batch_decodeが呼ばれる
    mock_processor.batch_decode = MagicMock(return_value=["text1"])
    mock_tensor = MagicMock()
    token_ids = [mock_tensor]
    out = annotator._format_predictions(token_ids)
    assert out == ["text1"]
    mock_processor.batch_decode.assert_called_once()


@pytest.mark.standard
def test_format_predictions_no_batch_decode():
    annotator = DummyTransformersAnnotator("dummy-model")

    # batch_decodeなし、CLIPProcessor型のisinstanceチェックを避けるためMagicMockで上書き
    class DummyCLIPProcessor:
        pass

    processor_mock = DummyCLIPProcessor()
    annotator.components["processor"] = processor_mock
    mock_tensor = MagicMock()
    with pytest.raises(ValueError):
        annotator._format_predictions([mock_tensor])


# --- _generate_tags ---
@pytest.mark.standard
def test_generate_tags_str():
    annotator = DummyTransformersAnnotator("dummy-model")
    out = annotator._generate_tags("caption")
    assert out == ["caption"]


@pytest.mark.standard
def test_generate_tags_not_str():
    annotator = DummyTransformersAnnotator("dummy-model")
    out = annotator._generate_tags(["a", "b"])
    assert out == []


# --- 例外系 ---
@pytest.mark.standard
def test_preprocess_images_no_processor():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components.pop("processor", None)
    with pytest.raises(Exception):
        annotator._preprocess_images([Image.new("RGB", (32, 32))])


@pytest.mark.standard
def test_run_inference_no_model():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components["model"] = None
    mock_tensor = MagicMock()
    with pytest.raises(Exception):
        annotator._run_inference([{"input_ids": mock_tensor}])


@pytest.mark.standard
def test_format_predictions_no_processor():
    annotator = DummyTransformersAnnotator("dummy-model")
    if annotator.components is not None:
        annotator.components["processor"] = None
    mock_tensor = MagicMock()
    with pytest.raises(Exception):
        annotator._format_predictions([mock_tensor])
