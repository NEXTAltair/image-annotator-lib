"""テスト用の小さな ONNX モデルを `onnx` パッケージ無しで組み立てる。

`onnx` は依存に無いため、ONNX の protobuf (ModelProto) をバイト列で直接組み立てる。
フィールド番号は onnx.proto (https://github.com/onnx/onnx/blob/main/onnx/onnx.proto) による。

モデルは ``入力 [B, 3, H, W] -> Flatten -> MatMul(重み) -> Sigmoid -> 出力 [B, K]`` で、
行ごとの出力が同じ行の入力だけに依存する。入力の batch 軸 (B) の宣言を切り替えられる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FLOAT = 1


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _field_varint(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _field_bytes(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _field_str(number: int, text: str) -> bytes:
    return _field_bytes(number, text.encode("utf-8"))


def _dimension(dim: int | str | None) -> bytes:
    """TensorShapeProto.Dimension。None は dim_value も dim_param も持たない (shape に None が出る)。"""
    if dim is None:
        return b""
    if isinstance(dim, str):
        return _field_str(2, dim)
    return _field_varint(1, dim)


def _value_info(name: str, shape: list[int | str | None]) -> bytes:
    shape_proto = b"".join(_field_bytes(1, _dimension(dim)) for dim in shape)
    tensor_type = _field_varint(1, _FLOAT) + _field_bytes(2, shape_proto)
    type_proto = _field_bytes(1, tensor_type)
    return _field_str(1, name) + _field_bytes(2, type_proto)


def _node(
    op_type: str, inputs: list[str], outputs: list[str], attributes: list[bytes] | None = None
) -> bytes:
    body = b"".join(_field_str(1, name) for name in inputs)
    body += b"".join(_field_str(2, name) for name in outputs)
    body += _field_str(3, f"{op_type}_{outputs[0]}") + _field_str(4, op_type)
    body += b"".join(_field_bytes(5, attribute) for attribute in attributes or [])
    return body


def _int_attribute(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return _field_str(1, name) + _field_varint(3, value) + _field_varint(20, 2)


def _initializer(name: str, array: np.ndarray) -> bytes:
    data = np.ascontiguousarray(array, dtype=np.float32)
    dims = b"".join(_field_varint(1, int(dim)) for dim in data.shape)
    return dims + _field_varint(2, _FLOAT) + _field_str(8, name) + _field_bytes(9, data.tobytes())


def tagger_weights(input_elements: int, tag_count: int) -> np.ndarray:
    """決定的な重み。画像ごとに出力が変わるよう、乱数でなく固定の系列から作る。"""
    rng = np.random.default_rng(1234)
    return (rng.standard_normal((input_elements, tag_count)) * 0.05).astype(np.float32)


def build_tagger_onnx(
    path: Path,
    *,
    batch_dim: int | str | None,
    height: int = 8,
    width: int = 8,
    tag_count: int = 6,
    collapse_batch: bool = False,
) -> Path:
    """小さな NCHW タガー形式の ONNX モデルを ``path`` に書き出す。

    Args:
        path: 出力先のファイルパス。
        batch_dim: 入力の batch 軸の宣言。int は静的、str は動的 (dim_param)、None は未記載。
        height: 入力の高さ。
        width: 入力の幅。
        tag_count: 出力のタグ数 K。
        collapse_batch: True なら出力を batch 軸で平均して [K] にする
            (入力は動的でも、出力が画像ごとに分割できないモデルを作る)。

    Returns:
        書き出したファイルパス。
    """
    elements = 3 * height * width
    weights = tagger_weights(elements, tag_count)
    nodes = [
        _node("Flatten", ["input"], ["flat"], [_int_attribute("axis", 1)]),
        _node("MatMul", ["flat", "weights"], ["logits"]),
        _node("Sigmoid", ["logits"], ["probs"]),
    ]
    output_name = "probs"
    output_shape: list[int | str | None] = [batch_dim, tag_count]
    if collapse_batch:
        # ReduceMean(axes=[0], keepdims=0): opset 13 では axes は属性。
        axes = _field_str(1, "axes") + _field_varint(8, 0) + _field_varint(20, 7)
        keepdims = _int_attribute("keepdims", 0)
        nodes.append(_node("ReduceMean", ["probs"], ["collapsed"], [axes, keepdims]))
        output_name = "collapsed"
        output_shape = [tag_count]
    graph = b"".join(_field_bytes(1, node) for node in nodes)
    graph += _field_str(2, "tagger")
    graph += _field_bytes(5, _initializer("weights", weights))
    graph += _field_bytes(11, _value_info("input", [batch_dim, 3, height, width]))
    graph += _field_bytes(12, _value_info(output_name, output_shape))
    opset = _field_str(1, "") + _field_varint(2, 13)
    model = (
        _field_varint(1, 8) + _field_str(2, "lib164-test") + _field_bytes(7, graph) + _field_bytes(8, opset)
    )
    path.write_bytes(model)
    return path


def reference_probabilities(batch: np.ndarray, tag_count: int = 6) -> np.ndarray:
    """モデルと同じ計算を numpy で行った参照値 (動作確認用)。"""
    flat = batch.reshape(batch.shape[0], -1).astype(np.float32)
    logits = flat @ tagger_weights(flat.shape[1], tag_count)
    return 1.0 / (1.0 + np.exp(-logits))
