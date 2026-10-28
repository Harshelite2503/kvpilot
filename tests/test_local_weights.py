"""Direct-loader integrity tests with tiny blobs and a fake tensor engine."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("local_weights", Path(__file__).resolve().parents[1] / "experiments/local/weights.py")
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)

PREFIX = "language_model.model."


def blob(models_dir, name, content, media="application/vnd.ollama.image.tensor"):
    digest = sha256(content).hexdigest()
    path = models_dir / "blobs" / ("sha256-" + digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"name": name, "digest": "sha256:" + digest, "size": len(content), "mediaType": media}


def tensor_blob(models_dir, name, specs, metadata=None):
    header, offset = {}, 0
    for tensor_name, dtype, shape in specs:
        length = int(np.prod(shape)) * weights._DTYPE_BYTES[dtype]
        header[tensor_name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    return blob(models_dir, name, struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def model_fixture(tmp_path):
    base = tmp_path / "models"
    embed = PREFIX + "embed_tokens.weight"
    projection = PREFIX + "layers.0.mlp.up_proj.weight"
    norm = PREFIX + "layers.0.input_layernorm.weight"
    alog = PREFIX + "layers.0.linear_attn.A_log"
    layers = [
        tensor_blob(base, embed, [(embed, "BF16", [2, 16])]),
        tensor_blob(base, projection, [(projection, "U32", [2, 2]), (projection + ".scale", "U8", [2, 1])],
                    {"quant_type": "nvfp4", "group_size": "16"}),
        tensor_blob(base, norm, [(norm, "BF16", [16])]),
        tensor_blob(base, alog, [(alog, "F32", [2])]),
        # An unreadable vision payload proves the loader never examines/loads it.
        blob(base, "vision_tower.invalid.weight", b"not a safetensors file"),
        blob(base, "config.json", json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 16}}).encode(),
             "application/vnd.ollama.image.json"),
        blob(base, "tokenizer.json", b'{"fixture":true}', "application/vnd.ollama.image.json"),
    ]
    manifest = base / "manifests/registry.ollama.ai/library/qwen3.5/4b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"runner": "mlx", "format": "safetensors", "layers": layers}))
    return base, manifest, layers


class FakeMLX:
    float16 = np.float16
    float32 = np.float32

    def __init__(self):
        self.loads = []
        self.evaluations = []
        self.quantized_calls = []

    def load(self, path, *, format, return_metadata):
        assert format == "safetensors" and return_metadata is True
        self.loads.append(path)
        header = weights.read_tensor_header(path)
        dtypes = {"BF16": np.float32, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "U8": np.uint8}
        arrays = {name: np.full(spec["shape"], 0.5 if spec["dtype"] == "BF16" else 1, dtype=dtypes[spec["dtype"]])
                  for name, spec in header.items() if name != "__metadata__"}
        return arrays, header.get("__metadata__", {})

    def eval(self, *arrays):
        self.evaluations.append(arrays)

    def quantized_matmul(self, x, weight, scales, **kwargs):
        self.quantized_calls.append((x, weight, scales, kwargs))
        return np.full((*x.shape[:-1], weight.shape[0]), 3, dtype=np.float16)


class FakeTokenizer:
    @classmethod
    def from_file(cls, path):
        assert json.loads(Path(path).read_text()) == {"fixture": True}
        return {"tokenizer_path": path}


def test_loader_preserves_packed_weights_skips_vision_and_records_casts(tmp_path, monkeypatch):
    base, manifest, layers = model_fixture(tmp_path)
    engine = FakeMLX()
    monkeypatch.setattr(weights, "_mlx", lambda: engine)
    monkeypatch.setattr(weights, "_tokenizer_class", lambda: FakeTokenizer)
    config, tensors, tokenizer, metadata = weights.load_local_model(models_dir=base)
    assert config["text_config"]["hidden_size"] == 16
    assert len(engine.loads) == 4 and len(tensors) == 5
    assert tensors[PREFIX + "layers.0.mlp.up_proj.weight"].dtype == np.uint32
    assert tensors[PREFIX + "layers.0.mlp.up_proj.weight.scale"].dtype == np.uint8
    assert tensors[PREFIX + "embed_tokens.weight"].dtype == np.float16
    assert tensors[PREFIX + "layers.0.linear_attn.A_log"].dtype == np.float32
    assert np.all(tensors[PREFIX + "layers.0.input_layernorm.weight"] == 0.5)  # No implicit +1.
    assert len(engine.evaluations) == 2
    assert metadata["activation_dtype"] == "float16" and metadata["nvfp4_matrix_count"] == 1
    assert metadata["vision_loaded"] is False and metadata["norm_sanitization_applied"] is False
    assert metadata["quantized_weights_converted"] is False
    assert metadata["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    assert metadata["selected_blob_bytes"] == sum(layer["size"] for layer in layers[:4])
    assert tokenizer["tokenizer_path"].startswith(str(base))


def test_nvfp4_projection_uses_exact_dispatch_and_scales_output_before_bias(monkeypatch):
    engine = FakeMLX()
    monkeypatch.setattr(weights, "_mlx", lambda: engine)
    name = PREFIX + "projection.weight"
    tensors = weights.TensorStore({name: np.ones((2, 2), dtype=np.uint32),
                                   name + ".scale": np.ones((2, 1), dtype=np.uint8),
                                   name + ".global_scale": np.array(2 * 2688, dtype=np.float32),
                                   PREFIX + "projection.bias": np.array([1, 2], dtype=np.float16)},
                                  quantization={name: {"mode": "nvfp4", "group_size": 16, "bits": 4,
                                                      "scale_name": name + ".scale",
                                                      "global_scale_name": name + ".global_scale",
                                                      "global_scale_value": None}})
    x = np.ones((1, 16), dtype=np.float16)
    output = weights.linear(x, PREFIX + "projection", tensors)
    np.testing.assert_array_equal(output, [[7, 8]])
    _, packed, scales, kwargs = engine.quantized_calls[0]
    assert packed is tensors[name] and scales is tensors[name + ".scale"]
    assert kwargs == {"biases": None, "transpose": True, "group_size": 16, "bits": 4, "mode": "nvfp4"}
    assert output.dtype == np.float16


def test_dense_projection_and_missing_quantization_metadata(monkeypatch):
    monkeypatch.setattr(weights, "_mlx", lambda: FakeMLX())
    name = PREFIX + "projection.weight"
    tensors = weights.TensorStore({name: np.array([[2, 3], [4, 5]], dtype=np.float16)})
    np.testing.assert_array_equal(weights.linear(np.array([[1, 2]], dtype=np.float16), name, tensors), [[8, 14]])
    tensors[name + ".scale"] = np.ones((2, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="format metadata"):
        weights.linear(np.ones((1, 2)), name, tensors)


def test_bad_size_or_unsupported_quantization_rejected_before_engine_load(tmp_path, monkeypatch):
    base, manifest, _ = model_fixture(tmp_path)
    monkeypatch.setattr(weights, "_mlx", lambda: pytest.fail("engine must not be loaded"))
    data = json.loads(manifest.read_text())
    data["layers"][0]["size"] += 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="blob size"):
        weights.load_local_model(models_dir=base)
    name = PREFIX + "matrix.weight"
    header = {name: {"dtype": "U32", "shape": [2, 2]}, name + ".scale": {"dtype": "U8", "shape": [2, 1]},
              "__metadata__": {"quant_type": "affine", "group_size": "16"}}
    with pytest.raises(ValueError, match="only NVFP4"):
        weights._quantization_spec(name, header)


def test_tensor_header_rejects_out_of_bounds_and_overlapping_ranges(tmp_path):
    path = tmp_path / "bad"
    header = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(4))
    with pytest.raises(ValueError, match="offsets/shape"):
        weights.read_tensor_header(path)
    header["b"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))
    with pytest.raises(ValueError, match="overlapping"):
        weights.read_tensor_header(path)


def test_nvfp4_shape_and_global_scale_validation():
    name = PREFIX + "matrix.weight"
    header = {name: {"dtype": "U32", "shape": [2, 2]}, name + ".scale": {"dtype": "U8", "shape": [2, 1]},
              "__metadata__": {"quant_type": "nvfp4", "group_size": "16", "global_scale": "2.5"}}
    assert weights._quantization_spec(name, header)["global_scale_value"] == 2.5
    header["__metadata__"]["global_scale"] = "nan"
    with pytest.raises(ValueError, match="finite and positive"):
        weights._quantization_spec(name, header)
    del header["__metadata__"]["global_scale"]
    header[name + ".scale"]["shape"] = [2, 2]
    with pytest.raises(ValueError, match="shapes/dtypes"):
        weights._quantization_spec(name, header)


def test_local_manifest_resolution_does_not_accept_traversal(tmp_path):
    assert weights._manifest_path("qwen3.5:4b", tmp_path) == tmp_path / "manifests/registry.ollama.ai/library/qwen3.5/4b"
    assert weights._manifest_path("library/qwen3.5:4b", tmp_path) == weights._manifest_path("qwen3.5:4b", tmp_path)
    with pytest.raises(ValueError, match="local name"):
        weights._manifest_path("../qwen3.5:4b", tmp_path)
