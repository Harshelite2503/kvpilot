"""Load existing local Ollama text weights using MLX tensor operations only.

No Ollama API, model download, MLX-LM model/cache class, or weight re-quantization
is used. NVFP4 packed words and scale bytes remain in their original format.
The caller owns model execution and every inference-state/cache object.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
_TENSOR_MEDIA = "application/vnd.ollama.image.tensor"
# Ollama's stored NVFP4 global scale includes max(E4M3)*max(E2M1).
# https://github.com/ollama/ollama/blob/v0.40.11/mlx/ops_extra.go#L11
_NVFP4_MAX_PRODUCT = 448 * 6


def _mlx():
    import mlx.core as mx
    return mx


def _tokenizer_class():
    from tokenizers import Tokenizer
    return Tokenizer


class TensorStore(dict):
    """Ordinary named arrays plus per-weight format metadata; no inference state."""

    def __init__(self, *args, quantization: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantization = {} if quantization is None else quantization


def _manifest_path(model: str, models_dir: Path) -> Path:
    name, sep, tag = model.rpartition(":")
    if not sep:
        name, tag = model, "latest"
    parts = name.split("/")
    if not all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) for part in [*parts, tag]):
        raise ValueError("model must be a local name[:tag], namespace/name[:tag], or registry/namespace/name[:tag]")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", *parts]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    elif len(parts) != 3:
        raise ValueError("unsupported local model name")
    return models_dir / "manifests" / Path(*parts) / tag


def _blob_path(layer: dict, models_dir: Path) -> Path:
    digest = layer.get("digest", "")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("model layer must reference a SHA256 blob")
    path = models_dir / "blobs" / digest.replace(":", "-")
    if not isinstance(layer.get("size"), int) or layer["size"] != path.stat().st_size:
        raise ValueError(f"declared blob size does not match file: {layer.get('name', digest)}")
    return path


def read_tensor_header(path: str | Path) -> dict:
    """Read bounded safetensors metadata only, validating shape/range consistency."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 1 <= header_size <= min(1024 * 1024, size - 8):
            raise ValueError("invalid or oversized safetensors header")
        header = json.loads(source.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in metadata.items()):
        raise ValueError("safetensors metadata must contain strings")
    intervals = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict) or spec.get("dtype") not in _DTYPE_BYTES:
            raise ValueError(f"unsupported tensor dtype: {name}")
        shape, offsets = spec.get("shape"), spec.get("data_offsets")
        if (not isinstance(shape, list) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)):
            raise ValueError(f"invalid tensor shape/offsets: {name}")
        begin, end = offsets
        if (not 0 <= begin <= end <= size - 8 - header_size
                or end - begin != math.prod(shape) * _DTYPE_BYTES[spec["dtype"]]):
            raise ValueError(f"tensor offsets/shape exceed or disagree with blob: {name}")
        if end > begin:
            intervals.append((begin, end))
    intervals.sort()
    if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
        raise ValueError("overlapping tensor data offsets")
    return header


def _quantization_spec(weight_name: str, header: dict) -> dict | None:
    metadata = header.get("__metadata__", {})
    quant_type = metadata.get("quant_type")
    if quant_type is None:
        if any(spec.get("dtype") in {"U32", "U8"} for name, spec in header.items() if name != "__metadata__"):
            raise ValueError("packed tensors require explicit supported quantization metadata")
        return None
    if quant_type != "nvfp4" or metadata.get("group_size") != "16":
        raise ValueError("only NVFP4 with group_size=16 is supported")
    scale_name = weight_name + ".scale"
    weight, scale = header.get(weight_name, {}), header.get(scale_name, {})
    if (not weight_name.endswith(".weight") or weight.get("dtype") != "U32"
            or scale.get("dtype") != "U8" or len(weight.get("shape", [])) != 2
            or len(scale.get("shape", [])) != 2
            or weight["shape"][1] % 2
            or scale["shape"] != [weight["shape"][0], weight["shape"][1] // 2]):
        raise ValueError("NVFP4 packed weight and scale shapes/dtypes disagree")
    global_name = weight_name + ".global_scale"
    global_value = metadata.get("global_scale")
    if global_name in header:
        if global_value is not None:
            raise ValueError("ambiguous tensor and metadata global_scale")
        if header[global_name]["dtype"] != "F32" or math.prod(header[global_name]["shape"]) != 1:
            raise ValueError("NVFP4 global scale must be a scalar float32 tensor")
    elif global_value is not None:
        global_value = float(global_value)
        if not math.isfinite(global_value) or global_value <= 0:
            raise ValueError("NVFP4 global scale must be finite and positive")
    return {"mode": "nvfp4", "group_size": 16, "bits": 4, "scale_name": scale_name,
            "global_scale_name": global_name if global_name in header else None,
            "global_scale_value": global_value}


def load_local_model(model: str = "qwen3.5:4b", models_dir: str | Path = Path.home() / ".ollama/models"):
    """Return (full config JSON, TensorStore, Tokenizer, load metadata).

    Retain exact language_model.* names. BF16 arrays become FP16, with exact
    values wherever representable in FP16; no all-values range audit is claimed.
    A_log stays FP32. Norm weights are deliberately left unsanitized: the caller
    decides which architecture-specific norms need their additive one.
    """
    models_dir = Path(models_dir).expanduser().resolve()
    manifest_path = _manifest_path(model, models_dir)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "safetensors" or manifest.get("runner") != "mlx":
        raise ValueError("expected an existing MLX/safetensors Ollama manifest")
    layers = manifest["layers"]
    by_name = {}
    for layer in layers:
        if "name" in layer:
            if layer["name"] in by_name:
                raise ValueError(f"duplicate named layer: {layer['name']}")
            by_name[layer["name"]] = layer
    config_layer, tokenizer_layer = by_name["config.json"], by_name["tokenizer.json"]
    config_path = _blob_path(config_layer, models_dir)
    tokenizer_path = _blob_path(tokenizer_layer, models_dir)
    config = json.loads(config_path.read_text())
    selected = [layer for layer in layers if layer.get("mediaType") == _TENSOR_MEDIA
                and layer.get("name", "").startswith("language_model.")]
    if not selected:
        raise ValueError("no language_model tensor layers found")
    # Validate every selected header before asking the tensor engine to load data.
    described = []
    seen = set()
    for layer in selected:
        path = _blob_path(layer, models_dir)
        header = read_tensor_header(path)
        names = set(header) - {"__metadata__"}
        if layer["name"] not in names or any(not n.startswith("language_model.") for n in names):
            raise ValueError("text blob contains an unexpected tensor name")
        if seen & names:
            raise ValueError("duplicate tensor names across blobs")
        seen.update(names)
        described.append((layer, path, header, _quantization_spec(layer["name"], header)))
    mx = _mlx()
    tokenizer = _tokenizer_class().from_file(str(tokenizer_path))
    tensors = TensorStore()
    converted = []
    blob_records = []
    for layer, path, header, quantization in described:
        loaded, loaded_metadata = mx.load(str(path), format="safetensors", return_metadata=True)
        if set(loaded) != set(header) - {"__metadata__"} or loaded_metadata != header.get("__metadata__", {}):
            raise ValueError("tensor-engine load disagrees with inspected safetensors header")
        cast_arrays = []
        for name, value in loaded.items():
            if list(value.shape) != header[name]["shape"]:
                raise ValueError(f"loaded tensor shape mismatch: {name}")
            if name.endswith(".A_log"):
                value = value.astype(mx.float32)
            elif header[name]["dtype"] == "BF16":
                value = value.astype(mx.float16)
                converted.append(name)
                cast_arrays.append(value)
            tensors[name] = value
        # Evaluate conversions per blob so graphs do not retain a second entire
        # BF16 model until the first forward pass; packed weights are untouched.
        if cast_arrays:
            mx.eval(*cast_arrays)
        if quantization is not None:
            tensors.quantization[layer["name"]] = quantization
        blob_records.append({"name": layer["name"], "path": str(path), "digest": layer["digest"],
                             "bytes": layer["size"], "tensor_names": list(loaded),
                             "safetensors_metadata": header.get("__metadata__", {})})
    metadata = {
        "model": model, "models_dir": str(models_dir), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "config": {"path": str(config_path), "digest": config_layer["digest"], "bytes": config_layer["size"]},
        "tokenizer": {"path": str(tokenizer_path), "digest": tokenizer_layer["digest"], "bytes": tokenizer_layer["size"]},
        "text_blob_count": len(described), "tensor_count": len(tensors),
        "selected_blob_bytes": sum(layer["size"] for layer in selected),
        "nvfp4_matrix_count": len(tensors.quantization), "activation_dtype": "float16",
        "bf16_to_float16_tensor_names": converted,
        "casting": "BF16 to FP16 is exact where representable; underflow/overflow was not audited. A_log remains FP32.",
        "norm_sanitization_applied": False, "quantized_weights_converted": False,
        "vision_loaded": False, "external_cache_manager_used": False,
        "blob_hash_verification": "Manifest digests recorded; file sizes and tensor headers checked. Full weight payloads were not rehashed.",
        "blobs": blob_records,
    }
    return config, tensors, tokenizer, metadata


def linear(x, base_name: str, tensors: TensorStore):
    """Apply a local dense/NVFP4 projection; base_name normally omits .weight."""
    mx = _mlx()
    weight_name = base_name if base_name.endswith(".weight") else base_name + ".weight"
    base_name = weight_name[:-len(".weight")]
    weight = tensors[weight_name]
    spec = tensors.quantization.get(weight_name)
    if spec is not None:
        result = mx.quantized_matmul(x, weight, tensors[spec["scale_name"]], biases=None,
                                     transpose=True, group_size=spec["group_size"], bits=spec["bits"], mode=spec["mode"])
        global_scale = tensors[spec["global_scale_name"]] if spec["global_scale_name"] is not None else spec["global_scale_value"]
        if global_scale is not None:
            # Match Ollama's scaleAndCast, then add bias below. The stored global
            # scale is not the direct output multiplier; it includes 448*6.
            result = (result * (global_scale / _NVFP4_MAX_PRODUCT)).astype(result.dtype)
    else:
        if weight_name + ".scale" in tensors:
            raise ValueError("quantized projection is missing its validated format metadata")
        result = x @ weight.T
    if base_name + ".bias" in tensors:
        result = result + tensors[base_name + ".bias"]
    return result
