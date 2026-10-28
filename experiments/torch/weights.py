"""Read a local, unquantized Hugging Face Qwen2 checkpoint without model libraries.

No downloads, pickle loading, Transformers, or remote code execution. All source
files are SHA256 hashed, and optional caller-supplied pins must match exactly.
The tokenizer library handles file parsing and tokenization only.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict) -> dict:
    """Accept dense, full-attention Qwen2 with ordinary full-head RoPE only."""
    if not isinstance(config, dict) or config.get("model_type") != "qwen2":
        raise ValueError("only dense Qwen2/Qwen2.5 text checkpoints are supported")
    c = dict(config)
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "max_position_embeddings"):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f"invalid Qwen2 configuration field: {key}")
    if (c["hidden_size"] % c["num_attention_heads"] or
            c["num_attention_heads"] % c["num_key_value_heads"]):
        raise ValueError("incompatible hidden size or grouped attention heads")
    derived = c["hidden_size"] // c["num_attention_heads"]
    if c.get("head_dim", derived) != derived or derived % 2:
        raise ValueError("only even, full hidden-size attention head dimensions are supported")
    c["head_dim"] = derived
    if (c.get("hidden_act", "silu") != "silu" or c.get("use_sliding_window", False)
            or c.get("rope_scaling") is not None or c.get("quantization_config") is not None
            or c.get("num_experts", 0) or c.get("num_local_experts", 0)
            or c.get("partial_rotary_factor", 1.0) != 1.0):
        raise ValueError("unsupported activation, sliding window, quantization, MoE, or RoPE variant")
    for key, default in (("rms_norm_eps", 1e-6), ("rope_theta", 1_000_000.0)):
        value = c.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}")
        c[key] = float(value)
    if type(c.get("tie_word_embeddings", False)) is not bool:
        raise ValueError("tie_word_embeddings must be Boolean")
    c.setdefault("tie_word_embeddings", False)
    if c.get("attention_bias", True) is not True:
        raise ValueError("this Qwen2 runner requires Q/K/V projection biases")
    return c


def tensor_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    c = validate_config(config)
    h, intermediate, vocabulary = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    shapes = {"model.embed_tokens.weight": (vocabulary, h), "model.norm.weight": (h,)}
    if not c["tie_word_embeddings"]:
        shapes["lm_head.weight"] = (vocabulary, h)
    for layer in range(c["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (h,)
        shapes[prefix + "post_attention_layernorm.weight"] = (h,)
        for name, size in (("q_proj", h), ("k_proj", kv), ("v_proj", kv)):
            shapes[prefix + f"self_attn.{name}.weight"] = (size, h)
            shapes[prefix + f"self_attn.{name}.bias"] = (size,)
        shapes[prefix + "self_attn.o_proj.weight"] = (h, h)
        shapes[prefix + "mlp.gate_proj.weight"] = (intermediate, h)
        shapes[prefix + "mlp.up_proj.weight"] = (intermediate, h)
        shapes[prefix + "mlp.down_proj.weight"] = (h, intermediate)
    return shapes


def resolve_device(device) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be CPU, CUDA, or MPS")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device, torch.float32 if device.type == "cpu" else torch.float16


def validate_weight_keys(names, config):
    required = tensor_shapes(config)
    optional = {"lm_head.weight"} if config["tie_word_embeddings"] else set()
    if set(names) - set(required) - optional or set(required) - set(names):
        raise ValueError("checkpoint has missing or unsupported tensor names")
    if "lm_head.weight" in names:
        required["lm_head.weight"] = (config["vocab_size"], config["hidden_size"])
    return required


def load_local_model(model_dir, *, device="cpu", expected_hashes=None):
    """Return ``(config, tensors, tokenizer, provenance)`` from existing files.

    With no external pins, the computed hashes pin this load for the experiment;
    they are not a claim that the files match an authenticated upstream revision.
    Optional ``expected_hashes`` maps every selected relative filename to SHA256.
    """
    from safetensors import safe_open
    from tokenizers import Tokenizer

    root = Path(model_dir).expanduser().resolve()
    device, dtype = resolve_device(device)
    config_path, tokenizer_path = root / "config.json", root / "tokenizer.json"
    config_bytes = config_path.read_bytes()
    config = validate_config(json.loads(config_bytes))
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes() if index_path.is_file() else None
    index = json.loads(index_bytes) if index_bytes is not None else None
    if index is not None:
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(k, str) or not isinstance(v, str) or Path(v).name != v
            or not v.endswith(".safetensors") for k, v in mapping.items()
        ):
            raise ValueError("invalid safetensors shard index")
        weight_paths = [root / name for name in sorted(set(mapping.values()))]
    else:
        weight_paths = sorted(root.glob("*.safetensors"))
        if len(weight_paths) != 1:
            raise ValueError("need exactly one safetensors file or an explicit shard index")
    files = [config_path, tokenizer_path, *weight_paths]
    files += [root / name for name in ("tokenizer_config.json", "generation_config.json") if (root / name).is_file()]
    if index is not None:
        files.append(index_path)
    hashes = {path.name: file_sha256(path) for path in files}
    hashes[config_path.name] = sha256(config_bytes).hexdigest()
    if index_bytes is not None:
        hashes[index_path.name] = sha256(index_bytes).hexdigest()
    if expected_hashes is not None and expected_hashes != hashes:
        raise ValueError("local model files do not match the exact expected SHA256 pins")
    # Validate every name, shape and dtype before copying model weights to a device.
    locations, shapes = {}, {}
    for path in weight_paths:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                if name in locations or (index is not None and index["weight_map"].get(name) != path.name):
                    raise ValueError("duplicate tensor or inconsistent shard index")
                view = file.get_slice(name)
                if view.get_dtype() not in {"F16", "BF16", "F32"}:
                    raise ValueError("only ordinary FP16, BF16, or FP32 weights are supported")
                locations[name], shapes[name] = path, tuple(view.get_shape())
    if index is not None and set(locations) != set(index["weight_map"]):
        raise ValueError("shard index contains missing tensors")
    required = validate_weight_keys(locations, config)
    if shapes != required:
        raise ValueError("checkpoint tensor shapes disagree with the model configuration")
    tensors = {}
    for path in weight_paths:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                value = file.get_tensor(name).to(dtype=dtype)
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"weight is nonfinite or outside activation precision: {name}")
                tensors[name] = value.to(device=device).contiguous().detach()
    if config["tie_word_embeddings"] and "lm_head.weight" in tensors:
        if not torch.equal(tensors["lm_head.weight"], tensors["model.embed_tokens.weight"]):
            raise ValueError("tied language head disagrees with embedding weights")
        del tensors["lm_head.weight"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) > config["vocab_size"]:
        raise ValueError("tokenizer vocabulary exceeds the embedding table")
    if any(file_sha256(path) != hashes[path.name] for path in files):
        raise ValueError("model files changed during loading")
    provenance = {
        "source": "local_huggingface_safetensors", "model_dir": str(root),
        "file_sha256": hashes, "expected_hashes_verified": expected_hashes is not None,
        "model_digest": sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_file_bytes": {path.name: path.stat().st_size for path in files},
        "activation_dtype": str(dtype).removeprefix("torch."), "device": str(device),
        "torch_version": torch.__version__,
        "conversion": "all unquantized floating weights converted to execution dtype",
    }
    return config, tensors, tokenizer, provenance
