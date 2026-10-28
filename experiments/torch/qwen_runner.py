"""Single-sequence dense Qwen2.5 math with owned, explicit key/value tensors.

Torch supplies tensor arithmetic, not a model, cache manager, or generation loop.
The runner has no prefix reuse beyond a LiveState explicitly supplied by callers.
Only ordinary full-head RoPE/full attention, greedy decoding, and <=8192 tokens
are supported. FP32 attention scores/softmax favor correctness over kernel speed.
See THIRD_PARTY_NOTICES.md for consulted architecture references.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

from .weights import load_local_model, resolve_device, validate_config, validate_weight_keys


INITIAL_DIGEST = sha256(b"kvpilot-token-v1").hexdigest()
LAYOUT = "kvpilot_qwen25_single_sequence_bhtd_v1"


@dataclass
class LiveState:
    offset: int = 0
    tensors: dict = field(default_factory=dict)
    token_digest: str = INITIAL_DIGEST
    model_identity: dict | None = None

    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.tensors.values())

    def release(self):
        """Drop this state's references; do not flush the device allocator."""
        self.tensors.clear()
        self.offset, self.token_digest, self.model_identity = 0, INITIAL_DIGEST, None


def synchronize(device="cpu"):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class QwenRunner:
    def __init__(self, model_dir, *, device="cpu", expected_hashes=None):
        config, weights, tokenizer, provenance = load_local_model(
            model_dir, device=device, expected_hashes=expected_hashes)
        self._initialize(config, weights, tokenizer, provenance, device)

    @classmethod
    def from_components(cls, config, weights, tokenizer, *, device="cpu", provenance=None):
        """Build an owned tiny-fixture runner without file or network access."""
        device, dtype = resolve_device(device)
        config = validate_config(config)
        required = validate_weight_keys(weights, config)
        if any(not isinstance(weights[name], torch.Tensor) or tuple(weights[name].shape) != shape
               for name, shape in required.items()):
            raise ValueError("component tensor shapes disagree with configuration")
        owned = {name: value.detach().to(device=device, dtype=dtype).clone().contiguous()
                 for name, value in weights.items()}
        if any(not torch.isfinite(value).all().item() for value in owned.values()):
            raise ValueError("component tensors must be finite in execution precision")
        if config["tie_word_embeddings"] and "lm_head.weight" in owned:
            if not torch.equal(owned["lm_head.weight"], owned["model.embed_tokens.weight"]):
                raise ValueError("tied language head disagrees with embedding weights")
            del owned["lm_head.weight"]
        digest = sha256(json.dumps(config, sort_keys=True).encode())
        for name, value in sorted(owned.items()):
            digest.update(name.encode())
            digest.update(value.cpu().numpy().tobytes())
        provenance = dict(provenance or {})
        provenance.update({"source": "explicit_components", "model_digest": digest.hexdigest(),
                           "activation_dtype": str(dtype).removeprefix("torch."),
                           "torch_version": torch.__version__, "device": str(device)})
        obj = cls.__new__(cls)
        obj._initialize(config, owned, tokenizer, provenance, device)
        return obj

    def _initialize(self, config, weights, tokenizer, provenance, device):
        self.device, self.dtype = resolve_device(device)
        self.config = validate_config(config)
        self.weights, self.tokenizer, self.provenance = weights, tokenizer, provenance
        self.max_context = min(8192, self.config["max_position_embeddings"])
        self.identity = {"model_digest": provenance["model_digest"], "layout": LAYOUT,
                         "execution": "kvpilot_qwen25_torch_eager_v1", "device_type": self.device.type,
                         "activation_dtype": str(self.dtype).removeprefix("torch."),
                         "torch_version": str(torch.__version__),
                         "prompt_format": "explicit_user_only_qwen_chatml_v1"}
        eos = self.config.get("eos_token_id", [])
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        for token in ("<|im_end|>", "<|endoftext|>"):
            value = tokenizer.token_to_id(token)
            if value is not None:
                self.stop_ids.add(value)
        if any(type(value) is not int or not 0 <= value < self.config["vocab_size"] for value in self.stop_ids):
            raise ValueError("invalid configured stop token")
        self.synchronize()

    def synchronize(self):
        synchronize(self.device)

    def clear_cache(self):
        """Explicit allocator cleanup hook; never called inside prefill/decode."""
        self.synchronize()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _tokens(self, tokens):
        if not isinstance(tokens, (list, tuple)) or not tokens or any(
            type(value) is not int or not 0 <= value < self.config["vocab_size"] for value in tokens
        ):
            raise ValueError("tokens must be a nonempty sequence of vocabulary IDs")

    def _validate_state(self, state):
        if not isinstance(state, LiveState) or type(state.offset) is not int or not 0 <= state.offset <= self.max_context:
            raise ValueError("invalid live state or offset")
        if not isinstance(state.token_digest, str) or len(state.token_digest) != 64 or any(
            c not in "0123456789abcdef" for c in state.token_digest
        ):
            raise ValueError("invalid token-prefix digest")
        if not state.offset:
            if state.tensors or state.token_digest != INITIAL_DIGEST or state.model_identity not in (None, self.identity):
                raise ValueError("empty state has unexpected tensors or identity")
            return
        if state.model_identity != self.identity:
            raise ValueError("live state belongs to a different model or execution identity")
        expected = {f"{i}.{kind}" for i in range(self.config["num_hidden_layers"]) for kind in ("k", "v")}
        shape = (1, self.config["num_key_value_heads"], state.offset, self.config["head_dim"])
        if set(state.tensors) != expected or any(
            not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
            value.dtype != self.dtype or value.device != self.weights["model.embed_tokens.weight"].device
            for value in state.tensors.values()
        ):
            raise ValueError("incomplete or incompatible live KV tensors")

    def norm(self, x, name):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.config["rms_norm_eps"])
        return value.to(x.dtype) * self.weights[name + ".weight"]

    def linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"], self.weights.get(name + ".bias"))

    def rope(self, value, offset):
        dimension = self.config["head_dim"]
        positions = torch.arange(offset, offset + value.shape[2], dtype=torch.float32, device=self.device)
        inverse = self.config["rope_theta"] ** (-torch.arange(0, dimension, 2, dtype=torch.float32, device=self.device) / dimension)
        angle = positions[:, None] * inverse[None, :]
        angle = torch.cat((angle, angle), dim=-1)[None, None]
        left, right = value.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return value * angle.cos().to(value.dtype) + rotated * angle.sin().to(value.dtype)

    def attention(self, x, layer, state, pending):
        c, length = self.config, x.shape[1]
        heads, kv_heads, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        name = f"model.layers.{layer}.self_attn"
        q = self.linear(x, name + ".q_proj").reshape(1, length, heads, dim).transpose(1, 2)
        k = self.linear(x, name + ".k_proj").reshape(1, length, kv_heads, dim).transpose(1, 2)
        v = self.linear(x, name + ".v_proj").reshape(1, length, kv_heads, dim).transpose(1, 2)
        q, k = self.rope(q, state.offset), self.rope(k, state.offset)
        if state.offset:
            k = torch.cat((state.tensors[f"{layer}.k"], k), dim=2)
            v = torch.cat((state.tensors[f"{layer}.v"], v), dim=2)
        pending[f"{layer}.k"], pending[f"{layer}.v"] = k.contiguous(), v.contiguous()
        repeated_k = k.repeat_interleave(heads // kv_heads, dim=1)
        repeated_v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.matmul(q.float(), repeated_k.float().transpose(-1, -2)) * dim ** -0.5
        queries = torch.arange(state.offset, state.offset + length, device=self.device)
        keys = torch.arange(state.offset + length, device=self.device)
        scores.masked_fill_(keys[None, :] > queries[:, None], float("-inf"))
        probabilities = scores.softmax(dim=-1)
        result = torch.matmul(probabilities, repeated_v.float()).to(x.dtype)
        result = result.transpose(1, 2).contiguous().reshape(1, length, heads * dim)
        return self.linear(result, name + ".o_proj")

    @torch.inference_mode()
    def forward(self, tokens, state: LiveState):
        self._tokens(tokens)
        self._validate_state(state)
        if state.offset + len(tokens) > self.max_context:
            raise ValueError("request exceeds the bounded context window")
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        x = F.embedding(ids, self.weights["model.embed_tokens.weight"])
        pending = {}
        for layer in range(self.config["num_hidden_layers"]):
            name = f"model.layers.{layer}"
            x = x + self.attention(self.norm(x, name + ".input_layernorm"), layer, state, pending)
            h = self.norm(x, name + ".post_attention_layernorm")
            x = x + self.linear(F.silu(self.linear(h, name + ".mlp.gate_proj")) *
                                self.linear(h, name + ".mlp.up_proj"), name + ".mlp.down_proj")
        h = self.norm(x[:, -1, :], "model.norm")
        output_weight = self.weights["model.embed_tokens.weight"] if self.config["tie_word_embeddings"] else self.weights["lm_head.weight"]
        logits = F.linear(h, output_weight)[0].float()
        self.synchronize()
        if not torch.isfinite(logits).all().item():
            raise ValueError("nonfinite model logits; live state was not updated")
        digest = state.token_digest
        for token in tokens:
            digest = sha256(bytes.fromhex(digest) + struct.pack("<I", token)).hexdigest()
        state.tensors, state.offset = pending, state.offset + len(tokens)
        state.token_digest, state.model_identity = digest, deepcopy(self.identity)
        return logits

    def prefill(self, tokens, state=None, chunk_size=128):
        self._tokens(tokens)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        state = LiveState() if state is None else state
        self._validate_state(state)
        if state.offset + len(tokens) > self.max_context:
            raise ValueError("request exceeds the bounded context window")
        logits = None
        for start in range(0, len(tokens), chunk_size):
            logits = self.forward(tokens[start:start + chunk_size], state)
        return logits, state

    def generate(self, tokens, state=None, max_tokens=32, chunk_size=128):
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self._tokens(tokens)
        if (state.offset if state else 0) + len(tokens) + max_tokens > self.max_context:
            raise ValueError("prompt plus generation exceeds the bounded context window")
        self.synchronize()
        started = time.perf_counter()
        logits, state = self.prefill(tokens, state, chunk_size)
        emitted, first_ms, stopped = [], None, False
        for _ in range(max_tokens):
            token = int(logits.argmax().item())
            if first_ms is None:
                first_ms = (time.perf_counter() - started) * 1000
            if token in self.stop_ids:
                stopped = True
                break
            emitted.append(token)
            # Returned state includes every emitted non-stop token.
            logits = self.forward([token], state)
        return {"answer": self.tokenizer.decode(emitted), "tokens": emitted,
                "ttft_ms": first_ms, "wall_ms": (time.perf_counter() - started) * 1000,
                "truncated": not stopped and len(emitted) == max_tokens}, state

    def prompt_tokens(self, text):
        if not isinstance(text, str):
            raise ValueError("prompt must be text")
        rendered = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        return self.tokenizer.encode(rendered, add_special_tokens=False).ids

    def state_metadata(self, state):
        self._validate_state(state)
        return {"offset": state.offset, "model": deepcopy(self.identity), "prefix_hash": state.token_digest,
                "activation_dtype": str(self.dtype).removeprefix("torch."), "layout": LAYOUT}

    def state_specs(self, state):
        """Metadata-only sizing: no export, device copy, synchronization or scan."""
        metadata = self.state_metadata(state)
        dtype = np.dtype("float32" if self.dtype == torch.float32 else "float16")
        specs = {name: (tuple(value.shape), dtype) for name, value in state.tensors.items()}
        return specs, {name: "attention" for name in specs}, metadata

    def expected_state_specs(self, offset, prefix_hash="0" * 64):
        """Exact prospective framing from config alone, without creating a state.

        The default hash is a length-matched sizing placeholder, never evidence
        of a real prefix. No live tensors are created, scanned or exported.
        """
        if type(offset) is not int or not 1 <= offset <= self.max_context:
            raise ValueError("invalid prospective state offset")
        if not isinstance(prefix_hash, str) or len(prefix_hash) != 64 or any(
            c not in "0123456789abcdef" for c in prefix_hash
        ):
            raise ValueError("invalid prospective prefix hash")
        shape = (1, self.config["num_key_value_heads"], offset, self.config["head_dim"])
        dtype = np.dtype("float32" if self.dtype == torch.float32 else "float16")
        specs = {f"{i}.{kind}": (shape, dtype) for i in range(self.config["num_hidden_layers"])
                 for kind in ("k", "v")}
        metadata = {"offset": offset, "model": deepcopy(self.identity), "prefix_hash": prefix_hash,
                    "activation_dtype": str(self.dtype).removeprefix("torch."), "layout": LAYOUT}
        return specs, {name: "attention" for name in specs}, metadata

    def export_state(self, state):
        metadata = self.state_metadata(state)
        self.synchronize()
        tensors = {name: value.detach().cpu().numpy().copy() for name, value in state.tensors.items()}
        return tensors, {name: "attention" for name in tensors}, metadata

    def import_state(self, tensors, roles, metadata, *, expected_prefix_hash=None):
        if not isinstance(metadata, dict) or metadata.get("model") != self.identity or metadata.get("layout") != LAYOUT:
            raise ValueError("incompatible model or state layout")
        if metadata.get("activation_dtype") != str(self.dtype).removeprefix("torch."):
            raise ValueError("incompatible declared state precision")
        digest, offset = metadata.get("prefix_hash"), metadata.get("offset")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid token-prefix identity")
        if expected_prefix_hash is not None and digest != expected_prefix_hash:
            raise ValueError("restored state belongs to another token prefix")
        if type(offset) is not int or not 1 <= offset <= self.max_context:
            raise ValueError("invalid restored offset")
        names = {f"{i}.{kind}" for i in range(self.config["num_hidden_layers"]) for kind in ("k", "v")}
        if not isinstance(tensors, dict) or not isinstance(roles, dict) or set(tensors) != names or set(roles) != names:
            raise ValueError("incomplete restored state")
        shape = (1, self.config["num_key_value_heads"], offset, self.config["head_dim"])
        dtype = np.dtype("float32" if self.dtype == torch.float32 else "float16")
        for name, value in tensors.items():
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or roles[name] != "attention" or not np.isfinite(value).all():
                raise ValueError(f"invalid restored tensor: {name}")
        restored = {name: torch.from_numpy(value.copy(order="C")).to(self.device)
                    for name, value in tensors.items()}
        self.synchronize()
        return LiveState(offset, restored, digest, deepcopy(self.identity))
