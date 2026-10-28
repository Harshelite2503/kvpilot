"""Minimal single-sequence Qwen3.5 text forward pass with KVPilot-owned state.

Only MLX tensor math and the local tokenizer/weight loader are dependencies.
No Ollama service, MLX-LM model/cache/generation code or prefix cache is invoked.
Architecture follows Qwen and Apple's MIT mathematical reference; see notices.
This research runner uses FP16 activations and FP32 recurrent state, greedy decode,
and no vision, padding, batching, speculative decoding, or automatic reuse.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import gc
from hashlib import sha256
import time
import struct
import mlx.core as mx
import numpy as np
from .weights import load_local_model, linear
from .gated_delta import gated_delta


@dataclass
class LiveState:
    offset: int = 0
    tensors: dict = field(default_factory=dict)
    token_digest: str = sha256(b"kvpilot-token-v1").hexdigest()

    def nbytes(self):
        return sum(x.nbytes for x in self.tensors.values())

    def release(self):
        self.tensors.clear()
        self.offset = 0
        self.token_digest = sha256(b"kvpilot-token-v1").hexdigest()
        gc.collect()
        mx.clear_cache()


def synchronize(*arrays):
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


class QwenRunner:
    def __init__(self, model="qwen3.5:4b", models_dir=None):
        kwargs = {} if models_dir is None else {"models_dir": models_dir}
        config, self.weights, self.tokenizer, self.provenance = load_local_model(model, **kwargs)
        self.config = config["text_config"]
        self.identity = {"manifest_sha256": self.provenance["manifest_sha256"],
                         "config_digest": self.provenance["config"]["digest"],
                         "tokenizer_digest": self.provenance["tokenizer"]["digest"],
                         "execution": "kvpilot_qwen35_fp16_v1"}
        c = self.config
        expected = {"num_hidden_layers": 32, "num_attention_heads": 16,
                    "num_key_value_heads": 4, "head_dim": 256,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                    "tie_word_embeddings": True}
        if c.get("num_experts", 0) or any(c.get(k) != v for k, v in expected.items()):
            raise ValueError("this prototype supports only the audited dense Qwen3.5:4b configuration")
        self.prefix = "language_model.model."
        self.eps = c["rms_norm_eps"]
        self.stop_ids = {c["eos_token_id"], self.tokenizer.token_to_id("<|im_end|>")}
        # Hugging Face Qwen stores RMSNorm offsets and [channels,1,kernel] convs.
        # Ollama's per-tensor export retains this unsanitized checkpoint layout.
        conv_name = self.prefix + "layers.0.linear_attn.conv1d.weight"
        self.unsanitized = self.weights[conv_name].shape[-1] != 1
        suffixes = ("input_layernorm.weight", "post_attention_layernorm.weight",
                    "model.norm.weight", "q_norm.weight", "k_norm.weight")
        for name, value in list(self.weights.items()):
            if self.unsanitized and any(name.endswith(s) for s in suffixes):
                self.weights[name] = (value.astype(mx.float32) + 1).astype(mx.float16)
            if "conv1d.weight" in name and value.shape[-1] != 1:
                self.weights[name] = mx.swapaxes(value, 1, 2)
        synchronize(*self.weights.values())
        mx.clear_cache()

    def norm(self, x, name):
        return mx.fast.rms_norm(x, self.weights[name + ".weight"], self.eps)

    def attention(self, x, layer, state):
        c = self.config
        name = self.prefix + f"layers.{layer}.self_attn"
        b, length, _ = x.shape
        h, hk, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q, gate = mx.split(linear(x, name + ".q_proj", self.weights).reshape(b, length, h, 2*d), 2, axis=-1)
        k = linear(x, name + ".k_proj", self.weights).reshape(b, length, hk, d)
        v = linear(x, name + ".v_proj", self.weights).reshape(b, length, hk, d).transpose(0, 2, 1, 3)
        q = self.norm(q, name + ".q_norm").transpose(0, 2, 1, 3)
        k = self.norm(k, name + ".k_norm").transpose(0, 2, 1, 3)
        rope = c["rope_parameters"]
        kw = dict(dims=int(d * rope["partial_rotary_factor"]), traditional=False,
                  base=float(rope["rope_theta"]), scale=1.0, offset=state.offset)
        q, k = mx.fast.rope(q, **kw), mx.fast.rope(k, **kw)
        key, value = f"{layer}.k", f"{layer}.v"
        if key in state.tensors:
            k = mx.concatenate([state.tensors[key], k], axis=2)
            v = mx.concatenate([state.tensors[value], v], axis=2)
        state.tensors[key], state.tensors[value] = mx.contiguous(k), mx.contiguous(v)
        # Explicit positions make resumed/chunked causal masking unambiguous.
        mask = None if length == 1 else (
            mx.arange(state.offset + length)[None, :] <=
            (state.offset + mx.arange(length))[:, None])
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, h*d)
        return linear(out * mx.sigmoid(gate.reshape(b, length, h*d)), name + ".o_proj", self.weights)

    def recurrent(self, x, layer, state):
        c = self.config
        name = self.prefix + f"layers.{layer}.linear_attn"
        b, length, _ = x.shape
        hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        kd, vd = hk*dk, hv*dv
        qkv = linear(x, name + ".in_proj_qkv", self.weights)
        z = linear(x, name + ".in_proj_z", self.weights).reshape(b, length, hv, dv)
        a = linear(x, name + ".in_proj_a", self.weights)
        beta = mx.sigmoid(linear(x, name + ".in_proj_b", self.weights))
        conv_key, rec_key = f"{layer}.conv", f"{layer}.recurrent"
        conv = state.tensors.get(conv_key)
        keep = c["linear_conv_kernel_dim"] - 1
        if conv is None:
            conv = mx.zeros((b, keep, 2*kd+vd), dtype=x.dtype)
        conv_input = mx.concatenate([conv, qkv], axis=1)
        state.tensors[conv_key] = mx.contiguous(conv_input[:, -keep:, :])
        qkv = mx.conv1d(conv_input, self.weights[name + ".conv1d.weight"], groups=2*kd+vd)
        qkv = qkv * mx.sigmoid(qkv)
        q, k, v = mx.split(qkv, [kd, 2*kd], axis=-1)
        q, k, v = q.reshape(b, length, hk, dk), k.reshape(b, length, hk, dk), v.reshape(b, length, hv, dv)
        inv = dk**-0.5
        q = inv**2 * mx.fast.rms_norm(q, None, 1e-6 * inv**2)
        k = inv * mx.fast.rms_norm(k, None, 1e-6 * inv**2)
        dt = (a + self.weights[name + ".dt_bias"]).astype(mx.float32)
        softplus = mx.maximum(dt, 0) + mx.log1p(mx.exp(-mx.abs(dt)))
        decay = mx.exp(-mx.exp(self.weights[name + ".A_log"].astype(mx.float32)) * softplus)
        recurrent_state = state.tensors.get(rec_key)
        if recurrent_state is None:
            recurrent_state = mx.zeros((b, hv, dv, dk), dtype=mx.float32)
        out, recurrent_state = gated_delta(q, k, v, decay, beta, recurrent_state)
        state.tensors[rec_key] = recurrent_state
        normed = mx.fast.rms_norm(out.astype(mx.float32), self.weights[name + ".norm.weight"].astype(mx.float32), self.eps)
        z32 = z.astype(mx.float32)
        out = (normed * z32 * mx.sigmoid(z32)).astype(x.dtype).reshape(b, length, vd)
        return linear(out, name + ".out_proj", self.weights)

    def forward(self, tokens, state: LiveState):
        if not tokens:
            raise ValueError("forward requires at least one new token")
        if state.offset and not state.tensors:
            raise ValueError("nonzero offset cannot have missing state")
        x = self.weights[self.prefix + "embed_tokens.weight"][mx.array([tokens], dtype=mx.int32)]
        for i in range(self.config["num_hidden_layers"]):
            name = self.prefix + f"layers.{i}"
            normed = self.norm(x, name + ".input_layernorm")
            attention = self.attention(normed, i, state) if (i+1) % 4 == 0 else self.recurrent(normed, i, state)
            x = x + attention
            h = self.norm(x, name + ".post_attention_layernorm")
            gate = linear(h, name + ".mlp.gate_proj", self.weights)
            up = linear(h, name + ".mlp.up_proj", self.weights)
            x = x + linear((gate * mx.sigmoid(gate)) * up, name + ".mlp.down_proj", self.weights)
        x = self.norm(x[:, -1:, :], self.prefix + "norm")
        logits = (x @ self.weights[self.prefix + "embed_tokens.weight"].T)[0, 0].astype(mx.float32)
        synchronize(logits, *state.tensors.values())
        state.offset += len(tokens)
        for token in tokens:
            state.token_digest = sha256(bytes.fromhex(state.token_digest) + struct.pack("<I", token)).hexdigest()
        return logits

    def prefill(self, tokens, state=None, chunk_size=128):
        state = LiveState() if state is None else state
        logits = None
        for start in range(0, len(tokens), chunk_size):
            logits = self.forward(tokens[start:start+chunk_size], state)
            mx.clear_cache()
        return logits, state

    def generate(self, tokens, state=None, max_tokens=32, chunk_size=128):
        started = time.perf_counter()
        logits, state = self.prefill(tokens, state, chunk_size)
        emitted = []
        first_ms = None
        for _ in range(max_tokens):
            token = int(mx.argmax(logits).item())
            if first_ms is None:
                first_ms = (time.perf_counter()-started)*1000
            if token in self.stop_ids:
                break
            emitted.append(token)
            logits = self.forward([token], state)
        return {"answer": self.tokenizer.decode(emitted), "tokens": emitted,
                "ttft_ms": first_ms, "wall_ms": (time.perf_counter()-started)*1000,
                "truncated": len(emitted) == max_tokens}, state

    def prompt_tokens(self, text):
        rendered = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.tokenizer.encode(rendered, add_special_tokens=False).ids

    def export_state(self, state: LiveState):
        synchronize(*state.tensors.values())
        tensors = {key: np.array(value) for key, value in state.tensors.items()}
        roles = {key: "attention" if key.endswith((".k", ".v")) else "recurrent" for key in tensors}
        return tensors, roles, self.state_metadata(state)

    def state_metadata(self, state: LiveState):
        return {"offset": state.offset, "model": self.identity, "prefix_hash": state.token_digest,
                    "activation_dtype": "float16", "recurrent_dtype": "float32",
                    "layout": "kvpilot_qwen35_single_sequence_v1"}

    def import_state(self, tensors, roles, metadata, *, expected_prefix_hash=None):
        if metadata.get("model") != self.identity or metadata.get("layout") != "kvpilot_qwen35_single_sequence_v1":
            raise ValueError("incompatible model or state layout")
        if metadata.get("activation_dtype") != "float16" or metadata.get("recurrent_dtype") != "float32":
            raise ValueError("incompatible declared state precision")
        prefix_hash = metadata.get("prefix_hash")
        if not isinstance(prefix_hash, str) or len(prefix_hash) != 64 or any(c not in "0123456789abcdef" for c in prefix_hash):
            raise ValueError("invalid prefix identity")
        if expected_prefix_hash is not None and prefix_hash != expected_prefix_hash:
            raise ValueError("restored cache belongs to a different token prefix")
        offset = metadata.get("offset")
        if type(offset) is not int or offset < 1:
            raise ValueError("invalid restored offset")
        expected = {f"{i}.{kind}" for i in range(32) for kind in (("k", "v") if (i+1)%4 == 0 else ("conv", "recurrent"))}
        if set(tensors) != expected or set(roles) != expected:
            raise ValueError("incomplete restored state")
        restored = {}
        c = self.config
        for key, value in tensors.items():
            kind = key.split(".")[1]
            shape = {"k": (1,4,offset,256), "v": (1,4,offset,256),
                     "conv": (1,3,8192), "recurrent": (1,32,128,128)}[kind]
            dtype = np.dtype("float32") if kind == "recurrent" else np.dtype("float16")
            role = "attention" if kind in ("k", "v") else "recurrent"
            if value.shape != shape or value.dtype != dtype or roles[key] != role:
                raise ValueError(f"invalid restored tensor {key}")
            restored[key] = mx.array(value)
        synchronize(*restored.values())
        return LiveState(offset, restored, prefix_hash)
