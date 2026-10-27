"""Tiny tensor checks and allocation-free restore-schema checks; no model load."""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="Optional local MLX tensor engine")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Local Qwen arithmetic requires Apple Metal"
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runner_module = importlib.import_module("experiments.local.qwen_runner")
weights_module = importlib.import_module("experiments.local.weights")
QwenRunner, LiveState = runner_module.QwenRunner, runner_module.LiveState
PREFIX = "language_model.model."


def _runner():
    runner = QwenRunner.__new__(QwenRunner)
    runner.config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "tie_word_embeddings": True,
    }
    runner.provenance = {"model": "fixture", "manifest_sha256": "fixture-digest",
                         "config": {"digest": "fixture-config"},
                         "tokenizer": {"digest": "fixture-tokenizer"}}
    runner.identity = {"manifest_sha256": "fixture-digest", "config_digest": "fixture-config",
                       "tokenizer_digest": "fixture-tokenizer", "execution": "kvpilot_qwen35_fp16_v1"}
    runner.prefix = PREFIX
    runner.eps = 1e-6
    return runner


def _snapshot(runner, offset=3):
    """Broadcast scalar arrays carry native shapes without allocating 50 MB."""
    tensors, roles = {}, {}
    shapes = {
        "k": (1, 4, offset, 256),
        "v": (1, 4, offset, 256),
        "conv": (1, 3, 8192),
        "recurrent": (1, 32, 128, 128),
    }
    for layer in range(32):
        kinds = ("k", "v") if (layer + 1) % 4 == 0 else ("conv", "recurrent")
        for kind in kinds:
            key = f"{layer}.{kind}"
            dtype = np.float32 if kind == "recurrent" else np.float16
            tensors[key] = np.broadcast_to(np.array(0.25, dtype=dtype), shapes[kind])
            roles[key] = "attention" if kind in ("k", "v") else "recurrent"
    metadata = {
        "offset": offset,
        "model": deepcopy(runner.identity),
        "prefix_hash": "a" * 64,
        "activation_dtype": "float16",
        "recurrent_dtype": "float32",
        "layout": "kvpilot_qwen35_single_sequence_v1",
    }
    return tensors, roles, metadata


@pytest.fixture
def no_device_restore(monkeypatch):
    """Exercise schema checks without materializing native-size MLX arrays."""
    converted = []

    def array(value):
        converted.append(value)
        return value

    monkeypatch.setattr(runner_module, "mx", SimpleNamespace(array=array))
    monkeypatch.setattr(runner_module, "synchronize", lambda *arrays: None)
    return converted


def test_restore_accepts_complete_state_and_preserves_offset(no_device_restore):
    runner = _runner()
    tensors, roles, metadata = _snapshot(runner, offset=11)
    restored = runner.import_state(tensors, roles, metadata)
    assert restored.offset == 11
    assert set(restored.tensors) == set(tensors)
    assert len(no_device_restore) == 64
    assert restored.tensors["3.k"].shape[2] == restored.offset
    assert restored.token_digest == metadata["prefix_hash"]


@pytest.mark.parametrize(
    "mutation",
    ["model", "layout", "missing_tensor", "missing_role", "wrong_role", "shape", "dtype"],
)
def test_restore_rejects_incompatible_or_incomplete_state(no_device_restore, mutation):
    runner = _runner()
    tensors, roles, metadata = _snapshot(runner)
    if mutation == "model":
        metadata["model"]["manifest_sha256"] = "other-weights"
    elif mutation == "layout":
        metadata["layout"] = "unknown-layout"
    elif mutation == "missing_tensor":
        tensors.pop("0.recurrent")
    elif mutation == "missing_role":
        roles.pop("0.recurrent")
    elif mutation == "wrong_role":
        roles["0.conv"] = "attention"
    elif mutation == "shape":
        tensors["3.k"] = tensors["3.k"][:, :, :-1, :]
    else:
        tensors["0.conv"] = tensors["0.conv"].astype(np.float32)
    with pytest.raises(ValueError):
        runner.import_state(tensors, roles, metadata)


@pytest.mark.parametrize("offset", [-1, 0, 1.5, True, "3"])
def test_restore_rejects_invalid_offset(no_device_restore, offset):
    runner = _runner()
    tensors, roles, metadata = _snapshot(runner, offset=1)
    metadata["offset"] = offset
    with pytest.raises(ValueError):
        runner.import_state(tensors, roles, metadata)


@pytest.mark.parametrize(
    "field,value", [("activation_dtype", "bfloat16"), ("recurrent_dtype", "float16")]
)
def test_restore_rejects_incompatible_declared_precision(no_device_restore, field, value):
    runner = _runner()
    tensors, roles, metadata = _snapshot(runner)
    metadata[field] = value
    with pytest.raises(ValueError):
        runner.import_state(tensors, roles, metadata)


@pytest.mark.parametrize("prefix", [None, "short", "x" * 64, "b" * 64])
def test_restore_rejects_invalid_or_wrong_prefix_identity(no_device_restore, prefix):
    runner = _runner()
    tensors, roles, metadata = _snapshot(runner)
    metadata["prefix_hash"] = prefix
    with pytest.raises(ValueError):
        runner.import_state(tensors, roles, metadata, expected_prefix_hash="a" * 64)


def test_export_records_offset_and_tensor_roles_without_aliasing_numpy_payloads():
    runner = _runner()
    original = mx.array([[1, 2]], dtype=mx.float16)
    state = LiveState(17, {"3.k": original, "0.recurrent": mx.ones((1,), dtype=mx.float32)})
    tensors, roles, metadata = runner.export_state(state)
    assert metadata["offset"] == 17
    assert metadata["model"] == runner.identity
    assert metadata["prefix_hash"] == state.token_digest
    assert metadata["layout"] == "kvpilot_qwen35_single_sequence_v1"
    assert metadata["activation_dtype"] == "float16"
    assert metadata["recurrent_dtype"] == "float32"
    assert roles == {"3.k": "attention", "0.recurrent": "recurrent"}
    tensors["3.k"][0, 0] = 99
    assert np.asarray(original)[0, 0] == 1


def test_checkpoint_sanitization_shifts_only_offset_norms(monkeypatch):
    weights = weights_module.TensorStore({
        PREFIX + "layers.0.linear_attn.conv1d.weight": mx.ones((5, 1, 4), dtype=mx.float16),
        PREFIX + "layers.0.input_layernorm.weight": mx.zeros((4,), dtype=mx.float16),
        PREFIX + "layers.0.post_attention_layernorm.weight": mx.zeros((4,), dtype=mx.float16),
        PREFIX + "layers.3.self_attn.q_norm.weight": mx.zeros((4,), dtype=mx.float16),
        PREFIX + "layers.3.self_attn.k_norm.weight": mx.zeros((4,), dtype=mx.float16),
        PREFIX + "norm.weight": mx.zeros((4,), dtype=mx.float16),
        PREFIX + "layers.0.linear_attn.norm.weight": mx.full((4,), 2, dtype=mx.float16),
    })
    config = {"text_config": {**_runner().config, "rms_norm_eps": 1e-6, "eos_token_id": 2}}
    tokenizer = SimpleNamespace(token_to_id=lambda token: 3)
    monkeypatch.setattr(runner_module, "load_local_model", lambda *a, **k: (config, weights, tokenizer, _runner().provenance))
    runner = QwenRunner()
    assert runner.unsanitized
    assert weights[PREFIX + "layers.0.linear_attn.conv1d.weight"].shape == (5, 4, 1)
    for key, value in weights.items():
        if "conv1d" in key:
            continue
        expected = 2 if key.endswith("linear_attn.norm.weight") else 1
        np.testing.assert_array_equal(np.asarray(value), np.full((4,), expected))


def test_attention_chunked_resume_preserves_positions_and_causality():
    runner = _runner()
    runner.config.update({"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                          "rope_parameters": {"partial_rotary_factor": 0.5, "rope_theta": 10000}})
    rng = np.random.default_rng(281)
    name = PREFIX + "layers.3.self_attn"
    shapes = {"q_proj": (128, 8), "k_proj": (32, 8), "v_proj": (32, 8), "o_proj": (8, 64)}
    runner.weights = weights_module.TensorStore({
        name + "." + key + ".weight": mx.array(rng.normal(scale=0.08, size=shape), dtype=mx.float16)
        for key, shape in shapes.items()
    })
    for key in ("q_norm", "k_norm"):
        runner.weights[name + "." + key + ".weight"] = mx.ones((32,), dtype=mx.float16)
    x = mx.array(rng.normal(size=(1, 7, 8)), dtype=mx.float16)
    whole_state = LiveState()
    whole = runner.attention(x, 3, whole_state)
    chunk_state = LiveState()
    first = runner.attention(x[:, :3], 3, chunk_state)
    chunk_state.offset = 3
    second = runner.attention(x[:, 3:], 3, chunk_state)
    combined = mx.concatenate([first, second], axis=1)
    runner_module.synchronize(whole, combined)
    np.testing.assert_allclose(np.asarray(combined), np.asarray(whole), rtol=0.006, atol=2e-4)
    np.testing.assert_array_equal(np.asarray(chunk_state.tensors["3.k"]), np.asarray(whole_state.tensors["3.k"]))
    np.testing.assert_array_equal(np.asarray(chunk_state.tensors["3.v"]), np.asarray(whole_state.tensors["3.v"]))


def test_prefix_identity_is_independent_of_prefill_chunking():
    runner = _runner()
    # A zero-layer arithmetic fixture isolates token accounting from model math.
    runner.config["num_hidden_layers"] = 0
    runner.weights = weights_module.TensorStore({
        PREFIX + "embed_tokens.weight": mx.array(np.arange(24).reshape(6, 4) / 32, dtype=mx.float16),
        PREFIX + "norm.weight": mx.ones((4,), dtype=mx.float16),
    })
    whole = LiveState(tensors={"fixture": mx.zeros((1,))})
    split = LiveState(tensors={"fixture": mx.zeros((1,))})
    reordered = LiveState(tensors={"fixture": mx.zeros((1,))})
    runner.forward([1, 2, 3], whole)
    runner.forward([1], split)
    runner.forward([2, 3], split)
    runner.forward([2, 1, 3], reordered)
    assert whole.offset == split.offset == reordered.offset == 3
    assert whole.token_digest == split.token_digest
    assert whole.token_digest != reordered.token_digest
    assert runner.export_state(split)[2]["prefix_hash"] == whole.token_digest


def test_recurrent_chunked_resume_keeps_convolution_tail_and_accumulator():
    runner = _runner()
    runner.config.update({"linear_num_key_heads": 1, "linear_num_value_heads": 2,
                          "linear_key_head_dim": 32, "linear_value_head_dim": 32})
    rng = np.random.default_rng(319)
    name = PREFIX + "layers.0.linear_attn"
    shapes = {"in_proj_qkv": (128, 8), "in_proj_z": (64, 8),
              "in_proj_a": (2, 8), "in_proj_b": (2, 8), "out_proj": (8, 64)}
    runner.weights = weights_module.TensorStore({
        name + "." + key + ".weight": mx.array(rng.normal(scale=0.08, size=shape), dtype=mx.float16)
        for key, shape in shapes.items()
    })
    runner.weights[name + ".conv1d.weight"] = mx.array(rng.normal(scale=0.3, size=(128, 4, 1)), dtype=mx.float16)
    runner.weights[name + ".norm.weight"] = mx.ones((32,), dtype=mx.float16)
    runner.weights[name + ".dt_bias"] = mx.zeros((2,), dtype=mx.float16)
    runner.weights[name + ".A_log"] = mx.zeros((2,), dtype=mx.float32)
    x = mx.array(rng.normal(size=(1, 9, 8)), dtype=mx.float16)
    whole_state, split_state = LiveState(), LiveState()
    whole = runner.recurrent(x, 0, whole_state)
    first = runner.recurrent(x[:, :4], 0, split_state)
    second = runner.recurrent(x[:, 4:], 0, split_state)
    resumed = mx.concatenate([first, second], axis=1)
    runner_module.synchronize(whole, resumed, *whole_state.tensors.values(), *split_state.tensors.values())
    np.testing.assert_allclose(np.asarray(resumed), np.asarray(whole), rtol=0.006, atol=2e-4)
    np.testing.assert_array_equal(np.asarray(split_state.tensors["0.conv"]), np.asarray(whole_state.tensors["0.conv"]))
    np.testing.assert_allclose(np.asarray(split_state.tensors["0.recurrent"]),
                               np.asarray(whole_state.tensors["0.recurrent"]), rtol=1e-4, atol=2e-6)


def test_prompt_framing_disables_thinking_and_adds_no_extra_special_tokens():
    runner = _runner()
    calls = []

    def encode(text, *, add_special_tokens):
        calls.append((text, add_special_tokens))
        return SimpleNamespace(ids=[7, 11])

    runner.tokenizer = SimpleNamespace(encode=encode)
    assert runner.prompt_tokens("Question here") == [7, 11]
    assert calls == [(
        "<|im_start|>user\nQuestion here<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        False,
    )]
