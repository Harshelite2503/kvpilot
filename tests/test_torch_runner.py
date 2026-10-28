"""Independent tiny-model math and snapshot tests; no model downloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional tensor-engine experiment")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.torch.qwen_runner import QwenRunner, LiveState, INITIAL_DIGEST
from experiments.torch.weights import tensor_shapes, validate_config, load_local_model
from experiments.local.state_codec import encode_state, decode_state, encoded_size_from_specs


class TinyTokenizer:
    def token_to_id(self, token):
        return None

    def decode(self, tokens):
        return " ".join(map(str, tokens))

    def encode(self, text, add_special_tokens=False):
        self.last_text = text
        return SimpleNamespace(ids=[1, 2, 3])


def components(*, tied=True):
    config = {"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 12,
              "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
              "vocab_size": 17, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "hidden_act": "silu", "tie_word_embeddings": tied,
              "eos_token_id": 16}
    rng = np.random.default_rng(712)
    weights = {}
    for name, shape in tensor_shapes(config).items():
        array = rng.normal(0, 0.15, size=shape).astype(np.float32)
        if "norm.weight" in name:
            array = 1 + array * 0.1
        weights[name] = torch.from_numpy(array)
    return config, weights, TinyTokenizer()


@pytest.fixture
def runner():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    result = QwenRunner.from_components(*components())
    yield result
    torch.set_num_threads(previous)


def independent_numpy_logits(config, weights, tokens):
    """Uncached FP64 reference, explicit per-head and per-query attention."""
    weights = {name: value.cpu().numpy().astype(np.float64) for name, value in weights.items()}
    hidden, heads, kv_heads = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
    dimension, length = hidden // heads, len(tokens)

    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights.get(name + ".bias", 0)

    def norm(x, name):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config["rms_norm_eps"]) * weights[name + ".weight"]

    def rotate(x):
        out = x.copy()
        for position in range(length):
            for j in range(dimension // 2):
                angle = position / config["rope_theta"] ** (2 * j / dimension)
                left, right = x[position, :, j], x[position, :, j + dimension // 2]
                out[position, :, j] = left * np.cos(angle) - right * np.sin(angle)
                out[position, :, j + dimension // 2] = right * np.cos(angle) + left * np.sin(angle)
        return out

    x = weights["model.embed_tokens.weight"][tokens]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normalized = norm(x, prefix + ".input_layernorm")
        q = rotate(linear(normalized, prefix + ".self_attn.q_proj").reshape(length, heads, dimension))
        k = rotate(linear(normalized, prefix + ".self_attn.k_proj").reshape(length, kv_heads, dimension))
        v = linear(normalized, prefix + ".self_attn.v_proj").reshape(length, kv_heads, dimension)
        attended = np.zeros((length, heads, dimension))
        for t in range(length):
            for h in range(heads):
                kv = h // (heads // kv_heads)
                scores = k[:t + 1, kv] @ q[t, h] / np.sqrt(dimension)
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                attended[t, h] = probabilities @ v[:t + 1, kv]
        x += linear(attended.reshape(length, hidden), prefix + ".self_attn.o_proj")
        normalized = norm(x, prefix + ".post_attention_layernorm")
        gate = linear(normalized, prefix + ".mlp.gate_proj")
        gated = gate / (1 + np.exp(-gate)) * linear(normalized, prefix + ".mlp.up_proj")
        x += linear(gated, prefix + ".mlp.down_proj")
    output = norm(x, "model.norm")
    head = "model.embed_tokens.weight" if config["tie_word_embeddings"] else "lm_head.weight"
    return output @ weights[head].T


@pytest.mark.parametrize("tied", [True, False])
def test_full_context_matches_independent_per_head_numpy_reference(tied):
    config, weights, tokenizer = components(tied=tied)
    runner = QwenRunner.from_components(config, weights, tokenizer)
    tokens = [2, 5, 3, 9, 4]
    expected = independent_numpy_logits(config, weights, tokens)
    actual = runner.forward(tokens, LiveState()).numpy()
    np.testing.assert_allclose(actual, expected[-1], atol=1e-6, rtol=1e-5)


def test_chunked_prefill_and_incremental_decode_match_full_context(runner):
    tokens = [2, 5, 3, 9, 4, 6, 1]
    full, full_state = runner.prefill(tokens, chunk_size=99)
    chunked, chunked_state = runner.prefill(tokens, chunk_size=2)
    single, single_state = runner.prefill(tokens, chunk_size=1)
    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(single, full, atol=1e-6, rtol=1e-5)
    assert full_state.token_digest == chunked_state.token_digest == single_state.token_digest
    for name in full_state.tensors:
        torch.testing.assert_close(chunked_state.tensors[name], full_state.tensors[name], atol=1e-6, rtol=1e-5)
    resumed = runner.forward([11], chunked_state)
    fresh = runner.forward(tokens + [11], LiveState())
    torch.testing.assert_close(resumed, fresh, atol=1e-6, rtol=1e-5)
    assert chunked_state.offset == 8


def test_causal_mask_protects_earlier_tokens_from_future_suffix(runner):
    _, prefix = runner.prefill([1, 5, 2], chunk_size=99)
    _, longer = runner.prefill([1, 5, 2, 10, 9], chunk_size=99)
    for name, value in prefix.tensors.items():
        torch.testing.assert_close(value, longer.tensors[name][:, :, :3], atol=1e-6, rtol=1e-5)


def test_raw_snapshot_is_bit_exact_owned_and_resumes_same_generation(runner):
    _, live = runner.prefill([1, 2, 4, 5])
    arrays, roles, meta = runner.export_state(live)
    blob = encode_state(arrays, roles, meta, "raw_native")
    restored = runner.import_state(*decode_state(blob), expected_prefix_hash=live.token_digest)
    for key in live.tensors:
        assert torch.equal(live.tensors[key], restored.tensors[key])
    first_key = next(iter(arrays))
    arrays[first_key].fill(100)
    assert not np.all(restored.tensors[first_key].numpy() == 100)
    runner.stop_ids.clear()
    original_result, original_state = runner.generate([7], live, max_tokens=3)
    restored_result, restored_state = runner.generate([7], restored, max_tokens=3)
    assert original_result["tokens"] == restored_result["tokens"]
    assert original_state.token_digest == restored_state.token_digest
    assert original_result["truncated"] is True
    assert 0 <= original_result["ttft_ms"] <= original_result["wall_ms"]


@pytest.mark.parametrize("codec", ["raw_native", "raw_fp16", "group_int8", "group_int4"])
def test_exact_sizing_without_export_and_codec_dtype_restore(runner, monkeypatch, codec):
    _, live = runner.prefill([2, 7, 4])
    arrays, roles, metadata = runner.export_state(live)
    blob = encode_state(arrays, roles, metadata, codec)
    decoded = decode_state(blob)
    # The outer state envelope restores recorded FP32 dtype after FP16 decoding.
    assert all(array.dtype == np.float32 for array in decoded[0].values())
    restored = runner.import_state(*decoded, expected_prefix_hash=live.token_digest)
    assert all(value.dtype == torch.float32 for value in restored.tensors.values())
    monkeypatch.setattr(runner, "export_state", lambda *_: pytest.fail("sizing exported data"))
    monkeypatch.setattr(runner, "synchronize", lambda: pytest.fail("sizing synchronized device"))
    actual = runner.state_specs(live)
    prospective = runner.expected_state_specs(live.offset, live.token_digest)
    assert actual == prospective
    assert encoded_size_from_specs(*actual, codec) == len(blob)
    assert encoded_size_from_specs(*runner.expected_state_specs(live.offset), codec) == len(blob)


@pytest.mark.parametrize("corruption", ["model", "layout", "dtype_declared", "prefix", "offset",
                                        "missing", "role", "shape", "dtype", "nan"])
def test_import_rejects_corrupt_or_incompatible_snapshot(runner, corruption):
    _, state = runner.prefill([1, 4, 7])
    arrays, roles, metadata = runner.export_state(state)
    key = next(iter(arrays))
    if corruption == "model":
        metadata["model"]["model_digest"] = "other"
    elif corruption == "layout":
        metadata["layout"] = "other"
    elif corruption == "dtype_declared":
        metadata["activation_dtype"] = "float16"
    elif corruption == "prefix":
        metadata["prefix_hash"] = "b" * 64
    elif corruption == "offset":
        metadata["offset"] = True
    elif corruption == "missing":
        del arrays[key]
    elif corruption == "role":
        roles[key] = "recurrent"
    elif corruption == "shape":
        arrays[key] = arrays[key][:, :, :2]
    elif corruption == "dtype":
        arrays[key] = arrays[key].astype(np.float16)
    else:
        arrays[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        runner.import_state(arrays, roles, metadata, expected_prefix_hash=state.token_digest)


def test_release_drops_only_live_state_and_rejects_foreign_or_partial_state(runner):
    _, state = runner.prefill([1, 2])
    saved = runner.export_state(state)
    assert state.nbytes() == 2 * 2 * 1 * 1 * 2 * 4 * 4
    state.release()
    assert state.nbytes() == 0 and state.offset == 0 and state.token_digest == INITIAL_DIGEST
    restored = runner.import_state(*saved)
    restored.model_identity["model_digest"] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        runner.forward([3], restored)
    with pytest.raises(ValueError):
        runner.forward([3], LiveState(offset=1))


def test_invalid_requests_leave_state_unmodified(runner):
    _, state = runner.prefill([1, 2])
    digest, count = state.token_digest, state.offset
    for tokens in ([], [-1], [17], [True]):
        with pytest.raises(ValueError):
            runner.forward(tokens, state)
    with pytest.raises(ValueError):
        runner.prefill([1], state, chunk_size=0)
    with pytest.raises(ValueError):
        runner.prefill([1] * 129, state)
    with pytest.raises(ValueError):
        runner.generate([1], state, max_tokens=0)
    assert (state.token_digest, state.offset) == (digest, count)


def test_eos_does_not_emit_or_append_stop_token(runner, monkeypatch):
    logits = torch.full((17,), -1.0)
    logits[16] = 1
    state = LiveState()
    monkeypatch.setattr(runner, "prefill", lambda *_: (logits, state))
    result, _ = runner.generate([1], max_tokens=3)
    assert result["tokens"] == [] and result["truncated"] is False
    assert state.offset == 0


def test_equal_work_uses_fixed_tokens_even_when_every_prediction_is_a_stop(runner, monkeypatch):
    prefix, suffix, targets = [2, 5, 3], [8, 10], [16, 4, 16, 2]
    _, state = runner.prefill(prefix)
    runner.stop_ids = set(range(runner.config["vocab_size"]))
    observed = []
    forward = runner.forward

    def recording_forward(tokens, current_state):
        observed.append(list(tokens))
        return forward(tokens, current_state)

    monkeypatch.setattr(runner, "forward", recording_forward)
    result, returned = runner.continue_fixed(suffix, state, targets)
    assert returned is state
    assert observed == [suffix] + [[token] for token in targets[:-1]]
    expected_predictions = [int(independent_numpy_logits(
        runner.config, runner.weights, prefix + suffix + targets[:index])[-1].argmax())
        for index in range(len(targets))]
    assert result["predicted_tokens"] == expected_predictions
    assert result["tokens"] == targets and result["tokens"] is not targets
    assert result["mode"] == "equal_work"
    assert "answer" not in result and result["truncated"] is False
    assert len(result["per_token_ms"]) == len(targets)
    assert all(value >= 0 for value in result["per_token_ms"])
    assert result["first_token_ms"] == result["ttft_ms"] == result["per_token_ms"][0]
    assert result["total_ms"] == result["wall_ms"] == pytest.approx(sum(result["per_token_ms"]))
    _, expected_state = runner.prefill(prefix + suffix + targets[:-1])
    assert state.offset == expected_state.offset
    assert state.token_digest == expected_state.token_digest
    for name in state.tensors:
        torch.testing.assert_close(state.tensors[name], expected_state.tensors[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("codec", ["raw_native", "group_int4"])
def test_equal_work_after_restore_consumes_same_ids_and_steps(runner, codec):
    _, state = runner.prefill([2, 5, 3, 8])
    restored = runner.import_state(*decode_state(encode_state(*runner.export_state(state), codec)),
                                   expected_prefix_hash=state.token_digest)
    original, _ = runner.continue_fixed([6, 2], state, [16, 5, 2])
    replay, _ = runner.continue_fixed([6, 2], restored, [16, 5, 2])
    assert original["tokens"] == replay["tokens"] == [16, 5, 2]
    assert len(original["per_token_ms"]) == len(replay["per_token_ms"]) == 3
    assert state.offset == restored.offset == 8
    assert state.token_digest == restored.token_digest
    if codec == "raw_native":
        assert original["predicted_tokens"] == replay["predicted_tokens"]


@pytest.mark.parametrize("suffix,targets", [([], [1]), ([1], []), ([True], [2]),
                                           ([1], [False]), ([1], [-1]), ([1], [17]),
                                           ([1], [2, 3, -1]), ([1], "2"),
                                           ([1], [2] * 127)])
def test_invalid_equal_work_request_does_not_mutate_state(runner, monkeypatch, suffix, targets):
    _, state = runner.prefill([1, 2])
    before = (state.offset, state.token_digest, state.model_identity, dict(state.tensors))
    monkeypatch.setattr(runner, "forward", lambda *_: pytest.fail("invalid request executed a forward"))
    with pytest.raises(ValueError):
        runner.continue_fixed(suffix, state, targets)
    assert (state.offset, state.token_digest, state.model_identity) == before[:3]
    assert state.tensors.keys() == before[3].keys()
    assert all(state.tensors[name] is value for name, value in before[3].items())


def test_equal_work_validates_state_before_timing_or_forward(runner, monkeypatch):
    _, state = runner.prefill([1, 2])
    state.model_identity["model_digest"] = "foreign"
    monkeypatch.setattr(runner, "synchronize", lambda: pytest.fail("invalid request began timing"))
    with pytest.raises(ValueError, match="identity"):
        runner.continue_fixed([3], state, [4])


@pytest.mark.parametrize("targets", [[16], [4, 5, 6, 16]])
def test_equal_work_checks_exact_consumed_context_and_does_not_append_last_target(runner, targets):
    runner.max_context = 2 + 1 + len(targets) - 1
    _, state = runner.prefill([1, 2])
    result, _ = runner.continue_fixed([3], state, targets)
    assert state.offset == runner.max_context
    assert len(result["predicted_tokens"]) == len(targets)
    _, expected = runner.prefill([1, 2, 3] + targets[:-1])
    assert state.token_digest == expected.token_digest


def test_prompt_format_and_config_restrictions(runner):
    assert runner.prompt_tokens("read this") == [1, 2, 3]
    assert runner.tokenizer.last_text == "<|im_start|>user\nread this<|im_end|>\n<|im_start|>assistant\n"
    for changes in ({"rope_scaling": {"type": "linear"}}, {"use_sliding_window": True},
                    {"model_type": "qwen2_moe"}, {"num_key_value_heads": 3}, {"head_dim": 5},
                    {"hidden_act": "relu"}, {"rms_norm_eps": float("nan")}):
        with pytest.raises(ValueError):
            validate_config(runner.config | changes)
    for offset in (0, True, -1, 129):
        with pytest.raises(ValueError):
            runner.expected_state_specs(offset)


def local_checkpoint(tmp_path, *, sharded=False):
    safetensors = pytest.importorskip("safetensors.torch")
    tokenizers = pytest.importorskip("tokenizers")
    config, weights, _ = components()
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({f"t{i}": i for i in range(17)}, unk_token="t0"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    if sharded:
        names = sorted(weights)
        mapping = {}
        for index, chunk in enumerate((names[::2], names[1::2])):
            filename = f"model-{index}.safetensors"
            safetensors.save_file({name: weights[name] for name in chunk}, str(tmp_path / filename))
            mapping.update({name: filename for name in chunk})
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        safetensors.save_file(weights, str(tmp_path / "model.safetensors"))
    return config, weights


@pytest.mark.parametrize("sharded", [False, True])
def test_loader_hashes_local_files_and_agrees_with_components(tmp_path, sharded):
    config, weights = local_checkpoint(tmp_path, sharded=sharded)
    loaded = QwenRunner(tmp_path)
    expected = independent_numpy_logits(config, weights, [1, 2, 5])[-1]
    np.testing.assert_allclose(loaded.forward([1, 2, 5], LiveState()).numpy(), expected, atol=1e-6, rtol=1e-5)
    hashes = loaded.provenance["file_sha256"]
    pinned = QwenRunner(tmp_path, expected_hashes=hashes)
    assert pinned.provenance["expected_hashes_verified"] is True
    assert pinned.identity == loaded.identity
    (tmp_path / "config.json").write_text(json.dumps(config | {"max_position_embeddings": 129}))
    with pytest.raises(ValueError, match="SHA256"):
        QwenRunner(tmp_path, expected_hashes=hashes)


def test_loader_rejects_inconsistent_or_traversing_shard_map(tmp_path):
    local_checkpoint(tmp_path, sharded=True)
    path = tmp_path / "model.safetensors.index.json"
    original = json.loads(path.read_text())
    key = next(iter(original["weight_map"]))
    corrupted = deepcopy(original)
    corrupted["weight_map"][key] = "../outside.safetensors"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="index"):
        load_local_model(tmp_path)
    corrupted = deepcopy(original)
    corrupted["weight_map"][key] = "model-1.safetensors"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="index"):
        load_local_model(tmp_path)


def test_loader_rejects_missing_tensor_and_components_reject_bad_tied_head(tmp_path):
    config, weights = local_checkpoint(tmp_path)
    from safetensors.torch import save_file
    with pytest.raises(ValueError, match="tied"):
        QwenRunner.from_components(config, weights | {"lm_head.weight": weights["model.embed_tokens.weight"] + 1}, TinyTokenizer())
    del weights["model.layers.0.self_attn.q_proj.bias"]
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="names"):
        load_local_model(tmp_path)


def test_nonfinite_forward_does_not_commit_partial_state(runner):
    state = LiveState()
    runner.weights["model.layers.1.mlp.down_proj.weight"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        runner.forward([1, 2, 3], state)
    assert state.offset == 0 and not state.tensors and state.token_digest == INITIAL_DIGEST


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_fp16_math_and_raw_restore_when_available(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} tensor device is unavailable")
    config, weights, tokenizer = components()
    runner = QwenRunner.from_components(config, weights, tokenizer, device=device)
    tokens = [2, 5, 3, 9, 4]
    expected = independent_numpy_logits(config, runner.weights, tokens)[-1]
    logits, state = runner.prefill(tokens, chunk_size=2)
    assert runner.dtype == torch.float16
    np.testing.assert_allclose(logits.cpu().numpy(), expected, atol=3e-3, rtol=2e-2)
    arrays, roles, metadata = runner.export_state(state)
    assert all(array.dtype == np.float16 for array in arrays.values())
    restored = runner.import_state(*decode_state(encode_state(arrays, roles, metadata, "raw_native")),
                                   expected_prefix_hash=state.token_digest)
    assert all(torch.equal(state.tensors[key], restored.tensors[key]) for key in state.tensors)
    live_logits, restored_logits = runner.forward([7], state), runner.forward([7], restored)
    assert torch.equal(live_logits, restored_logits)
    state.release()
    restored.release()
