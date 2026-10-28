"""Optional real-Metal checks; the ordinary CPU development environment skips."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="Optional local MLX tensor engine")
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Explicit-state kernel requires Apple Metal"
)

_path = Path(__file__).resolve().parents[1] / "experiments/local/gated_delta.py"
_spec = importlib.util.spec_from_file_location("kvpilot_local_gated_delta", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
gated_delta = _module.gated_delta


def _reference(q, k, v, decay, beta, state):
    """Independent NumPy matrix recurrence, including grouped key/value heads."""
    q, k, v, decay, beta, state = [
        np.asarray(x, dtype=np.float32) for x in (q, k, v, decay, beta, state)
    ]
    state = state.copy()
    output = np.empty(v.shape, dtype=np.float32)
    key_for_value = np.arange(v.shape[2]) // (v.shape[2] // q.shape[2])
    for token in range(q.shape[1]):
        qt = q[:, token][:, key_for_value]
        kt = k[:, token][:, key_for_value]
        state *= decay[:, token, :, None, None]
        predicted = np.einsum("bhvk,bhk->bhv", state, kt)
        correction = beta[:, token, :, None] * (v[:, token] - predicted)
        state += np.einsum("bhv,bhk->bhvk", correction, kt)
        output[:, token] = np.einsum("bhvk,bhk->bhv", state, qt)
    return output, state


def _inputs(shape, seed=761):
    batch, tokens, key_heads, value_heads, key_dim, value_dim = shape
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(batch, tokens, key_heads, key_dim)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * np.sqrt(key_dim)
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    v = rng.normal(size=(batch, tokens, value_heads, value_dim)).astype(np.float32)
    decay = rng.uniform(0.2, 1, size=(batch, tokens, value_heads)).astype(np.float32)
    beta = rng.uniform(0, 1, size=decay.shape).astype(np.float32)
    state = rng.normal(
        scale=0.1, size=(batch, value_heads, value_dim, key_dim)
    ).astype(np.float32)
    return q, k, v, decay, beta, state


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1, 32, 4),
        (2, 7, 2, 4, 64, 7),
        (1, 67, 2, 4, 128, 16),
        (1, 17, 16, 32, 128, 128),
    ],
)
def test_kernel_matches_numpy_recurrence(shape):
    arrays = _inputs(shape)
    expected, expected_state = _reference(*arrays)
    inputs = [mx.array(x) for x in arrays]
    output, state = gated_delta(*inputs)
    mx.eval(output, state)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(state), expected_state, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(inputs[-1]), arrays[-1])


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_low_precision_activations_keep_float32_state(dtype):
    arrays = [mx.array(x) for x in _inputs((1, 11, 2, 4, 128, 8))]
    arrays[:3] = [x.astype(dtype) for x in arrays[:3]]
    rounded = [np.asarray(x.astype(mx.float32)) for x in arrays]
    expected, expected_state = _reference(*rounded)
    output, state = gated_delta(*arrays)
    mx.eval(output, state)
    assert output.dtype == dtype
    assert state.dtype == mx.float32
    tolerance = 0.009 if dtype == mx.bfloat16 else 0.001
    np.testing.assert_allclose(
        np.asarray(output.astype(mx.float32)), expected, rtol=tolerance, atol=3e-4
    )
    np.testing.assert_allclose(np.asarray(state), expected_state, rtol=2e-5, atol=2e-6)


def test_chunked_resume_matches_single_forward():
    arrays = [mx.array(x) for x in _inputs((2, 19, 2, 4, 128, 8))]
    whole, whole_state = gated_delta(*arrays)
    split = 7
    first, first_state = gated_delta(*[x[:, :split] for x in arrays[:5]], arrays[5])
    second, resumed_state = gated_delta(
        *[x[:, split:] for x in arrays[:5]], first_state
    )
    resumed = mx.concatenate([first, second], axis=1)
    mx.eval(whole, whole_state, resumed, resumed_state)
    np.testing.assert_array_equal(np.asarray(resumed), np.asarray(whole))
    np.testing.assert_array_equal(np.asarray(resumed_state), np.asarray(whole_state))


@pytest.mark.parametrize("decay,beta", [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_gate_extremes(decay, beta):
    arrays = list(_inputs((1, 5, 1, 2, 32, 5)))
    arrays[3].fill(decay)
    arrays[4].fill(beta)
    expected, expected_state = _reference(*arrays)
    output, state = gated_delta(*[mx.array(x) for x in arrays])
    mx.eval(output, state)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(state), expected_state, rtol=2e-5, atol=2e-6)


def test_rejects_wrong_gate_shape_and_reduced_precision_state():
    arrays = [mx.array(x) for x in _inputs((1, 3, 1, 2, 32, 4))]
    with pytest.raises(ValueError, match="Decay and beta"):
        gated_delta(*arrays[:3], arrays[3][..., None], *arrays[4:])
    with pytest.raises(ValueError, match="remain float32"):
        gated_delta(*arrays[:5], arrays[5].astype(mx.float16))


def test_empty_forward_preserves_supplied_state():
    arrays = [mx.array(x) for x in _inputs((1, 0, 1, 2, 32, 4))]
    output, state = gated_delta(*arrays)
    assert output.shape == (1, 0, 2, 4)
    assert state is arrays[-1]
