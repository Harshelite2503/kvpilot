"""Explicit-state gated-delta math for the local Qwen3.5 experiment.

Only ``mlx.core`` supplies tensor operations and Metal execution. This module
does not import MLX-LM, create cache objects, retain model state, or choose cache
policies. The caller owns every input and returned state array.

The Metal recurrence is adapted from Apple's generic scalar-gate kernel:
https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/gated_delta.py
Copyright © 2025 Apple Inc. MIT license; see THIRD_PARTY_NOTICES.md.
Changes: one explicit unmasked operator, FP32 state/gates, shape checks,
non-divisible value-row bounds, lazy kernel creation, no cache/runtime helpers.
"""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        raise RuntimeError("The local gated-delta operator requires Apple Metal")
    return mx.fast.metal_kernel(
        name="kvpilot_explicit_gated_delta",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=r"""
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            auto dv_idx = thread_position_in_grid.y;
            if (dv_idx >= Dv) return;
            auto lane = thread_position_in_threadgroup.x;
            constexpr int n_per_lane = Dk / 32;

            // Q/K [B,T,Hk,Dk]; V/output [B,T,Hv,Dv].
            auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk;
            auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk;
            auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
            auto y_ = y + (b_idx * T * Hv + hv_idx) * Dv;
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // FP32 accumulator [B,Hv,Dv,Dk], one SIMD group per value row.
            auto input_state = state_in + (n * Dv + dv_idx) * Dk;
            auto output_state = state_out + (n * Dv + dv_idx) * Dk;
            float state[n_per_lane];
            for (int i = 0; i < n_per_lane; ++i) {
                state[i] = input_state[n_per_lane * lane + i];
            }

            for (int t = 0; t < T; ++t) {
                float prediction = 0.0f;
                for (int i = 0; i < n_per_lane; ++i) {
                    auto j = n_per_lane * lane + i;
                    state[i] = state[i] * g_[hv_idx];
                    prediction += state[i] * static_cast<float>(k_[j]);
                }
                prediction = simd_sum(prediction);
                float delta = (static_cast<float>(v_[dv_idx]) - prediction)
                              * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_lane; ++i) {
                    auto j = n_per_lane * lane + i;
                    state[i] = state[i] + static_cast<float>(k_[j]) * delta;
                    out += state[i] * static_cast<float>(q_[j]);
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                    y_[dv_idx] = static_cast<InT>(out);
                }
                q_ += Hk * Dk;
                k_ += Hk * Dk;
                v_ += Hv * Dv;
                y_ += Hv * Dv;
                g_ += Hv;
                beta_ += Hv;
            }
            for (int i = 0; i < n_per_lane; ++i) {
                output_state[n_per_lane * lane + i] = state[i];
            }
        """,
    )


def gated_delta(q, k, v, g, beta, state):
    """Return ``(output, next_state)`` without modifying the supplied state.

    Shapes: Q/K ``[B,T,Hk,Dk]``, V ``[B,T,Hv,Dv]``, decay/beta
    ``[B,T,Hv]``, FP32 state ``[B,Hv,Dv,Dk]``. ``Hv`` must be a multiple
    of ``Hk`` and ``Dk`` a multiple of 32. Q, K and V share a floating dtype;
    output has that dtype, while the recurrence accumulates in FP32.

    ``g`` is the *decay factor*, already exp(-exp(A_log)*softplus(a+dt_bias)),
    not its logarithm. ``beta`` is already sigmoid-transformed. Q/K have
    already been L2-normalized and Q scaled by ``1/sqrt(Dk)``. This function
    applies no normalization, gate activation, convolution or output norm.

    For each token: S <- g*S; delta <- beta*(v-S@k);
    S <- S + delta outer k; output <- S@q. Token positions, pause/resume,
    serialization and cache ownership belong to the calling experiment.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Q/K must share [B,T,Hk,Dk]; V must be [B,T,Hv,Dv]")
    batch, tokens, key_heads, key_dim = q.shape
    if v.shape[:2] != (batch, tokens):
        raise ValueError("Q/K/V batch and token dimensions must match")
    value_heads, value_dim = v.shape[2:]
    if min(batch, key_heads, key_dim, value_heads, value_dim) <= 0:
        raise ValueError("Batch, head and head-width dimensions must be positive")
    if value_heads % key_heads:
        raise ValueError("Value-head count must be a multiple of key-head count")
    if key_dim % 32:
        raise ValueError("Key width must be divisible by the 32-lane SIMD width")
    if g.shape != (batch, tokens, value_heads) or beta.shape != g.shape:
        raise ValueError("Decay and beta must have shape [B,T,Hv]")
    expected_state = (batch, value_heads, value_dim, key_dim)
    if state.shape != expected_state:
        raise ValueError(f"State must have shape {expected_state}, got {state.shape}")
    floating = (mx.float16, mx.bfloat16, mx.float32)
    if q.dtype not in floating or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V must share float16, bfloat16 or float32 dtype")
    if g.dtype not in floating or beta.dtype not in floating:
        raise ValueError("Decay and beta must be floating-point arrays")
    if state.dtype != mx.float32:
        raise ValueError("Recurrent state must remain float32")
    if tokens == 0:
        return mx.zeros(v.shape, dtype=v.dtype), state

    # Inputs may be slices of a larger prefill. The kernel uses dense indexing.
    arrays = [mx.contiguous(x) for x in (q, k, v)]
    arrays.extend(mx.contiguous(x.astype(mx.float32)) for x in (g, beta, state))
    output, next_state = _kernel()(
        inputs=[*arrays, tokens],
        template=[
            ("InT", v.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[v.shape, state.shape],
        output_dtypes=[v.dtype, mx.float32],
    )
    return output, next_state
