# Third-party mathematical references

`qwen_runner.py` follows the text-only Qwen3.5 layer equations and checkpoint
sanitization described in Apple's mathematical reference files below, at the
same pinned commit `4cfc6478473b0668ede3eb0cc93f2a9181a15052`:

- [qwen3_5.py](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/qwen3_5.py), Copyright © 2026 Apple Inc.
- [qwen3_next.py](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/qwen3_next.py), Copyright © 2025 Apple Inc.
- [activations.py](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/activations.py), Copyright © 2023 Apple Inc.

The local runner implements explicit state tensors, forward/prefill/decode
control, and serialization itself; it does not import the reference model or
cache implementations. It deliberately uses FP16 activations and FP32 recurrent
state, and is not asserted to reproduce native Ollama BF16 logits bit for bit.
The model architecture, normalization and rotary-position mathematics are prior
work, not KVpilot inventions. These references use the MIT license reproduced
below.

## Gated-delta kernel

`gated_delta.py` adapts the generic, unmasked scalar-gate Metal mathematical
recurrence from Apple's MLX-LM `mlx_lm/models/gated_delta.py`, pinned to commit
`4cfc6478473b0668ede3eb0cc93f2a9181a15052`:

- [Source](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/gated_delta.py)
- [License](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/LICENSE)
- Source SHA-256: `f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428`

The adaptation removes model, cache, mask, environment and dispatch helpers;
keeps state and gates in FP32; adds explicit input validation and value-row
bounds; and accepts/returns caller-owned state tensors. It imports only
`mlx.core`, not MLX-LM. No upstream cache or cache-management implementation is
included in this module. This arithmetic kernel is prior work, not KVpilot's
novel contribution.

Original file copyright: Copyright © 2025 Apple Inc.

## MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
