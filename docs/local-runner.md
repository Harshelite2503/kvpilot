# Reproducing the local KVPilot state experiment

This experiment runs a real local Qwen3.5 text model, saves its actual inference state, releases the live state, and either restores the saved bytes or recomputes the prefix before continuing generation. It compares KVPilot's own adaptive store with fixed native-state storage, fixed INT4 attention storage, and recomputation. Timings come from execution on the local machine; the policy's cost predictions remain estimates.

The implementation is in [benchmark_local_kvpilot.py](../scripts/benchmark_local_kvpilot.py) and [experiments/local](../experiments/local/). It is separate from both the earlier analytical replay and the native Ollama cache benchmarks. Those experiments retain their own protocols and evidence.

The runner requires an Apple Silicon Mac with a working Metal backend and the existing `qwen3.5:4b` MLX/safetensors model files under `~/.ollama/models`. It reads those files directly. It does not start or call Ollama, download a model, or use an inference server. A GGUF checkpoint or another model configuration is not interchangeable with the supported checkpoint.

The development environment uses Python 3.13.0 with these package versions:

| Package | Version | Purpose |
| --- | --- | --- |
| `mlx` | 0.32.4 | Tensor operations, weight loading and Metal execution |
| `mlx-metal` | 0.32.4 | Apple Metal backend |
| `numpy` | 2.5.4 | CPU snapshot serialization and reference quantization |
| `tokenizers` | 0.23.3 | Tokenization using the existing local tokenizer |
| `pytest` | 9.1.1 | Tests; not required by the benchmark itself |

The runtime versions are pinned in [requirements.txt](../experiments/local/requirements.txt). For a fresh development environment, run these commands from the repository root. Installing dependencies requires access to the package index; the benchmark itself makes no network calls.

```sh
python3.13 -m venv .venv-local
.venv-local/bin/python -m pip install -e . \
  -r experiments/local/requirements.txt pytest==9.1.1
```

No Python `safetensors` package or MLX-LM package is required: MLX loads the checked tensor files. The model architecture is prior work. The gated-delta arithmetic kernel is adapted from Apple's MIT-licensed reference; attribution and the pinned source are in [THIRD_PARTY_NOTICES.md](../experiments/local/THIRD_PARTY_NOTICES.md). KVPilot owns the live state, cache update rules, pause/resume hooks, serialized payloads, admission decisions and evictions. It imports no external model cache, prefix-cache manager or generation implementation.

The text forward pass uses the checkpoint's NVFP4 matrix weights, FP16 activations and FP32 recurrent state. Some checkpoint BF16 arrays are converted to FP16, while designated recurrent parameters remain FP32. This is a deliberately narrow local runner, with greedy decoding and a single sequence at a time. It is not claimed to reproduce native Ollama's numerical behavior exactly. The loader records manifest identity, tensor metadata, declared blob digests and file sizes; it does not independently rehash every full weight blob.

Each paused session has one complete snapshot containing all 32 layers' state, the current offset, model/layout identity and cumulative token-prefix hash. Eight layers have attention K/V arrays; the other 24 retain convolution and recurrent state. Import rejects incomplete snapshots, incompatible identity, wrong shapes/dtypes and a mismatched expected prefix hash. It cannot split a recurrent snapshot to recover an earlier prefix.

The bounded reference serializer caps a complete decoded state at 256 MiB. With this model's layout that limits snapshots to roughly 6,600 prefix tokens, despite the model's larger advertised context window. The measured cases use about 1,500 prefix tokens. This is not yet a long-context serving implementation.

| Policy or representation | Actual retained state |
| --- | --- |
| `recompute` | No paused payload; the full prefix is recomputed at resume |
| `raw_native` | Original dtype and bytes for every state tensor, plus metadata |
| `group_int4` | Symmetric INT4 attention K/V, FP16 scale per last-axis group of 64, and exact native convolution/recurrent state |
| `group_int8` | The corresponding INT8 representation; available to the adaptive policy |
| `adaptive_exact` | Selects among the same candidates using exact byte sizes and calibrated costs, without charging harm to evicted owners |
| `adaptive_eviction` | Selects recomputation, native storage, INT8 or INT4 using calibrated costs and current store occupancy |

“INT4” therefore does not mean that the whole model state uses four bits per value. Recurrent FP32 state remains exact and places a floor on total snapshot size. The reference quantizers run on the CPU; their performance does not represent optimized GPU quantization kernels. Compression can change subsequent model outputs, so answer quality and token agreement are measured alongside speed and bytes.

The store calculates candidate sizes from tensor shapes, dtypes and metadata before exporting data. Only the selected candidate is exported and encoded. A recompute decision avoids that copy. Commit verifies the complete payload length against the decision and executes the same oldest-first victim plan that was scored. There is no shared-prefix deduplication, partial cache hit, asynchronous offload or queue model in this local store.

The eviction penalty estimates the restoration benefit that currently resident paused owners would lose. It uses their past measured prefill costs, calibrated restore costs and a forecast learned only from resolved sessions. Outstanding or evicted sessions are not automatically treated as cancellations. A resume with no complete payload performs full recomputation. The benchmark provides no future resume outcome or future answer to the controller.

Run the focused tests first. These use small fixtures and schema checks; they do not load the full model. Tensor arithmetic checks require Metal.

```sh
.venv-local/bin/python -m pytest -q \
  tests/test_local_weights.py tests/test_local_gated_delta.py \
  tests/test_local_qwen_runner.py tests/test_local_state_codec.py \
  tests/test_local_store.py tests/test_local_cases.py
```

A shorter smoke run can use four cases and one occupancy scenario:

```sh
.venv-local/bin/python scripts/benchmark_local_kvpilot.py \
  --output results/qwen-kvpilot-smoke-new \
  --case-limit 4 --repeats 1 --scenarios pressure \
  --policies recompute group_int4 adaptive_eviction
```

The full default workload uses eight controlled repository questions, two repetitions, two capacities and five policies, producing 160 resumed responses:

```sh
.venv-local/bin/python scripts/benchmark_local_kvpilot.py \
  --output results/qwen-kvpilot-run-new \
  --repeats 2 --scenarios roomy pressure \
  --policies recompute raw_native group_int4 adaptive_exact adaptive_eviction \
  --chunk-size 128 --max-tokens 32
```

Choose a new output directory for each run; the script rejects an existing directory. It captures source/fixture hashes before calibration, warms the model, calibrates export/encoding and decoding/import on separate text, and checks native snapshot continuation against the original live state. It writes a protocol with source hashes, package versions, job order, budgets and calibration before the timed workload. Source/fixture changes are checked before that protocol and again at completion; a mismatch invalidates the run.

The workload pauses four sessions, then resumes them in order `0, 2, 1, 3`. Every session eventually resumes. The roomy capacity fits four native snapshots; the pressure capacity is constructed around three INT4 snapshots. These deliberately chosen occupancy boundaries test behavior under a limited store. They are not sampled production traffic, and they do not test cancellation prediction or repeated long-lived agent conversations. Repetitions share the same small synthetic repository and questions; 160 responses are not 160 independent tasks. Policy/scenario blocks are shuffled with a fixed seed, while case order stays fixed.

Interpret the output fields as follows:

- `calibration.json` contains measured codec/export/import samples, derived policy profiles and the native-continuation integrity check. Transfer cost is effectively zero in policy scoring on this unified-memory experiment; no PCIe or CXL delay is simulated.
- `rows.jsonl` retains every response, selected action, alternative predictions, actual victims, timing components, expected answer, correctness, generated tokens and sampled engine memory. Initial prefill, pause work, restoration/recomputation and continuation remain distinguishable.
- `blocks.json` contains store occupancy and allocation counters for each policy/scenario/repetition. `peak_paused_payload_bytes` counts complete retained serialized bytes, including scales and metadata. `byte_event_sum / event_count` is mean occupancy after lifecycle events, not time-weighted memory.
- `summary.json` aggregates the raw rows. Exact answer matching is case-sensitive and normalizes whitespace only; truncated answers fail. Token agreement compares each response with the same case/repetition under recomputation, when that baseline is present. A response can agree with a wrong baseline answer, so agreement and correctness are separate metrics.
- `model-provenance.json` records the local model artifacts and execution conversions. `fixture.json` preserves the input cases and `protocol.json` defines the run; keep both with the raw records and summary.

Independently check a completed archive with the standard-library validator. It does not execute the model or import the benchmark/codec implementations:

```sh
.venv-local/bin/python scripts/validate_local_results.py \
  results/qwen-kvpilot-run-new \
  --output results/qwen-kvpilot-run-new/validation.json
```

The validator rejects incomplete or changed archives, recomputes answer scores and aggregates, and reconstructs framed sizes, LRU residency, hits and occupancy from the recorded events. It cannot re-decode payload contents that were not archived or authenticate elapsed timings. The actual [measured report](local-results.md) keeps those limits explicit.

Store byte counters exclude weights, live state, Python objects, codec scratch space, an incoming encoded payload before commit, and a returned payload during restoration. MLX active/cache/peak allocation counters are reported separately; they do not measure CPU NumPy/Python allocations or total process/system memory. Old resident payloads and the new payload can coexist while encoding, so the store capacity is not a bound on transient process memory. The latency totals include real export/encoding/restore costs and state release; they should be considered together with resume TTFT when deciding whether saving a snapshot was worthwhile.

This benchmark can demonstrate actual state compression and resumption and compare these policies on this machine and fixture. It cannot establish CUDA or production-serving throughput, long-term cognitive memory, general language-model quality, customer demand, or a memristor advantage. Compare `adaptive_eviction` with `adaptive_exact` to examine the eviction penalty under matched candidate sizes and victim order. An adaptive-versus-INT4 difference alone does not isolate that effect, and identical selected actions provide no evidence that the penalty improved this workload.
