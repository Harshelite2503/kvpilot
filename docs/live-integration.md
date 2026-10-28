# Live GPU integration: status

See [`docs/phase1-report.md`](phase1-report.md) for the full, detailed
record of Phase 1's approach, every command run, every failure and fix, and
the complete results. This file is the short status summary.

This tracks real-GPU validation of KVPilot's decision policy against vLLM —
the work the main README's "Next two weeks" section calls out as not yet
started ("Inspect and pin one compatible vLLM/LMCache integration..."). This
file states only what has actually run, not the intent; see the repository's
own eviction/Ollama docs for the established style this follows.

## What exists now (built, zero GPU cost, runs on any machine)

- [`src/kvpilot/live/calibration.py`](../src/kvpilot/live/calibration.py):
  `measure_codec()` times kvpilot's real `codecs.encode`/`decode` round trip
  on a real array and reports real bytes, real wall-clock ms, and the
  codec's real error. `.to_profile()` turns that into a
  `kvpilot.policy.CodecProfile` built from measurements, never from the
  illustrative numbers in `examples/assumed-costs.json`. Unit tested in
  [`tests/test_live_calibration.py`](../tests/test_live_calibration.py)
  against plain NumPy arrays — no GPU or vLLM needed to exercise this.
- [`src/kvpilot/live/recorder.py`](../src/kvpilot/live/recorder.py):
  `LiveRecorder` accumulates real pause/resume/cancel outcomes (measured
  delay, real transferred bytes, real cache hit/miss) into the same
  `metrics`/`decisions` shape as `kvpilot.replay.replay()`'s output, labeled
  `evidence_type: live_measured_session` and `measured_*` metric names so it
  is never confused with the modeled replay output it is shaped to sit
  next to. Unit tested in
  [`tests/test_live_recorder.py`](../tests/test_live_recorder.py).
- [`src/kvpilot/integrations/vllm_connector.py`](../src/kvpilot/integrations/vllm_connector.py):
  a `KVConnectorBase_V1` subclass scaffold that wires calibrated profiles, a
  `CostModel`, and a `LiveRecorder` together, and raises a clear
  `ImportError` if vLLM is not installed.

## Phase 1 result: real GPU prefix-cache baseline + real codec calibration (done)

Ran on a rented Vast.ai on-demand RTX 3090 (24GB VRAM), vLLM 0.32.0,
Qwen2.5-0.5B-Instruct, CUDA 13.0 / torch 2.13.0. Total cost: ~$0.35 of a $5
prepaid balance.

**Prefix-cache baseline** (same shared-prefix-repeated methodology as the
existing `docs/ollama-results.md`): a cold request against a ~1,600-token
shared prefix took 1490 ms to first-complete; an identical second request
hit vLLM's native prefix cache and took 604 ms — a real 2.47x speedup. (A
second run with the engine forced in-process, needed only to get Python
access to the raw KV tensors, measured 634 ms cold / 499 ms warm; the
multiprocess number above is the one representative of a real deployment.)

**Real KV tensor capture**: vLLM's V1 engine runs the model in a child
process by default, so the raw KV cache isn't reachable from a normal `LLM`
handle. Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps it in-process,
reachable at
`llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.kv_caches`
(a list of 24 per-layer tensors, each `[num_blocks, 2, num_heads, head_dim]`
— here `[56139, 2, 16, 128]`, bfloat16). That tensor is vLLM's whole
preallocated pool; only the rows written by an actual prompt are real
data (everything else is unwritten zero padding), so the capture script
masks for nonzero rows and saves just that slice. Captured 153 real
block-rows (`(153, 2, 16, 128)`, 99.9% nonzero, values ranging -130 to
121.5) from the same ~1,600-token prefix above.

**Real calibration** (`kvpilot.live.calibration.measure_codec`, run locally
on this dev machine against the downloaded real tensor — pure CPU/NumPy,
needs no GPU): saved to
[`results/measured-costs-v1.json`](../results/measured-costs-v1.json).

| Codec | ratio: measured vs. assumed | encode ms/MiB: measured vs. assumed | decode ms/MiB: measured vs. assumed | max abs error |
|---|---|---|---|---|
| raw_fp16 | 1.00 vs. 1.00 | 6.45 vs. 0.00 | 2.75 vs. 0.00 | 0 (lossless) |
| zlib_fp16 | 3.25 vs. 1.08 | 28.25 vs. 20.00 | 5.62 vs. 4.00 | 0 (lossless) |
| group_int8 | 1.94 vs. 1.94 | 15.78 vs. 0.15 | 8.12 vs. 0.08 | 0.51 |
| group_int4 | 3.76 vs. 3.76 | 17.06 vs. 0.25 | 13.51 vs. 0.15 | 9.28 |

**Headline finding**: the assumed *compression ratios* in
`examples/assumed-costs.json` were basically right for `group_int8` and
`group_int4` (near-exact match), and real KV data compresses with zlib
much better than assumed (3.25x vs. the assumed 1.08x). But the assumed
*encode/decode timings* for the quantized codecs were off by roughly
**100x** on the optimistic side (e.g. int4 decode assumed at 0.15 ms/MiB,
measured at 13.5 ms/MiB on real data) — because `examples/assumed-costs.json`
was illustrative, and kvpilot's actual reference codecs
(`src/kvpilot/codecs.py`) use an unvectorized per-group Python loop, not a
GPU kernel. Any `choose_action` decision driven by the assumed numbers
would have badly underestimated real compression overhead and likely
over-selected compression. The `group_int4` max absolute error of 9.28
(against a real value range of roughly -130 to 121.5, so ~3.7% of range)
is also the first real signal that lossy restore isn't free quality-wise —
exactly the risk Phase 3's task-quality check exists to catch.

## What still does not exist

No decision from `kvpilot.policy.choose_action` has been applied to a real
model's KV cache yet — only measured and compared to the assumed numbers.
The vLLM lifecycle methods a connector must implement
(`update_state_after_alloc`, `wait_for_layer_load`, `wait_for_save`,
`request_finished`, `build_connector_meta`, `start_load_kv`) are still not
implemented, for the same reason as before: writing them without the real
API in front of us risks code that looks finished but silently does
nothing.

## Next steps

1. Implement the real `KVConnectorBase_V1` lifecycle methods against the
   actual installed vLLM 0.32.0 API (now confirmed reachable, see above),
   driven by the existing synthetic workload generator
   (`scripts/make_demo_trace.py`) replayed as real HTTP requests, recorded
   through `LiveRecorder`.
2. Add the task-quality exact-answer check (same pattern as
   `examples/ollama_quality_cases.json`) comparing lossless vs. lossy
   (INT8/INT4) restore before trusting any lossy result — the measured
   int4 error above makes this more urgent, not less.
3. Consider vectorizing `codecs.py`'s group loop before any live
   latency-sensitive use, given the ~100x gap between assumed and measured
   encode/decode timing found above.

No external implementation code was copied for any of the above.
