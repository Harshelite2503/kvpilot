# Live GPU integration: status

See [`docs/phase1-report.md`](phase1-report.md) for the full, detailed
record of Phase 1's approach, every command run, every failure and fix, and
the complete results. This file is the short status summary.

This tracks the vLLM integration path. Phase 1 measured native vLLM request
timings and CPU codec costs on a captured pool slice; it did not apply
KVPilot decisions to a live vLLM cache. The separate [owned MLX experiment](local-results.md)
already tested KVPilot's own local state path. The [P100 preparation](p100-runner.md)
uses an owned PyTorch runner. Its completed real-cache results are in the
[P100 record](p100-results-2026-10-28.md), including lossy INT4 restoration and
answer checks. The new [admission/equal-work suite](p100-admission-experiment.md)
continues that owned-runner path. Neither is a working vLLM integration.

## What exists now (built, zero GPU cost, runs on any machine)

- [`src/kvpilot/live/calibration.py`](../src/kvpilot/live/calibration.py):
  `measure_codec()` times kvpilot's real `codecs.encode`/`decode` round trip
  on a real array and reports real bytes, real wall-clock ms, and the
  codec's error against an FP16 reference. Source conversion error is reported
  separately by the current implementation. `.to_profile()` turns that into a
  `kvpilot.policy.CodecProfile` built from measurements, never from the
  illustrative numbers in `examples/assumed-costs.json`. Unit tested in
  [`tests/test_live_calibration.py`](../tests/test_live_calibration.py)
  against plain NumPy arrays — no GPU or vLLM needed to exercise this.
- [`src/kvpilot/live/recorder.py`](../src/kvpilot/live/recorder.py):
  `LiveRecorder` accumulates caller-supplied pause/resume/cancel observations into the same
  `metrics`/`decisions` shape as `kvpilot.replay.replay()`'s output, labeled
  `evidence_type: live_measured_session` and `measured_*` metric names.
  Review found that its original pause method incorrectly counted decision
  estimates as measurements. The current method requires explicit measured
  encoding time and transferred bytes; the recorder cannot itself verify
  that the caller timed real work. Unit tested in
  [`tests/test_live_recorder.py`](../tests/test_live_recorder.py).
- [`src/kvpilot/integrations/vllm_connector.py`](../src/kvpilot/integrations/vllm_connector.py):
  a `KVConnectorBase_V1` subclass scaffold that wires calibrated profiles, a
  `CostModel`, and a `LiveRecorder` together, and raises a clear
  `ImportError` if vLLM is not installed. Its lifecycle hooks are not implemented;
  this is not a functional cache connector.

## Phase 1 result: real GPU prefix-cache baseline + real codec calibration (done)

Ran on a rented Vast.ai on-demand RTX 3090 (24GB VRAM), vLLM 0.32.0,
Qwen2.5-0.5B-Instruct, CUDA 13.0 / torch 2.13.0. Total cost: ~$0.35 of a $5
prepaid balance.

**Native first/repeated-request observation**: a first request against a
~1,600-token prefix took 1490 ms and an identical second request took 604 ms,
an observed **2.47x first/repeat latency ratio** with native vLLM prefix
caching enabled. These are request-completion timings, not measured TTFT.
KVPilot controlled neither request's cache. Two sequential requests do not
separate cache reuse from warmup and order effects or establish production
performance. An in-process run, used for tensor access, recorded 634 ms and
499 ms (1.27x). The difference between the two ratios is not causally
attributed to process overhead.

**Real KV tensor capture**: vLLM's V1 engine runs the model in a child
process by default, so the raw KV cache isn't reachable from a normal `LLM`
handle. Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps it in-process,
reachable at
`llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.kv_caches`
(a list of 24 tensors; observed first-layer shape `[56139, 2, 16, 128]`,
bfloat16). Axis meanings were not established from the backend's layout
descriptor and should not be guessed from these dimensions. The tensor is
the preallocated pool. The capture used a **nonzero-row heuristic**, selecting
153 rows (`(153, 2, 16, 128)`, 99.9% nonzero, values -130 to 121.5).
Nonzero is not proof of ownership by the target request: warmup/stale rows
can be included and valid zero rows omitted. Unused storage is not guaranteed
to be zero. Exact request block-table mapping remains unverified.

**Real calibration** (`kvpilot.live.calibration.measure_codec`, run locally
on this dev machine against the downloaded real tensor — pure CPU/NumPy,
needs no GPU): saved to
[`results/measured-costs-v1.json`](../results/measured-costs-v1.json).
The source file was float32; all codecs used an FP16 reference. Ratios and
per-MiB timings use that FP16 byte count. The RTX 3090 label identifies the
capture's origin, not the machine executing compression. The artifact lacks
a complete host/runtime/source-hash provenance record.

| Codec | ratio: measured vs. assumed | encode ms/MiB: measured vs. assumed | decode ms/MiB: measured vs. assumed | max abs error |
|---|---|---|---|---|
| raw_fp16 | 1.00 vs. 1.00 | 6.45 vs. 0.00 | 2.75 vs. 0.00 | 0 (vs. FP16 reference) |
| zlib_fp16 | 3.25 vs. 1.08 | 28.25 vs. 20.00 | 5.62 vs. 4.00 | 0 (vs. FP16 reference) |
| group_int8 | 1.94 vs. 1.94 | 15.78 vs. 0.15 | 8.12 vs. 0.08 | 0.51 |
| group_int4 | 3.76 vs. 3.76 | 17.06 vs. 0.25 | 13.51 vs. 0.15 | 9.28 |

**Finding**: this CPU reference implementation was much slower than the
illustrative codec costs (INT4 decode 13.5 vs. 0.15 ms/MiB). This is a
host-implementation comparison, not a measured GPU-codec result. INT8/INT4
ratios largely follow format and shape; zlib's 3.25x ratio applies to this
one heuristically selected array. The 9.28 INT4 numerical error does not
by itself measure answer quality. Replaying CPU codec costs with otherwise
assumed transfer/recompute costs changed the policy on the development
trace; that is a modeled sensitivity check, not live GPU validation.

Review also found that the original calibration inferred `lossless` from
zero error on a single sample. Current code instead classifies the codec
semantically: only raw/zlib FP16 are lossless relative to FP16, and neither
promises preservation of arbitrary FP32/BF16 source values. Historical JSON
numbers are retained, not silently relabeled as a new experiment.

## What still does not exist

No decision from `kvpilot.policy.choose_action` was applied to a live vLLM
cache in Phase 1. This statement does not describe the separate MLX experiment.
The vLLM lifecycle methods a connector must implement
(`update_state_after_alloc`, `wait_for_layer_load`, `wait_for_save`,
`request_finished`, `build_connector_meta`, `start_load_kv`) are still not
implemented, for the same reason as before: writing them without the real
API in front of us risks code that looks finished but silently does
nothing.

## Next steps

1. Run the new [admission/equal-work protocol](p100-admission-experiment.md)
   on the available P100. The earlier owned P100 run is already completed and
   preserved; this next run adds a simple baseline and controls answer length.
2. For a later RTX 3090 serving test, implement the real `KVConnectorBase_V1` lifecycle methods against the
   actual installed vLLM 0.32.0 API (now confirmed reachable, see above),
   driven by the existing synthetic workload generator
   (`scripts/make_demo_trace.py`) replayed as real HTTP requests, recorded
   through `LiveRecorder`.
3. Add the task-quality exact-answer check (same pattern as
   `examples/ollama_quality_cases.json`) comparing lossless vs. lossy
   (INT8/INT4) restore in that vLLM integration before trusting any lossy result.
4. Profile the actual selected codec path on the target machine before
   optimizing it or assigning performance costs to a policy.

No external implementation code was copied for any of the above.
