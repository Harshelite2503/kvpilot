# Initial findings — 27 October 2026

**Decision: continue a bounded local investigation; no startup-performance claim is supported yet.** We have runnable reference code and a concrete failure to investigate. This is the starting point of the two-week experiment, not its completed outcome.

## What ran

Four original NumPy CPU codecs were exercised on synthetic Gaussian and correlated arrays at 512 KiB and 4 MiB of FP16 data, with two warmups and 30 timed repetitions per codec/input. These are actual local wall-clock measurements on the recorded macOS arm64/Python environment. Timings include Python validation, framing and allocation. There was no isolated CPU reservation, GPU, model forward pass, real KV capture or inference engine.

For the 512 KiB fixtures, serialized grouped INT8 is 270,429 bytes, or about **1.94×** smaller than raw FP16; grouped INT4 is 139,357 bytes, or about **3.76×** smaller. Both include scales and metadata and both are lossy. Raw framing adds 75 bytes. Lossless zlib ratios depend on the input distribution (about 1.08× Gaussian and 1.12× correlated).

This shows that the implementations encode, decode, and expose overhead/error. It does not show that the quantized state preserves coding accuracy, that this is a better codec than existing work, or that the CPU implementation is fast enough for an actual memory path. The single-head random-query diagnostic is deliberately not an LLM quality test.

Raw reports:

- [Gaussian, 512 KiB](../results/codecs-gaussian.json)
- [Correlated, 512 KiB](../results/codecs-correlated.json)
- [Gaussian, 4 MiB](../results/codecs-gaussian-large.json)
- [Correlated, 4 MiB](../results/codecs-correlated-large.json)

## The first policy result is negative

The included seeded fixture has 60 synthetic sessions, two tenants, a 4 MiB shared prefix and 4 MiB distinct tail per session, tool/human pause categories, and explicit resume/cancel events. It is a development fixture, not a production trace. The example supplies a 32 MiB paused-store budget, 35 ms full-recompute estimate, and hypothetical codec/transfer costs. Those costs were **not** calibrated from the CPU measurements.

| Policy | Modeled p95 resume delay | Modeled bytes transferred | Peak paused-store reservation |
|---|---:|---:|---:|
| Recompute | 35.00 ms | 0 | 0 |
| Raw FP16 | 35.00 ms | 549,453,824 | 33,554,432 |
| zlib FP16 | 35.00 ms | 361,176,195 | 31,068,920 |
| Grouped INT8 | 35.00 ms | 317,815,911 | 32,430,195 |
| Grouped INT4 | 1.40 ms | 172,903,585 | 30,118,689 |
| Adaptive | 35.00 ms | 549,453,824 | 33,554,432 |

These are analytical outputs, **not measured TTFTs**. In particular, the apparent INT4 win has no model-quality validation and excludes queueing, GPU contention and staging capacity. It cannot be claimed as a serving speedup.

The adaptive policy favors cheap immediate raw movement under these assumptions. It prices its own occupancy with a fixed penalty, but does not price the future recomputation caused when its admission evicts another session's pages. Fixed INT4 retains more sessions within this small budget. The adaptive policy therefore fails its first competitive test; it is not a stronger product than fixed compression on this fixture.

The fair next question is whether a capacity-aware marginal eviction cost improves decisions on held-out workloads **after** queueing and cancellation are represented. A possible next scoring term charges the predicted recomputation penalty to affected existing sessions, accounting for shared pages. This is a standard opportunity-cost direction to evaluate, not a claimed invention. Compare it with a fixed codec, static idle threshold, TTL and simple LRU before making the policy more elaborate.

See [full decision logs and assumptions](../results/policy-replay.json) and the [exploratory sensitivity cells](../results/sensitivity.json). The sweep uses seeds 7/42/99, host budgets 8/32/128 MiB, and assumed effective bandwidths 1/16/32 GiB/s. These cells are development diagnostics, not a frozen holdout or evidence of a general advantage. All policies receive the same cell inputs.

## What remains untested

| Question | Status |
|---|---|
| Can the CPU reference formats round-trip their intended inputs? | Tested; see the test suite and raw microbenchmarks |
| Does the local analytical controller avoid future-event leakage and obey its paused-store budget? | Tested on deterministic invariants, including shared branches and early resumes |
| Does the adaptive policy beat a fixed codec on the initial pressure fixture? | No |
| Does any lossy path preserve model/coding-task quality? | Not run |
| Is there a functioning vLLM/LMCache connector? | Not implemented |
| Do codec work and transfer overlap improve actual serving economics? | Not run |
| Does the approach beat CacheGen, LMCache settings, TensorRT cold compression or retention research? | Not run |
| Will an inference operator pay for this? | Unknown; no customer outreach performed |

Proceed locally to better cost/accounting and workload evidence. A GPU test becomes useful when the local harness can express the actual bottleneck and a simple baseline no longer explains the apparent gain. Keep the [evaluation contract](evaluation.md) fixed before a final holdout; avoid tuning repeatedly to the same fixture.
