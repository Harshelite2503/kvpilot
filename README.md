# KVPilot

**A local experiment in deciding which paused agent KV caches are worth retaining, in what format, and for how long.**

Status: CPU prototype and research, 27 October 2026. No GPU, live vLLM/LMCache integration, real-model quality result, customer trace, or demonstrated startup advantage yet. The original adaptive policy and its negative result are preserved. The new eviction-aware experiment has a frozen source/config manifest and a paired synthetic holdout protocol; see [eviction results](docs/eviction-results.md).

## The question worth testing

A coding agent pauses for a tool or a person. Keeping its cache consumes scarce memory; encoding it costs compute; restoring it costs bandwidth; dropping it means another prefill. Branches may share most of the same physical cache. Can observable lifecycle events and calibrated costs support better choices than existing policies on a particular customer's workload?

CacheGen already does bandwidth-adaptive KV compression. LMCache already provides tiered storage, compression, policy extension points and lazy offload. TensorRT-LLM already compresses cold pages, and recent research already studies retention during human approval. **Neither compression nor session awareness is our claimed invention.** See the [prior-art review](docs/research.md) and [18-group source register](docs/sources.json).

The candidate product is a measurable policy and diagnostics layer on an existing engine: select a supported representation, decline uneconomic storage, explain the decision, and prove the benefit under a fixed memory budget. The [product hypothesis](docs/product-hypothesis.md) defines the first user and validation sequence. The immediate milestone is evidence that this layer adds value. It may turn out to be an upstream contribution rather than a standalone company.

## What runs today

| Component | Implemented | Boundary |
|---|---|---|
| Reference codecs | FP16 framing, zlib FP16, grouped INT8, packed grouped INT4; decode, metadata and error measurement | NumPy CPU implementations; not CacheGen or TurboQuant |
| Replay | Ordered pause/resume/cancel events; finite paused-store budget; physical shared-prefix accounting; LRU eviction | Analytical, no actual tensors transferred and no concurrent resource queues |
| Policy | Raw/compressed/recompute choices; exact planned LRU victims; expected lost-cache benefit per affected owner; historical resume forecast; decision log | Greedy heuristic, no survival model, future-arrival planning or global SLO guarantee |
| Baselines | Recompute, raw retention, fixed codecs, original `adaptive`, exact-sizing zero-penalty `adaptive_exact` | `adaptive_eviction` is the candidate; these are not actual vLLM/LMCache/TensorRT benchmark arms |
| Experiments | Codec benchmark, sensitivity sweep, seven-family seeded holdout protocol, source freeze and paired whole-run bootstrap | Synthetic assumptions; new event sequences do not establish production generalization |
| Verification | Codec round trips, malformed input, isolation, shared prefixes, cancellation, capacity, early return and causal decisions | Does not establish LLM task quality or production correctness |

The policy rejects lossy profiles by default. The example explicitly enables **unvalidated lossy exploration** so we can study assumed cost tradeoffs. This flag is not quality approval.

## Run locally

Python 3.11+; Python 3.13 was used for the checked-in results. No CUDA dependency or model download is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/make_demo_trace.py
python scripts/compare_policies.py
python scripts/sweep_policies.py
python scripts/benchmark_codecs.py --fixture gaussian --repeats 30 --output results/codecs-gaussian.json
python scripts/benchmark_codecs.py --fixture correlated --repeats 30 --output results/codecs-correlated.json
```

The [dependency snapshot](requirements-dev.lock.txt) records the local versions. To recreate it on a compatible interpreter, install that file before `pip install -e . --no-build-isolation`. Compatibility on other operating systems is untested.

Run one policy with the CLI:

```bash
kvpilot replay --trace examples/synthetic-sessions.jsonl --config examples/assumed-costs.json --policy adaptive --output results/local-adaptive.json
```

An optional `--input your-capture.npy` argument to the codec benchmark accepts a single numeric array. Record the model/layer/token/dtype provenance yourself; the program labels external input as unverified. `.npy` and `.npz` captures are excluded from Git by default.

## Initial evidence

The [initial findings](docs/initial-findings.md) explain the original measurements and the failed first adaptive-policy hypothesis. The CPU INT4 codec produces about **3.76×** smaller serialized data than unframed FP16 for the included 512 KiB fixtures. It is lossy; this is a byte ratio, not an inference speedup or quality result. Current local verification passed **130 tests plus 89 subtests**. The [original run manifest](results/manifest.json) describes files at initial commit `d103494`, not the changed current source. The [eviction freeze](results/eviction-freeze-v1.json) binds the new experiment. Cloud CI has not run.

All policy timings are **modeled from assumptions** in [assumed-costs.json](examples/assumed-costs.json). They do not use the CPU benchmark timings as invented GPU timings. `measured_gpu_itl_ms` and `measured_llm_task_success` remain null. Fully restoring a compressed cache to FP16 does not reduce its active attention footprint.

## Eviction-aware experiment

Read the [design and limits](docs/eviction-design.md), [predeclared workload protocol](examples/holdout-protocol.json), and [results](docs/eviction-results.md). `adaptive_eviction` charges the expected lost restoration benefit to each complete owner harmed by an admission, once per owner. `adaptive_exact` provides a matched sizing control with that charge disabled.

Frozen evaluation: 140 unseen synthetic traces, 33,600 sessions and six policies. Mean run-level p95 improves versus fixed INT4 in five of seven families, but cancellation-heavy traffic regresses and branching is inconclusive. Fast-return/low-pressure wins use 3.76× the stored-byte time and transfer volume and exactly match fixed raw FP16; the matched ablation attributes none of those two wins to eviction pricing. No family meets the original joint latency/memory target. Raw results, intervals and adverse secondary metrics are retained.

Run a new development archive, or reproduce the already evaluated frozen holdout:

```bash
python scripts/evaluate_eviction.py --split development --output-dir results/local-eviction-dev
python scripts/evaluate_eviction.py --split holdout --freeze-manifest results/eviction-freeze-v1.json --output-dir results/local-eviction-reproduction
```

Output directories must not already exist. The holdout command refuses source/config changes; it never silently refreshes a freeze. Rerunning the frozen experiment is reproduction, not a new unseen test. After algorithm changes, declare a new holdout and retain these results.

## Next two weeks

Follow the [14-day prototype plan](docs/prototype-plan.md), [evaluation contract](docs/evaluation.md), and [implemented trace contract](docs/implementation.md). The plan separates local preparation from later GPU validation.

1. Validate accounting and profile codecs on a wider range of inputs, including actual KV captures when available.
2. Add bounded movement/codec queues, refine eviction forecasts and compare simple TTL/static-threshold policies. Any new algorithm needs a new declared holdout after tuning on the revealed results.
3. Inspect and pin one compatible vLLM/LMCache integration, then prepare the live measurement and coding-quality protocol.
4. Publish negative cases and decide whether a GPU experiment is justified.

Later live target: at least 20% lower p95 resume first-token latency and 15% lower time-weighted stored bytes versus a strong compatible baseline, with ITL and task-quality guardrails. These are decision thresholds, not forecasts or achieved results. If a simple static policy wins, the controller has not earned its complexity.

No external implementation code was copied. This private prototype has no open-source license selected yet.
