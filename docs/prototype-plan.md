# KVPilot: a 14-day experiment

**Decision to make:** can an online policy for retaining, compressing and offloading paused coding-session KV caches improve resume latency at a fixed resource budget, beyond strong existing cache policies?

**Current execution scope: local preparation without a GPU.** This is a plan for validating the prototype. A working CPU codec benchmark or trace simulator is not evidence of a working vLLM connector or a serving speedup. Use the evidence labels and gates in [evaluation.md](evaluation.md). No GPU purchase, rental, cloud deployment or paid model call is authorized by this plan. The live-serving phase below remains a later validation path.

## Product hypothesis and scope

The initial customer is a team operating its own open-model coding agents. Agents pause for tools or people, resume, and branch from shared context. The product would choose which inactive cache blocks deserve scarce memory and which representation is economical for their expected next use. It would expose the decision, avoided recomputation and full costs to the operator.

The two-week local experiment uses CPU reference codecs and analytical trace replay. Its follow-on live trial targets one dense 4–8B model, one GPU, one host-memory tier and optionally a local SSD. It keeps the model, tokenizer and adapter version fixed within a trial. It does not change attention semantics, train the model, introduce memristors or promise longer cognitive memory.

This space has substantial incumbents. [LMCache](https://github.com/LMCache/LMCache) already provides tiered offload, reuse, management APIs and extensible serialization. [CacheGen](https://arxiv.org/abs/2310.07240) already studies compressed KV transfer and adaptive compression. [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) already shares matching prefix blocks. Merely exposing compression or CPU offload is not differentiation.

Session awareness is not novel either. [TensorRT-LLM cold-page compression](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog30_KV_Cache_Compression_in_TensorRT-LLM.html) stores inactive attention pages in NVFP4 and fuses conversion with movement. [Adaptive KV Retention for LLM Agents at Human-Approval Timescales](https://arxiv.org/abs/2608.30830) studies uncertain waits and opportunity-cost-based retention. [Dynamo ThunderAgent](https://docs.dynamo.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler) provides program-level admission and tool-boundary pause/resume. The open [vLLM programmable-cache RFC](https://github.com/vllm-project/vllm/issues/57103) proposes composable lifecycle-aware policies; an RFC is not proof that every interface is shipped.

The remaining product hypothesis is a measurable, portable decision and evaluation layer for a particular customer's traces. It must earn a benefit over these approaches, or provide integration/observability value customers will pay for. The local prototype establishes neither novelty nor that willingness to pay.

## Four hypotheses

| ID | Testable hypothesis | Evidence that would reject it |
|---|---|---|
| H1 | Past session activity and an explicit pause signal predict useful reuse better than recency alone. | Gains disappear on held-out session families or when future-trace information is removed. |
| H2 | Selective compression saves enough transfer or residency cost to cover encoding, decoding and contention. | The same codec with a simple static threshold matches the policy; CPU/GPU work erases savings. |
| H3 | Protecting physically shared prefixes helps branching sessions at fixed bytes. | A fair reference-counted LRU or existing prefix caching achieves the same result. |
| H4 | The policy can refrain from expensive work when reuse is unlikely or imminent. | No-reuse workloads waste write bandwidth; rapid resumes wait for encoding; active decodes slow down. |

The first policy should be inspectable: estimate reuse from elapsed idle time, prior observed reuse gaps and known active references; compare retention/offload/recompute costs; impose a minimum expected gain and a bounded work queue. Hand-set reuse priors and latency/cost parameters are acceptable in a local experiment only when labeled assumptions and swept over a range. They are not measured predictions. A learned predictor is optional only after simple policies establish a baseline.

## Intended architecture

```text
Agent runner: request / completion / pause / branch / resume events
                         |
                         v
             Versioned block identity and causal ledger
                         |
                  Online policy + cost model
                         |
             Bounded admission and codec work queues
                         |
            LMCache backend/serialization integration
                  /                         \
           host-memory store             optional SSD
                         |
             restore -> compatible vLLM KV blocks

Client token timings + engine metrics + transfer/codec spans -> report
```

The policy sees only events that have occurred. The ledger tracks physical blocks, valid bytes, codec version, tier, reference count, pending work and last access. Blocks used by active requests are protected. A branch adds references to matching prefix blocks rather than copying them. Eviction frees memory only when the last relevant reference or retained copy is removed. Partial final blocks and branch-specific tails remain distinct.

Use exact cache identity: model/weight revision, adapter revision, tokenizer/chat-template revision, position/attention configuration, KV dtype/layout, tenant isolation salt, preceding-prefix hash and block token IDs. A matching text fragment at a different position is not automatically reusable. Changes to weights or adapters invalidate affected KV entries.

Encoding is scheduled after inactivity, with cancellation or bypass if a request resumes. Keep a valid original until a replacement has been committed; charge both copies and staging buffers during overlap. Set queue and worker limits. If a format is incompatible, a checksum fails, or a lookup misses, recompute normally and log the reason. The two-week system must not depend on arbitrary non-prefix KV reuse.

## Integration research, checked 27 October 2026

LMCache's [current quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) recommends a separate MP server connected through `LMCacheMPConnector`. It documents dynamic loading via `lmcache.integration.vllm.lmcache_mp_connector` for vLLM versions supporting that path. Older `LMCacheConnectorV1` examples describe a different deployment mode. Pin a tested pair and record the connector class actually imported; do not mix configuration snippets across modes.

The connector configuration boundary is `KVTransferConfig` / `--kv-transfer-config`, including connector name, role and optional module path. The package's planned policy adapter should sit behind a small interface for lookup, admission, store, retrieve and completion events. Identify the exact backend/serialization hook in the pinned source before implementing it. The public docs do not by themselves prove that arbitrary per-block codec switching is available in the chosen MP release.

[LMCache MP request metadata](https://docs.lmcache.ai/mp/configuration.html) documents `kv_transfer_params` entries prefixed with `lmcache.`. Use supported request tags to correlate sessions if available; maintain an external request-to-session ledger otherwise. A tag is not a guarantee of tenant isolation or an eviction-policy hook.

[LMCache compression docs](https://docs.lmcache.ai/kv_cache_optimizations/compression/index.html) list CacheGen. Its compatibility with the selected MP mode, GPU, KV dtype and backend must pass a smoke test. An unavailable CacheGen arm is explicitly missing evidence; an INT8-plus-zlib proxy must never be named CacheGen.

Observability should combine the client's streamed-token timestamps with [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics/) and [LMCache observability](https://docs.lmcache.ai/mp/observability/index.html). MP mode has metrics, logs and optional tracing; endpoint location depends on launch mode. Save a metrics inventory after traffic and inspect actual series, rather than assuming current documentation matches an installed release. Enable detailed traces for diagnostic runs, then measure their overhead.

## Hardware and spending boundary

The local CPU path needs no GPU: run codec round trips, benchmark CPU work, and replay synthetic events. Label its inputs as synthetic tensors unless they were captured from a named model. Apple Silicon results do not establish CUDA serving behavior.

For a live trial, an existing supported CUDA GPU with roughly 24 GB VRAM is a possible starting point for a 4B model or a carefully configured 7–8B model with limited context/concurrency. It is not a fit guarantee. A 48 GB GPU gives more headroom for BF16 weights, KV blocks and temporary buffers. Host memory of 32–64 GB is a practical starting envelope, but cap cache use explicitly and reserve memory for the OS and runner. Use local storage only if it already exists.

Estimate before loading: BF16 weights need approximately two bytes per parameter, and a conventional KV cache needs `2 × layers × KV_heads × head_dim × tokens × bytes_per_element` per distinct sequence, before alignment, metadata and temporary buffers. Read dimensions from the exact model config; grouped-query attention and sharing matter. Measure a one-request load, then increase context and concurrency until the preset budget is reached. Keep the same weight precision in every comparable arm; FP8 KV and weight quantization are different changes.

With no GPU available, finish the CPU and simulation deliverables and prepare the live-run manifest fields. Pin exact versions once a compatible GPU environment is selected. Mark the serving hypothesis untested; this local preparation incurs no GPU rental or paid inference.

## Day-by-day execution

| Day | Work and concrete output | Gate |
|---|---|---|
| 1 | Freeze workload, metric definitions and baseline matrix. Inventory available hardware and dependency compatibility. Create an experiment manifest. | No claimed serving result without a live engine; no spend. |
| 2 | Define request/session/block events and separate the policy-visible stream from evaluator labels. Create tiny hand-audited branch and eviction traces. | Future-trace mutation cannot change earlier decisions. |
| 3 | Run actual CPU codec round trips and timed encode/decode on several tensor sizes and distributions. Save bytes, error and timings. | Codec names, input provenance and measurement units are explicit. |
| 4 | Implement LRU, no-offload/recompute and static compression baselines in the simulator. Account for shared blocks and bounded queues. | Resident bytes never exceed budget; shared-prefix accounting is correct. |
| 5 | Implement a conservative adaptive admission/representation policy using historical features. Freeze its first configuration. | Every decision has a reason and a charged cost. |
| 6 | Replay held-out workloads: no reuse, rapid return, long pauses, branches and memory pressure. Sweep bandwidth and codec cost assumptions. | Reject a policy that wins only with zero codec cost or advance knowledge. |
| 7 | Inspect current integration sources; prepare a pinned candidate vLLM/LMCache manifest and smoke-test recipe. | A source-reviewed recipe is labeled unexecuted without a GPU. |
| 8 | Sweep finite host capacity, transfer rates, codec costs and recomputation estimates in the simulator. | Publish assumed versus CPU-measured parameters; do not invent GPU calibration. |
| 9 | Map FP8, CacheGen, cold-page NVFP4 and retention comparators to future live arms. Compare only actual local codecs locally. | Proxy codecs are named proxies; missing incumbent runs stay missing. |
| 10 | Specify the adapter boundary and validate event normalization/accounting locally. Inspect queue and cancellation edge cases. | No claim of live connector integration from a contract or stub. |
| 11 | Execute paired simulation repetitions and CPU benchmark repetitions; report no-reuse and rapid-resume losses. | Simulated TTFT remains a modeled estimate, not a measured request latency. |
| 12 | Prepare coding-task quality fixtures and identity/corruption cases for the live phase. Run locally testable invariants. | No model-quality score exists until a model actually generates outputs. |
| 13 | Freeze tuning and replay the final held-out synthetic workload. Publish sensitivity and negative cases. | No selecting the best test seed, codec or workload after seeing results. |
| 14 | Write the evidence table and go/no-go memo; deliver the local demo and an exact follow-on live validation checklist. | Separate local completeness from untested serving hypotheses. |

The timetable is a sequence of bounded local experiments, not a promise of two-week serving results. If suitable GPU access is later authorized, the live sequence is: verify cold/hot-prefix retrieval and instrumentation; establish B0–B4 under matched budgets; implement and smoke-test the actual policy hook; measure codec/transfer contention; run paired performance trials and coding-quality tests; then apply the live gates in [evaluation.md](evaluation.md). Do not compress this sequence into an unverified demo.

## Delivery and startup decision

Deliver executable local demonstrations, machine-readable results, an explicit integration status, the workload definitions, a pinned live-run manifest when exercised, and a decision memo. Keep raw traces and run-level results so reviewers can reproduce denominators. Distinguish mechanisms implemented in the local package from planned production integration.

A two-week local success is a reproducible causal harness, measured CPU codec results, honest parameter sensitivity and a live protocol capable of disproving the idea. The subsequent live target is at least 20% lower observed p95 resume TTFT and 15% lower time-weighted stored KV bytes than a competitive matched-budget baseline on a preregistered paused/resumed workload, with uncertainty and the guardrails in [evaluation.md](evaluation.md). These are targets, not current results. A gain only over an uncompressed baseline, or only in simulation, supports further engineering but not a competitive performance claim.

If a static policy or incumbent matches the result, stop calling the policy a product differentiator. The report or integration may still be useful. If a meaningful gain survives, the next step is a customer's anonymized trace and a paid-pilot conversation about measured latency/capacity value; neither outreach nor deployment is performed automatically by this plan.
