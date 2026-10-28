# KVpilot: prior art, feasibility, and the hypothesis to test

Research cutoff: **27 October 2026**. This review uses primary papers, official documentation, public proposals, and selected upstream source files. It establishes substantial prior art; it does **not** establish a unique invention, production readiness, or customer demand. The accompanying [source register](sources.json) records dates, code and license evidence, and limits for 18 source groups.

**28 October follow-up:** the [first P100 audit](p100-results-2026-10-28.md) and [next-experiment research plan](next-experiments.md) add measured evidence, the missing no-eviction baseline, and direct comparisons with InferCept, Continuum, CacheWise, KIVI and KVQuant. This original review remains the dated background; the follow-up distinguishes proposals from implemented capabilities.

The defensible first project is a reproducible experiment: **can a causal controller choose when and how to retain paused-session KV state more effectively than existing simple policies, after charging for codec work, transfers, memory occupancy, and recomputation?** A useful negative result would prevent building another cache layer whose overhead exceeds its benefit.

## What is already being built

The original idea of compressing cold KV before offload is already implemented or explicitly studied. The following are the closest overlaps, with their maturity distinguished.

| System | Capability relevant to KVpilot | Evidence status on the research date | Consequence |
| --- | --- | --- | --- |
| [TensorRT-LLM cold-page compression](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog30_KV_Cache_Compression_in_TensorRT-LLM.html) | Fuses NVFP4 quantization with offload, and dequantization with reload; targets cold pages from idle sessions and other inactive state. | Official implementation documentation and vendor measurements; not reproduced here and exact release not pinned. | “Compress cold cache before transfer” is not an open invention. |
| [LMCache MP serde](https://docs.lmcache.ai/mp/serde.html) and [storage policies](https://docs.lmcache.ai/mp/l2_storage/index.html) | FP8/TurboQuant L2 encoding, configurable tiering, custom store/prefetch policies, multiple eviction policies. | Documented capabilities plus selected `dev` source inspected. | A new codec registry or generic tier manager alone offers little differentiation. |
| [Adaptive KV Retention for LLM Agents at Human-Approval Timescales](https://arxiv.org/abs/2608.30830) | Retention decisions based on GPU opportunity cost, host retention, expiration, and long human waits. | August 2026 preprint; no independent reproduction here. | Workload economics during pauses is already direct research prior art. |
| [vLLM programmable-KV RFC](https://github.com/vllm-project/vllm/issues/57103) | Session/branch lifecycle, retention, demotion, prefetch, shared ownership, and resource approval contracts. | **Open proposal**, not proof of merged support. | Lifecycle correctness is an active upstream design area. |
| [Dynamo ThunderAgent](https://docs.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler) | Program-aware admission, worker placement, and pause/resume decisions using tier capacity. | Official **experimental** development documentation. | Agent-aware scheduling is already being productized. |
| [kvpolicy](https://github.com/abidedavana/kvpolicy) | Small local-agent library for quantized KV persistence and resumption. | Public MIT-licensed prototype; README performance claims not independently verified. | Even a small persistence library has direct prior art. |

None of these sources proves that all useful policy combinations are solved. Equally, failure to find an exact combination is not evidence of a defensible startup gap. The immediate opportunity could be an upstream contribution or a better evaluation tool. A separate company needs customer evidence and a measurable advantage that survives upstream improvements.

## What compression does—and does not—buy

There are two different execution paths:

1. **Cold storage encoding:** encode a cache while inactive, move/store fewer bytes, then reconstruct ordinary KV before attention. This can improve retained-session capacity and resume latency. Once fully reconstructed, it generally does not reduce the active attention cache footprint.
2. **Attention over compact state:** change the representation consumed by attention, such as quantized keys/values or fewer synthetic tokens. This can reduce active memory traffic, but requires compatible kernels, positions, attention semantics, and quality evaluation.

[KVTC](https://arxiv.org/abs/2511.01815) belongs directly in the cold-cache comparison. [Attention Matching](https://arxiv.org/abs/2602.16284) changes the attended representation. Treating both as interchangeable serialization plugins hides their different integration and correctness costs.

Compression ratio is not concurrency gain. Let `h` be the fraction of a particular cache budget that stays uncompressed and `r` the compression ratio of the rest. Even ignoring metadata, its aggregate compression ratio is `1 / (h + (1-h)/r)`. That says nothing yet about weights, temporary buffers, scheduling limits, or compute saturation. For example, a 20% hot fraction and 10× cold compression yield about 3.57× for that cache budget, not 10× for the server. This is arithmetic, not a performance prediction.

A minimal transfer break-even calculation is also useful. For one unoverlapped offload/reload cycle of `B` bytes over a symmetric effective bandwidth `W`, compressing by `r` saves transfer time `2B(1-1/r)/W`. Codec encode and decode time must fit inside that saving, unless storage pressure or avoided recomputation provides additional value. Real systems need measured queueing, overlap, copy directions, contention, and temporary memory; advertised PCIe bandwidth is not an adequate profile.

Keep three latency metrics separate: first-token latency on a fresh request, first-token latency when a session resumes, and inter-token latency during generation. Improving one does not imply improving the others. A codec consuming GPU time during another request's decode can improve its own transfer while worsening fleet tail latency.

This project does not need memristors. KV caching concerns computed attention state; it is also distinct from persistent user facts, RAG documents, and learned model weights. A future hardware experiment should follow a demonstrated data-movement bottleneck and an end-to-end advantage over digital memory, not be a prerequisite for this software experiment.

## LMCache: the extension points actually inspected

The current MP documentation distinguishes an L1 store from L2 adapters, with asynchronous store/prefetch controllers. It documents LRU, ARC, IsolatedLRU, and noop eviction choices, subject to adapter support. Legacy LMCache documentation instead lists LRU/LFU/FIFO/MRU; these lists must not be mixed when selecting an integration target. [MP storage](https://docs.lmcache.ai/mp/l2_storage/index.html), [legacy policies](https://docs.lmcache.ai/legacy/kv_cache/caching_policies.html).

| Surface | Observed interface or behavior | Implication for a later adapter |
| --- | --- | --- |
| Store policy | `StorePolicy.select_store_targets` can select adapter targets or omit keys; a separate method selects L1 deletions. | Admission and tier selection are already expressible. Inspect actual ownership/lifecycle constraints before wiring a policy. |
| Prefetch policy | `PrefetchPolicy.plan_load` plans against L1/L2 snapshots; `plan_l1_retention` controls retention after loading. | Planning and movement execution are separate; a plan is not completed residency. |
| Serde | FP8 and TurboQuant factories, configurable per L2 adapter; synchronous serializer/deserializer interfaces with asynchronous orchestration. | Start by measuring existing codecs. This interface alone does not prove GPU-side compression before the GPU→host copy. |
| Lazy offload | `LMCacheMPConnector` supports eviction-aware or FIFO deferred GPU→LMCache stores, with block-hash validation and in-flight protection. | Avoiding premature copies is already a baseline; compare against it. |
| Runtime plugins | Subprocess-based runtime extension mechanism. | Do not confuse a monitoring/management subprocess with the tensor codec hot path. |

Selected raw files were read: [store policy](https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/distributed/storage_controllers/store_policy.py), [prefetch policy](https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/distributed/storage_controllers/prefetch_policy.py), [serde exports](https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/distributed/serde/__init__.py), and [factory registry](https://raw.githubusercontent.com/LMCache/LMCache/dev/lmcache/v1/distributed/serde/factory.py). See also [serde documentation](https://docs.lmcache.ai/mp/serde.html), [lazy offload](https://docs.lmcache.ai/mp/lazy_offload.html), and [runtime plugins](https://docs.lmcache.ai/developer_guide/extending_lmcache/runtime_plugins.html).

This was a **partial source inspection**, not a complete concurrency or kernel audit. In particular, TurboQuant exports and documentation were verified, but its complete kernel path was not audited. Rolling `dev` references were not pinned to immutable commit hashes. Pin engine, LMCache, connector, attention backend, and model revisions before claiming a working integration.

[SGLang's HostKVCache plugin RFC](https://github.com/sgl-project/sglang/issues/36522), opened 26 August 2026, also proposes extensible compression across memory tiers, per-page state, and asynchronous loading. It is an open RFC; build against a verified merged API, not an assumed proposal. The vLLM proposal likewise explicitly separates session association, residency, and active references. These are useful design inputs, not completed integration work.

## Method comparison

The rows below identify what should be compared and what cannot be inferred from a headline result. Paper results remain the authors' results; no reported speedup is a KVpilot measurement.

| Method | Mechanism and relevance | Main limitation for the proposed workload |
| --- | --- | --- |
| [CacheGen](https://arxiv.org/abs/2310.07240), SIGCOMM 2024 | Distribution-aware tensor encoding, streaming, and bandwidth-adaptive compression. | Adaptivity is existing prior art. Reproduce the relevant model, link, and quality configuration rather than a static compression-only baseline. |
| [TurboQuant](https://arxiv.org/abs/2504.19874), ICLR 2026 | Low-bit quantization with rotation/error-control techniques; current LMCache offers corresponding serde options. | Bits per scalar omit metadata and kernel costs. A quantized attention-kernel result is not full-server throughput. |
| [KVTC](https://arxiv.org/abs/2511.01815), ICLR 2026 | PCA decorrelation, adaptive quantization, entropy coding, and brief calibration; explicitly targets reusable chat/code caches. | A very close cold-storage competitor. Model-specific calibration, encode/decode overhead, and quality need reproduction. |
| [KVCodec](https://arxiv.org/abs/2602.09725), SIGCOMM 2026 | Uses GPU-native video codec hardware and pipelines remote transfer/restoration. | Requires relevant hardware and layout; CPU timings cannot validate its intended overlap or contention benefit. |
| [Attention Matching](https://arxiv.org/abs/2602.16284), February 2026 | Fits compact K/V state against reference queries, including token-specific attention bias. | High compression is task-dependent. Requires model/kernel integration and fitting work; not a universally safe byte codec. |
| [KVzip](https://arxiv.org/abs/2505.23416), NeurIPS 2025 | Reconstructs context to score and evict KV for reuse across multiple queries. | Query-agnostic reuse is already studied. Scoring cost and supported kernels matter; token removal is lossy. |
| [PyramidKV](https://arxiv.org/abs/2406.02069), June 2024 | Allocates token budgets differently across layers. | Eviction changes information available to later queries. It is not reversible cache storage. |
| [VarRate](https://arxiv.org/abs/2607.15498), July 2026 preprint | Allocates variable low-rank budgets by salience while retaining a nonzero representation of each token. | Keeping every token is not lossless. Published model/task coverage does not establish long-lived coding-agent reliability. |
| [ReadKV](https://arxiv.org/abs/2610.11245), October 2026 preprint | Stores progressive codes and varies fetched precision by query without rewriting stored entries. | Separates retained bytes from read bytes. Its reported latency experiment is a restricted single-layer workload, not a serving system. |
| [CacheBlend](https://arxiv.org/abs/2405.16444), EuroSys 2025 | Reuses non-prefix context KV while selectively recomputing to restore cross-chunk attention. | Context composition rather than general cold persistence. Check engine-version availability; LMCache's FAQ qualifies latest-vLLM availability. |
| [CacheFocus](https://arxiv.org/abs/2502.11101), February 2025 | RAG cache reuse with repositioning, pruning, and positional allocation. | RAG-specific context processing; not evidence that arbitrary cached branches can be replayed unchanged. |
| [CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration](https://arxiv.org/abs/2604.25080), April 2026 | Schedules restoration/recomputation across tokens, layers, and GPUs. | Restoration contention is already a research topic. This is distinct from the unrelated video-generation project named CacheFlow. |

The [Attention Matching repository](https://github.com/adamzweiger/compaction) is a research implementation, not a drop-in vLLM connector. A [Baseten investigation](https://labs.baseten.co/articles/repeated-kv-cache-for-long-running-agents) reports additional degradation from repeated compaction in its experiments, with implementation differences from the original method. In contrast, TensorRT-LLM documents idempotent repeated encode/decode for its specific cold-page NVFP4 representation after the first quantization. Therefore the safe requirement is to **track and test transformation provenance**. Do not assume every lossy format compounds error, or assume any format is harmless without validation.

## A narrow, falsifiable project hypothesis

For a deployment with frequent pauses, limited host/disk capacity, and costly resumes, a controller may improve the cost/latency tradeoff by selecting among already-calibrated representations and deciding when storage work is worth doing. The possible advantage is the quality of a concrete policy under measured workload constraints. Neither the policy category nor any individual mechanism is claimed to be new.

Useful candidate inputs are currently observable pause events, cache size, physical sharing, elapsed pause time, current load, memory pressure, measured bandwidth, codec profiles, and aggregate prior resume/cancellation statistics. An optional predicted wait must be marked as a prediction. Exact future resume/cancel timestamps from the replay trace must never enter an online decision.

The controller should be allowed to keep raw state, use a configured codec, defer a decision, decline storage, evict, or choose recomputation. Otherwise an experiment can artificially favor compression by excluding cheaper alternatives. A prefix shared by several branches must be charged once in physical memory accounting. Canceling one branch releases its association; it does not invalidate another branch's valid prefix.

Preserve immutable encoded records, or recompress from canonical state when available. A decoded lossy cache is not an unmodified canonical cache. Version identity should cover the model/configuration, tokenizer and exact token sequence, position semantics, cache layout, precision/codec, and isolation scope. A hash hit is not proof that a transfer completed or that all required preceding state is available.

## Evaluation gates

### Gate 1: CPU-only policy and accounting experiment

Start with a deterministic event replay and small synthetic tensors. This can test policy causality, byte accounting, finite capacity, deterministic serialization, cancellation, shared-prefix reference counts, and codec-selection break-even behavior. Label profiled CPU codec measurements separately from modeled transfer times. Synthetic distortion is an implementation diagnostic, not proof of language-model quality.

Use at least raw retention, immediate recomputation/no retention, fixed codec, age/LRU or simple TTL, and an adaptive policy. Include a hindsight oracle only as an explicitly labeled upper-bound comparator. Give all online policies the same observations and capacities. Evaluate both favorable and adverse traces: quick resumes, never-resumed sessions, sudden bandwidth changes, sustained pressure, high cancellation, shared branches, and repeated pauses.

Do not describe this stage as a vLLM plugin, GPU benchmark, real 128k agent evaluation, or measured serving savings. Any feature listed here but absent from the implementation remains a planned gate.

### Gate 2: a pinned real-engine experiment

Choose one supported model and one engine/connector version first. Capture actual cache layout and positions. Establish uncompressed behavior, then reproduce an existing codec and simple policy before adding an adaptive controller. Measure total payload plus headers, scales, retained raw copies, staging buffers, and workspace. Confirm whether compression occurs before or after each expensive link.

Record successful task completions per unit time/cost, resume first-token latency, fresh-request first-token latency, inter-token latency, hit rate, recomputation, wasted encoding after cancellations, peak tier memory, and GPU codec/copy contention. Report distributions and confidence intervals across workload repetitions, rather than only a mean or best-case ratio.

Quality tests should include exact rare facts, edits that reverse earlier instructions, varied future queries, long code contexts with executable tests, and several pause/resume/branch cycles. Compare against the same model with full cache and against legitimate recomputation. Establish a quality acceptance threshold before tuning on the evaluation set. A cosine-similarity or perplexity result alone does not establish agent correctness.

### Gate 3: a reason for a product

Obtain representative traces and cost constraints from prospective users, with data handling agreed in advance. Determine whether an existing LMCache setting, TensorRT feature, simple TTL, or the published adaptive-retention baseline already solves the problem. A positive simulated result is insufficient; a positive real-engine result still needs operational value, maintainability, and demand.

Stop or narrow the project if the adaptive controller does not beat a fixed codec/simple retention policy on held-out workloads; if gains disappear after charging codec contention and shared physical bytes; if quality fails the chosen threshold; or if upstream functionality already achieves the customer's objective with less integration work. These are decision criteria, not predicted outcomes.

## Remaining uncertainty

- No GPU serving experiment or full-model quality reproduction was performed for this research review.
- Rolling upstream documents/source were inspected without immutable commit pins; feature availability must be rechecked for the chosen released versions.
- Some papers do not expose an author-maintained implementation in the sources inspected. An unverified code/license field means unknown, not unavailable or unlicensed.
- LMCache's complete TurboQuant path, engine hooks, cancellation races, and cross-tier ownership were not exhaustively audited.
- [CacheBlend availability](https://docs.lmcache.ai/community/faq.html) is qualified by the project's own FAQ; confirm the target engine implementation directly.
- Commercial demand, representative pause distributions, model-specific quality limits, and repeatable end-to-end savings remain unproven.
- A competitor matrix is not an exhaustive patent, freedom-to-operate, or novelty search. No such legal conclusion is made here.

The next useful result is a transparent, reproducible comparison that can fail. The research supports doing that experiment; it does not support promising 10–50× concurrency, lossless arbitrary compaction, or an uncontested startup category.
