# KVPilot: a sellable next experiment

Research checked **29 October 2026**. This is a proposed product and validation plan,
not implemented functionality, measured customer savings, or a promise of funding.
It supersedes the implementation priorities in earlier strategy notes where they conflict.

## What the evidence now permits

The [P100/RTX replication](policy-replication-2026-10-29.md) changes the priority:

- On the balanced equal-work fixture, adaptive took **3.66% more time than simple
  native admission on P100** and **0.29% more on RTX**. These are descriptive results.
- No GPU-memory reduction was measured. Smaller serialized host payload is a different metric.
- Adaptive selected native snapshots or recomputation; **no adaptive lossy codec was exercised**.
- Adaptive and simple admission preserved recompute's free-generation tokens on the observed
  fixture. There were 24 underlying questions, not hundreds of independent quality samples.
- Beating our slow CPU INT4 implementation establishes a tradeoff in that implementation;
  it does not establish a superior production scheduler or codec.

Keep simple admission as the default reference. The adaptive policy has not earned deployment.
The useful asset is an auditable experiment harness and a way to reject bad cache decisions.

## Recommendation: sell a deployment audit before a controller

The first offer should be **a fixed-scope cache and resume-cost audit for a self-hosted
agent deployment**, followed by a customer-run validation of one recommended change.
Deliver a reproducible diagnosis, an actionable configuration or admission recommendation,
and the evidence needed to accept or reject it. A report saying “keep your current settings”
is legitimate; a report claiming savings from uncalibrated replay is not.

The hypothesis is that a platform team will pay to identify which changes actually reduce
its cost per successful agent task, without spending weeks correlating harness and engine logs.
Begin as an engineering service with reusable software. Recurring software is justified only
if customers need this whenever workloads, models, engine versions or capacity change.

KVPilot would sit beside the customer's agent harness and inference deployment. It would sell
measurement and validated decisions about retained state, rather than route purchases among
model providers. A customer using only a closed model API cannot expose its provider's KV manager.

## Existing work already covers much of the mechanism

All sources below were accessed on **2026-10-29**. Documentation is version-dependent;
pin the actual customer engine versions before promising that an operation is supported.

| Primary source | Verified capability or research result | Consequence for KVPilot |
| --- | --- | --- |
| [LMCache controller](https://docs.lmcache.ai/kv_cache_management/controller.html) | Documents lookup, move, pin, clear and compression control APIs. | A cache-control API by itself is not differentiation. |
| [LMCache lazy offload](https://docs.lmcache.ai/mp/lazy_offload.html) | Documents deferred offload with an eviction-aware default policy; currently specific to vLLM's MP connector and disabled by default. | Benchmark the applicable native option before recommending an extra controller. |
| [Dynamo agent hints](https://docs.nvidia.com/dynamo/agents/agent-hints) | Carries priority, strict priority, output-length estimates and speculative-prefill intent. Its support table lists priority eviction for SGLang and planned support for vLLM/TensorRT-LLM. | Do not advertise interchangeable cache semantics across engines. |
| [Dynamo agent trace replay](https://docs.nvidia.com/dynamo/agents/agent-simulation) | Documents metadata capture, live AIPerf replay and offline DynoSim; it distinguishes simulation from final live measurements. | A generic trace collector, replay dashboard or common hints SDK is already occupied territory. |
| [InferCept, ICML 2024](https://proceedings.mlr.press/v235/abhyankar24a.html) | Studies wasted work during augmented-model interruptions and serving support for those interruptions. | Tool-wait memory management and recomputation avoidance are established prior art. |
| [MORI, May 2026 preprint](https://arxiv.org/abs/2606.00866v1) | Ranks programs by relative idleness, partitions GPU/CPU residency and applies admission control. | Compare an idleness baseline; a binary “paused” flag is insufficient. |
| [CacheWise, June 2026 preprint](https://arxiv.org/abs/2606.16824v1) | Combines prefix-aware scheduling with reuse-aware eviction informed by tool metadata. | Tool-aware retention is direct competition, not an unclaimed invention. |

Dynamo's [agentic-inference documentation](https://docs.nvidia.com/dynamo/dev/digest/agentic-inference)
specifically distinguishes currently supported priority hints from future TTL/token-range
retention directives. An annotation with a familiar name must not be treated as a pinning guarantee.

The [author-linked CacheWise traces](https://github.com/cachewise-project/cachewise-coding-traces)
are a candidate workload input. Inspect applicable data/code licenses and privacy constraints
before use or redistribution; an accessible repository and a paper's license are not a blanket
license for every artifact. Trace replay does not rerun the original agent's decisions or tools.

Our proposed differentiation is narrower: verified deployment-specific recommendations,
low-effort evidence collection, accounting that exposes uncertainty, and regression protection.
This is a buyer hypothesis. Existing upstream tools may already satisfy it for a given team.

## Buyer, pain and a concrete first offer

Target the engineer responsible for latency and capacity of a **self-hosted coding or workflow
agent service**, with a platform lead controlling its infrastructure budget and configuration.
Qualify for frequent multi-turn reuse, tool waits, observable cache misses, and meaningful spend
or a capacity constraint. These conditions must be demonstrated in their deployment.

Do not start with teams whose dominant expense is external API pricing, tool execution,
training, or idle hardware that cannot be resized. Those need a different intervention.

Proposed pilot: one deployment, one week of metadata, one ranked recommendation and one
validation handoff. Test a **$1,000 fixed audit fee** as a willingness-to-pay hypothesis,
not established market pricing. Agree on deliverables; do not guarantee a percentage saving.

Ask five qualified teams, with the founder conducting outreach:

1. Which recent incident or bill made cache reuse or resume latency a priority?
2. Can you show the slow task, cache events and allocated infrastructure behind it?
3. Which native settings or upstream tools have you tried, and what remained unresolved?
4. Who can change the configuration, approve the data export and validate the result?
5. Would you pay for this scoped investigation now, and retain it after the first fix?

Planning targets: two usable trace exports and one paid audit or clearly budgeted pilot.
Interest, GitHub stars and a request for a free report are not purchase evidence.

## What we build first

Start with a **local CLI importing existing metadata**, not a required production proxy.
This avoids introducing a request-path dependency before there is evidence of value.

- A versioned JSONL schema links tenant, session, branch, turn and parent request IDs.
- Record arrival, pause, resume, cancellation and tool-completion times, plus the time
  each fact became observable. Preserve unfinished sessions as censored observations.
- Join input/output lengths, scoped prefix/block fingerprints, hit/miss events, cache
  tier, bytes, transfer work and engine configuration. Scope identities by tenant,
  model, tokenizer, template and cache layout; reject incompatible state reuse.
- Import measured resource profiles and their uncertainty. Missing cache counters
  remain missing; repeated input tokens alone do not prove a particular cache miss.
- Validate coverage, duplicate/out-of-order events, missing terminal events, shared
  prefixes and accounting conservation before allowing a quantitative recommendation.
- Produce one evidence bundle: observed facts, modeled alternatives, assumptions,
  calibration error, candidate change, validation commands and rollback condition.

Keep raw prompts, responses and tool arguments out of the default export. Fingerprints
still reveal relationships and need tenant scoping and customer-approved handling.
An engine adapter initially reads logs or documented telemetry; it does not own live state.

All **reference-runner cache management remains ours**: serialization, storage, admission,
eviction and restoration. Do not silently substitute LMCache or an engine's cache as a
KVPilot implementation. Customer-native configurations are comparison targets or recommendations;
a future live KVPilot cache adapter requires its own correctness and compatibility work.

## Economics that can be tested

Report three separate quantities:

1. **Observed avoidable work:** measured repeated-prefill, snapshot and restore work,
   with attribution limits. Overlapping requests prevent summing request wall time
   into billed GPU-seconds.
2. **Modeled opportunity:** replayed resource demand under explicit service-time,
   concurrency and cache assumptions, with sensitivity bounds and holdout error.
3. **Realized economics:** a customer-run change meets task-quality and latency targets
   while reducing allocated resource-hours, or serving more successful tasks at the same cost.

Use `cost per successful task = allocated infrastructure and operating cost / successful tasks`.
For a proposed change, net benefit includes changes in GPU, CPU/DRAM, storage and transfer costs,
less KVPilot fees, integration and operational overhead. Use the customer's effective prices.
An illustrative 100 fewer billed GPU-hours at $1/hour is $100 gross, before other costs;
neither number is a measured result or a quoted hardware price.

A fixed cluster with spare capacity may save no cash when a request becomes faster.
On the free P100, reduced work is useful evidence, but its rental-charge saving is zero.
API token counts, byte-time occupancy and sequential benchmark milliseconds do not directly
convert to GPU dollars. Report “unidentified” when the evidence cannot support that conversion.

## Mechanism improvements worth testing

Prioritize the observed failure: extra admissions and replacements without useful extra hits.
Keep the current policy and its failed results as a frozen regression case.

- **Admission margin:** replace an incumbent only when estimated avoided recomputation
  exceeds snapshot, restore, transfer and displaced-owner costs by a conservative margin.
  Costs must use consistent units; do not count transfers twice or price only payload bytes.
- **Uncertainty fallback:** default to simple admission when candidate benefit is within
  profiling noise or reuse estimates are poorly supported. Compare this directly with
  no replacement, TTL, recency/idleness and recompute baselines.
- **Causal lifecycle hints:** distinguish a tool wait, approval wait and explicit termination
  only when the harness has reported that state. Do not use future cancellation outcomes.
- **Age-conditioned reuse:** after a simple rule, test a survival/return model learned on
  past resolved sessions. Evaluate calibration and decision value, including censored waits.
- **Opportunity bound:** run a perfect-future-information oracle only as a labeled offline
  upper bound. If even it has little room over simple admission, stop optimizing this mechanism.

These are proposed ablations, not new algorithms or achieved improvements. A cheap rule that
solves the customer's problem is a successful product outcome. Defer new quantization formats,
GPU packing kernels and cross-engine adapters until a measured bottleneck requires them.

## Two-week implementation and decision gates

| Days | Deliverable | Advance only if |
| --- | --- | --- |
| 1–2 | Freeze these results; define trace schema, buyer checklist, evidence vocabulary and audit offer. | We can name the buyer, configuration owner and measurable failure. |
| 3–4 | Local importer, validation report and one small consented or appropriately licensed trace fixture. Add coverage and temporal-leakage tests. | Missing records cannot silently manufacture hits or savings. |
| 5–6 | Join measured profiles; replay simple admission, recompute, TTL/idleness and current policy. Add oracle and cost/noise sensitivity. | The current-config replay reproduces held-out observed resource/latency behavior within a predeclared tolerance. |
| 7–9 | Implement the admission-margin ablation in our reference manager; preserve all baselines and instrument decision components. | It improves decisions on development traces without using future labels; freeze it before confirmation. |
| 10–12 | Customer-run confirmation on existing infrastructure; collect costs, quality, tails and cancellation outcomes. | Results support an actionable recommendation rather than only a synthetic win. |
| 13–14 | Deliver the audit, actual-versus-predicted comparison and a purchase/continue/stop decision. | A qualified buyer pays or commits a concrete budget and validation owner. |

Customer discovery runs in parallel; lack of a customer trace is not permission to call
generated traces production evidence. No new GPU rental is required for this plan.
If a live deployment cannot be validated in two weeks, deliver the bounded diagnosis and
record the unresolved gate; do not label modeled savings realized.

Suggested engineering gates, to freeze with the customer before confirmation: at least 10%
lower cost per successful task **or** 10% higher SLO-compliant task throughput at fixed resources,
no more than 5% p95 resume regression, and the agreed quality/error budget. These are proposed
decision thresholds, not achieved results. Exact-cache reference tests require exact outputs.

## Measurement controls and release discipline

- Preserve model/tokenizer hashes, source commit, initial state, concurrency, seeds, request
  dependencies and identical continuation IDs for equal-work cross-system comparisons.
- Record CPU affinity/threads/cgroup limits, GPU clocks/utilization, driver/runtime, TF32 flags,
  copy/codec/recompute costs and profile variability. Never treat host variance as policy value.
- Separate fixed-work timing from normal-answer quality. Keep strict scoring, and predeclare
  any additional task-value/format rubric; do not replace a failed metric after inspection.
- Use independent session/day/workload clusters for confirmation and uncertainty. Freeze
  whole sessions before tuning; an observed holdout is no longer an unseen test.
- Keep default-policy behavior, accounting/isolation, cancellation, shared-prefix lifecycle,
  malformed traces and artifact auditing behind the required regression gate.
- A metadata replay preserves workload shape, not answer quality or alternate tool branches.
  Validate meaningful recommendations live before billing them as realized savings.

The next sale is an evidence-backed deployment improvement. A recurring product becomes
credible when multiple customers repeatedly need that improvement and keep paying for it.
Until then, a universal controller, an open hints standard and “free agent cold starts” are
expansion ideas, not claims supported by this repository.
