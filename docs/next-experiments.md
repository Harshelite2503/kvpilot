# From the first P100 run to a useful product

> **Historical plan, superseded on 29 October 2026.** The proposed admission/equal-work experiment is now complete on P100 and RTX 5060 Ti. Adaptive loses to simple admission in the balanced workload on both systems. Follow the [new measured findings](policy-replication-2026-10-29.md) and [current product plan](product-plan-2026-10-29.md). In particular, the stronger-codec/GPU-compression sequence below is deferred until a measured deployment bottleneck justifies it. This document is retained to show what was proposed before seeing the results.

Research checked **28 October 2026**. This is a proposed engineering and customer-validation plan, not a list of implemented features or a promise of YC acceptance. The [existing review](research.md) covers CacheGen, KVTC, Attention Matching, current storage frameworks and other close prior art. The [P100 record](p100-results-2026-10-28.md) is the empirical starting point.

## Decision: test admission value before inventing another compressor

The current adaptive policy never compressed in the first P100 run. Its strongest measured benefit is avoiding wasted native snapshots under cancellation. In the analytical holdout with measured codec costs, no policy beats recomputation by more than 2.6% on p95 and eviction pricing gives no p95 benefit; one changed decision increases modeled work and peak reservation ([measured-cost rerun](eviction-results-measured-costs.md)). A simple admit-only-if-space rule has the same modeled hit counts on the first P100 traces. Before making the controller more complicated, measure that baseline and expose workloads where each policy fails.

The candidate product is **cost control for self-hosted agent sessions while they wait for tools or people**: decide whether retaining state is worth its bytes, copy work and eventual restore cost. The customer must control the inference runtime. A developer using only a closed model API cannot install our cache manager behind it.

This is a customer hypothesis in a crowded category. The potential advantage would be verified savings on a specific deployment, reliable state handling, and low integration effort. An adaptive label, a mixture of published methods, or a favorable comparison with our slow INT4 reference is not a moat.

## What the literature changes about our plan

| Verified source | Relevant finding | Concrete implication |
|---|---|---|
| [InferCept, ICML 2024](https://proceedings.mlr.press/v235/abhyankar24a.html) | Already studies tool interruptions, retention/swap/discard and profiled resource costs. | Measure occupancy over time and compare a cheap cost-based rule. Retain-versus-recompute is established prior art. |
| [KVCache Cache in the Wild, USENIX ATC 2025](https://www.usenix.org/conference/atc25/presentation/wang-jiahao) | Real reuse differs by category, request history and age; policy effectiveness depends on workload. | Obtain actual trace shapes and split by whole session/time. Synthetic same-size groups are insufficient. |
| [Continuum, arXiv v7, September 2026](https://arxiv.org/abs/2511.02230v7) | Retention TTL accounts for reload/recompute costs, queueing and uncertainty in tool duration. | Include TTL and uncertainty stress tests when the harness supports time. Queueing benefits require a concurrent runner. Conference acceptance was not verified. |
| [CacheWise, June 2026 preprint](https://arxiv.org/abs/2606.16824v1) | Coding-agent scheduling and eviction use prefix reuse and tool metadata. | Tool-aware eviction is already direct competition. Author-linked [coding traces](https://github.com/cachewise-project/cachewise-coding-traces) are a candidate workload source, subject to verifying licensing and suitability. |
| [LMCache lazy offload, official documentation](https://docs.lmcache.ai/mp/lazy_offload.html) | Already defers stores, checks block identity, and supports eviction-aware draining. | Skipping unnecessary copies alone is not new. A later competitor benchmark must enable relevant current features and pin versions. |
| [KIVI, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html) | Different quantization grouping for keys and values, with a native residual cache. | Build one stronger fixed INT4 snapshot codec ourselves; distinguish it from the paper's active quantized-attention implementation. |
| [KVQuant, NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | Uses channel-sensitive/pre-RoPE keys, calibrated nonuniform quantization and separate outliers. | After the first codec, test an outlier side-buffer as one controlled ablation. Pre-RoPE caching needs explicit runner/state changes. |
| [TurboQuant, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) | Random rotation, scalar quantization and residual correction target vector/inner-product distortion. | A possible later codec baseline. Distortion guarantees do not imply arbitrary answer preservation or P100 speedups. |

Two attractive distractions should wait. [H2O, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html) and [SnapKV, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf) remove tokens within a session. Our current decision concerns whole saved sessions. Removing context needed by a later, unknown agent question changes the problem and needs different position handling and quality tests. [ShadowKV, ICML 2025](https://proceedings.mlr.press/v267/sun25b.html) and [Palu, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7da6e0e00702c60607a6ae05c802ef85-Abstract-Conference.html) also mean generic low-rank cache/offload is occupied territory, not an unexplored invention.

## First experiment: can adaptive beat cheap exact policies?

Keep the current native snapshot codec and model fixed. Add these **owned** policies to the same state store; do not import a cache-management library:

1. `raw_admit_if_fits`: store only when currently free capacity fits; otherwise recompute; never evict.
2. Native LRU and immediate recomputation: retain the current controls.
3. A simple size/cost rule: rank measured saved recomputation work per retained byte, without a learned predictor. Charge the work imposed on displaced entries.
4. Current adaptive policy, including its decision overhead.
5. Simple expiry/TTL only after timestamped lifecycle events exist. An unimplemented timer must not be approximated by giving the policy future events.

Use a deliberately simple value estimate before trying a neural policy. For a snapshot, a one-future-resume approximation is

`net_value = p_resume × (recompute_ms − restore_ms) − snapshot_ms − displaced_value_ms`.

Measure snapshot/restore/recompute cost; estimate return probability using only resolved past observations and currently available metadata. This approximation omits repeated reuse, queueing and retention duration; it is a baseline, not an invention or complete optimizer. A later age-conditioned reuse model should earn its complexity through an ablation. If predictions are uncertain or benefits are within measurement noise, prefer the cheaper conservative rule.

For later tier/time decisions, add measured memory occupancy cost and copy/GPU contention without counting transfers twice. Delaying a snapshot requires keeping canonical state somewhere until the decision; that residency is not free. Shared prefixes also require physical reference accounting before any shared-byte claim.

### Workloads that can disprove our hypothesis

The existing trace pauses eight almost equal sessions and then resolves the group. That structure makes no-eviction admission competitive. Add variable sizes (initially 256–4,096 tokens; later up to 8,000 within the runner's context limit), interleaved arrivals, repeated resumes and cancellations invisible to the policy.

Freeze families that include: first-admitted sessions never returning; frequently reused long contexts competing with many short ones; LIFO and FIFO returns; bursts with no reuse; changes in reuse/cancellation rates; and repeated pause/resume cycles. Include conditions that hurt each candidate, not only the adaptive controller. Cancellation rates and future order belong to the workload generator/evaluator, never the online policy inputs.

Split development and evaluation by whole session and workload family before running. A new seed in an unchanged template is useful repetition but weak generalization evidence. Public trace replay through our model/cache is still trace-shaped experimentation, not execution of the original coding agent. Verify dataset terms before redistributing records; do not download customer code for a metadata-only trace study.

### Two complementary measurements

**Equal-work timing:** every policy processes the same predetermined continuation token IDs and number of steps. Separately measure pause, transfer, restore/recompute and decision overhead. Teacher forcing is a computational control, not an answer-quality test.

**Free-generation quality:** preserve normal greedy generation and the original strict score. Also report a separately specified task-value/format rubric, truncation, token count, token agreement and repeated-resume failures. A semantic metric must not silently replace a strict metric after a failure. The 0.5B model is useful for mechanism testing; its low baseline accuracy cannot establish coding-agent quality.

Report full-session time per started session, completed responses, resume p50/p95, actual encoded and tensor-copy bytes, retained host bytes, total GPU allocated/reserved peaks, staging buffers and measured occupancy over time. A sampled RSS maximum has limits. Logical trace waits and modeled byte-seconds must be labeled separately from measured wall time. A sequential runner cannot establish serving throughput, queueing latency, or reduced fleet cost.

Randomize paired run order and use independent workload/session clusters as the unit of uncertainty. Repeating the same 24 questions does not create hundreds of independent quality observations. Freeze the policy, baseline settings, corpus and analysis rules before the confirmation run; publish failures and tails as well as means.

## Second experiment: a credible fixed compression baseline

If admission provides a useful signal, implement a **KIVI-inspired INT4 snapshot codec** using only our tensor engine and byte format: affine min/max quantization, keys grouped over consecutive tokens separately per channel, values grouped over channels separately per token, and a small native recent tail. Count scale/offset arrays, padding, tail and framing. Keep the old symmetric last-axis INT4 as a historical control.

Start with 4 bits rather than chasing a 2-bit headline. Freeze group size and tail settings on development data. Validate offset, shape, model identity, positions, residual-tail restoration and repeated round trips. Quantization error, teacher-forced log-probability and first-token agreement help diagnose faults but do not certify task quality. Our decode-to-dense snapshot path is not a reproduction of fused quantized attention, and cannot claim its active-memory gains.

Only after quality is acceptable, move encoding **before** device-to-host transfer and decoding **after** host-to-device transfer. Ordinary Torch arithmetic and packed `uint8` storage are sufficient for a first P100 implementation; the storage codec does not need native INT4 matrix multiplication. Measure the complete synchronized path, all temporary buffers and contention. No speedup is guaranteed. Avoid a full CPU export followed by compression if the claim concerns transfer savings.

Then choose one outlier or rotation ablation. Do not combine low rank, attention-based deletion, entropy selection, learned prediction and quantization in one change: a win or failure would be hard to attribute. If an improved fixed codec wins and adaptive adds no value, ship the simpler result.

## Two-week sequence and gates

These are planning targets, not completed work or guaranteed durations.

| Window | Deliverable | Decision |
|---|---|---|
| Days 1–3 | Measured no-eviction and cost/byte baselines; equal-work timing; frozen variable-size traces | Does adaptive add value beyond simpler exact policies? |
| Days 4–6 | Free P100 development run; inspect stage costs, answer failures and tails; freeze confirmation protocol | Remove ineffective features. Keep the already published experiment unchanged. |
| Days 7–9 | Unseen session-family confirmation; repeated-resume correctness; larger supported model only if resources allow | Is there a repeatable benefit at equal capacity and quality? |
| Days 10–14 | One stronger owned codec if indicated; otherwise an exact-cache pilot with a design partner | Follow the measured bottleneck rather than forcing the original compression story. |

Suggested advance criteria to freeze before confirmation: at least **10% lower mean session time versus the best simple exact baseline** on two independent representative workload families, no more than **5% p95 resume regression**, and exact output preservation for exact-cache policies. These are our proposed engineering thresholds, not industry/YC requirements. For a lossy codec, specify its acceptable task-quality difference separately and gather enough distinct tasks to assess it; do not grant it a blanket quality exemption.

Move to a paid 3090 only after a meaningful P100 result or a specific kernel/production compatibility question that the P100 cannot answer. The 3090 run should have a fixed hypothesis and spending cap. Modern serving economics ultimately require concurrent scheduling and a representative model/workload; speed on a 0.5B eager runner alone is insufficient.

## Build toward customers and YC in parallel

The first proposed customer profile is a team self-hosting coding or workflow agents with long contexts, frequent tool waits, and a measured prefill/retention bottleneck. Interview five such teams and seek two willing to provide sanitized event traces and run a pilot. These counts are planning targets. An enterprise that mainly pays a closed API bill has different problems and cannot deploy this component directly.

Ask about their most recent cache-related slowdown, observed pause durations/context growth, reuse and cancellation rates, current engine/cache settings, p95 latency target, constrained resource, and previous fixes. Request concrete logs/cost measurements where appropriate. Ask who can install and pay for a change. A design partner committing engineering time or a paid pilot is stronger evidence than interest in a benchmark. This emphasis is consistent with [YC's advice to build and talk to users](https://www.ycombinator.com/library/4D-yc-s-essential-startup-advice).

The initial product could be a trace collector and replay report that identifies wasted retention/recomputation, followed by a measured policy deployment with rollback. This is a candidate route to a customer, not a claim of novel observability; existing cache systems already expose metrics and policies. Keep all cache management in our implementation for the experiment. Any production runtime hooks need their own compatibility/correctness tests, and comparative claims require actually running the competitor, not substituting our weak INT4 baseline.

A credible eventual pitch is: **“We reduce avoidable inference work for self-hosted agents while they wait, under a fixed memory and latency budget.”** It becomes persuasive with a customer, an installation they keep using, reproducible cost per successful task, and evidence that simple settings cannot match it. Today we have an owned-cache prototype and a narrow P100 admission result. We do not yet have proven product demand, a unique algorithm, general quality preservation, or fleet cost savings. If simple rules solve the customer's problem, adopt them; if current upstream features solve it, narrow the product or contribute the result rather than manufacture differentiation.
