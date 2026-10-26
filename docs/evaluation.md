# KVPilot evaluation contract

This contract defines what each experiment can establish. It is stricter than a demonstration: **synthetic replay, a real CPU codec measurement, and live LLM serving are separate evidence classes.** Current authorized execution is local and has no GPU. Live metrics, model quality and incumbent serving comparisons are prospective and must be `not-run` until performed. The initial package may implement only part of this contract; missing fields or benchmark arms must remain visible as gaps.

## Evidence labels

| Label | What actually ran | Permitted conclusion | Not established |
|---|---|---|---|
| `cpu_codec_microbenchmark` | A codec encoded and decoded byte/tensor buffers on the named CPU. | Measured byte ratio, CPU latency, throughput and reconstruction error for those inputs. | GPU throughput, real-model quality, vLLM compatibility or serving TTFT. |
| `analytical_trace_replay` | A deterministic event model replayed generated or recorded lifecycle events with supplied costs. The current implementation has no resource queues. | Policy behavior under stated workload and cost assumptions. | Actual engine speed, real GPU interference or model quality. |
| `live_serving` | A named model generated responses through the pinned engine/connector/backend. | Observed request latency, physical memory/traffic and task performance for that configuration. | Universal savings, other GPUs, other model families or production reliability. |

A simulation calibrated with CPU measurements is still a simulation. A microbenchmark using model-captured KV is still not a serving benchmark. Reconstruction RMSE, cosine error or token agreement alone does not establish coding-task correctness. Never pool these evidence classes into a single speedup chart.

## Workload and causal trace rules

Start with one uniform-attention 4–8B model. Use reproducible coding tasks with repository snapshots, fixed prompts, tool-result fixtures and executable tests. For performance replay, keep arrival times, tokenized prefixes and requested output lengths fixed across policies. For quality evaluation, allow the model's actual outputs and tool calls to determine outcomes; do not force a known successful continuation.

Split by complete session family, including all branches, before tuning. Do not put a parent's prefixes in training and its children in held-out evaluation. A useful initial split is 60% development, 20% validation and 20% final holdout, with seeds and corpus revisions recorded. If only synthetic traces exist, label them synthetic and report their generative assumptions.

The policy receives events one at a time in monotonic order. It may use elapsed time, observed access counts, known active references, past queue times, current sizes and a pause reason actually emitted by the runner. It must not access `next_resume_time`, future branch count, final task success, future attention scores, eventual reuse labels or the remainder of a trace. Dataset columns with such fields belong to an evaluator-only sidecar.

Test causality: replace every event after time `t` while preserving the prefix and seed. All policy decisions and state up to `t` must remain identical. Predictions trained offline must use a training split, and their calibration costs must be reported. An oracle using future reuse is an explicitly marked unattainable upper bound, never a product baseline.

### Extended trace schema for live integration

These fields specify the target trace contract; they are not a claim that every field is present in the lightweight local simulator.

| Record | Required fields and meaning |
|---|---|
| Run manifest | `run_id`, evidence class, policy/version/config hash, workload hash, seed, exact model/tokenizer/adapter revisions, engine/connector/codec versions, GPU/CPU/driver, byte budgets, arrival schedule, instrumentation settings. |
| Event envelope | `schema_version`, `event_id`, `ts_monotonic_ns`, `run_id`, `request_id`, `tenant_id`, `session_id`, `branch_id`, `parent_branch_id`, `event_type`, `source`. Wall time is optional and not used for durations. |
| Lifecycle event | Request accepted/completed/cancelled; session paused/resumed/closed; branch created. Include only the known prefix token count, completed output count and observable pause reason. |
| Block event | Block content/prefix identity, model namespace, token span, allocated and logical bytes, dtype/layout, source/destination tier, codec/version, reference count, state, checksum, operation/queue timestamps and completion status. |
| Policy decision | Decision time, feature snapshot, action, threshold/config version, predicted reuse/cost with units, rejected alternatives, queue occupancy and reason. |
| Request outcome | Client submission/first-content-token/end times, streamed output timestamps, server queue/prefill/decode spans when available, cache-hit tokens by tier, recomputed tokens, errors/timeouts and quality task ID. |
| Evaluator sidecar | Future reuse, true next-access time, task outcome and oracle labels. Never handed to the online policy. |

Hash real token prefixes, not just a session ID. Include compatible model/adapter/config identity and tenant namespace. Store parent/child sharing explicitly. A shared physical block consumes its bytes once even if several requests reference it; logical per-session usage is a separate statistic. Snapshot the raw schema and any normalization mapping in the report.

### Required workload cells

| Cell | Purpose and initial synthetic settings |
|---|---|
| No reuse | Sessions never resume; measures useless encode/store work. Include at least 20% of the mixed development trace and a 100% no-reuse stress cell. |
| Rapid resume | Resume after 0.1–2 seconds; expose races, transfer startup and encode cancellation. |
| Paused/resumed | Resume after 10–120 seconds; initial prompts 2K/8K/16K tokens and new tails 128–1K tokens, subject to actual capacity. |
| Long idle/expiry | Human approval delays of 5–60 minutes, plus expiry/never-return cases; examine eviction and graceful recomputation rather than assuming persistence. |
| Branching | Two or four children share 50–90% of the prefix, then diverge; test exact shared-byte accounting and cancellation. |
| Pressure and queueing | Concurrency 1, 2, 4, then the largest stable value; repeated distinct prefixes; host tier at 25/50/100% of the trace working set. |
| Identity invalidation | Same text under a changed model/adapter/template/tenant namespace; must miss safely. |

These are deliberately declared synthetic starting ranges, not estimates of typical customer behavior. Freeze the mixed-workload weights before final evaluation; publish per-cell results so a favorable mixture cannot conceal regressions.

## Baseline matrix

| ID | Policy and representation | Role |
|---|---|---|
| B0 | Native vLLM automatic prefix caching, BF16/FP16 KV, no external cache; evicted prefixes recomputed. | Necessary practical baseline. |
| B1 | Same engine plus raw host-memory offload with LRU and bounded asynchronous workers. | Separates offload value from compression/policy value. |
| B2 | Supported native FP8 KV with explicit scaling configuration; native prefix cache and, if compatible, the same offload tier. | Strong lower-byte baseline; do not conflate KV and model-weight precision. |
| B3 | Actual CacheGen through a compatible LMCache configuration and documented settings. | Compression incumbent. If unavailable, mark missing and limit conclusions. |
| B4 | Same tested codec as the candidate, applied using a fixed idle threshold/size threshold plus LRU. | Tests whether adaptive policy adds anything. |
| B5 | Calibrated wait-distribution/TTL retention, with explicit GPU opportunity cost where modeled. | Compare against the approach in human-approval retention research; label any local approximation rather than claiming a reproduction. |
| X1 | TensorRT-LLM native cold-page NVFP4 enabled versus disabled on that engine. | Follow-on incumbent test when supported; compare within that engine, not an unqualified cross-engine ratio. |
| X2 | Dynamo ThunderAgent admission/retention settings with its supported backend. | Follow-on session-aware scheduling comparison; unsupported integration is an evidence gap. |
| P | KVPilot adaptive retention/admission/representation policy with the same supported codecs. | Candidate. |
| O | Evaluator-only perfect future reuse. | Optional upper bound, excluded from victory claims. |

[vLLM's quantized-KV documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) describes supported formats and scaling behavior. Record what the installed engine actually uses and whether calibration is required. A CPU INT8 proxy is not an FP8 serving result. [CacheGen](https://arxiv.org/abs/2310.07240) is a real codec and streaming system; a generic entropy compressor is not a faithful reproduction.

Select the strongest compatible baseline on the validation split at the same resource budget and quality constraint. Freeze this selection before the final holdout. Report every baseline separately. If FP8 or CacheGen cannot run, say exactly which comparison remains unknown; do not announce superiority over the incumbent category.

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog30_KV_Cache_Compression_in_TensorRT-LLM.html) already compresses cold pages during offload, while [Dynamo ThunderAgent](https://docs.dynamo.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler) uses session lifecycle for admission and placement. The [human-approval retention paper](https://arxiv.org/abs/2608.30830) is directly relevant to pause-time policies. Their presence rules out treating either session awareness or cold-cache compression as new by itself. Their published wins are not results of this prototype.

## Fair resource and timing accounting

Fix the model precision, maximum context, generation settings, scheduler limits, batch settings, prefill configuration and arrival process. Fix physical GPU and host byte caps, rather than granting equal numbers of logical blocks to codecs with different sizes. Record total GPU memory and the usable KV budget; the candidate's staging allocations reduce its usable capacity. Also report maximum feasible concurrency under each policy as a separate capacity experiment.

Charge compressed payloads, scales, headers, indexes, allocator rounding, dedup metadata, pinned buffers and simultaneous source/destination copies. Measure both peak bytes and time-weighted occupancy (byte-seconds divided by elapsed run time). Count physical bytes transferred, including retries, headers and wasted work; report logical token-hit rate separately.

Encoding and decoding are not free because they run asynchronously. Measure CPU time, queue residence, copy times and GPU kernel elapsed time with appropriate synchronization at microbenchmark boundaries. During serving, preserve natural overlap and use timelines/profiling rather than serializing the workload merely to time it. A separate CUDA stream can still contend for SMs, memory bandwidth and copy engines. Report its effect on concurrently active decodes.

Use an open-loop arrival schedule for the primary latency test so a slower system does not receive less work. Record offered and completed request rates, queue depth, preemptions and dropped/timed-out requests. Run at a low-load point and a fixed moderate-load point chosen from baseline calibration. Do not compare different policies at different offered loads. A closed-loop user/session experiment can be additional evidence but must be labeled separately.

Warm model kernels and allocations before measurement. Define cold-cache, steady-state and controlled-resume trials separately, with identical initial state and cache-reset procedure across arms. Do not count a repeated hot prefix as evidence of offload retrieval unless the GPU cache was actually displaced or the engine records the source tier. Randomize policy run order to reduce thermal and background-load bias.

## Metrics and implementation hooks

**Primary latency:** resume TTFT is the time from the client's resumed-request submission to its first nonempty generated content token. It includes admission queueing, lookup, restore and any remaining prefill. The prior idle interval is not included. Report p50/p95/p99 and counts; p99 is exploratory with small samples. Server-side phase spans explain the result, but do not replace client latency.

**Decode guardrail:** inter-token latency is the time between consecutive generated tokens within a response. Report the pooled token-gap distribution and the distribution of each request's p95 so a few long responses cannot hide degraded sessions. Do not include tool/user pauses between requests as token gaps. Streaming chunks may contain multiple tokens: if token-level timing is unavailable, label the measurement chunk latency and use engine ITL for the token metric.

**Capacity/cost:** allocated KV bytes by tier, time-weighted stored KV bytes, transfer bytes, encoded bytes never reused, codec CPU/GPU time, queueing, repeated-prefill tokens, throughput and timeout rate. Separate storage compression ratio from total-device memory savings.

[vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) include TTFT, ITL, queue/prefill/decode times and prefix-cache statistics. Counter names vary by release; inventory `/metrics` after smoke traffic and persist the mapping. Where present, distinguish local and external prefix hits and cached tokens by source. Histograms are diagnostic; calculate final paired quantiles from request-level records rather than subtracting histogram percentiles.

[LMCache MP observability](https://docs.lmcache.ai/mp/observability/index.html) exposes cache operations, storage activity and optional request tracing. Preserve request/session correlation and inspect event drops. Do a short instrumentation-on/off comparison; if full traces materially slow the server, use sampling for performance runs and full traces for diagnosis, recording the difference.

## CPU codec measurement protocol

Run multiple payload sizes and distributions: random normal, correlated/heavy-tailed synthetic data, zeros as an explicit easy case, and real KV captures if available. Never report all-zero compression as representative. Record shape, source dtype, seed and whether conversion to/from the codec format is inside the timer. Warm up, repeat at least 30 times per cell, use a monotonic timer and report median/p95 encode/decode latency, bytes including metadata, and effective original-byte throughput. Record allocation/copy costs or report them as separately timed components.

For a lossless path, require byte-exact round trips. For lossy paths, report absolute/relative error and nonfinite values; attach no task-quality claim. Compare synthetic and captured-input results separately. A CPU result is a local measured cost that may parameterize a simulation; GPU costs remain assumed until independently measured.

## Statistics and success gates

Use paired replay of the same session families and arrival schedule. Initial performance target: at least 200 independent session families and 1,000 resumed requests per policy across at least five randomized-order repetitions. If the two-week budget cannot reach this, report a pilot estimate with wider uncertainty. Correlated requests from one session are not independent samples.

Bootstrap paired differences/ratios by session family, preserving branches and requests together, and include run-level variation; use at least 2,000 bootstrap resamples for reported 95% intervals. Report the actual sample counts, seeds and interval method. For dependent open-loop queueing, also inspect paired whole-run summaries. A small p95 point estimate is not strong evidence if only a handful of tail events were observed.

The exploratory follow-on live performance target, to test after the local two-week preparation on a preregistered held-out mixed workload, is:

- At least **20% lower observed p95 resume TTFT**, with the 95% confidence interval supporting a positive improvement, versus the frozen strongest compatible matched-budget baseline.
- At least **15% fewer time-weighted stored KV bytes**, including metadata and temporary copies, without increasing the allowed GPU/host budgets.
- At most **5% observed p95 ITL regression**, with an upper confidence bound no greater than 5% for a confirmed guardrail pass. If uncertainty crosses 5%, the result is inconclusive.
- No increase in errors or stale/incorrect cache reuse; throughput must not fall by more than 5% at the same offered load. No-reuse and rapid-resume p95 TTFT regressions must be reported, with a 5% target ceiling and a bypass policy when compression is not worthwhile.

Passing all targets is not assumed. A latency/storage tradeoff can still justify further work, but must be described as a tradeoff. If the confidence interval supports some improvement but not 20%, call it promising, not a confirmed 20% gain.

### Quality and correctness

Begin with 100–200 held-out coding tasks as a screen, using executable tests and predetermined timeouts. Pair the same task, prompt and sampling seeds across arms. Include exact identifiers, distant instructions, long tool output and branch-specific requirements. Record pass/fail, valid tool arguments, required-fact recall and output agreement. Correctness tests must include wrong-tenant and changed-model/adapter cache collisions, partial blocks, cancelled writes, corrupt payloads and resumed branches.

The desired quality non-inferiority margin is no worse than **2 percentage points** in paired task success versus the full-precision baseline. A small screen will usually be too weak to certify that margin. Report a paired confidence interval for the success-rate difference and classify it as pass/fail/inconclusive. Choose the needed larger sample from observed discordance; several hundred or thousands of tasks may be needed. Do not equate “no statistically significant loss” with “no quality loss.” Do not silently exclude tasks that timed out or produced invalid outputs.

Lossless paths should restore exact stored KV bytes, while output determinism still depends on kernels and scheduling. Lossy paths need the task evaluation even when tensor errors look small. Repeated resumption must not repeatedly quantize an already degraded cache without accounting for cumulative error.

## Decision report

For each result provide evidence class, provenance, run manifest, baseline, budget, workload cell, counts, point estimate, 95% interval and status. Status is one of `measured-pass`, `measured-fail`, `inconclusive` or `not-run`. Include a failure table showing where recomputation or a simpler baseline wins.

Proceed to a customer-trace pilot only if a useful workload-specific advantage survives comparison and the quality evidence is adequate for that pilot. If only CPU/simulation evidence exists, the deliverable is a feasibility harness and a live-validation plan. If a static policy matches the candidate, or gains vanish after queue/GPU/metadata costs, reject the adaptive-policy startup claim rather than tuning the holdout until it looks positive.
