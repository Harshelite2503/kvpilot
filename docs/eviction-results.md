# Eviction-aware policy: frozen synthetic holdout results

27 October 2026. **The policy can beat fixed INT4 in some declared synthetic workloads, but it is not a general winner and has not met the original combined latency/memory target.** These are local analytical results. No GPU, inference engine, real KV capture or model-quality test ran.

## What changed and what was frozen

`adaptive_eviction` prices the expected restoration benefit lost by already paused sessions when admitting a new cache evicts their pages. It charges once per affected owner, counts shared physical pages once, plans exact LRU victims without mutating state, and executes the selected plan. See [the design](eviction-design.md).

The source, workload definitions, cost assumptions, metrics and evaluator were frozen in commit **`34414dd`** before generating evaluation seeds. The [freeze manifest](../results/eviction-freeze-v1.json) binds nine source/config artifacts. The evaluator verified those hashes before and after the run. Development used seeds 101–103; evaluation used previously unexecuted seeds 1101–1120. No code or policy coefficient was tuned after viewing holdout results.

Seven workload families × 20 independent seeds × 240 sessions = **140 unique traces containing 33,600 synthetic sessions**. Each trace ran through all six policies, producing **840 policy/run records**. Every policy in a pair received identical hashed events, cost assumptions and memory limits. History and residency reset between runs.

The policies were fixed INT4 (primary comparator), raw FP16, recompute, legacy `adaptive`, exact-sizing/no-penalty `adaptive_exact`, and the new `adaptive_eviction`. The last two differ only in the charged eviction penalty; this isolates the effect of eviction pricing from the separate page-rounding correction. The original baselines also reproduce every metric in the original checked-in demo.

Verification: **130 tests and 89 subtests passed** before the freeze, including shared victims, incomplete owners, readiness, causal decisions, protected prefixes, exact physical sizing, source-freeze refusal and paired statistical accounting.

## Primary result versus fixed INT4

The preregistered primary statistic is the **mean of each run's p95 modeled resume delay**, compared in pairs across seeds within a workload family. It is not a pooled percentile across all requests. Relative improvement is `1 - mean(candidate_run_p95) / mean(INT4_run_p95)`. Positive values favor the candidate.

Intervals are nominal 95% paired bootstrap intervals using 2,000 resamples of whole runs. They describe variation across these synthetic seeds under fixed assumptions, not hardware/codec uncertainty or customer generalization. Per-family intervals are unadjusted for multiple comparisons.

| Workload | Fixed INT4 mean run-p95 | New policy mean run-p95 | Relative improvement, 95% CI | Run wins / ties / losses |
|---|---:|---:|---:|---:|
| Sustained pressure | 94.56 ms | 91.57 ms | **3.17%** [1.57%, 5.42%] | 14 / 6 / 0 |
| Shared branching pressure | 123.02 ms | 123.44 ms | **−0.34%** [−0.94%, 0.07%] | 1 / 17 / 2 |
| Fast returns | 14.72 ms | 7.79 ms | **47.06%** [47.05%, 47.08%] | 20 / 0 / 0 |
| Cancellation-heavy | 89.13 ms | 90.39 ms | **−1.41%** [−2.89%, −0.06%] | 3 / 9 / 8 |
| Low pressure | 5.33 ms | 2.00 ms | **62.49%** [62.49%, 62.49%] | 20 / 0 / 0 |
| Heterogeneous sizes/recompute costs | 235.35 ms | 208.82 ms | **11.27%** [6.02%, 18.09%] | 17 / 3 / 0 |
| Shifted pause mix | 79.32 ms | 74.50 ms | **6.08%** [2.50%, 10.22%] | 15 / 5 / 0 |

The branching result is inconclusive on the primary metric, with a slightly worse point estimate. Cancellation-heavy traffic has a small primary regression under these assumptions. We report both rather than selecting only favorable workloads. No blended overall winner is declared.

The extremely narrow fast-return/low-pressure intervals are a consequence of the fixed codec cost equations and workload shape, not evidence of precise real-server predictions.

## Was the improvement caused by eviction pricing?

The matched `adaptive_exact` control keeps sizing, forecasts, target filtering and victim planning identical, but charges zero eviction penalty. Relative reductions in mean run-p95 versus this control were:

| Workload | Eviction-pricing effect, 95% CI |
|---|---:|
| Sustained pressure | 4.66% [2.94%, 7.06%] |
| Shared branching pressure | 1.45% [0.49%, 2.69%] |
| Fast returns | 0.00% [0.00%, 0.00%] |
| Cancellation-heavy | 4.18% [1.78%, 6.84%] |
| Low pressure | 0.00% [0.00%, 0.00%] |
| Heterogeneous sizes/recompute costs | 17.06% [10.57%, 24.58%] |
| Shifted pause mix | 22.10% [17.16%, 26.85%] |

Thus eviction pricing improves this heuristic in five pressure/shift families. It still does not beat fixed INT4 on every family. The impressive fast-return and low-pressure comparisons with INT4 are **not caused by the eviction fix**: the zero-penalty controller makes the same decisions. There, avoiding the assumed codec cost is enough.

In both fast-return and low-pressure families, the new policy also **exactly matches fixed raw FP16 on every recorded metric in all 20 seeds**. A simple static raw configuration achieves those same outcomes. These two families provide no evidence of an adaptive-policy advantage over that stronger baseline.

## Resource tradeoffs and regressions

The next table reports reductions versus fixed INT4; negative means the new policy consumed more. “Service work” adds assumed encode, decode, recompute and transfer time. It is summed modeled resource work, not runtime, GPU utilization, throughput or actual spending.

| Workload | Service-work reduction | Stored-byte-time reduction | Transfer-byte reduction |
|---|---:|---:|---:|
| Sustained pressure | 10.47% | −6.38% | 22.95% |
| Shared branching pressure | 0.33% | −3.56% | −12.67% |
| Fast returns | 46.61% | −276.00% | −276.00% |
| Cancellation-heavy | 22.87% | 7.94% | −31.60% |
| Low pressure | 71.22% | −276.00% | −276.00% |
| Heterogeneous sizes/recompute costs | 15.07% | −5.67% | 22.12% |
| Shifted pause mix | 6.52% | −4.51% | −4.02% |

Fast returns and low pressure use **3.76×** the stored-byte time and transfer bytes of fixed INT4 while staying within the same configured capacity ceiling. They are latency/resource tradeoffs, not compression wins. No family achieves the original joint target of at least 20% lower tail latency and 15% lower time-weighted stored bytes; live quality/decode guardrails remain untested regardless.

Adverse secondary results matter:

- Branching: mean resume delay is 2.20% worse; transfers rise 12.67% and full recomputes rise 4.13%. Its 0.33% service-work point improvement is inconclusive (95% interval approximately −0.33% to 0.93%).
- Cancellation-heavy: mean resume delay is 13.77% worse and full recomputes rise 21.77%, although lower wasted encoding reduces total modeled service work. The apparent 7.94% byte-time saving has an interval crossing zero.
- Shifted mix: p95 improves but mean resume delay worsens 7.08% and full recomputes rise 23.00%. Protecting expensive sessions can improve the tail while making more other sessions recompute.
- The new policy slightly increases peak reservation relative to fixed INT4 in the pressure families. Both obey the same byte ceiling; equal capacity limits do not mean equal actual occupancy.

## Decision and remaining work

**Keep the eviction-cost fix as a useful local heuristic improvement. Do not claim a validated serving or startup advantage.** The promising cases for further investigation are heterogeneous recompute values and changing pause mixes. The branching and cancellation regressions are evidence that this greedy score is incomplete.

Potential follow-up hypotheses are better censoring-aware forecasts, pricing actual GPU/transfer queue contention, and reconsidering victim selection or recompression of earlier raw admissions. They should be compared with strong simple threshold/TTL policies, then tested on new declared seeds and representative real traces. The revealed holdout is now available for diagnosis; it cannot be reused as fresh unseen evidence after tuning.

All codec ratios, durations and bandwidths here are assumed. There are no bounded resource queues, active decodes, source/staging memory costs, partial-prefill execution, model-quality measurements or live CacheGen/LMCache/TensorRT comparisons. Policy-planning CPU time is also uncharged; its overhead could erase a small modeled gain. Fixed INT4 itself is an illustrative profile, not an implemented inference-engine codec benchmark.

## Reproduction and raw artifacts

- [Protocol and costs](../examples/holdout-protocol.json)
- [Source/config freeze](../results/eviction-freeze-v1.json), committed before evaluation in `34414dd`
- [Held-out run manifest and input trace hashes](../results/eviction-holdout-v1/manifest.json)
- [All 840 per-run records](../results/eviction-holdout-v1/runs.jsonl)
- [All paired comparisons and intervals](../results/eviction-holdout-v1/summary.json)
- [Development comparisons](../results/eviction-development-v1/summary.json)

From the repository root with the dependencies installed:

```bash
python scripts/evaluate_eviction.py --split holdout \
  --freeze-manifest results/eviction-freeze-v1.json \
  --output-dir results/local-eviction-reproduction
```

Use a new output directory. This reproduces the frozen experiment; it does not create a new holdout. A changed frozen source/config file is rejected before workload generation. No paid service or external inference endpoint is used.
