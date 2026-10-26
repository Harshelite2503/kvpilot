# Eviction-aware admission experiment

The initial `adaptive` policy priced the incoming cache's encoding, movement, expected restoration and occupancy, but did not price harm to existing owners when admitting it evicted their pages. The new `adaptive_eviction` policy adds that external cost. The original `adaptive` remains available as a historical baseline, while `adaptive_exact` uses identical per-page sizing and victim planning with zero eviction penalty for a matched ablation. Fixed INT4 is the main comparator.

## The change

For every eligible codec, the replay plans admission against the same current store snapshot. It uses exact, separately rounded physical page sizes, protects the incoming session's resident shared prefix, and identifies the LRU pages that would be evicted. Planning changes neither recency, references, memory nor history. Execution applies the selected victim plan.

For each currently paused owner that has a complete cache before the admission but an incomplete cache after it, the added penalty is:

```text
resume_probability * max(0, full_recompute_ms - predicted_restore_ms_without_eviction)
```

Sum this once per affected owner, not once per evicted page. Evicting one shared page can harm several owners while freeing only one physical allocation. An already incomplete owner incurs no additional penalty under this simulator's full-recompute fallback. Repairing an incomplete owner receives no positive credit in this version; the penalty is conservative, not the complete change in system value.

Predicted restoration includes remaining offload readiness, transfer and decoding. Probability uses resolved history for the observed pause kind. Predicted remaining pause is `max(mean_resolved_pause - elapsed_pause, 0)`. These are deliberately simple plug-in forecasts, **not** survival-conditioned probability or residual-life estimates. In particular, long-unresolved sessions can differ from the resolved sample.

The candidate score adds the eviction penalty to the original expected-cost score with coefficient one: both terms are expressed in modeled milliseconds. No coefficient is fitted to the held-out outcomes. Exact per-page sizing also removes a small aggregate-rounding feasibility error for the new policy. The old policy is retained unchanged, including its original aggregate screening and later physical admission check, so it is an ablation of eviction accounting plus this sizing correction. Legacy metric regression tests compare all original baselines with the original checked-in report.

## Scope and limits

Incoming-session resume-target filtering is unchanged. If recomputation violates that target but a cache admission satisfies it, the policy can still evict an expensive existing owner. This is incoming-session priority, **not** a global SLO guarantee. A regression test makes that choice explicit.

The new policy does not alter the LRU victim order, recompress existing pages, anticipate future arrivals, add queues, or optimize a full retention horizon. It can therefore lose to fixed INT4 because earlier raw admissions consume space before pressure arrives. It may also overprotect expensive but unlikely-to-return sessions if its forecast is poor. Comparing against INT4 is necessary even when the new policy beats the old one.

All prior analytical limitations remain: no real LLM, no GPU or actual PCIe, no active decode, no bounded compute/copy queues, no staging capacity, no partial-prefill restoration and no quality validation. Lossy exploration must be explicitly enabled. The change concerns an analytical cost-accounting omission; it does not prove that production evictions or serving economics are solved.

## Held-out procedure

Workload definitions, seeds, policies, assumed costs, metrics and bootstrap settings are declared in [the protocol](../examples/holdout-protocol.json). Development and evaluation seeds are disjoint. The new policy and fixed baseline use identical trace events and budgets. Each run resets history and residency; online learning from already resolved events within that run is allowed and causal.

Before evaluation, freeze source, evaluator, generator and protocol hashes. The evaluator rejects a mismatch. Commit the freeze before running the held-out seeds, then commit every result. If a correctness bug requires a later change, label the original experiment invalid/superseded and use a new declared holdout; do not silently reuse it as unseen.

The primary endpoint is the paired difference in run-level p95 modeled resume delay, summarized across independent seeds within each workload family. This is **not** the pooled p95 over all requests. Bootstrap whole paired runs because branches and requests within a run are dependent. Report all scenario families, confidence intervals and win/tie/loss counts; nominal per-family intervals do not correct for multiple comparisons.

Secondary measurements include mean resume delay, full recomputes, transfer bytes, peak paused-store bytes, byte-time and total modeled service work. Service work sums encoding, decoding, recomputation and transferred bytes divided by bandwidth. It is aggregate estimated resource work, **not** elapsed wall-clock time, measured cost per token, or a replacement primary endpoint after inspecting results.

“Unseen” means previously unexecuted seeded event sequences drawn from the declared synthetic families. It does not mean independently collected customer workloads or arbitrary out-of-distribution generalization. Neither a positive result nor a narrow confidence interval validates the assumed codec costs or model quality.
