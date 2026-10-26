# Product hypothesis

## Proposed first user

A team self-hosting open-model coding agents that has measurable repeated prefill or tail-latency pain when sessions resume. It must control the inference engine and cache configuration. A team that only calls a hosted model API generally cannot install this kind of KV controller.

The initial workload has a mix of fast tool waits, longer human waits, cancellations and shared-prefix branches. The investigation must determine how much this mix actually matters in that team's deployment. Do not infer an addressable market from synthetic pause distributions.

## The narrow product experiment

Start as a **pause-cost advisor**: ingest anonymized lifecycle and cache telemetry, estimate the alternatives, replay supported policies, and show which waiting sessions cause avoidable memory/transfer/recompute cost. Then implement a policy plugin only if an improvement survives live measurement.

The eventual operator action is concrete: “Keep this shared prefix raw; encode this inactive branch with an already validated codec; discard this other branch and recompute if it returns.” Every choice must explain the observation, estimated cost, capacity impact, quality-approved format, and fallback.

Prefer LMCache's existing storage/policy machinery. The incremental work being tested is workload measurement, joint admission/representation decisions, eviction externalities, cancellation-safe accounting and quality-aware rollout for a specific operator. Each individual mechanism has prior art. A durable product advantage would have to come from better measured outcomes, maintained integrations and workload understanding, not simply assembling those words.

## Competitive obligation

Before asking anyone to adopt another component, establish whether their existing LMCache configuration, CacheGen, fixed FP8/INT4 representation, TensorRT feature, static TTL or agent scheduler already meets the same objective. The [research review](research.md) lists the direct overlaps.

Sell a measured improvement in successful coding tasks per dollar and resume latency at matched quality and budgets. Do not sell an unsupported universal compression ratio. If diagnostics alone reveal a useful configuration change, that may be a useful service or upstream contribution; whether it supports a company is a separate question.

## Evidence to seek from prospective users

- Engine/model versions, who controls them, and why existing caching is insufficient.
- Pause/return/cancel distributions and shared-prefix relationships, without needing prompt contents at the first stage.
- Measured cache budgets, eviction/recompute rates, resume tail latency, codec overhead and operational costs.
- A target worth paying to fix, integration constraints, and the simplest incumbent configuration already tried.

No outreach has been sent. Once the local experiment is credible, a small set of operator conversations and a representative trace are more valuable than adding more speculative codecs.

## Decision sequence

1. Local correctness and causal replay; retain negative results.
2. Capacity/queue modeling, stronger simple baselines, realistic traces, frozen evaluation.
3. One pinned live engine/model/GPU experiment with quality guardrails.
4. A customer-specific pilot if the value survives incumbent comparison.
5. Broader integrations and a product only after repeatable demand and outcomes.

There is a plausible research and engineering project here. The standalone startup case remains an open hypothesis.
