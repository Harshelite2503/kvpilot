# KVPilot: how to make this big (YC strategy memo)

> **Historical memo.** The [29 October product plan](product-plan-2026-10-29.md) and [P100/RTX comparison](policy-replication-2026-10-29.md) supersede this memo's implementation status and near-term recommendations. Both completed systems show no adaptive advantage over simple admission in the balanced workload. Claims about current vendor capabilities or pricing below are historical and should be rechecked before reuse; the new plan uses freshly checked primary sources.

Written 28 October 2026 from the current `main`, the open PR
[Harshelite2503/kvpilot#1](https://github.com/Harshelite2503/kvpilot/pull/1), Astra's
recommendation list, and a fresh scan of papers, vendor docs and funding news. Sources are
linked inline; claims from papers are the authors' claims, not ours.

## 1. Where KVPilot actually stands (what a YC partner will read in ten minutes)

What exists is an unusually honest research prototype:

- A causal replay simulator for pause/resume/cancel events with shared-prefix accounting,
  an eviction-aware admission policy, and a frozen holdout protocol.
- Four CPU reference codecs (FP16, zlib, INT8, INT4) with measured byte ratios.
- Native Ollama cache measurements on a Mac (6.6 s fresh vs 0.08 s exact repeat vs 0.27 s
  after a checkpoint). This measures Ollama, not KVPilot.
- PR #1: the first real-GPU run. Two sequential vLLM requests had a 2.47× ratio of full
  completion times (1,490 to 604 ms). This is not an isolated cache speedup or TTFT result:
  warmup and order were uncontrolled. A real KV tensor was captured and CPU reference-codec
  costs were calibrated against it. See [live-integration.md](live-integration.md).

What PR #1 also shows, and what the pitch must not hide:

- The assumed codec timings were about 100x too optimistic. Under measured costs the
  adaptive policy picks "recompute everything" on every one of 60 pause events, and the
  published five-of-seven-families eviction result is highly cost-sensitive. The same-day
  [rerun](eviction-results-measured-costs.md) removes the two large p95 wins.
- At the time of writing, no KVPilot decision had been applied to a live cache in this repo.
  **Superseded the same day:** the P100 branch, now merged, runs the policy inside an owned
  PyTorch runner on a Tesla P100 with Qwen2.5-0.5B. Eviction-aware admission takes 6.4 to 7.9%
  less full-session time than always-store-native LRU in cancellation scenarios with identical
  outputs, ties when memory is roomy, and never chose INT4 or INT8. No GPU-memory saving was
  measured, and a one-line "store only if it fits, never evict" rule is predicted to match its
  hit counts. See [p100-results-2026-10-28.md](p100-results-2026-10-28.md). That observation and
  the measured-cost simulator rerun agree: the value is admission control, not compression.

Conclusion: the current asset is not a faster cache. It is a rigorous evaluation harness,
a cost model that already proved it can change its own mind when fed real numbers, and a
team that refuses to overclaim. That is the right foundation for a company, but it is not
yet a product. The pitch should be built around the problem and the wedge, not the policy.

## 2. Why now (the part of the deck that writes itself)

In the last six months the problem KVPilot targets became measured, named, and priced.

**Measured at production scale.** Microsoft's August 2026 characterization of GitHub
Copilot's coding agent (13.5M sessions, 3.2M users, 761M LLM calls) reports: prefix cache
hit rate averages 90% within a turn but drops to 55% at turn boundaries and 8% after a model
switch; idle time is bimodal, seconds within turns and minutes across turns, with the KV
cache sitting idle about 2.9 minutes at a turn boundary; median session 4.2 minutes but the
p90 exceeds three hours; a lightweight idle-time predictor captures 86 to 90% of idle time.
Their stated takeaway is that turn boundaries are "the single most actionable signal" for
KV-cache eviction and offloading.
([ghcp_traces paper, Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/08/ghcp_traces-6.pdf))

**Named by the engine vendors.** vLLM's September 2026 AgentX post describes agentic
sessions with a median of 43 turns and 142K input tokens, requiring prefix hit rates above
96%, and shipped "session-aware retention" (interval and second-sight checkpointing) plus
hierarchical offloading to Mooncake.
([vLLM x AgentX](https://vllm.ai/blog/2026-09-08-vllm-agentx)) NVIDIA Dynamo's agentic
digest has a section literally titled "Selective Cache Retention": LRU cannot tell a system
prompt from a dead subagent's KV, a 2 to 30 second tool pause can age out an agent's prefix,
and they want harnesses to tag sessions with priority hints. TTL pinning is listed as future
work. ([Dynamo agentic inference digest](https://docs.dynamo.nvidia.com/dynamo/dev/digest/agentic-inference))

**Priced by the hosted providers.** Anthropic charges 1.25x base input to keep a cache for
5 minutes and 2x to keep it for 1 hour; reads are 10%.
([Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching))
OpenAI moved GPT-5.6+ to a 30-minute minimum TTL with paid cache writes and 24-hour default
retention. ([OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching))
That is the whole KVPilot question, retention versus recompute during an agent's wait,
turned into a line item on an invoice. Teams that self-host have no equivalent control and no
equivalent bill, which means they have no idea what their idle agents cost.

**Called out by YC.** The Summer 2026 RFS includes inference infrastructure for agent
workflows on the grounds that agents create "bursty, looping, memory-heavy" inference
loads. ([VC Cafe summary of YC RFS S26](https://www.vccafe.com/requests-for-startups-summer-2026-edition/))

## 3. The competitive map, honestly

| Who | What they do | What it means for KVPilot |
|---|---|---|
| [Tensormesh / LMCache](https://pulse2.com/tensormesh-20-million-raised-to-scale-kv-caching-infrastructure-for-enterprise-ai-inference/) | $24.5M raised (Laude, AMD Ventures, CoreWeave, NVentures). KV storage tiers, serde (FP8, TurboQuant), "zero-cost cached tokens" SaaS. | The storage and codec mechanism layer is funded and ahead. Do not compete on bytes moved. |
| NVIDIA Dynamo KVBM, vLLM session-aware retention, SGLang priority eviction, TensorRT-LLM token-range retention | Engine-specific controls exist; [Dynamo agent hints](https://docs.nvidia.com/dynamo/agents/agent-hints) already span multiple backends, with feature-dependent support. | Integration and measured savings must beat these existing capabilities; a common hints API alone is not an unoccupied market. |
| [Continuum](https://arxiv.org/abs/2511.02230) (arXiv v7, September 2026; conference acceptance not verified) | TTL pinning in GPU memory for tool calls. | Prior art for "keep during short waits". |
| [MORI](https://arxiv.org/abs/2606.00866) (Berkeley, Stoica lab, May 2026) | Continuous idleness ranking, GPU/CPU tiering, on replayed Claude Code SWE-bench Pro traces: 20 to 71% higher throughput, 18 to 43% lower TTFT vs best offloading baseline. | The strongest academic competitor to an adaptive policy. Any KVPilot number must be compared to it. |
| [Adaptive KV Retention at Human-Approval Timescales](https://arxiv.org/abs/2608.30830) (JHU, Aug 2026) | Prices "GPU opportunity cost": residency costs 41% goodput, eviction costs about 10x resume latency; their controller beats vLLM by 23 to 51%, MORI by 22 to 29%. | Validates the exact tradeoff KVPilot models. Two authors, a preprint, no product. |
| [CacheWise](https://arxiv.org/abs/2606.16824) (June 2026) | Coding-agent traces, reuse-aware eviction from tool metadata: 2 to 2.6x fewer evictions, up to 3.5x session completion. Traces open-sourced. | Free realistic workloads for Astra's step 3. |
| [OneTriangle](https://dealroom.co/news/talk-nfubxNVmQYY-onetriangle-launches-with-yc-s26/) (YC S26) | KV transfer between models (small prefill, large decode). | Proof YC funds KV-layer companies; not a direct overlap. |

The lesson is the same one `docs/research.md` already drew: compression, offload, TTL,
priority and agent-aware controls have prior art. Replay is also occupied:
[Dynamo agent simulation](https://docs.nvidia.com/dynamo/agents/agent-simulation) supports
offline and live replay, and [LMCache tracing](https://docs.lmcache.ai/mp/tracing_and_debugging.html)
supports cache-configuration replay. Deployment-specific savings and low integration cost
are hypotheses to validate, not evidence that nobody is building this layer.

## 4. Three ways to make it big, in the order to pursue them

### Wedge 1 (do first): the inference FinOps and "what-if" product

KVPilot's replay simulator is a counterfactual engine. Turn it outward:

- Ingest traces from a thin OpenAI-compatible proxy (request timestamps, token counts,
  cached-token counts, tool-call durations, session and turn IDs) plus engine metrics.
  Exclude prompt contents. Obtain permission, minimize identifiers and review retention:
  metadata can still be sensitive; another organization's study does not make it safe to share.
- Replay the team's own week against policies: current config, TTL, priority hints,
  offload tier sizes, recompute. Report GPU-seconds wasted on re-prefill after turn-boundary
  eviction and p95 resume latency. Request elapsed time is not GPU busy time or allocated
  GPU-hours. Report dollars only after a live experiment connects the saved work to fewer
  billed GPU-hours, or more completed work at the same spend and service target.
- Output a recommendation they can apply today in vLLM, SGLang, Dynamo or TRT-LLM
  settings. Charge for the report and the dashboard before any code of yours touches their
  serving path.

This wedge can be tested through platform-team interviews and offline pilots before a live
connector. The current simulator still needs customer-trace ingestion and deployment cost
calibration; a sale or policy improvement is not guaranteed. It is the "pause-cost advisor" in
`docs/product-hypothesis.md`, promoted from a step to the product. The analogy for the
deck: Datadog or Vantage, but for agent inference memory.

### Wedge 2 (next): the cross-engine "agent hints" layer

Anthropic has `cache_control`. Dynamo already exposes `nvext.agent_hints` for serving
intent, priority and speculative prefill across backends; supported actions differ by engine.
SGLang has priorities. vLLM's
[`prefix_cache_retention_interval`](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)
is a checkpoint interval in tokens, not a time-based TTL. Agent harnesses can supply useful
lifecycle information, but future tool and approval durations remain predictions. First
identify a valuable lifecycle signal or deployment requirement these existing interfaces
do not handle, rather than assuming no signal reaches an engine today.

Ship a small open-source SDK and sidecar that annotates requests with lifecycle hints and
translates them into each engine's native retention API, using KVPilot's cost model to set
TTLs and priorities per deployment. This rides upstream instead of fighting it, and it
makes the product sticky in both the harness and the serving stack. Proposing it as a
de facto standard ("cache_control for self-hosted models") is a credible open-source
story for a YC application.

### Wedge 3 (the vision slide): suspend and resume for agents

Agents spend most of their life waiting, for tools, for tests, for humans. At any moment a
fleet running thousands of coding agents has most of its KV memory held by sessions that
are not generating. The long-term product is the layer that checkpoints an entire agent
(KV state, sandbox, tool state), places it on the cheapest tier that still meets its resume
target, and resumes it on any replica or node. The analogy is serverless and Firecracker
snapshots: "we made cold starts free for agents". This is where the Microsoft p90 session of
three hours and the JHU human-approval paper point. It is also where the GPU-economics
pitch becomes a fleet-utilization pitch, which is the size of story YC funds.

## 5. The one-liners to test

- "Agents spend 90% of their life waiting. We make waiting free for teams that run their
  own models."
- "Prompt-caching economics for self-hosted agents."
- "Autopilot for agent inference memory: pin, offload, or drop, per session, in dollars."

## 6. What to change in Astra's plan

Astra's steps 1 to 3 are right and should happen in that order: beat the simplest credible
baselines first, separate cache speed from answer length, and test workloads designed to
make the policy fail. Two changes:

- **Use real workload shapes, not more synthetic families.** The Microsoft paper gives the
  distributions (turn counts, idle bimodality, archetype shares) and CacheWise and MORI
  have replayable coding-agent traces. Freeze holdouts drawn from those.
- **Rerun the frozen eviction holdout under the measured costs from PR #1 before quoting any
  policy number anywhere.** Done on 28 October 2026, see
  [eviction-results-measured-costs.md](eviction-results-measured-costs.md): the +47% and +62%
  wins become exact ties and no policy beats plain recomputation by more than 2.6% on p95.
  Scenario-mean adaptive work is 1.01× to 2.65× recomputation's. Fixed INT4 ranges from
  5.28× to 37.02× and native from 2.64× to 14.74×. The eviction-pricing ablation has
  20 p95 ties in every scenario, but one seed's changed decision increases work and peak
  reservation. The "five of seven families" headline alone does not establish savings.

Steps 4 and 5 (a stronger owned INT4 codec, GPU-side packing before transfer) should be
cut. They are the mechanism layer where Tensormesh, TurboQuant, KVTC and TensorRT-LLM's
NVFP4 cold pages already compete, PR #1 shows the codec path is a cost center that makes
the policy prefer recompute, and every week spent there is a week not spent on traces and
customers. Use the engine's own codecs as fixed inputs to the policy.

Astra's positioning note is correct and should be taken further: Continuum, MORI and the
JHU paper mean "adaptive retention" is not the invention. The defensible asset is the
decision layer plus the data plus the harness integration.

## 7. Six weeks to a fundable demo

| Week | Deliverable | Proof it produces |
|---|---|---|
| 1 to 2 | Trace proxy and cost report. Run your own Claude Code or OpenHands sessions against vLLM with a current open coding model on a rented GPU. | One real sentence: "N% of GPU-seconds in these sessions were re-prefill after turn-boundary eviction; policy X recovers Y% at Z ms p95." |
| 2 to 3 | Rerun the eviction holdout under measured costs; add TTL, priority-hint and MORI-style idleness baselines to the replay. | A table where KVPilot either beats or loses to each named baseline. Both outcomes are usable; a loss redirects the product to Wedge 1 only. |
| 3 to 4 | Hint translation for one engine (vLLM retention or SGLang priority). Replay CacheWise or MORI traces live. | First live number attributable to a KVPilot decision. |
| 3 to 6 | Five conversations with teams self-hosting coding agents (platform teams at enterprises that must self-host, neoclouds, inference providers). Two pilots with sanitized traces. | Letters of intent and a trace corpus. This is the traction slide. |
| 6 | Dashboard demo with one pilot's real dollars. | The demo. |

Target customers to go after first: companies running open coding models (Qwen3.x-Coder,
Kimi K2.x and K3, GLM-5, DeepSeek V4) on vLLM or SGLang for internal agents; regulated
enterprises that cannot use hosted APIs; and inference providers who would like to offer
Anthropic-style retention pricing on open models but have no policy layer to do it.

## 8. Risks a partner will raise, and the answers

- **"vLLM just shipped session-aware retention. Why won't upstream absorb you?"** This is
  a real risk. Dynamo already covers cross-backend hints and replay. KVPilot would need
  customer-verified savings or operational capabilities beyond the upstream baseline;
  neither an independent SDK nor a dashboard establishes that advantage by itself.
- **"Tensormesh raised $24.5M for this."** They sell storage and a hosted inference
  product. KVPilot sits above storage and is engine-agnostic; it can drive LMCache as one of
  its backends.
- **"The hosted providers already solved it with prompt caching."** That is the market
  proof: retention has a price. The self-hosted market, which is the entire enterprise and
  neocloud segment, has no equivalent.
- **"Your own PR says the policy picked recompute everywhere."** Yes, because the assumed
  costs were wrong and the policy corrected itself. The company is the harness that catches
  that. Show the before and after table proudly.
- **"Where is the live win?"** Not yet. The six-week plan above exists to produce it, and
  Wedge 1 is sellable without it.

## 9. What not to do

- Do not pitch compression ratios, concurrency multipliers, or memristors.
- Do not pitch "adaptive KV management" as the invention; it is a 2026 paper genre.
- Do not build a hosted inference provider first; that is a capital game Tensormesh,
  Baseten and Fireworks are already playing.
- Do not quote simulator numbers in the deck. Quote Microsoft, AgentX, Anthropic pricing,
  and your own live measurements.
