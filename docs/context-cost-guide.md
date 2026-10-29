# Context-cost MVP: from exported usage to a tested change

This guide covers `kvpilot context`. The [repository README](../README.md) has a
short introduction and the complete synthetic quickstart. The earlier
[owned-cache runner](owned-cache-research.md) is a separate research experiment.

The current tool imports local records, prices supported usage, compares task
outcomes and durations, and calculates hypothetical compaction break-even points.
It does not run a model, call a provider, generate summaries or rewrite prompts.

## What counts as evidence

An **audit** answers what the supplied usage would cost at the supplied rates.
A **comparison** checks whether a candidate meets declared cost, task-outcome and
latency thresholds on matching records. A **plan** estimates a hypothetical future.
None independently establishes a lower customer invoice or general answer quality.

Use `evidence_type: "synthetic"` for generated fixtures and `"observed"` for exports
from actual runs. This label is a declaration by the exporter, not an authentication
mechanism. The included examples are synthetic and use illustrative prices.

## Input 1: one complete run in JSONL

JSONL means one JSON object per line, encoded as UTF-8. Do not add blank lines or
comments. The first record is `run`; the last is `run_end`. Between them, record
requests and one terminal `task_end` for every task. Interleaved tasks are allowed.

This is a minimal **synthetic** trace:

```jsonl
{"schema_version":1,"type":"run","run_id":"example-baseline","dataset_id":"fixture-v1","workload_version":"1","evaluator_id":"fixture-evaluator-v1","agent_revision":"example-agent-v1","provider":"anthropic","model":"example-model","evidence_type":"synthetic","cache_protocol":"5m-explicit-v1"}
{"type":"request","request_id":"r1","task_id":"t1","session_id":"s1","sequence":0,"kind":"inference","usage":{"input_tokens":100,"output_tokens":20}}
{"type":"task_end","task_id":"t1","success":true,"duration_ms":1000}
{"type":"run_end","complete":true,"request_count":1,"task_count":1}
```

### Run identity and controls

All header fields shown above are required. Identifiers are nonempty strings.

| Field | What to record |
| --- | --- |
| `schema_version`, `type` | Exactly `1` and `"run"`. |
| `run_id` | A unique identity for this execution; baseline and candidate must differ. |
| `dataset_id`, `workload_version` | The task collection and its version. |
| `evaluator_id` | The exact success rubric or evaluator version. |
| `agent_revision` | The tested agent revision; this may differ for the candidate change. |
| `provider`, `model` | Currently `"anthropic"` and the exact model identity matching the rate card. |
| `evidence_type` | `"synthetic"` or `"observed"`. |
| `cache_protocol` | A documented test protocol, including cache warmth, TTL and relevant routing conditions. |

A protocol name alone cannot prove the runs followed it. Preserve the run procedure
and source records. The comparison refuses mismatched controls rather than guessing
how to adjust costs across models, workloads or cache conditions.

### Requests and task outcomes

Each request requires `request_id`, `task_id`, `session_id`, `sequence`, `kind` and
`usage`, in addition to `type: "request"`.

- Request IDs must be unique within the run. Record every billable attempt, including
  retries and requests belonging to tasks that fail.
- `sequence` is a nonnegative integer that strictly increases within each session.
  Gaps are allowed; the sequence is not proof that nothing was omitted.
- `kind` is `"inference"` or `"compaction"`. Include summary-generation requests as
  compaction. They are priced and included once in the task's total.
- Optional `latency_ms` records a request duration. It does not supply the whole-task
  duration used by the latency gate.
- `task_end.success` is a JSON boolean supplied by your evaluator. `duration_ms` is
  optional for an audit but required for **every task in both runs** to evaluate p95.
- End each task once. A request cannot appear after its task ends. A task that fails
  before making a model request may still have a terminal record and zero model cost.

The final record requires `type: "run_end"`, `complete: true`, and integer
`request_count` / `task_count` matching the exported records. Unknown event fields,
duplicate JSON keys, invalid counts and non-finite numbers are rejected.

These checks catch incomplete framing and inconsistent records. An exporter can
still omit a request and adjust its counts; KVPilot cannot detect that from this
file alone. Reconcile against original provider records before claiming savings.

### Native Anthropic usage and explicit cache-write TTLs

Supply ordinary text Messages usage, not an invented total-input field.
`input_tokens` and `output_tokens` are required nonnegative integers. In Anthropic's
usage format, ordinary `input_tokens` excludes cache reads and writes. Missing
cache-read and cache-write counters are treated as zero.

For nonzero cache creation, include the provider's explicit TTL breakdown:

```json
{
  "input_tokens": 100,
  "cache_read_input_tokens": 2000,
  "cache_creation_input_tokens": 1000,
  "cache_creation": {
    "ephemeral_5m_input_tokens": 800,
    "ephemeral_1h_input_tokens": 200
  },
  "output_tokens": 20
}
```

Both TTL fields are required when the `cache_creation` object is present, and their
sum must equal `cache_creation_input_tokens`. Do not guess the billed TTL from a
request setting or assign all writes to the cheaper category. If your export lacks
the breakdown, obtain it before auditing mixed or nonzero writes.

Reported thinking tokens are included in `output_tokens`; an optional
`output_tokens_details.thinking_tokens` is not billed again. Nonzero server-tool,
server-side iteration and nontext usage counters are outside this MVP and rejected.
It cannot detect unsupported charges that the exporter hides or never records.

### Optional prefix diagnostics

A request may include `prefix_fingerprint` for a **caller-designated stable prefix**
and `prefix_change_reason` with one of:
`intentional_compaction`, `intentional_edit`, `unexpected`, `unknown`.

Define which bytes the fingerprint covers and use a consistent scheme. A change
between adjacent fingerprinted requests in the same task/session is a reason to
investigate. It is not proof of a provider cache miss, its cause or avoidable cost.
Missing fingerprints mean missing diagnostic coverage. Raw prompts are not required;
even fingerprints and task identifiers can reveal relationships, so handle exports
under the customer's data rules.

## Input 2: an explicit model-specific price card

The following numbers are **illustrative test rates, not a provider quote**:

```json
{
  "schema_version": 1,
  "pricing_id": "illustrative-v1",
  "currency": "USD",
  "source": "Illustrative test rates; not a provider quote",
  "effective_date": "2026-10-29",
  "provider": "anthropic",
  "model": "example-model",
  "rates_usd_per_million": {
    "input": "3",
    "cache_read": "0.3",
    "cache_write_5m": "3.75",
    "cache_write_1h": "6",
    "output": "15"
  }
}
```

All five categories are required. Decimal strings preserve the intended prices.
Rates must be finite and nonnegative; provider and model must match the trace.
The tool does not look up current prices or infer a model's billing tier. Use the
actual returned model identity, not an alias that may resolve differently. A single
flat card cannot model mixed long-context, premium or other per-request price tiers.

For each request, it sums `category tokens × category rate / 1,000,000`.
The expanded usage example above costs **$0.0054 at these illustrative rates**.
Cache writes are split by TTL, not added again as an aggregate charge.

The calculation excludes taxes, credits, contractual discounts, batch/service-tier
and geography adjustments, external tools and runtime charges. Use an appropriate
flat rate card only when that model fits your run. Report other costs separately;
token accounting is not GPU-hour accounting or invoice reconciliation.

## Reports and decisions

Each command writes `report.json`, `report.md` and a standalone `report.html` into
its output directory. Use a fresh directory for each saved experiment; reports in
an existing directory are overwritten. JSON preserves structured values and decimal
money strings; the human-readable reports round displayed amounts for readability.

| Report | Fields worth reading |
| --- | --- |
| Audit | `metadata`, `trace_sha256`, `pricing`, per-request breakdowns, `tasks`, `totals`, `findings`, `coverage`, `diagnostic_limits`, `scope_exclusions`. |
| Comparison | `status`, `passed`, `comparable`, `thresholds`, `summary`, `paired_outcomes`, `gates`, `reasons`, `limitations`, both trace hashes and metadata, shared `pricing`. |
| Plan | `inputs`, `keep_cost_usd`, `compact_cost_usd`, `estimated_net_savings_usd`, `first_strictly_cheaper_turn`, `assumptions`. |

Audit totals retain all recorded request cost, failed-task work and compaction cost.
The compaction subtotal is already included in the total; do not add it twice.
Cost per successful task divides the full cost by the successful-task count and is
`null` when no tasks succeed. Inspect outcomes alongside any cost reduction.

Comparison requires exactly the same task IDs, model, provider, workload version,
dataset, evaluator, cache protocol, evidence type and rate card. The agent revision
may change; run IDs must differ. An incompatible comparison is inconclusive and
does not calculate cross-run deltas.

The gates are:

1. **Comparability:** matching controls and task identities.
2. **Quality:** no baseline-successful task becomes a failure. New successes do not
   cancel out new failures elsewhere. This checks supplied labels, not answers.
3. **Cost:** cost-per-success savings meets `--min-savings-pct`.
4. **Latency:** p95 task-duration regression stays within `--max-p95-regression-pct`.

p95 uses the nearest-rank value over all task wall-clock durations, including failed
tasks. It does not sum request latencies. A missing duration prevents that gate from
passing. Zero baseline denominators or no successful tasks can make a decision
inconclusive. Any failed gate makes the comparison fail; otherwise an unresolved
gate makes it inconclusive. A pass is descriptive, without a statistical or causal
claim. `certifies_customer_savings` remains false.

The CLI returns exit code **2** for invalid inputs or report-writing errors. With
`--fail-on-regression`, a failed or inconclusive comparison returns **1** after writing
its report; a passing comparison returns **0**. Without that flag, inspect `status`
in the report: a completed command is not necessarily a passing comparison.

## What the compaction planner assumes

```text
net saving = future turns × removed tokens × cache-read rate / 1,000,000
             − rewrite cost − summary cost
```

The README's hypothetical inputs keep 1,000 of 15,000 tokens for 12 future turns.
At the illustrative $1/million cache-read rate, avoided reads cost $0.168. Subtract
$0.12 of rewriting and $0.04 of summarization: estimated net savings are **$0.008**.
The first strictly cheaper future turn is 12. This is arithmetic, not an experiment.

All modeled later reads are assumed cache hits at one constant rate. The rewrite
estimate must include the **entire invalidated suffix**, not just the shortened
tool output. Future turns are supplied, not predicted. Output costs, retries,
behavior and quality are assumed unchanged; unrelated task costs are excluded.
Expiry, routing misses or a summary that makes the agent fail can reverse the result.
Planner counts must be integers from zero to one billion. Monetary inputs must be
finite and nonnegative, at most one trillion, with at most 100 digits and decimal
exponent magnitude 100. Strict break-even arithmetic preserves small differences;
the human report still rounds displayed dollars, so inspect JSON at a close boundary.

## The first useful customer experiment

Start with one agent workflow and a customer-approved export from existing logs.
Agree on the complete task boundary, evaluator and latency/cost thresholds first.
Inspect native provider cache diagnostics before choosing one small change to the
request builder or context policy. KVPilot does not choose or apply that change yet.

Run baseline and candidate on the same held-out tasks under a documented cache
protocol. Keep model identity, tool environment and evaluator fixed, account for all
attempts, and use repeated or randomized ordering where state and load can matter.
Save the inputs, rate card, reports, exporter version and patch. Review failures and
individual task changes, not only an aggregate percentage. A small fixture cannot
establish statistical significance or general production quality.

Proceed only if the measured benefit survives the customer's outcome and latency
requirements and matters after other costs. A paid, repeatable need for this process
is still a hypothesis. The earlier [self-hosted plan](product-plan-2026-10-29.md)
documents a different buyer path and is retained for history.

## Relevant existing features

- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): native cache behavior and usage categories; consult the current model-specific pricing separately.
- [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching): existing provider functionality; an OpenAI usage importer is not part of this MVP.
- [LMCache CacheBlend](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html): existing non-prefix reuse with selective recomputation.
- [Dynamo agent simulation](https://docs.nvidia.com/dynamo/agents/agent-simulation): existing workload replay tools for self-hosted inference.

These links explain why KVPilot claims neither a new caching invention nor an empty
market. The proposed value is a verified workflow improvement beyond existing tools.
