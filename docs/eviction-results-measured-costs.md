# Eviction holdout rerun under measured codec costs

Run 28 October 2026 on `main` (`b2dc97b`) with Python 3.13.0 and numpy 2.5.4. This is a
**cost-sensitivity reanalysis on already revealed evaluation seeds**, not a new unseen holdout.
It answers the question raised in PR #1: does the published eviction result survive when the
assumed codec costs are replaced with the ones measured on a real KV tensor?

## What changed and what did not

- **Unchanged:** source code, workloads, scenarios, seeds 1101 to 1120, 240 sessions per run,
  six policies, predeclared metrics, bootstrap settings. The published v1 archive was first
  reproduced from the frozen manifest; `summary.json` and `runs.jsonl` were byte-identical to
  `results/eviction-holdout-v1/`.
- **Changed:** only the four codec entries in the protocol's base cost config, taken verbatim
  from PR #1's `examples/measured-costs.json`. Those numbers are the CPU timings of KVPilot's
  own NumPy reference codecs on a Qwen2.5-0.5B KV slice captured from vLLM on an RTX 3090.
  They are not GPU kernel timings. Bandwidth, budgets, resume target, pause distributions and
  storage penalties remain assumptions.

| Codec | Ratio assumed → measured | Encode ms/MiB assumed → measured | Decode ms/MiB assumed → measured |
|---|---|---|---|
| raw_fp16 | 1.00 → 1.00 | 0 → 6.45 | 0 → 2.75 |
| zlib_fp16 | 1.08 → 3.25 | 20 → 28.2 | 4 → 5.62 |
| group_int8 | 1.94 → 1.94 | 0.15 → 15.8 | 0.08 → 8.12 |
| group_int4 | 3.76 → 3.76 | 0.25 → 17.1 | 0.15 → 13.5 |

Protocol: `examples/holdout-protocol-measured-costs.json`. Freeze manifest:
`results/eviction-freeze-measured-costs-v1.json` (approval note states the sensitivity-rerun
status). Archive: `results/eviction-holdout-measured-costs-v1/`.

## Headline: the published result does not survive

Primary metric, paired mean per-run p95 modeled resume delay, `adaptive_eviction` versus fixed
`group_int4`, 20 seeds per scenario, 95% bootstrap intervals:

| Scenario | Assumed costs (published v1) | Measured costs (this run) |
|---|---|---|
| sustained_pressure | 14/6/0, +3.2% [+1.6, +5.4] | 12/8/0, +1.4% [+0.8, +2.2] |
| branching_pressure | 1/17/2, −0.3% [−0.9, +0.1] | 13/7/0, +1.5% [+0.9, +2.3] |
| fast_return | 20/0/0, **+47.1%** | 0/20/0, **+0.0%** |
| cancel_heavy | 3/9/8, −1.4% [−2.9, −0.1] | 2/18/0, +0.5% [+0.0, +1.4] |
| low_pressure | 20/0/0, **+62.5%** | 0/20/0, **+0.0%** |
| heterogeneous | 17/3/0, +11.3% [+6.0, +18.1] | 13/7/0, +2.6% [+1.1, +4.8] |
| shifted_pause_mix | 15/5/0, +6.1% [+2.5, +10.2] | 12/8/0, +0.9% [+0.4, +1.3] |

(W/T/L = wins/ties/losses by seed; positive is lower p95 for the candidate.)

Three things happen at once:

1. **The two large wins vanish.** In `fast_return` and `low_pressure`, the +47% and +62% p95
   reductions become exact ties for the candidate versus INT4 and recompute (129.9 ms and
   130.5 ms). This does not mean every resume recomputes: across the 20 `low_pressure` runs,
   the candidate restores 40 of 4,688 resumes and fixed native restores 603. Native's mean
   run-p95 is 130.44047 ms, compared with recompute's 130.53575 ms.
2. **The remaining "wins" are tiny and against a degenerate baseline.** Fixed INT4 still
   encodes everything on pause but never restores anything on resume, so its p95 is just
   recompute's p95 exactly. The simulator models no service queues. Beating this baseline
   by 0.5 to 2.6% does not establish a useful improvement in total work.
3. **The cancel-heavy p95 regression disappears.** The candidate's small p95 improvement
   still comes with substantially more modeled work than plain recomputation.

Counting families, the candidate still "wins five of seven" on the primary metric. That
statement is technically true in both runs and meaningless in the second. The honest
headline is: **under measured codec costs no policy, including the candidate, materially
improves modeled resume latency over plain recomputation, and every caching policy's
scenario-mean total work exceeds recomputation's.**

## The comparison that matters: candidate versus plain recompute

The protocol never compared the candidate to the simplest policy directly. Computed from the
same run rows (paired by seed):

| Scenario | p95 impr. assumed | p95 impr. measured | Total service work vs recompute, measured |
|---|---|---|---|
| sustained_pressure | +5.0% (18/2/0) | +1.4% (12/8/0) | **1.82×** more work |
| branching_pressure | +2.9% (16/4/0) | +1.5% (13/7/0) | 1.87× |
| fast_return | +94.0% (20/0/0) | +0.0% (0/20/0) | 1.01× |
| cancel_heavy | +4.5% (11/9/0) | +0.5% (2/18/0) | **2.65×** |
| low_pressure | +98.5% (20/0/0) | +0.0% (0/20/0) | 1.04× |
| heterogeneous | +36.2% (20/0/0) | +2.6% (13/7/0) | 1.57× |
| shifted_pause_mix | +23.0% (20/0/0) | +0.9% (12/8/0) | 2.02× |

Under measured costs the candidate buys at most 2.6% lower p95 and at most 3.8% lower mean
resume delay. Its ratio of scenario-mean total modeled service work to recomputation ranges
from 1.01× to 2.65× (1.57× to 2.65× in the five families with a p95 improvement). It
does not pass Astra's first gate ("beat the simplest credible policy").

Fixed INT4 is far worse: its ratio of scenario-mean total work is 5.28× to 37.02×
recomputation's, because it pays encode on every pause and then recomputes anyway.
Fixed raw FP16 ranges from 2.64× to 14.74× on the same estimator.
The adaptive policies at least avoid most of that waste, which is the only positive reading of
this run.

## Eviction pricing gives no p95 improvement under measured costs

The predeclared attribution ablation (`adaptive_eviction` versus `adaptive_exact`, identical
except for the eviction externality charge) is **0 wins, 20 ties, 0 losses in every scenario
on p95 resume delay**. Decisions and secondary metrics are not all identical. For
`heterogeneous`, seed 1105, event `heterogeneous-s1105-f0162-b0`, `adaptive_exact` chooses
`raw_fp16` while `adaptive_eviction` chooses `zlib_fp16`. Total modeled service work rises
from 19,245.800033 to 19,418.763735 ms, and peak reserved bytes rise from 92,280,360 to
94,861,717, without changing p95. Thus this run provides no primary-metric benefit from
eviction pricing and includes a secondary-resource regression. Under assumed costs its
p95 attribution was +4.7% to +22.1% in four scenarios.

## What this does and does not mean

- It does **not** show that KV retention during agent waits is worthless. It shows that with
  CPU-speed reference codecs and these pause distributions, none of the evaluated caching
  policies reduces scenario-mean total work below recomputation. Individual restores can
  still be worthwhile. A GPU codec kernel, or no codec at all (raw transfer over a fast link), is a
  different regime, and `raw_fp16` here still carries a measured 6.45 ms/MiB "encode" cost
  that a real memcpy path would not.
- It **does** show that the large latency advantages depend on the assumed codec costs.
  The frozen holdout's "five of seven families" headline alone is not evidence of savings.
- It confirms the harness works as intended: fed real numbers, it changed the conclusion.
- The useful next experiment is not a better codec. It is to sweep codec cost from zero to the
  measured values and report the break-even cost per MiB at which any caching policy beats
  recomputation under each scenario. That number, compared against what a GPU kernel or plain
  transfer actually costs, decides whether a decision layer has anything to decide.

All prior limitations apply: synthetic traces, analytical replay, no live engine, no quality
validation, lossy codecs enabled only for sensitivity. Evaluation seeds are now revealed for
both cost configurations; any policy change requires a new predeclared holdout.

## Reproduce

Both freezes bind the `src/` tree of commit `b2dc97b`. The current tree contains additional
modules (live measurement, vLLM connector) added by later commits, so running either command
from the current checkout is refused with `freeze mismatch in component_sha256`. Reproduce from
the bound revision in a throwaway worktree, copying in the two files this rerun added:

```bash
git worktree add /tmp/kvpilot-b2dc97b b2dc97b
cp examples/holdout-protocol-measured-costs.json /tmp/kvpilot-b2dc97b/examples/
cp results/eviction-freeze-measured-costs-v1.json /tmp/kvpilot-b2dc97b/results/
cd /tmp/kvpilot-b2dc97b
python scripts/evaluate_eviction.py --split holdout --freeze-manifest results/eviction-freeze-v1.json --output-dir /tmp/kvpilot-v1-reproduction
python scripts/evaluate_eviction.py --split holdout --protocol examples/holdout-protocol-measured-costs.json --freeze-manifest results/eviction-freeze-measured-costs-v1.json --output-dir /tmp/kvpilot-measured-reproduction
```

Verified on 28 October 2026 after the P100 branch was merged: both regenerated `summary.json`
files are byte-identical to the checked-in archives. Each run takes about 20 seconds on an
M1 Pro. Output directories must not exist.
