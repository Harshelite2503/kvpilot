# P100 and RTX 5060 Ti: adaptive has not beaten simple admission

The completed admission experiment tests our own cache management in a small Qwen2.5-0.5B-Instruct runner. Each system has 2,592 equal-work timing sessions and 2,592 free-generation sessions, plus two smoke runs. The two modes answer different questions: fixed continuation inputs control computational work; normal greedy generation checks answer changes. The underlying dataset has **24 generated questions**, repeated across contexts, scenarios and three runs.

## The measured result

These means weight all six workload cells equally, with 72 started sessions per policy per cell. They include original prefill, pause/export/codec work, restore or recomputation, generation and canceled work. They do not represent a customer workload distribution or billing savings.

| Adaptive relative to baseline | P100 | RTX 5060 Ti |
|---|---:|---:|
| Mean full-session time vs owned CPU INT4 | 16.11% lower | 16.82% lower |
| Mean full-session time vs simple admission | **3.66% higher** | **0.29% higher** |
| Peak GPU allocation reduction | None measured | None measured |
| Free-generation tokens identical to recompute | 396/396 | 396/396 |
| Adaptive INT4 / INT8 decisions | 0 / 0 | 0 / 0 |

Simple admission retains a native snapshot when space is free and otherwise recomputes; it never evicts. Adaptive is slower than this baseline in the balanced mixture in all three repeats on both systems. This is descriptive evidence; repeated questions and scenarios are not independent samples, and no significance claim is made.

| Context | Scenario | P100 adaptive time vs simple admission | RTX adaptive time vs simple admission |
|---:|---|---:|---:|
| 512 | Cancellation | 1.09% more | 0.72% more |
| 512 | Pressure | 6.97% more | 0.13% more |
| 512 | Roomy | 2.47% less | 0.29% more |
| 2,048 | Cancellation | 6.21% more | 2.20% more |
| 2,048 | Pressure | 6.36% more | 0.66% less |
| 2,048 | Roomy | 0.56% more | 0.60% less |

Adaptive beats fixed CPU INT4 in every cell, but that reference spends substantial CPU time encoding and decoding. This is not a comparison against an optimized GPU codec, LMCache or a production serving engine. A win over this codec alone is insufficient to justify the adaptive controller.

## What the decision records explain

In P100 equal-work timing, adaptive makes **233 native admissions and 199 recompute decisions**, compared with simple admission's **216 native admissions and 216 recomputes**. The extra 17 admissions yield five fewer hits overall. In pressure cases, adaptive admits 20 and 19 snapshots versus simple admission's 18, while all still get 18 hits. In cancellation cases, it admits 24 and 26 versus 18, but gets 12 and 13 hits versus 15.

The RTX timing run makes 228 native admissions and 204 recompute decisions. Relative to simple admission, that is 12 extra admissions and six fewer hits. Roomy and pressure action traces match simple admission; their small timing differences do not show smarter policy behavior.

This identifies wasted decisions worth investigating, not the exact cause of every millisecond. P100 prefill and generation times vary materially across blocks despite equal token work. Extra admissions, noisy cost estimates and host/runtime variation must be separated before assigning the entire slowdown to the policy.

Timing and quality calibrate separately. In **free generation**, both systems' adaptive choices are 228 native admissions and 204 recomputations. It would be incorrect to use those quality-run decisions to describe P100 timing.

## Memory and answer quality

Every policy has the same peak CUDA allocation within a context: approximately 977.93 MiB at context 512 and 1,044.45 MiB at context 2,048, on both systems. The pressure limit controls retained **host snapshot bytes**, not GPU VRAM. At context 2,048 with a roomy store, adaptive retains 188.24 MiB versus INT4's 50.07 MiB, about 3.76 times as much. CPU compression occurs after native GPU export and therefore does not reduce transferred GPU-to-host bytes.

Adaptive and simple admission each preserve recomputation's tokens on all 396 response observations per system. Their strict exact-answer score is 243/396; fixed INT4 scores 105/396 on P100 and 132/396 on RTX. No policy truncates a response in these runs.

The strict score rejects extra wording, even when the expected number is present. Of INT4's 138 P100 exact-match regressions, 57 contain the expected number; RTX has 111 regressions, with 42 containing it. Numeric occurrence is a diagnostic, not a semantic rescore, and those regressions must not all be described as factual errors. Baseline preservation on a small synthetic set is not broad quality certification. No adaptive lossy path was exercised.

## What was controlled

- Model files, all 25 recorded benchmark source hashes, prompts, expected answers, schedules, budgets and workload arguments match. The commit labels differ: P100 `b91ebb37ead55d21dabe23ef0311ae2a9d2dc697`, RTX `ca6f6feaaf7faef4b942bd37d78ee29ded99a27a`. Intervening changes are outside the benchmark implementation.
- Both full modes use contexts 512/2,048, 24 cases, three repeats, groups of eight and roomy/pressure/cancellation schedules. Equal-work uses 24 forced steps; free generation allows at most 24 output tokens.
- Frozen continuations match for **47/48** context-case pairs. Context 512, `case-0002`, diverges at the eighth token, after EOS at the third token. Within each machine every policy uses identical inputs. Across machines, token inputs are not fully identical.
- The normal free-generation recompute tokens match across machines on all 396 paired responses.
- P100 uses PyTorch 2.7.1+cu126; RTX uses 2.7.1+cu128. Operating systems differ, and CPU/PCIe/driver settings are not matched or fully documented. P100 lacks the later CPU/thread/cgroup runtime sidecar. This is not a GPU-only causal comparison.

All eight supplied smoke/full runs pass the independent artifact/accounting/score audit. P100's test log records **512 passed, 11 skipped, 89 subtests passed**; RTX records **548 passed, 11 skipped**. These are historical test results, not the current PR's test count. Audits establish record consistency, not independently signed hardware execution.

The P100 folder has individual run manifests but no enclosing suite manifest. Its import hashes record what was received. RTX has an enclosing manifest. Original archive bytes are retained unchanged in the [evidence record](../results/policy-replication-2026-10-29/README.md).

## Counterfactuals and failed attempts

Both completed suites include a break-even sweep that lowers modeled lossy codec cost. These are simulations, not extra measurements. Faster codecs also improve fixed INT4, so it must remain a baseline at the same cost scale. The [HTML report](../results/policy-replication-2026-10-29/report.html) lists scales and comparator sets explicitly. The sweep proves neither a viable implementation nor a quality-preserving adaptive win.

The first RTX rental produced an audited equal-work run, but its full-quality bundle was not retrieved before shutdown. Its timing and later offline sweep are preserved separately and excluded from the paired comparison. A second interrupted attempt produced no useful benchmark evidence; its disappearance cause was not established. The third attempt supplied both completed modes. Current provider-reported total spend was $0.657, including $0.224 for the completed attempt; that snapshot can receive later billing adjustments. These are experiment costs, not customer savings.

## Decision

Keep simple admission as the immediate exact-cache baseline. Improve instrumentation and causal trace replay before adding a learned predictor or another codec. Test replacements against the value of keeping current entries, using only signals available when the decision is made. Evaluate any change on fresh, frozen workload families and include TTL, idleness and recomputation baselines where the harness actually implements their time semantics.

The commercial direction is now defined in the [product plan](product-plan-2026-10-29.md). Lower sequential milliseconds only become lower spend when a real deployment can provision fewer paid resources at unchanged throughput, latency and quality.
