# KVPilot test and benchmark plan

**Every change to KVPilot must show, with reproducible evidence, how much KVPilot's own code improves on the best existing alternative, and nothing else.** Gains produced by vLLM, LMCache, Ollama, MLX, CUDA kernels, the GPU, the model, the workload's shape, or the plain fact of reusing a cache are not KVPilot's and are never reported as KVPilot results.

This plan builds on [evaluation.md](evaluation.md) and does not replace it. That contract already defines the evidence classes, the incumbent baseline matrix (B0–B5), the metrics and the statistical gates. This document adds four things:

- the attribution rule that decides what counts as KVPilot's contribution (§1–2);
- an audit of every existing claim against that rule, each verdict backed by a reproducible measurement (§3);
- the baselines and controls that make the rule measurable (§4–8);
- a per-change gate that says which tests each kind of change must pass (§9–11).

## 1. The attribution rule

> KVPilot is credited with an improvement in metric *m* only when **(a)** two arms differ in exactly one KVPilot-owned layer, **(b)** the comparison arm is the strongest non-KVPilot alternative for that layer under the same configured budget and quality constraint, **(c)** counters prove the KVPilot code path actually served the work, **(d)** the paired confidence interval of the pre-registered comparison excludes zero, and **(e)** that interval comes from at least the minimum number of independent units (§7).

Rule (a) applies to each attribution term in §2. "KVPilot total" is the one term that spans layers, and it is compared against all non-KVPilot arms at once.

```
credit(m) = m(candidate) − m(strongest alternative under the same configured budget)
```

Every claim reports two comparisons:

- **Same budget (measured).** All arms in a cell run under one configured budget, so this compares the candidate with the fastest baseline that actually ran. It needs no model of what might have happened, and it is the default primary comparison.
- **Frontier (memory trade-off).** The frontier is the lower envelope of the baseline arms on the (stored bytes, metric) plane. At a given byte level, it is the best result that one baseline, or a split of sessions between two baselines, could reach. Use it when the claim is about memory. A frontier point between two arms is an *interpolation*: nobody ran that split, and under a capacity limit its real time is not linear in bytes. An interpolated point can motivate a budget-matched run, but cannot grant credit by itself.

A candidate that trades more bytes for less time, or the reverse, is never compared against whichever baseline happens to flatter it most. [`scripts/audit_attribution.py frontier`](../scripts/audit_attribution.py) computes both comparisons. It marks each frontier point as measured or interpolated, and classifies each candidate as `credited`, `regression`, `inconclusive` or, below the unit floor, `insufficient_units`.

**Forbidden comparisons.** Each is illustrated by a case from this repository (§3):

| # | Never report as a KVPilot gain | Why |
|---|---|---|
| F1 | KVPilot versus recompute or versus no cache. | Any cache gives that speedup. It belongs to reuse, not to KVPilot. |
| F2 | A first request versus a repeat request. | Warmup, lazy initialisation and order effects dominate (§3, A1). |
| F3 | Arms run under different configured budgets, or a byte saving reported without its time cost (or the reverse). | In the A9 pressure cell, 9% fewer bytes than fixed INT4 came with 26.8% more time. [local-results.md](local-results.md) reports both; a one-axis summary would not. |
| F4 | Simulator output presented as a measurement, or different evidence classes pooled in one chart. | Restated from [evaluation.md](evaluation.md). |
| F5 | Costs measured on one host used to model a different host. | The same code runs 1.9–9.1× apart on two CPUs (§3, A5). |
| F6 | Results on degenerate or repetitive inputs presented as representative. | Repetition inflated a compression ratio 33× for half the tensor (§3, A4). |
| F7 | A component credited although its counter is zero. | A faster arm whose KVPilot path never ran proves nothing about KVPilot. |
| F8 | A policy credited against baselines that lack a codec the policy can choose. | The gain may come from the codec's availability, not the decision (§3, A9). |
| F9 | A reimplemented baseline that is weaker than the system it stands in for. | Whole-session LRU is not vLLM's offload, which works per block, loads partially and shares common prefixes. All 8 A9 sessions share one context (§3, A9; §4, R8). |
| F10 | Cases or repeats counted as independent units when they share a store or replay identical decisions. | Case-level intervals called a 1.4% gap significant. Per repeat, its sign flips (§3, A9). |

## 2. Layers: what is KVPilot's and what is not

| Layer | Owner | In this repository | Credited to KVPilot? |
|---|---|---|---|
| L0 Hardware, driver, OS, Python/NumPy build | External | — | Never. Control it by running every compared arm on the same host. |
| L1 Inference engine: attention kernels, paged KV, CUDA graphs, scheduler, native prefix cache | vLLM / MLX / PyTorch | Owned runners call PyTorch/MLX operations; the vLLM tier uses vLLM unchanged | Never. Hold it identical across arms. |
| L2 Offload store and transport | vLLM `OffloadingConnector` (per block, partial loads), vLLM prefix cache, LMCache, **or KVPilot's store** | `experiments/local/store.py`, `experiments/torch/`, `src/kvpilot/integrations/vllm_connector.py` (scaffold) | Only the delta against the strongest external store at the same budget. Integration overhead is charged against KVPilot (arm K0, §4). |
| L3 Representation / codec | vLLM FP8, CacheGen, **or KVPilot codecs** | `src/kvpilot/codecs.py`, `experiments/local/state_codec.py` | Only against the best external representation at equal quality. Never against raw FP16. |
| L4 Policy: admission, representation choice, eviction | LRU/ARC/TTL incumbents, **or KVPilot** | `src/kvpilot/policy.py`, `src/kvpilot/replay.py` (`estimate_eviction_cost`, `plan_lru_evictions`) | Only against fixed and simple policies over the same codec set (F8). |

Decompose every result into these terms. Each term is a paired difference between two arms (§4):

| Term | Arms compared | Question it answers |
|---|---|---|
| Integration overhead | K0 − E1 | What does routing through KVPilot cost when it decides nothing? |
| Codec value | K1 − frontier(E1, E2, E3) and R-codec arms | Does KVPilot's representation beat FP8/CacheGen at equal quality? |
| Policy value | K2 − best of (fixed arm of every codec in the set, static thresholds, ARC, TTL, and the shared-prefix store R8 when prefixes are shared) | Does deciding beat simple rules over the same options? |
| Forecast share (diagnostic) | K2 with its learned forecast − K2 with the true resume probability | How much of the policy's result comes from the forecast rather than the decision rule? Never a claim. |
| Eviction-pricing value | K3 − K2 (`adaptive_eviction` − `adaptive_exact`) | Does charging harm to evicted owners help? This is internal to KVPilot. |
| **KVPilot total** | K3 − frontier(all non-KVPilot arms) | The only number that may be called "KVPilot's improvement". |

## 3. Audit of existing claims

Each line below re-checks a published number against §1. Every figure is reproducible with the commands in §12, is computed directly from the archived rows, or comes from a recorded engine log line. The captured tensor `real_kv_capture.npy` (SHA-256 `f610c81f…77636`) is gitignored by the repository's existing `.npy` rule, so its hash is recorded here for identity.

| ID | Claim | Source | Whose effect | Verdict |
|---|---|---|---|---|
| A1 | "Real 2.47× speedup" | [phase1-report.md](phase1-report.md) | Multiprocess and first-request overhead, then vLLM's prefix cache | Not KVPilot. The cache's own share is not isolated (F1, F2) |
| A2 | "~1,600-token prefix" | phase1-report.md | — | Wrong: ≈2,426-token prompt. The capture's 2,445 rows include 19 generated tokens |
| A3 | Pool shape `[56139, 2, 16, 128]`, "axis semantics unverified" | phase1-report.md | — | Now verified per slice: `[blocks, kv_heads=2, block_size=16, K₆₄‖V₆₄]` |
| A4 | "zlib 3.25× on real KV" | phase1-report.md | Prompt repetition in layer 0 | Artifact (F6) |
| A5 | "Assumed timings ~100× too optimistic" | phase1-report.md | The laptop CPU | Category error (F5). The results file also names the wrong host |
| A6 | "Python loop over groups" as the explanation | phase1-report.md | NumPy dtype conversions | False |
| A7 | "`raw_fp16` is lossless" | codecs, calibration | — | Lossless only against the FP16 reference, not the engine's bf16 |
| A8 | "Adaptive flips to recompute 60/60" | phase1-report.md §7 | Laptop CPU costs plus assumed bandwidth | Simulation property, not a serving result (F4, F5) |
| A9 | Owned-runner adaptive vs fixed INT4: 58.5% less service time (roomy); 9% fewer bytes but 26.8% more time (pressure) | [local-results.md](local-results.md) | Choosing native storage (roomy); forecast and codec menu (pressure) | Reported correctly, including that no adaptive advantage was shown. Not gradeable: one replay of 8 sessions. Strongest baselines missing (F8, F9) |
| A10 | Eviction holdout: "better p95 than fixed INT4 in 5 of 7 families" | [eviction-results.md](eviction-results.md) | Assumed costs | Simulation only. Fixed INT4 is the right comparator for policy value, but there is no external comparator for a total claim |
| A11 | Ollama 0.079 s repeated-prompt TTFT | [ollama-results.md](ollama-results.md) | Ollama's native cache | Already labeled as not KVPilot; correct |

**A1, the 2.47× is mostly multiprocess and first-request overhead, and the cache's own share is not isolated.** The 1,490 ms first request ran in vLLM's default multiprocess mode; its repeat took 604 ms. Two later first requests on the identical prompt, each in a fresh in-process engine, took **634 ms** and **≈595 ms** (vLLM's progress line, 1.68 it/s). The in-process repeat took 499 ms. Switching to in-process mode coincided with ~850–900 ms less on the first request and 105 ms less on the repeat, from one sample each.

The remaining in-process gap of 96–135 ms (1.19–1.27×) is only an *upper bound* on the prefix cache's saving. Both in-process "cold" requests were also the first request after model load, so they include one-time work: the same first-versus-repeat error as F2. Isolating the cache needs a warmed engine and a cold control prompt of equal length.

There are four more problems:
- All requests were dominated by 20 decode tokens with CUDA graphs disabled (`enforce_eager=True`).
- The timer measured whole `generate()` calls, not TTFT.
- `num_cached_tokens` was never read to confirm a hit.
- Nothing was repeated.

None of it involved KVPilot.

**A2, the prompt was ≈2,426 tokens.** The capture holds 153 blocks × 16 slots = 2,448 slots. The last 3 were never written, which leaves 2,445 rows. Generation also writes KV for 19 of its 20 output tokens, so the prompt is ≈2,426 tokens. Two independent checks agree:
- V repeats with period 12 over rows 15–2,418: 200 copies of the 12-token function, plus boundary rows. That leaves 15 header rows and 26 tail rows, which hold the question and the 19 generated tokens.
- vLLM's input-throughput lines give 2,423–2,432 tokens (for example, 4,085 tok/s at 1.68 it/s, and 4,119.56 tok/s at 1.70 it/s).

**A3, axis semantics, from three independent checks.**
- *Arithmetic:* 56,139 × 16 = 898,224, exactly vLLM's logged token capacity, so axis 2 is `block_size`.
- *Config:* the pinned [model config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json) gives 24 layers, 2 KV heads, and head_dim = 896 / 14 = 64. Then 56,139 × 16 × 2 × 128 × 2 B × 24 layers = 11,037,376,512 B = **10.28 GiB**, which matches the engine log exactly.
- *Data, per slice:* both axis-1 slices behave identically. Channels 64–127 repeat exactly every 12 tokens (98.15% of rows, 47 unique of 2,445), which only position-free V can do. Channels 0–63 never repeat. Their rotate-half RoPE pair norms agree to 0.09% at the same lag, against 1.49% for the raw values, which is the signature of rotated K. If axis 1 were K/V instead, one slice of channels 64–127 would be rotated K and could not repeat.

The layout is `[blocks, kv_heads, block_size, K‖V]`. Calling axis 1 "K/V" and axis 2 "16 heads" was wrong.

**A4, the 3.25× zlib ratio is an artifact.** The capture is layer 0, where V depends on the token alone, and the prompt repeats one 12-token function 200 times. Split by component:
- V compresses **42.45×**. After duplicate rows are removed, it compresses only **1.28×**.
- K compresses **1.69×**.

The engine's native bf16 bytes give nearly the same ratios (K 1.64×, deduplicated V 1.26×, whole 3.16×). So the FP16 cast's three zero mantissa bits, present in 99.9% of values, are not what zlib exploits. Deeper layers, whose V depends on context, were never captured. No general zlib ratio for request KV has been measured.

**A5, the "100×" compared a laptop with an assumption.** [`examples/assumed-costs.json`](../examples/assumed-costs.json) never claimed to describe the NumPy reference codecs. The README says modeled timings do not reuse CPU benchmark timings. Rerunning the committed benchmark with the identical fixture, seed, shape and 30 repeats on the Windows laptop used for calibration gives medians **1.9× to 9.1× slower** than the committed M1 numbers:

| Codec | Encode, M1 / laptop | Decode, M1 / laptop |
|---|---|---|
| `raw_fp16` | 1.38 / 6.92 ms (5.0×) | 0.28 / 1.04 ms (3.7×) |
| `zlib_fp16` | 16.40 / 31.03 ms (1.9×) | 1.61 / 3.98 ms (2.5×) |
| `group_int8` | 1.91 / 10.76 ms (5.6×) | 0.40 / 3.64 ms (9.1×) |
| `group_int4` | 2.16 / 12.30 ms (5.7×) | 2.99 / 8.80 ms (2.9×) |

Three reruns on 2026-10-28 reproduced the laptop column within about 10%. The calibration also used 7 repeats with no warmup, against [evaluation.md](evaluation.md)'s ≥30 with warmup.

`results/measured-costs-v1.json` therefore describes a laptop that serves nothing. The file itself says `"gpu": "RTX 3090"` and "Vast.ai", but the session log shows that `measure_codec` ran in the laptop's checkout. Its provenance must be corrected (§11).

**A6, the group loop is not the cost.** With 128 channels and `group_size=64`, the Python loop runs twice per call. Profiling 30 `group_int4` round trips attributes **43%** of self time to `ndarray.astype` (16 dtype conversions per round trip). Nearly all the rest is elementwise NumPy passes inside `encode`/`decode`, `_clip` and reductions.

**A7, lossless relative to what.** Converting the captured bf16 values to FP16 changes 2 of 625,920 values (max error 2.98 × 10⁻⁸). The reviewer's fix correctly reports this conversion separately, but "lossless" in KVPilot still means *after conversion to FP16*. Bit-exactness must be tested against the engine's native dtype (T2).

**A9, the owned-runner result under the attribution rule.** This is the repository's strongest evidence that KVPilot's mechanism works: native restore reproduces continuation logits bit for bit. Its archived rows ([`results/qwen-kvpilot-v1/rows.jsonl`](../results/qwen-kvpilot-v1/rows.jsonl)) all come from one design:
- the same 8 questions, all asked about one identical 5,596-character repository context;
- sessions paused in two waves of four, and every session resumed;
- one configured budget per scenario, shared by every arm (roomy 400 MB, pressure 194 MB);
- each arm run twice, in shuffled job order.

The two repeats made identical decisions with identical hits (0 of 80 case pairs differ). They are timing replicas of one replay, not independent units. Service time is pause plus resume.

| Scenario | Arm | Actions | Hits | Peak stored | Service ms, repeat 0 / 1 | Mean resume TTFT |
|---|---|---|---|---|---|---|
| Roomy | fixed native | native ×8 | 8/8 | 379.5 MiB | 372 / 392 | 238 ms |
| Roomy | `adaptive_exact` | native ×8 | 8/8 | 379.5 MiB | 402 / 390 | 238 ms |
| Roomy | `adaptive_eviction` | native ×8 | 8/8 | 379.5 MiB | 400 / 375 | 241 ms |
| Roomy | fixed INT4 | INT4 ×8 | 8/8 | 245.1 MiB | 958 / 911 | 478 ms |
| Pressure | recompute | — | 0/8 | 0 | 4,377 / 4,259 | 4,226 ms |
| Pressure | fixed native | native ×8 | 2/8 | 95.1 MiB | 3,339 / 3,333 | 3,190 ms |
| Pressure | fixed INT4 | INT4 ×8 | 6/8 | 183.9 MiB | 1,864 / 1,951 | 1,456 ms |
| Pressure | `adaptive_exact` | native ×8 | 2/8 | 95.1 MiB | 3,434 / 3,459 | 3,281 ms |
| Pressure | `adaptive_eviction` | native 2, INT8 2, recompute 4 | 4/8 | 167.4 MiB | 2,446 / 2,393 | 2,234 ms |

Every arm answered 14 of 16 correctly, recompute included, so quality does not separate the arms on this set.

- **Roomy: no difference.** Both adaptive arms chose native for every session, exactly as fixed native did. Their measured decision time (`planning_ms`) is 1–2 ms in every arm, fixed ones included, and at most 0.8 ms above fixed native: about 0.2% of service time. The service gaps (+1.4% and +3.6%) come from pause-time noise on identical actions. Fixed native alone moved 5% between its two repeats (372 → 392 ms). Against it, `adaptive_eviction` is +7.5% in one repeat and −4.3% in the other. A case-level bootstrap calls the gap significant only because it treats the 8 cases as independent, although they share each repeat's timing drift (F10). The 58.5% gain over fixed INT4 is a gain of not compressing, which fixed native also gets.
- **Pressure: adaptive lost to the simplest baseline on its own menu.** Under the same 194 MB budget and with the same observed quality, `adaptive_eviction` took 26.8% longer than fixed INT4 in service time (2,419 vs 1,907 ms), and its resume TTFT was 53% higher (2,234 vs 1,456 ms). It lost in both repeats. INT4's smaller states kept three of each wave's four sessions. Adaptive kept two (one native, one INT8) and recomputed four. Against fixed native, adaptive took 27.5% less time. That gain is real, but fixed native is not the strongest baseline.
- **Likely cause: the forecast, not the decision rule.** Adaptive priced retention using a learned resume probability: 0.5 for each wave's first four decisions and 0.833 for the rest. In fact every session resumed. Underpricing resumes favours recompute, which is what it chose. This is a hypothesis; the oracle-forecast diagnostic (§4) tests it.
- **Missing baselines.** No fixed-INT8 arm exists, although adaptive chose INT8 (F8). No arm shares the common context, although every session repeats the same 5,596 characters. A prefix-sharing store would hold that context once instead of eight times (F9, R8).
- **Eviction pricing (internal).** `adaptive_eviction` used 29.8% less service time than `adaptive_exact` in both repeats. It is a valid internal comparison of two KVPilot variants, on this one replay.
- **Grading.** One replay of 8 sessions is below the 30-unit floor, so `audit_attribution.py frontier` reports every cell as `insufficient_units` (§7). The frontier view agrees in direction. At adaptive's 167 MiB, the interpolated recompute/INT4 split is 2,124 ms (+13.9%), but no arm ran that split.

[local-results.md](local-results.md) already states that these runs "do not show an adaptive advantage", and it reports the 26.8% slowdown next to the byte saving. This audit adds the missing baselines, the forecast confound and the unit count.

**A10.** The eviction holdout ([eviction-results.md](eviction-results.md)) compares KVPilot policies with KVPilot's own fixed INT4 codec, under assumed costs that A5 shows are host-dependent. It remains a simulation result with no external comparator.

**Pre-existing test portability (not a claim, but it affects reproducibility).** On this Windows host, two tests in `tests/test_local_cases.py` fail because `Path.read_text()` falls back to the cp1252 locale and corrupts the fixture's em dashes. They pass under `PYTHONUTF8=1`. One test in `tests/test_p100_tooling.py` needs symlink privileges (WinError 1314). Read fixtures with `encoding="utf-8"`, and skip symlink cases where the OS refuses them.

## 4. Baselines: the existing solutions

KVPilot is compared at two levels. Report both, and never substitute one for the other.

**Algorithm level, in KVPilot's owned runners.** The substrate (model, tensors, store, timers) is identical, so the only difference between arms is the algorithm. External algorithms are *reimplemented* here and labeled as reimplementations, never as the original system.

| ID | Arm | Stands in for | Status |
|---|---|---|---|
| R0 | `recompute` | No cache | Exists |
| R1 | `raw_native` with whole-session LRU | Plain raw retention. Not vLLM's offload or LMCache, which work per block with partial loads (use R8 for those) | Exists |
| R2 | Raw native with ARC | vLLM `OffloadingConnector` `eviction_policy: arc` | **To add** |
| R3 | Raw native with calibrated TTL | The human-approval retention line of work (B5) | **To add** |
| R4 | Fixed FP8 E4M3 with per-tensor scale | vLLM `--kv-cache-dtype fp8` storage format | **To add** |
| R5 | Fixed `group_int8` | Required by F8, because adaptive can choose INT8 | **To add** |
| R6 | Static idle/size threshold with each codec, at ≥3 thresholds | B4. Traces the real frontier instead of interpolating it | **To add** |
| R7 | KIVI-style asymmetric 2-bit (per-channel K, per-token V), or an off-the-shelf quantized KV cache (for example the one in Hugging Face Transformers) where it supports the model, version recorded | Published low-bit KV quantization | **To add**; required for any codec claim below 8 bits |
| R8 | Shared-prefix block store: raw blocks keyed by prefix hash, LRU, partial hits recompute only the missing tail. For hybrid models, a recurrent-state checkpoint at each shared boundary | vLLM automatic prefix caching with `OffloadingConnector`, SGLang RadixAttention, LMCache | **To add**; required whenever sessions share a prefix |
| K-arms | `group_int4`, `adaptive_exact`, `adaptive_eviction` | KVPilot | Exist |
| K-oracle | `adaptive_exact`/`adaptive_eviction` given the true resume probability | Diagnostic for the forecast term (§2); excluded from claims | **To add** |

**System level, in vLLM 0.32.0 on a CUDA GPU.** These arms answer "what would a user get?". vLLM's native prefix cache stays **on and identical in every arm**, so its benefit cancels out of every difference.

| ID | Arm | Configuration (record the exact JSON) | Status |
|---|---|---|---|
| E0 | vLLM native prefix cache only | `enable_prefix_caching=True`, no connector | Runnable |
| E1 | Native CPU offload, LRU | `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":<B>,"eviction_policy":"lru"}}'` per the [usage guide](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/) | Runnable |
| E1a | Same with ARC | `"eviction_policy":"arc"` | Runnable |
| E2 | FP8 KV plus E1 | `--kv-cache-dtype fp8` (e4m3), attention backend pinned and logged. On Ampere this is [route-dependent](https://github.com/vllm-project/vllm/issues/60262) and needs a working FlashInfer JIT or `flashinfer-cubin`. Preflight it, else `not-run` | Preflight |
| E3 | LMCache CPU offload | A pinned [compatible pair](https://docs.lmcache.ai/getting_started/compatibility.html); record the connector class actually imported | Preflight |
| E3c | LMCache with CacheGen | Only if its smoke test passes; never substitute a proxy | Preflight |
| K0 | KVPilot connector, passthrough (raw, LRU, no decisions) | Measures integration overhead against E1 | Needs connector |
| K1 | KVPilot connector, fixed codec | Codec value at system level | Needs connector |
| K2 / K3 | KVPilot connector, `adaptive_exact` / `adaptive_eviction` | Policy and eviction-pricing value | Needs connector |
| O | Oracle with future reuse | Upper bound, excluded from all claims | Optional |

vLLM's offload works per block: by default one chunk is one GPU block, and tokens beyond the aligned load cap are recomputed ([usage guide](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/)). The owned runner's hybrid model (Qwen3.5) keeps about half of each paused state in fixed-size recurrent tensors, so R8 must checkpoint them at shared boundaries rather than per token.

The vLLM connector's lifecycle methods are not implemented yet ([live-integration.md](live-integration.md)), so K0–K3 are `not-run` today. A baseline-only run of E0–E3 is still worth paying for, because it fixes the frontier KVPilot will have to beat on that hardware.

A CPU codec running after a device-to-host copy has already paid for the full raw PCIe transfer. When K1 is compared with E2, the comparison is between a CPU NumPy codec and fused GPU kernels. That is a fair system comparison, and the expected loss reflects implementation, not algorithm. R4 against the K-codec arms is the algorithm-level answer.

## 5. Controls that keep the attribution honest

- **One variable per comparison.** Pin and record across arms: engine version and config hash, weight revision and precision, KV dtype (except the arm under test), block size, maximum context, sampling (greedy), output cap, scheduler and batch limits, CUDA graphs (**on**, the production setting; Phase 1 ran eager), attention backend, GPU and host byte budgets in bytes, and the arrival schedule.
- **Same host for every arm in a comparison,** recorded by machine ID, GPU UUID and driver. Calibrate costs on that same host, not a development machine (F5).
- **Mechanism counters on every resume.** In the vLLM tier, record per request `num_cached_tokens` ([outputs](https://docs.vllm.ai/en/stable/api/vllm/outputs/)), plus server counters `vllm:prefix_cache_hits`, `vllm:external_prefix_cache_hits` and `vllm:prompt_tokens_cached_by_source` ([metrics](https://docs.vllm.ai/en/stable/usage/metrics/)), each as a before/after delta. Also record the connector's own load log keyed by request ID. In the owned runners, record `hit`, `decode_ms`, `recompute_ms` and victims, which already exist. A resume counts toward KVPilot only if its reused tokens came from KVPilot's source.
- **Displace the GPU cache in controlled-resume trials.** Between pause and resume, either run filler sessions sized to evict the paused prefix (realistic) or call the engine's prefix-cache reset (deterministic), and record which. Then verify that local GPU hits on the resumed prefix are zero in **every** arm. Otherwise vLLM's own cache serves the resume, and that resume is reported separately, outside the KVPilot comparison.
- **Warmup.** Discard at least 3 warmup sessions per arm block, use a fresh process or a recorded cache reset per block, and never time the first request after model load (F2).
- **Order.** Randomize arm order in counterbalanced blocks (ABBA) with a recorded seed. Interleave arms over time rather than running them in sequence, so thermal and background drift cancel.
- **Instrumentation overhead.** Run a short tracing-on/tracing-off pair per [evaluation.md](evaluation.md). If tracing slows the server, sample it during performance runs.
- **Freezing.** Write `protocol.json` with source hashes, budgets, arms, unit, seeds and thresholds *before* timing. The existing owned-runner freeze does this. Any mismatch at completion invalidates the run.

## 6. Metrics

| Metric | Definition | Measured by | Attribution role |
|---|---|---|---|
| Resume TTFT (primary) | Client submit of the resumed request to the first nonempty streamed token | Streaming client timestamps, not `generate()` wall time | Primary latency claim |
| Full-session latency | Initial prefill + pause work + resume + generation | Sum of synchronized spans | Prevents partial-span inflation: the reviewer's 58.5% service-time gain was 11.2% of full session |
| ITL p95 | Per [evaluation.md](evaluation.md) | Token timestamps or engine ITL | ≤5% regression guardrail |
| Stored bytes, peak and time-weighted | Retained payload including metadata, scales and headers; byte-seconds ÷ elapsed | Store counters | Memory axis of the frontier |
| Transient memory | Peak device allocation (reset before each block) and peak host RSS | `torch.cuda.max_memory_allocated`, MLX peak, psutil | Charges staging and coexisting copies |
| Transfer bytes | D2H and H2D bytes including retries; raw bytes for CPU codecs | Connector counters, cross-checked with profiler/NVML | Shows that CPU compression after export does not save PCIe |
| Codec CPU and GPU time | Process CPU-seconds and synchronized kernel time | `time.process_time`, CUDA events | Compression is not free |
| Served-by | Tokens served from the GPU prefix cache, KVPilot, another connector, or recompute | §5 counters | F7 enforcement |
| Quality | Exact answer, token agreement with the paired recompute run, logit KL and top-1 agreement over teacher-forced continuation | Owned runner (T2) | Lossy arms must pass before any speed claim |
| Cost-model calibration | Measured ÷ predicted for encode, restore and resume per decision | Decision log vs execution log | In the reviewer's archive, the median by arm was 1.04–1.36× for encode and 1.07–1.19× for restore on hits. Flag >1.25×, because decisions rest on mispriced costs |
| Forecast calibration | Realized resume rate against the forecast each decision used | Decision log vs outcomes | The A9 policy used 0.5–0.83 while every session resumed |

## 7. Workloads and statistics

**Workloads.** Use the cells in [evaluation.md](evaluation.md): no reuse, rapid resume, paused, long idle, branching, pressure and identity invalidation. Add these rules, each from a finding in §3:

- **Representative inputs only (F6).** Use real repository files and natural text, with at least 100 distinct prompts for any codec calibration corpus. A capture fails the representativeness check if the V half's best exact-repeat rate exceeds 5% (Phase 1's was 98.2%), unless it belongs to a declared repetition-stress cell.
- **Verified captures (A2, A3, A7).** Capture from the request's block table (block IDs and valid length), not a nonzero mask. Cover every layer, report K and V separately, keep the native dtype, and record the backend layout descriptor and model revision.
- **Declare prefix sharing (F9).** Report the fraction of each session's prefix shared with other live sessions. Run both a shared-context cell and a disjoint-context cell. The A9 sessions share 100% of their repository context, which favours any prefix-sharing store.
- **Context and pressure.** Test contexts of 512, 2K and 8K tokens on the owned runners, and up to 16K or more on the vLLM tier. Test host budgets at 25%, 50% and 100% of the working set; cancellation at 0%, 20% and 50%; and 2- and 4-way branching. Test return-order permutations, not one fixed order.
- **Freeze before tuning.** Use development seeds for tuning and a sealed holdout for claims. The repository's existing freeze manifest enforces this for the simulator; extend it to the live tiers.

**Statistics.**

- **Unit of analysis (F10).** The unit is whatever interacts. Sessions that share a store or a family belong to one unit, which is resampled as a whole. The owned runner must emit a `block_id` per row, so that `audit_attribution.py frontier --unit-field block_id` resamples correctly. A unit's sessions must also vary independently: different contexts, arrival orders and seeds. Repeats that replay identical decisions are timing replicas. They measure noise but add no units. The A9 archive is one replay of 8 sessions.
- **Spend budget on units, not repeats.** In the reviewer's archive, the case-level repeat-to-repeat difference in resume TTFT was 1.9% median (3.5% mean), so 2 repeats per unit suffice. Block-level drift is larger: one arm's mean service time moved 5% between repeats. Between-unit differences and the hit/miss split dominate: the pressure arms show a CV of 0.5–1.1, against 0.01 for recompute.
- **Size from a pilot, above a floor.** Screens need at least 30 units per arm and scenario, and the tool enforces this (`--min-units`). Thirty is a floor, not a power calculation. Size the run from a pilot as n ≥ (2.8·σ_d/δ)², where σ_d is the standard deviation of per-unit paired relative differences and δ is the smallest effect worth detecting (80% power, two-sided 5%). Claims also need ≥200 session families and ≥1,000 resumes per [evaluation.md](evaluation.md).
- **Paired bootstrap.** Use ≥2,000 resamples by unit and a 95% interval. A result is `credited` only if the whole interval of the pre-registered primary comparison is below zero. The primary comparison defaults to the measured same-budget one. Memory claims must also pass the frontier comparison, at a measured point (§1).
- **One primary comparison.** Each change pre-registers one primary metric and the frontier comparison. Everything else is descriptive, which avoids hunting across many baselines for a favourable one.
- **Report hits and misses separately.** Medians of a bimodal distribution hide which mode moved.

## 8. Test tiers

| Tier | What runs | Where | Cost | Pass criteria |
|---|---|---|---|---|
| T0 Static and unit | Full suite under `PYTHONUTF8=1`; regression tests for every fixed bug, including the two P1 review findings now covered by `tests/test_live_*.py`; the causality test; the budget-never-exceeded invariant; tamper tests for validators | Any machine | Free | All pass; any new skip is justified |
| T1 Codec microbenchmark | `benchmark_codecs.py` and `audit_attribution.py capture/codec-profile`, using ≥30 repeats, warmup and a verified corpus; CPU and GPU timings reported separately | Target host | Free / cheap | Ratio includes all metadata; error measured against the native dtype; the capture passes §7 representativeness |
| T2 Quality | Owned runner: bit-exact native restore (logit max diff 0); for lossy arms, exact answers on ≥100 distinct tasks, token agreement, logit KL, top-1, and repeated re-quantization | P100 (free) or Mac | Free | Native exact; lossy within the 2-point non-inferiority margin of [evaluation.md](evaluation.md), or declared failing |
| T3 Policy simulation | `evaluate_eviction.py` with costs calibrated **on the target host**, sweeps and a sealed holdout | Any machine | Free | Beats the frontier of fixed and simple policies in the simulator; labeled `analytical_trace_replay` |
| T4 Owned end-to-end | `benchmark_p100_kvpilot.py` / `benchmark_local_kvpilot.py` with arms R0–R8, K-arms and K-oracle, then the validator, then `audit_attribution.py frontier --unit-field block_id` | P100 / Mac | Free | Candidate `credited` on its pre-registered comparison with ≥30 units, quality from T2 |
| T5 vLLM serving | Arms E0–E3, K0–K3, with §5 controls and counters | RTX 3090 (paid) | ~$0.15–0.25/hr | K0 within noise of E1; K3 `credited` against frontier(E0–E3); [evaluation.md](evaluation.md) targets for claims |
| T6 Robustness | Wrong-tenant/model/adapter collisions, corrupt payloads, partial blocks, cancelled writes, OOM behaviour, resumed branches | T4/T5 hosts | Cheap | Always a safe miss or recompute; never a stale reuse |

## 9. Per-change gate

Every pull request names the KVPilot layer it changes and runs that row of tiers. A layer that is not touched is not re-credited.

| Change type | Required tiers | Required comparison |
|---|---|---|
| Codec (new or modified encode/decode, framing) | T0, T1, T2, T4; T5 before any serving claim | Fixed arm of the new codec against R4/R5/R7 and E2/E3c at equal quality |
| Policy, cost model or forecast | T0 (including causality), T3, T4; T5 before claims | Best of the fixed arms of **every** codec in the candidate set, plus R2/R3/R6, and R8 when prefixes are shared. Report K-oracle next to it |
| Eviction or accounting | T0 invariants, T3, T4 | `adaptive_eviction` against `adaptive_exact` (internal) **and** against R2 ARC (external) |
| Connector or integration | T0, T5 (K0), T6 | K0 against E1; overhead charged in every later K-arm |
| Measurement, harness or validator | T0, plus re-validating the last frozen archive | Summaries reproduce exactly; tamper tests fail when they should |
| Documentation or claims | Claim checklist below | Every number links to an archive, an evidence class and an attribution term (§2) |

Paste this into each pull request description:

```markdown
### KVPilot attribution report
- Layer changed (L2/L3/L4): …
- Tiers run: T… (archive paths: …)
- Primary metric and frontier comparison (pre-registered in protocol.json): …
- Attribution term(s) (§2) with point estimate, 95% CI, unit, status: …
- Independent units (definition, count ≥ floor, pilot σ_d): …
- Strongest external stand-in for this workload present (R8 if prefixes are shared, R7 below 8 bits): yes/no
- Served-by counters show the KVPilot path executed: yes/no (paths)
- Same host for all arms (ID): … ; costs calibrated on that host: yes/no
- Forbidden comparisons F1–F8 checked: yes
- Negative results and failed cells included: yes (paths)
```

## 10. Evidence package for any claim

- **Manifest:** git SHA, source hashes, Python/NumPy/PyTorch/vLLM/LMCache versions, GPU model, UUID, driver and CUDA, model revision and file hashes, config and workload hashes, seeds, budgets in bytes.
- **Raw rows:** every request and decision with predicted and measured costs and served-by counters. Never only aggregates.
- **Independent validation:** a standard-library validator that recomputes every summary cell from raw rows. This exists for the owned runners; extend it to T5.
- **Attribution ledger:** the `audit_attribution.py frontier` JSON output, which records the unit, arms, frontier, delta, interval and status for each candidate.
- **Negative results:** retained and linked. A run is never re-executed to replace an unfavourable one; the original stays in the archive.

## 11. Hardware, sequencing and spend

1. **Gate 0, free, before any paid run.**
   - Read fixtures as UTF-8.
   - Emit `block_id` in owned-runner rows.
   - Add arms R2–R8 and the K-oracle diagnostic to the owned runner, plus a disjoint-context workload cell.
   - Replace nonzero-mask capture with block-table capture across all layers.
   - Correct the Phase 1 report's prompt length, axis labels, and its zlib, "100×" and loop explanations (§3). Correct the host label in `results/measured-costs-v1.json`, or regenerate the file on the serving host.
2. **T4 on the free P100** with the new arms. This answers the policy and codec questions at algorithm level for zero cost. vLLM 0.32.0 does not support the P100 (compute capability 6.0, below its 7.5 minimum; see [p100-runner.md](p100-runner.md)), so the owned PyTorch runner is the right tool there.
3. **T5 baseline-only on a rented RTX 3090.**
   - Run E0–E3 to pin the external frontier.
   - Use a CUDA **devel** image. In Phase 1 the `pytorch/pytorch` runtime image had no C compiler, so Triton failed until `build-essential` was installed, and FlashInfer JIT needs `nvcc`/`ninja`. If FP8 cannot run on Ampere, rent an Ada-generation GPU for E2 or mark it `not-run`.
   - A 0.5B model needs well under a second of GPU time per session, so expect single-digit GPU-hours, roughly $1–3 at the observed rates. Phase 1 cost $0.46 for about three hours, mostly setup. Record actual spend.
4. **Implement the connector, K0 first,** then run T5 K-arms on the same host and image as step 3.
5. **Make performance claims only from the sealed holdout,** using the targets in [evaluation.md](evaluation.md) (≥20% lower p95 resume TTFT and ≥15% fewer time-weighted stored bytes than the strongest compatible baseline, ITL within 5%, quality non-inferior).

## 12. Reproducing the audit

```sh
# A3/A4/A7: axis semantics, repetition, zlib by component, bf16 conversion
python scripts/audit_attribution.py capture real_kv_capture.npy
# A6: where codec time goes
python scripts/audit_attribution.py codec-profile real_kv_capture.npy --codec group_int4
# A5: same fixture as the committed M1 run, on another host
python scripts/benchmark_codecs.py --fixture gaussian --shape 2,8,256,64 --group-size 64 \
  --repeats 30 --warmup 2 --seed 42 --output <new-file>.json
# A9: same-budget and frontier attribution on the owned-runner archive
# (reports insufficient_units: 8 cases; the per-repeat table groups rows.jsonl by scenario, policy, repeat)
python scripts/audit_attribution.py frontier results/qwen-kvpilot-v1/rows.jsonl
# T0 on Windows hosts
PYTHONUTF8=1 python -m pytest -q
```

A1 and A2 come from the vLLM log lines recorded in [phase1-report.md](phase1-report.md) and the Phase 1 session output. The capture file is available on request. Its SHA-256 is `f610c81ffd424b34377985f31d7499560793bbd1fd7259d3afc949fe97877636`.
