# Real local KVPilot results

**The integration works. The current adaptive policy does not beat fixed INT4 on both time and stored memory.** These are measurements from KVPilot's own state lifecycle with the user's local Qwen3.5:4b weights, not Ollama's native prefix cache or the analytical simulator.

On this Mac, the adaptive policy was faster with plenty of space because it chose exact snapshots, which also used more memory. Under a tight capacity it saved some stored bytes but had to recompute more sessions than fixed INT4. The same seven of eight distinct questions were correct in every arm. INT4 changed the answer to the remaining question, which both versions got wrong.

## What actually ran

Apple M1 Pro, 16 GB unified memory, macOS 27.0.1, Python 3.13.0, MLX/MLX-Metal 0.32.4, NumPy 2.5.4 and tokenizers 0.23.3. The direct text runner reads the existing NVFP4 checkpoint files and uses FP16 activations and FP32 recurrent state. It uses no Ollama API, MLX-LM cache/model classes, external cache-management library, rented GPU or cloud inference. Mathematical references are attributed in [the notices](../experiments/local/THIRD_PARTY_NOTICES.md); this is not a claim that we invented Qwen's numerical operations.

KVPilot runs the prefix into explicit owned tensors, selects a representation, serializes it, deletes live state and temporary NumPy snapshots, and retains only the selected byte payload in its store. On return it decodes and imports that state, or recomputes if it was discarded or evicted. Model identity, tensor layout, offset and token-prefix identity are checked before restoration. Candidate sizing uses shape/dtype metadata without encoding all formats for free.

The full-model integrity gate saved a **1,417-token prefix**, deleted its original live state, restored the exact snapshot, and processed the next 64 tokens. The resulting logits were **byte-for-byte identical**, with maximum absolute difference **0**. All 64 saved tensors also passed native bytewise roundtrip checks. This validates our exact save/restore boundary; it is not an equivalence test against native Ollama.

The measured fixture has eight new controlled repository questions with a shared 1,448-token context. Saved prefixes range from **1,453 to 1,472 tokens**; another 64 prompt tokens are processed after restoration, followed by greedy generation capped at 32 tokens. Five policies, two capacities and two repetitions produce **160 responses across 20 blocks**. Those are only eight distinct, correlated tasks, not 160 independent quality observations. These development fixtures are not an unseen production holdout.

Each block pauses four sessions and resumes them in order 0, 2, 1, 3, then repeats with the other four questions. Every session returns. The store holds whole snapshots with no prefix sharing. The roomy budget is **400,066,652 bytes**; the tight budget is **194,050,207 bytes**. These deliberately chosen boundaries fit four native or approximately three INT4 snapshots respectively. All policies receive the same cases, event order and scenario budget. Block order is shuffled with a fixed seed.

Codec/export/import costs were calibrated on separate text and frozen before the evaluated runs. Decisions use measured initial prefill cost, current residency and resolved-session history. They receive neither future return outcomes nor expected answers. Actual elapsed time is measured with synchronized model operations; no transfer delays are simulated.

## Measured time and stored bytes

“Mean pause + resume” includes planning, export, encoding, commit, state release, restoration or recomputation, suffix processing, answer generation and cleanup. It excludes the original prefix computation, which every arm performs before pausing. “Resume TTFT” starts when the single runner begins servicing the resume; it includes restoration or recomputation and ends at the first selected output token. These are sequential service measurements, without an external request queue or concurrency-throughput claim.

Peak payload memory is the sum of retained serialized snapshots, including metadata, scales and uncompressed recurrent state. It is not whole-process RAM or active GPU memory. Each row covers 16 resumed responses: eight questions repeated twice.

### Roomy store

| Policy | Mean pause + resume | Median resume TTFT | Peak paused payload | Hits |
| --- | ---: | ---: | ---: | ---: |
| Recompute | 4.383 s | 4.286 s | 0 MiB | 0/16 |
| Exact native snapshot | 0.382 s | 0.238 s | 379.50 MiB | 16/16 |
| Fixed INT4 | 0.934 s | 0.477 s | 245.14 MiB | 16/16 |
| Adaptive without eviction pricing | 0.396 s | 0.234 s | 379.50 MiB | 16/16 |
| Adaptive with eviction pricing | 0.388 s | 0.237 s | 379.50 MiB | 16/16 |

Both adaptive versions chose native storage for all 16 sessions. Against fixed INT4, eviction-aware adaptive reduced pause-plus-resume time **58.5%**, while increasing peak paused bytes **54.8%**. Mean occupancy after events rose from **122.50 to 189.50 MiB**. A simple fixed native policy achieved the same choices and was slightly faster in this run; these observations do not show an adaptive advantage over native storage or an eviction-pricing benefit in the roomy scenario.

Including the original prefix computation changes the adaptive-versus-INT4 comparison to **4.430 versus 4.986 seconds per session**, an **11.2%** reduction. The larger 58.5% number must not be presented as full-session latency improvement.

### Tight store

| Policy | Mean pause + resume | Median resume TTFT | Peak paused payload | Hits |
| --- | ---: | ---: | ---: | ---: |
| Recompute | 4.318 s | 4.213 s | 0 MiB | 0/16 |
| Exact native snapshot | 3.336 s | 4.171 s | 95.13 MiB | 4/16 |
| Fixed INT4 | 1.908 s | 0.495 s | 183.94 MiB | 12/16 |
| Adaptive without eviction pricing | 3.447 s | 4.239 s | 95.13 MiB | 4/16 |
| Adaptive with eviction pricing | 2.419 s | 2.219 s | 167.39 MiB | 8/16 |

Eviction-aware adaptive chose **4 native snapshots, 4 INT8 snapshots and 8 recomputations**. It retained two sessions in each four-session group and discarded the others. Fixed INT4 retained three, evicting the oldest when the fourth arrived. No previously stored snapshot was recompressed to make room.

Against fixed INT4, adaptive reduced peak retained bytes **9.0%** and mean occupancy after events **19.3%** (114.85 to 92.74 MiB), but increased pause-plus-resume time **26.8%**. Including original prefill, it took **6.435 versus 5.998 seconds per session**, **7.3% slower**. Median TTFT is much worse because half the adaptive resumes require full recomputation; report the misses alongside the fast cache hits.

Charging eviction harm helped relative to the matched `adaptive_exact` arm: pause-plus-resume work fell **29.8%** and hits rose from **4/16 to 8/16**. It also retained more bytes. This ablation supports the usefulness of the eviction charge in this controlled scenario, but it did not make the policy outperform fixed INT4 on latency.

These are descriptive observations from two repetitions on one machine. No statistical superiority, general workload prediction, p95 serving latency or throughput claim is made.

## Memory saving and its limit

Native snapshots occupied **94.54–95.13 MiB** each. Their INT4 equivalents occupied **61.20–61.35 MiB**, including both framing layers, quantization scales and all native recurrent state. The average full-snapshot ratio was **1.547×**, about **35% fewer stored bytes**.

Only 8 of the 32 model layers carry growing attention K/V. The other 24 have **49.125 MiB** of native convolution/recurrent state per session. We preserve that state exactly in every stored format. Quoting the attention-only compression ratio as a whole-state saving would overstate this result.

Every arm reported the same MLX peak active allocation during its workload blocks: **3,803,602,128 bytes**, about **3.54 GiB**. Those counters reset after model loading/calibration. The active model state returns to its original size after decoding. CPU NumPy/Python allocations, transient input/output codec buffers, returned payloads and total process/system RAM were not measured by that counter. Encoding can temporarily hold the old residents, live state and a new payload together; the store's capacity does not cap transient process memory. Mean occupancy after events is not a wall-time memory integral.

## Answer quality

Every policy/scenario scored **14/16**, representing the same **7/8 distinct questions**. No response was truncated. Against the paired recomputation control:

| Policy | Identical generated token sequences per scenario |
| --- | ---: |
| Exact native | 16/16 |
| Adaptive without eviction pricing | 16/16 |
| Adaptive with eviction pricing | 16/16 |
| Fixed INT4 | 14/16 |

The changed case is `relay-retry-total-wait`:

| Expected answer | Recompute/native/adaptive | Fixed INT4 |
| --- | --- | --- |
| `14` | `6` | `10` |

The native computation already failed this task. INT4 changed the wrong answer; it did not introduce an additional exact-match failure in this small fixture. An unchanged score therefore does **not** mean unchanged answers or establish broad quality preservation. No perplexity, coding pass@k, long-context recall or quality noninferiority study was run.

## Evidence and reproduction

- [Frozen protocol](../results/qwen-kvpilot-v1/protocol.json): 15 source hashes, fixture identity, calibrated profiles, arm order and budgets.
- [Calibration and integrity gate](../results/qwen-kvpilot-v1/calibration.json), [all response rows](../results/qwen-kvpilot-v1/rows.jsonl), [block counters](../results/qwen-kvpilot-v1/blocks.json), [summary](../results/qwen-kvpilot-v1/summary.json), and [preserved fixture](../results/qwen-kvpilot-v1/fixture.json).
- [Independent validation](../results/qwen-kvpilot-v1/validation.json): all 160 rows and 20 blocks checked, zero rows filtered. A separate standard-library validator recalculates framing bytes, scores, token agreement, allowed actions, LRU residency, capacity, denominators and all summary cells without importing the benchmark, codec or model code. Snapshot contents were not archived, so this is record/accounting verification rather than independent re-execution of inference.
- [Test results](../results/qwen-kvpilot-v1/tests.xml): **351 tests plus 89 subtests passed**, including real Metal numerical checks and tamper tests for the archive validator.
- [Runner setup and commands](local-runner.md). The earlier [two-case smoke archive](../results/qwen-kvpilot-smoke-v1/README.md) is retained separately and excluded from these totals.

```sh
.venv-local/bin/python scripts/benchmark_local_kvpilot.py \
  --output results/qwen-kvpilot-new-run --repeats 2
.venv/bin/python scripts/validate_local_results.py \
  results/qwen-kvpilot-new-run \
  --output results/qwen-kvpilot-new-run/validation.json
```

The runner supports this fixed single-sequence text architecture and bounded snapshots of roughly 6,600 prefix tokens. It does not establish 128K-context behavior, shared-page gains, cancellation prediction, multi-agent serving concurrency, CUDA/CXL performance, or an advantage over CacheGen/LMCache. Model precision and fixtures also differ from the earlier native Ollama experiment, so its times should not be directly compared to these.

## What this means for the next experiment

We can claim that our own code compresses real attention state, preserves the accompanying recurrent state, releases the live copy, restores it and continues a real LLM locally. We measured smaller paused snapshots and substantially faster resumes than recomputation on this fixture. The speed benefit over recomputation comes from reusing calculations; compression itself adds work, and exact snapshots are faster when space permits.

We cannot yet claim that adaptive KVPilot wins on both memory and latency against fixed INT4, or that compression is quality-neutral in general. A concrete next local test is to allow a resident native/INT8 snapshot to be recompressed when pressure rises, charging the full transcoding cost and transient memory. The present policy chooses each snapshot's format once and cannot revise early expensive choices. Test that change on a newly declared mix of pressure, return and cancellation patterns, keeping fixed INT4 and simple native/threshold policies as controls. This run's results should remain frozen even if the next policy succeeds.
