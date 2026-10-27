# Local Qwen experiment: native cache reuse, not KVPilot compression

We used the already installed `qwen3.5:4b` on an Apple M1 Pro MacBook Pro with 16 GB unified memory. Ollama 0.40.11 selected its MLX runner and the Mac's GPU. No rented GPU, model download, backend replacement, global setting change or new runtime dependency was needed.

The model's manifest identifies 4.5B parameters and NVFP4 **weights**. This does not describe KV-cache precision. Qwen3.5 also mixes attention KV with recurrent state, so treating every layer as an ordinary KV cache would be incorrect. The [version-pinned backend review](ollama-backend.md) explains these boundaries.

## The question this experiment can answer

Does native reuse avoid real prefill work on this laptop, and do the generated answers change on a small set of checks? We measure that directly using streaming time to first answer text, server cache counters, and known answers.

It cannot establish that KVPilot's adaptive eviction policy beats fixed INT4, or that our compression saves memory without changing answers. This installed MLX API has no supported raw-state export/import, codec injection or per-request KV precision control. Our NumPy codecs and analytical controller are not in the inference path.

## Protocol

The [fixture](../examples/ollama_quality_cases.json) contains twelve synthetic repository-reading questions sharing one document. It includes configuration precedence, literals, multi-hop lookups, arithmetic and Python behavior. These are development checks from a single fictional repository, not twelve independent production workloads or a hidden quality benchmark.

The [main harness](../scripts/benchmark_ollama.py) runs each question twice, for 24 trials. Each trial has four requests:

1. **Fresh content prefix:** a unique leading nonce makes the document new to the cache; model weights are already resident. Shared template tokens may still hit.
2. **Exact repeat:** immediately repeat the identical full prompt. This is a favorable reuse case and always comes second.
3. **Prime:** with another fresh nonce, read the document and answer a setup question.
4. **First changed question:** use that same document/nonce but ask the target question.

The fresh/repeat block and prime/changed-question block have randomized order. The paired fresh/repeat prompts are identical. The first-changed-question comparison uses a different nonce, so its answer-agreement comparison is weaker. All requests and failed cache hits remain in the archive. A verified-hit timing pair requires at most 5% cached fresh input, at least 80% cached reused input, present TTFT and both model-load durations below 250 ms. Summaries also include every timed pair, regardless of cache hit.

We set context to 4096, output limit to 64, temperature to zero, a fixed seed, and thinking off for short, controlled outputs. This is not an evaluation of the model's default reasoning mode. Actual prompt counts, generated token counts, load times, answers and termination reasons are retained. Exact-match scoring normalizes whitespace only; punctuation, paths, case and extra explanations remain significant. Truncated answers fail.

A model unload/reload and warmup precedes each trial to bound accumulated cache memory. Its time is in `setup.jsonl`, outside the compared resident-model requests. The model is unloaded at completion. This reset protocol deliberately cannot evaluate long-lived multi-session retention, eviction or concurrency. First-token timings are client measurements; they are not server prefill time, decode ITL or complete-answer latency.

The short warmup establishes resident weights, but does not prove that all long-prefill kernels or shapes are warmed. Exact repeats always follow their fresh request, so their observed latency ratio can include kernel warmup and order effects. Cache counters establish reused input and reduced prefill work; the entire timing ratio cannot be attributed exclusively to cache state.

## Results

The [main archive](../results/ollama-qwen35-4b-v2/) completed all **96 planned requests**: 24 fresh-prefix, 24 exact-repeat, 24 setup primes and 24 first-changed-question requests. The three scored conditions each answered **18/24 correctly**, representing **9/12 distinct questions**, repeated twice.

| Condition | Median first answer text | Median full request | Median input reused | Correct |
|---|---:|---:|---:|---:|
| Fresh content prefix | 6.582 s | 6.640 s | 0% | 18/24 |
| Exact repeat | 0.079 s | 0.138 s | 99.80% | 18/24 |
| First changed question after a prime | 6.567 s | 6.614 s | 0.25% | 18/24 |

All 24 exact-repeat pairs met the declared cache/load controls. Their median paired observed first-answer ratio was **82.62×**, with **98.79%** lower time. This is a favorable, sequential native-reuse result subject to the order/kernel caveat above, not an end-to-end application or KVPilot speedup. Fresh prompts had 2000–2031 tokens; exact repeats reused all but four. Median server prefill fell from 6510.8 ms to 39.1 ms. Compared-request model loads were at most 20.1 ms across the three conditions, separate from recorded setup loads.

The first changed question had **0/24 verified cache hits** under the 80% criterion; only 3–6 tokens were reused. Its all-pair median first-answer ratio was 1.003×, effectively no benefit in this run. Those misses remain in both raw records and descriptive summaries.

Answers matched across conditions in all 24 trials, including the same mistakes:

| Question type | Expected | Model output in every main condition/repetition |
|---|---|---|
| Remove all trailing slashes from a path | `/v1/tickets` | `/v1/tickets/` |
| Sum exponential waits between retry attempts | `15` | `4` |
| Use the health-route timeout exception | `250` | `1200` |

There were no observed correct-to-incorrect changes from exact reuse, but a 75% score on twelve correlated tasks is not broad quality preservation. No answer was truncated, and no thinking text was emitted. These checks do not evaluate lossy compression.

The 24 unload/reload/warmup setups took **54.94 s total**, median **2.258 s**. All 96 requests took **481.08 s** of client wall time in total, including the primes. Prime requests themselves had median wall time **6.550 s**. Priming from scratch is work; caching pays when earlier prefill was useful or amortized across later requests. We do not subtract setup work from a claimed total-cost saving.

Measured post-request aggregate allocation ranged from **4,288,288,272 to 4,730,983,952 bytes (3.99–4.41 GiB)** across the main run. This is model-plus-allocator reporting, not KV-only memory. Exact-repeat snapshots did not establish a memory reduction. Final cleanup unloaded the model, and the final model-memory lookup returned no loaded instance.

## Why the first changed question is different

The inspected hybrid-state implementation saves checkpoints at specific positions; it cannot simply rewind recurrent state to any matching token. A changed question can diverge before the useful checkpoint. The engine can then establish a checkpoint at that branch for subsequent questions. This is existing Ollama behavior.

The [branch diagnostic](../scripts/diagnose_ollama_branches.py) therefore runs twelve A/B/C trios with fresh nonces and resets before each trio. A and B have different setup questions; C is the original scored fixture question. It records all three requests, including both setup costs. A cold → B cold → C cached pattern would be consistent with the inspected implementation, not direct instrumentation proving its internal cause. Fixed order and different questions also prevent interpreting this diagnostic as an isolated causal speedup estimate.

The [completed diagnostic](../results/ollama-branches-v1/) observed that pattern in **12/12 trios**:

| Request | Median first answer text | Median reused input | Target quality |
|---|---:|---:|---:|
| A: initial setup question | 6.511 s | 0% | Setup only |
| B: first different setup question | 6.489 s | 0.15% | Setup only |
| C: new scored question after the branch | 0.269 s | 98.26% | 9/12 |

C reused 1973–1983 input tokens. It made the same three kinds of mistakes listed above. Its questions match the main fixture, but its prompts/nonces and preceding cache history differ; this is not a paired cold-target quality test. All 24 setup responses complied with `READY`, which is not counted as substantive task accuracy.

The 36 requests took **161.76 s** total, including A and B. Twelve reset/warmup operations added **26.26 s**, median **2.127 s**. The largest post-request reported allocation was **4,613,379,604 bytes (4.30 GiB)**. Thus the experiment demonstrates fast native reuse once the branch checkpoint exists, with substantial preceding setup work and retained state. It does not demonstrate a memory saving or a KVPilot contribution.

## Memory evidence and the stopped pilot

`/api/ps` reports aggregate MLX allocation, including model weights and allocator cache; it does not isolate KV storage. Total and GPU fields are overlapping views and must not be added. Readings can lag, and the before/after samples are not a peak-memory measurement. We make no KV-only memory-saving claim.

The first unbounded pilot was manually interrupted after 21 completed requests because its last recorded allocation had grown to 7,727,306,256 bytes (7.20 GiB). That version had no automatic memory stop. It was not an OOM or proof that a threshold had been breached. A canceled in-flight request may have incurred unrecorded work. Its [raw archive](../results/ollama-qwen35-4b-v1/) and [abort note](../results/ollama-qwen35-4b-v1/abort-note.json) remain available; its source is commit `a2975c3`. We do not pool it with the reset-based experiment.

The revised harness resets between trials and checks reported allocation against 8 GiB between requests. This guard cannot guarantee a transient peak cap. Native Ollama has a separate snapshot-byte eviction threshold, which does not bound total model memory on this 16 GB machine; see the backend review.

## Reproduce locally

Use a quiet Ollama instance with the model already installed. These commands unload and reload the named model between trials and unload it afterward; use the instance outside other active model work. They do not restart the daemon or change global settings.

```bash
.venv/bin/python scripts/benchmark_ollama.py \
  --model qwen3.5:4b --repeats 2 \
  --output-dir results/local-ollama-reproduction
.venv/bin/python scripts/diagnose_ollama_branches.py \
  --model qwen3.5:4b \
  --output-dir results/local-ollama-branches-reproduction
```

Archives must be new directories. Source/fixture hashes, model digest, requests, setup work and summaries are recorded. New nonces and timing variation mean a reproduction need not be byte-identical. The completed default protocol makes 96 main requests plus 24 model warmups, then 36 diagnostic requests plus twelve warmups. Unload calls and metadata reads are additional. No p95, broad quality noninferiority, production throughput or price-per-token claim is justified by this small sequential experiment.

The [archive validation](../results/ollama-validation-v1.json) records reconstruction of all 132 prompt hashes, recomputation of scores and summaries, matching source hashes, setup costs and failures. Source for both completed runs is commit `56e0edf`. Local verification passed **174 tests and 89 subtests**, including 44 offline harness tests; those tests do not invoke a model. The earlier eviction experiment's nine frozen component hashes remain unchanged. The stopped pilot is separate evidence, not an additional completed replication.

## What would establish a KVPilot benefit

The next missing capability is controlled access to actual saved attention and recurrent state: export, account for its bytes, encode/decode, restore and continue inference. Then compare fixed raw, fixed INT4 and the adaptive policy on the same inputs, including failed restores, setup and transfer work, with appropriate answer checks. An instrumented local runner could make that possible without renting a GPU, but that integration is not implemented here. Native cache speedups are the baseline it must beat.
