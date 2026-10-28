# What the local Ollama benchmark measures

Subsequent work added a [direct tensor runner](local-runner.md) that reads the same local weights and gives KVPilot ownership of the state. Its [results](local-results.md) test our codecs and policy. This document describes the earlier Ollama API experiment.

Source review: **2026-10-27**, pinned to **Ollama v0.40.11**. The local benchmark uses `qwen3.5:4b` on the MLX runner, Apple M1 Pro with 16 GB unified memory, with the experiment's context setting kept at 4096. These findings describe that backend and version; they are not claims about every Ollama runner or the upstream Python MLX-LM package.

The supported API can measure native prompt-cache reuse, request timings, generated answers, and aggregate model-memory reporting. It does **not** expose the raw cache or the controls needed to run KVpilot's compression/admission policy inside this runner. Native-cache results therefore establish a reference, not a measured KVpilot improvement.

## Hybrid state and the first-branch miss

Qwen3.5 creates a `RecurrentCache` for linear-attention layers and a `KVCache` for full-attention layers. Its retained state is not a uniform K/V tensor across all layers. The recurrent cache stores cumulative state: restoring a snapshot requires its exact offset, and an existing snapshot cannot be split to recover an earlier state. [Model cache construction, lines 1419–1430](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/model/qwen3_5/qwen3_5.go#L1419); [recurrent restore and split, lines 244–281](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/cache/recurrent.go#L244).

Prefill schedules checkpoints every **8192 tokens** and **four tokens before the end of the prompt**. The ordinary K/V cache's 256-token step is buffer-allocation growth, not recurrent-checkpoint spacing. The pipeline does not schedule a checkpoint at every chat-message boundary. [Prefill scheduling, lines 153–187](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/pipeline.go#L153); [K/V allocation, lines 38–88](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/cache/kvcache.go#L38).

Consequently, an exact repeat of a roughly 2000-token prompt can reuse almost all of it, while a different question after the same document can initially miss most of the document: the shared token prefix may end before any useful recurrent checkpoint. This is an explanation supported by the code, not a guarantee about every tokenization or cache state.

The cache learns a checkpoint at an observed branch point. After restoring the deepest available state, `beginAt` schedules a snapshot at the matched prefix if it had to resume earlier. Thus an **A → B → C** diagnostic is meaningful:

1. A evaluates a fresh document plus setup question.
2. B uses the identical document and a different question; this first divergence can require recomputation and captures the branch checkpoint.
3. C uses another question whose token prefix contains that checkpoint; it can now reuse the document state.

Use one fresh nonce per trio, identical document bytes and delimiters within the trio, and distinct question suffixes. Tokenization can affect the exact shared boundary, so use reported cached-token counts rather than assuming a hit. Keep B's miss in the results. This diagnostic tests an existing native behavior, not a new KVpilot policy. [Prefix-cache invariants, lines 1–23](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/prefix_cache.go#L1); [matching and branch capture, lines 108–145](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/prefix_cache.go#L108).

## Memory budget and measurement limits

The native prefix cache has a fixed **8 GiB eviction threshold for paged-out snapshot bytes** (`8 << 30`), not an 8 GiB bound on the entire runner and not a budget calculated from this computer's 16 GB RAM. Eligible snapshots are evicted oldest first, with depth and size as tie-breakers. The root, current frontier and multi-child branch points are excluded; the loop stops if no eligible victim remains. Therefore the threshold is not an unconditional bound even on that snapshot counter. [Threshold, lines 43–51](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/prefix_cache.go#L43); [eviction, lines 658–707](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/prefix_cache.go#L658).

Weights, active cache arrays, scratch allocations and allocator cache are additional. Here, “paged out” means retained cache snapshots in MLX arrays; it does not establish transfer to SSD. Snapshot accounting can also differ from physical unique bytes: lazy snapshots initially own no extra bytes, and shared backing storage can be charged conservatively. The wired-memory setting limits residency based on loaded active allocations and the device's recommended working set; it is not a total allocation cap. [Snapshot accounting, lines 55–68](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/cache/cache.go#L55); [wired-memory setup, lines 200–232](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/runner.go#L200).

Runner status reads `mlx.ActiveMemory() + mlx.CacheMemory()`. The client returns the same value as total and GPU memory, using a status value that can lag. A `/api/ps` memory sample is therefore an aggregate allocation proxy, not isolated KV bytes or an independent CPU/GPU split. Process RSS is another accounting view and must be labeled separately. Constant aggregate readings do not prove zero cache growth. Existing information logs report per-request MLX `peak` and `held`, also aggregate. [Status memory, lines 73–97](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/server.go#L73); [client memory reporting, lines 623–638](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/client.go#L623); [pipeline logging](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/pipeline.go).

A benchmark stop at 8 GiB of **model-reported aggregate memory** is an experiment's separate guard; it is not the native cache's eviction threshold and cannot guarantee against transient system-memory pressure. Do not fill the native 8 GiB snapshot allowance merely to observe eviction on this 16 GB machine.

## API and quantization boundary

The inspected MLX K/V cache allocates buffers with the incoming keys' and values' dtypes and appends those arrays without a quantized-cache conversion path. Qwen3.5 selects this implementation directly. Public options expose no KV dtype, cache admission, per-cache memory, or raw-state export/import control. `raw` means bypassing prompt formatting; the legacy `context` field is token history, not a raw KV snapshot. Weight quantization is a separate property. [K/V implementation, lines 24–91](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/cache/kvcache.go#L24); [public API types](https://github.com/ollama/ollama/blob/v0.40.11/api/types.go); [runner routes](https://github.com/ollama/ollama/blob/v0.40.11/mlxrunner/server.go).

The general FAQ documents `OLLAMA_KV_CACHE_TYPE`, but the inspected implementation reads that setting in the GGUF/llama-server construction path. It is not a supported per-request switch for this MLX benchmark. Accordingly, these measurements cannot be labeled native INT4-versus-FP16 cache tests. No global environment setting, daemon restart or backend replacement is required for the native-cache reference. [FAQ](https://docs.ollama.com/faq); [llama-server construction](https://github.com/ollama/ollama/blob/v0.40.11/llm/server.go).

## Timing and answer evidence

`prompt_eval_count` is the full logical prompt-token count; `prompt_eval_cached_count` counts tokens read from cache; `prompt_eval_duration` measures evaluation of the uncached prompt. An absent cached-count field is unreported, not automatically zero. Where defined, uncached prefill throughput is `(prompt_eval_count - prompt_eval_cached_count) / prompt_eval_duration_seconds`. Keep model load, prompt evaluation, generation, measured wall time and measured streaming time-to-first-token separate. Server prompt-evaluation time alone is not TTFT. [Generate API metrics](https://docs.ollama.com/api/generate); [cached-token reporting change](https://github.com/ollama/ollama/pull/17943).

Report answer correctness against the case's known answer separately from answer agreement between cache conditions. Two identical incorrect answers establish consistency, not quality. Native cache reuse still generates the answer; this experiment is not evidence of a stored-response cache, lossy compression, a new eviction policy, or isolated KV-memory savings.
