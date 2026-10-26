# Implemented contract and boundaries

This describes the current package, not the larger target architecture in the prototype plan.

## Trace

Input is JSONL, one event at a time, ordered by nonnegative `at_ms`. Equal timestamps are processed in file order. `session_id` identifies one paused association; the same identifier can pause again after a resume/cancel. A second outstanding pause or a resolution without a pause is an error.

```json
{"at_ms":0,"type":"pause","session_id":"s1","pause_kind":"tool","namespace":{"tenant":"t1","model_revision":"weights-and-config@revision","adapter_revision":"none","layout":"kv-layout-and-position-semantics-v1"},"blocks":[{"prefix_hash":"cumulative-token-prefix-hash","raw_bytes":4194304}],"recompute_ms":35}
{"at_ms":100,"type":"resume","session_id":"s1"}
```

`cancel` has the same envelope as `resume`. `pause_kind` defaults to `unknown`; it is an observed category, not a future reuse label. `recompute_ms` is a supplied estimate, not a measurement made by the replay engine.

Compatible physical identity is `(tenant, model_revision, adapter_revision, layout, prefix_hash)`. For real data, the producer must incorporate tokenizer/template, exact preceding tokens and position/attention semantics into these identity fields. The replay does **not** tokenize or compute the hash. Dummy hashes in the synthetic fixture are explicit test identities. Changed tenant/model/adapter/layout isolates blocks; conflicting byte sizes under the same identity are rejected.

Two paused sessions referencing the same compatible block consume its modeled bytes once. LRU can evict paused data; references identify owners and do not pin every paused block indefinitely. A new admission protects its own existing shared prefix while evicting other blocks. Cancellation releases an association and deletes a resident block only when no paused association remains. A missing block at resume causes full recomputation; partial-prefill reuse is not modeled.

## Decisions

The controller uses a Beta(1,1) estimate of resume probability and the mean pause among already resolved sessions of the same kind. Outstanding sessions are not treated as cancellations. This estimator is intentionally simple and may be biased by censoring; it is not survival analysis.

For a newly stored region of `B` raw bytes, representation ratio `r`, supplied bandwidth `W`, encode cost `E`, decode cost `D`, resume probability `p`, predicted pause `T`, and occupancy price `lambda`, it scores roughly:

```text
stored = ceil(B / r)
offload = stored / W
resume = remaining predicted offload wait + shared restore + stored / W + D
score = E + offload + p * resume + lambda * stored_MiB * T_seconds
recompute score = p * full_recompute_ms
```

This is an expected-cost heuristic in milliseconds, with a user-supplied occupancy penalty. It is not a calibrated monetary objective or queueing model. It does not price the harm of evicting another session's pages. That omission explains a failure in the checked-in demo.

The policy first chooses the lowest estimated score satisfying the supplied resume target; if none satisfy it, it chooses the lowest predicted resume delay. A target is not an SLA guarantee. Fixed baselines force a representation if it fits; otherwise they recompute. On actual replayed resume, every policy may choose full recomputation if it is estimated faster than waiting for restoration.

## Modeled resources

Reservation begins at pause. Each stored page has a ready time after its assumed encode and transfer duration. A premature resume can wait or recompute. Encoding and outbound bytes are charged in full even if later canceled/evicted, a conservative accounting choice. The canceled-owner encode metric is an upper bound: some work may still benefit another branch.

Host byte-time counts **paused-store reservations only**, ending at resume/cancel. It excludes unfinished restore/staging buffers. Raw GPU source copies, active KV, compute/transfer queues, overlap interference, disk, network retries, CPU heap/index overhead and partial restoration are absent. Different sessions' assumed operations may overlap without contention. Consequently the resulting p95 is a modeled restore/recompute delay, not measured TTFT.

Codec ratios in the replay config must include intended wire-format overhead; replay does not call the codecs or inspect tensors. It rounds each physical page separately. The example costs are invented sensitivity parameters and clearly labeled `analytical_assumptions`.

## Codec format

The CPU codecs frame shape/dtype metadata with a versioned binary header. Grouped quantization uses FP16 scales on last-axis groups, symmetric signed INT8 or packed INT4 codes. INT4 uses levels -7 through 7; it is not NVFP4. FP16 framing and zlib restore the FP16 input bytes exactly. Quantized paths restore an approximation.

The decoder validates sizes, shape, payload length, numeric limits and decompression bounds, without pickle. It is a reference research format, not an authenticated or production-hardened wire protocol. It does not have a payload checksum, encryption or a model compatibility registry.

## Evidence boundaries

- `cpu_codec_microbenchmark`: actual local CPU encode/decode timing and byte counts. Synthetic attention error uses random queries/values and only key approximation; it is not LLM accuracy.
- `analytical_trace_replay`: outcomes computed from explicit assumptions, with a causal online event stream.
- `live_serving`: reserved for future actual model/engine execution; none exists yet.

No current module imports vLLM, LMCache, Torch or a CUDA kernel. A future integration must use real lifecycle/completion callbacks and immutable, validated records; current correctness tests cannot certify that integration.
